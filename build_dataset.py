"""공시 본문 → 공시 유형 분류 데이터셋 생성.

최근 90일 list.json을 7일 단위 윈도우로 순회해 report_nm을 8개 라벨로
정규화하고, 라벨별 최대 250건을 rcept_dt 기준으로 고르게 뽑는다
(같은 회사가 한 라벨의 20%를 넘지 않도록 corp_code당 상한을 둔다).
각 건의 본문 앞부분을 받아 제목이 그대로 노출된 줄을 지우고 1,500자만
저장한다.

이어받기: classifier/data/filings.jsonl에 이미 있는 rcept_no는 본문을
다시 받지 않고 건너뛴다(list.json 재조회는 매번 하지만 비용이 적다).
"""

import json
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta

from tqdm import tqdm

import dart_io
from labels import LABELS, normalize_label

DAYS_LOOKBACK = 90
CHUNK_DAYS = 7
MAX_PER_LABEL = 250
CORP_CAP_RATIO = 0.2  # 라벨당 한 회사가 차지할 수 있는 최대 비율
MIN_TEXT_LEN = 300
TEXT_TRUNCATE = 1500
DOC_FETCH_SLEEP = 0.2

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_JSONL = os.path.join(DATA_DIR, "filings.jsonl")
STATS_MD = os.path.join(DATA_DIR, "stats.md")

_LEADING_BRACKET_RE = re.compile(r"^\[[^\]]*\]\s*")
_WORD_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def load_existing(path):
    """이미 저장된 건들을 rcept_no -> row dict로 로드 (이어받기용)."""
    existing = {}
    if not os.path.exists(path):
        return existing
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            existing[row["rcept_no"]] = row
    return existing


def collect_items(days_lookback, chunk_days):
    """최근 days_lookback일을 chunk_days 단위 윈도우로 나눠 list.json 전체를 모은다."""
    today = datetime.now()
    start = today - timedelta(days=days_lookback)
    windows = dart_io._date_windows(start, today, chunk_days)

    collected = {}  # rcept_no -> item (윈도우 경계 안전을 위한 dedup)
    for w_start, w_end in tqdm(windows, desc="list.json 윈도우"):
        items = dart_io.fetch_list_range(
            w_start.strftime("%Y%m%d"), w_end.strftime("%Y%m%d")
        )
        for it in items:
            collected[it.get("rcept_no", "")] = it
        time.sleep(0.2)
    return list(collected.values())


def group_by_label(items):
    grouped = defaultdict(list)
    for it in items:
        label = normalize_label(it.get("report_nm", ""))
        if label:
            grouped[label].append(it)
    return grouped


def sample_label_items(items, max_count, corp_cap):
    """rcept_dt 기준으로 고르게 뽑되, corp_code당 corp_cap건을 넘기지 않는다."""
    items = sorted(items, key=lambda x: x.get("rcept_dt", ""))
    n = len(items)
    if n == 0:
        return []

    target = min(max_count, n)
    step = n / target
    counts = defaultdict(int)
    selected = []
    selected_ids = set()

    next_target = 0.0
    for i, it in enumerate(items):
        if len(selected) >= max_count:
            break
        if i < next_target:
            continue
        corp = it.get("corp_code", "")
        if counts[corp] < corp_cap:
            selected.append(it)
            selected_ids.add(id(it))
            counts[corp] += 1
            next_target += step
        # corp_cap 초과로 이 슬롯을 못 채웠으면 next_target을 그대로 두고
        # 다음 항목으로 재시도한다.

    # 균등 샘플링에서 corp_cap에 걸려 채우지 못한 잔여 슬롯을 마저 채운다.
    if len(selected) < min(max_count, n):
        for it in items:
            if len(selected) >= max_count:
                break
            if id(it) in selected_ids:
                continue
            corp = it.get("corp_code", "")
            if counts[corp] < corp_cap:
                selected.append(it)
                selected_ids.add(id(it))
                counts[corp] += 1

    return selected


def strip_leaked_title_line(text, report_nm, label=None):
    """본문 첫 줄이 report_nm(제목)이나 라벨을 그대로 노출하고 있으면 그 줄을 제거한다.

    report_nm에는 "단일판매ㆍ공급계약체결(자회사의 주요경영사항)"처럼 본문에는
    안 나오는 부가 괄호가 붙기도 해서, 원제목 전체 / 괄호 앞부분 / 정규화된
    라벨 세 후보 중 하나라도 첫 줄과 겹치면 그 줄을 제거한다.
    """
    if not text:
        return text
    lines = text.split("\n", 1)
    first_line = lines[0]
    rest = lines[1] if len(lines) > 1 else ""

    core = _LEADING_BRACKET_RE.sub("", report_nm).strip()
    pre_paren = re.split(r"[(]", core)[0].strip()
    candidates = [c for c in {core, pre_paren, label or ""} if c]

    for cand in candidates:
        if cand in first_line:
            return rest.strip()
        tokens = [t for t in _WORD_RE.findall(cand) if len(t) >= 2]
        if tokens:
            matched = sum(1 for t in tokens if t in first_line)
            if matched / len(tokens) >= 0.5:
                return rest.strip()

    return text


def build():
    print(f"=== 데이터셋 생성 시작 ({datetime.now():%Y-%m-%d %H:%M}) ===")
    t0 = time.time()

    existing = load_existing(OUT_JSONL)
    print(f"기존 저장분: {len(existing)}건 (이어받기)")

    items = collect_items(DAYS_LOOKBACK, CHUNK_DAYS)
    print(f"list.json 수집 총 {len(items)}건, API 호출 {dart_io.call_count}회")

    grouped = group_by_label(items)
    corp_cap = max(1, math.ceil(MAX_PER_LABEL * CORP_CAP_RATIO))

    stats = {}  # label -> {"saved": n, "total_len": n, "dropped_short": n, "dropped_fetch": n}
    for label in LABELS:
        stats[label] = {"saved": 0, "total_len": 0, "dropped_short": 0, "dropped_fetch": 0}

    out_f = open(OUT_JSONL, "a", encoding="utf-8")
    try:
        for label in LABELS:
            candidates = sample_label_items(grouped.get(label, []), MAX_PER_LABEL, corp_cap)
            for it in tqdm(candidates, desc=label):
                rcept_no = it.get("rcept_no", "")
                if rcept_no in existing:
                    row = existing[rcept_no]
                    stats[label]["saved"] += 1
                    stats[label]["total_len"] += len(row.get("text", ""))
                    continue

                raw_text = dart_io.get_document_text(rcept_no)
                time.sleep(DOC_FETCH_SLEEP)
                if not raw_text:
                    stats[label]["dropped_fetch"] += 1
                    continue

                report_nm = it.get("report_nm", "")
                cleaned = strip_leaked_title_line(raw_text, report_nm, label)
                stored_text = cleaned[:TEXT_TRUNCATE].strip()

                if len(stored_text) < MIN_TEXT_LEN:
                    stats[label]["dropped_short"] += 1
                    continue

                row = {
                    "rcept_no": rcept_no,
                    "corp_name": it.get("corp_name", ""),
                    "rcept_dt": it.get("rcept_dt", ""),
                    "label": label,
                    "text": stored_text,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                existing[rcept_no] = row

                stats[label]["saved"] += 1
                stats[label]["total_len"] += len(stored_text)
    finally:
        out_f.close()

    elapsed = time.time() - t0
    write_stats_md(stats, elapsed, dart_io.call_count)

    print(f"\n=== 완료 ({elapsed:.1f}초, API 호출 {dart_io.call_count}회) ===")
    for label in LABELS:
        s = stats[label]
        avg_len = s["total_len"] / s["saved"] if s["saved"] else 0
        print(f"  {label}: 저장 {s['saved']}건, 평균길이 {avg_len:.0f}자, "
              f"버림(짧음) {s['dropped_short']}건, 버림(원문실패) {s['dropped_fetch']}건")


def write_stats_md(stats, elapsed_sec, api_calls):
    lines = [
        "# 데이터셋 통계",
        "",
        f"- 생성 시각: {datetime.now():%Y-%m-%d %H:%M}",
        f"- 소요 시간: {elapsed_sec:.1f}초",
        f"- DART API 호출: {api_calls}회",
        "",
        "| 라벨 | 저장 건수 | 평균 길이(자) | 버림(300자 미만) | 버림(원문 실패) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        s = stats[label]
        avg_len = s["total_len"] / s["saved"] if s["saved"] else 0
        lines.append(
            f"| {label} | {s['saved']} | {avg_len:.0f} | "
            f"{s['dropped_short']} | {s['dropped_fetch']} |"
        )
    with open(STATS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    build()
