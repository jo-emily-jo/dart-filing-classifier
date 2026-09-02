"""filings.jsonl을 train/val/test = 80/10/10으로 분할한다.

라벨 층화(각 라벨 내부에서 80/10/10 비율 유지) + 그룹 분할(같은 corp_code가
train/val/test 중 한 곳에만 속하도록) 두 조건을 동시에 만족시켜야 해서,
개별 row가 아니라 (label, corp_code) 그룹 단위로 먼저 배분한다.
"""

import json
import os
import random
from collections import defaultdict

IN_JSONL = os.path.join(os.path.dirname(__file__), "data", "filings.jsonl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
SPLIT_STATS_MD = os.path.join(OUT_DIR, "split_stats.md")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def corp_key(row):
    """corp_code가 없는 옛 데이터 대비: rcept_no 앞 8자리(회사 식별 대용 X) 대신
    corp_name을 그룹 키로 쓴다(동일 회사가 두 split에 걸치지 않게 하는 목적에는
    회사명 단위로도 충분하다)."""
    return row.get("corp_code") or row["corp_name"]


def split_rows(rows, seed=SEED):
    """라벨 층화 + corp 그룹 분할로 rows를 train/val/test로 나눈다.

    같은 회사가 여러 라벨에 걸쳐 공시를 낸 경우가 흔해서(예: 대기업이
    유상증자결정과 타법인주식취득결정을 둘 다 냄), 라벨 안에서만 그룹을
    나누면 "회사 단위로는" 여전히 train/val/test에 걸칠 수 있다. 그래서
    corp 그룹은 전체 라벨을 통틀어 한 곳에만 배정하고, 라벨별 80/10/10
    비율은 각 그룹을 "지금 가장 부족한 split"에 배정하는 그리디 방식으로
    맞춘다(정확히 80/10/10은 아니어도 근사).
    """
    rng = random.Random(seed)

    groups = defaultdict(list)
    for r in rows:
        groups[corp_key(r)].append(r)

    labels = sorted({r["label"] for r in rows})
    total_by_label = defaultdict(int)
    for r in rows:
        total_by_label[r["label"]] += 1

    target = {
        label: {
            "train": total_by_label[label] * TRAIN_RATIO,
            "val": total_by_label[label] * VAL_RATIO,
            "test": total_by_label[label] * TEST_RATIO,
        }
        for label in labels
    }
    assigned = {label: {"train": 0, "val": 0, "test": 0} for label in labels}

    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda kv: -len(kv[1]))  # 큰 그룹부터 배정해야 오차가 줄어듦

    out = {"train": [], "val": [], "test": []}
    for _, group_rows in group_items:
        label_counts = defaultdict(int)
        for r in group_rows:
            label_counts[r["label"]] += 1

        best_split, best_deficit = None, None
        for split in ("train", "val", "test"):
            deficit = sum(
                target[label][split] - assigned[label][split]
                for label in label_counts
            )
            if best_deficit is None or deficit > best_deficit:
                best_deficit, best_split = deficit, split

        out[best_split].extend(group_rows)
        for label, cnt in label_counts.items():
            assigned[label][best_split] += cnt

    return out["train"], out["val"], out["test"]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def check_group_leak(train, val, test):
    """같은 corp가 두 split 이상에 걸쳐 있는지 확인 (있으면 버그)."""
    train_corps = {corp_key(r) for r in train}
    val_corps = {corp_key(r) for r in val}
    test_corps = {corp_key(r) for r in test}
    return (
        train_corps & val_corps,
        train_corps & test_corps,
        val_corps & test_corps,
    )


def write_stats_md(train, val, test, path):
    labels = sorted({r["label"] for r in (train + val + test)})
    lines = [
        "# Split 통계",
        "",
        f"- train: {len(train)}건, val: {len(val)}건, test: {len(test)}건",
        "",
        "| 라벨 | train | val | test |",
        "|---|---:|---:|---:|",
    ]
    for label in labels:
        n_train = sum(1 for r in train if r["label"] == label)
        n_val = sum(1 for r in val if r["label"] == label)
        n_test = sum(1 for r in test if r["label"] == label)
        lines.append(f"| {label} | {n_train} | {n_val} | {n_test} |")

    leak_train_val, leak_train_test, leak_val_test = check_group_leak(train, val, test)
    lines += [
        "",
        f"- corp 그룹 누출 (train∩val): {len(leak_train_val)}건",
        f"- corp 그룹 누출 (train∩test): {len(leak_train_test)}건",
        f"- corp 그룹 누출 (val∩test): {len(leak_val_test)}건",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    rows = load_rows(IN_JSONL)
    print(f"입력: {len(rows)}건")

    train, val, test = split_rows(rows)
    print(f"분할: train {len(train)} / val {len(val)} / test {len(test)}")

    write_jsonl(os.path.join(OUT_DIR, "train.jsonl"), train)
    write_jsonl(os.path.join(OUT_DIR, "val.jsonl"), val)
    write_jsonl(os.path.join(OUT_DIR, "test.jsonl"), test)
    write_stats_md(train, val, test, SPLIT_STATS_MD)

    leak_train_val, leak_train_test, leak_val_test = check_group_leak(train, val, test)
    if leak_train_val or leak_train_test or leak_val_test:
        print("⚠ corp 그룹 누출 발견 — split_stats.md 확인 필요")
    else:
        print("✅ corp 그룹 누출 없음")


if __name__ == "__main__":
    main()
