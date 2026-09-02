"""dart_alert_bot.py의 제목 키워드 7종을 본문에 적용하는 규칙 베이스라인.

KEYWORDS는 `~/Korexis/dart-alert-bot/dart_alert_bot.py`에서 그대로 복사했다
(이 레포는 dart_alert_bot을 import하지 않는다 — README 참고).

두 가지를 계산한다:
1. 원래 봇처럼 report_nm(제목)에만 걸던 키워드를 여기서는 "본문"에 적용해
   8라벨 분류 규칙 베이스라인으로 쓰면 accuracy/macro-F1이 얼마나 되는지.
2. "제목 키워드 필터"가 실제로 얼마나 놓쳤을지: 기타경영사항(자율공시)·
   주요사항보고서(전체) 라벨은 애초에 report_nm에 이 키워드가 없어서
   normalize_label 단계에서 그 라벨로 분류된 건들이다. 그런데 그 본문
   안에 공급계약 관련 신호(단일판매/공급계약/수주)가 있다면, 제목만 보는
   원래 봇은 이런 건을 완전히 놓쳤을 것이다 — 그 비율을 계산한다.
"""

import json
import os

from sklearn.metrics import accuracy_score, f1_score

from data_utils import LABEL2ID, load_jsonl
from labels import LABELS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# dart_alert_bot.py KEYWORDS를 그대로 복사(순서 동일 — 원본은 순서대로
# 검사하다 첫 매칭에서 멈춘다. break 로직도 동일하게 재현한다.)
KEYWORDS = [
    "단일판매",
    "공급계약",
    "수주",
    "시설투자",
    "신규시설투자",
    "유형자산취득",
    "타법인 주식 및 출자증권 취득",
]

KEYWORD_TO_LABEL = {
    "단일판매": "단일판매ㆍ공급계약체결",
    "공급계약": "단일판매ㆍ공급계약체결",
    "수주": "단일판매ㆍ공급계약체결",
    "시설투자": "신규시설투자등",
    "신규시설투자": "신규시설투자등",
    "유형자산취득": "주요사항보고서(전체)",
    "타법인 주식 및 출자증권 취득": "타법인주식및출자증권취득결정",
}

SUPPLY_CONTRACT_KEYWORDS = ["단일판매", "공급계약", "수주"]

NO_MATCH = "NO_MATCH"


def predict_by_keyword(text):
    for kw in KEYWORDS:
        if kw in text:
            return KEYWORD_TO_LABEL[kw]
    return NO_MATCH


def main():
    rows = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
    gold_labels = [r["label"] for r in rows]
    pred_labels = [predict_by_keyword(r["text"]) for r in rows]

    label_ids = list(range(len(LABELS)))
    gold_ids = [LABEL2ID[l] for l in gold_labels]
    # NO_MATCH는 8개 실제 라벨 어디에도 속하지 않는 더미 id로 매핑한다
    # (sklearn이 labels=label_ids 범위 밖 예측은 자동으로 오답 처리한다).
    pred_ids = [LABEL2ID[l] if l in LABEL2ID else len(LABELS) for l in pred_labels]

    acc = accuracy_score(gold_ids, pred_ids)
    macro_f1 = f1_score(gold_ids, pred_ids, average="macro", labels=label_ids, zero_division=0)

    n_match = sum(1 for l in pred_labels if l != NO_MATCH)
    print(f"본문 키워드 매칭 건수: {n_match}/{len(rows)}")
    print(f"본문 키워드 규칙 베이스라인: accuracy={acc:.4f} macro_f1={macro_f1:.4f}")

    miss_targets = ["기타경영사항(자율공시)", "주요사항보고서(전체)"]
    title_filter_miss = {}
    for label in miss_targets:
        label_rows = [r for r in rows if r["label"] == label]
        if not label_rows:
            title_filter_miss[label] = None
            continue
        n_supply_signal = sum(
            1 for r in label_rows
            if any(kw in r["text"] for kw in SUPPLY_CONTRACT_KEYWORDS)
        )
        ratio = n_supply_signal / len(label_rows)
        title_filter_miss[label] = {
            "n_total": len(label_rows),
            "n_supply_contract_signal_in_body": n_supply_signal,
            "ratio": ratio,
        }
        print(f"  {label}: {n_supply_signal}/{len(label_rows)}건 본문에 공급계약 신호 "
              f"(제목 필터였다면 놓쳤을 비율 {ratio:.1%})")

    result = {
        "keyword_baseline": {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "n_match": n_match,
            "n_total": len(rows),
        },
        "title_keyword_filter_miss": title_filter_miss,
    }

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.json")
    existing = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(result)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"저장: {metrics_path}")


if __name__ == "__main__":
    main()
