"""train/evaluate/baseline/predict 스크립트가 공유하는 데이터 로딩 유틸."""

import json

from labels import LABELS

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def setup_korean_font():
    """matplotlib에 한글 폰트가 있으면 적용(없으면 조용히 넘어감 — 글자가
    네모로 깨져도 학습/평가 로직에는 영향 없음)."""
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    candidates = ("AppleGothic", "NanumGothic", "Malgun Gothic", "NanumBarunGothic")
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in candidates:
        if cand in available:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False
