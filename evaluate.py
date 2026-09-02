"""테스트셋 평가: accuracy, macro-F1, 라벨별 precision/recall, confusion matrix."""

import argparse
import json
import os

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_utils import LABEL2ID, load_jsonl, setup_korean_font
from labels import LABELS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MAX_LENGTH = 512


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict_all(model, tokenizer, texts, device, batch_size=16):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch, truncation=True, max_length=MAX_LENGTH,
                padding=True, return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return preds


def plot_confusion(cm, labels, out_path):
    setup_korean_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("예측")
    ax.set_ylabel("정답")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7,
            )
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_metrics(out_dir, section):
    """out/metrics.json에 section을 병합해서 쓴다(baseline_keyword.py와 공유)."""
    metrics_path = os.path.join(out_dir, "metrics.json")
    existing = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(section)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return metrics_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="out/best_model")
    parser.add_argument("--test", default=os.path.join(DATA_DIR, "test.jsonl"))
    parser.add_argument("--out", default="out")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = pick_device()
    print(f"device: {device}")

    rows = load_jsonl(args.test)
    texts = [r["text"] for r in rows]
    gold = [LABEL2ID[r["label"]] for r in rows]
    label_ids = list(range(len(LABELS)))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)

    preds = predict_all(model, tokenizer, texts, device)

    acc = accuracy_score(gold, preds)
    macro_f1 = f1_score(gold, preds, average="macro", labels=label_ids, zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, preds, labels=label_ids, zero_division=0
    )

    per_label = {
        LABELS[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(LABELS))
    }

    cm = confusion_matrix(gold, preds, labels=label_ids)
    confusion_path = os.path.join(args.out, "confusion.png")
    plot_confusion(cm, LABELS, confusion_path)

    metrics_path = write_metrics(args.out, {
        "model": {
            "model_dir": args.model_dir,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "per_label": per_label,
            "n_test": len(rows),
        }
    })

    print(f"accuracy={acc:.4f} macro_f1={macro_f1:.4f}")
    print(f"저장: {metrics_path}, {confusion_path}")


if __name__ == "__main__":
    main()
