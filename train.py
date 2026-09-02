"""klue/roberta-base 기반 공시 유형(8라벨) 분류기 파인튜닝.

Colab T4(cuda)에서 정식 학습하고, 로컬 MacBook Air(mps)에서는 --smoke로
32건·1epoch만 돌려 코드 경로가 도는지만 확인한다. device는
cuda > mps > cpu 순으로 자동 선택하고, cuda면 GPU 이름/VRAM을 로그에 남긴다.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data_utils import ID2LABEL, LABEL2ID, load_jsonl, setup_korean_font
from labels import LABELS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MAX_LENGTH = 512
SMOKE_N = 32


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def log_device(device):
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"device: cuda ({name}, {vram_gb:.1f}GB VRAM)")
    else:
        print(f"device: {device}")


def build_dataset_dict(smoke, seed):
    from datasets import Dataset, DatasetDict

    splits = {}
    for name in ("train", "val", "test"):
        rows = load_jsonl(os.path.join(DATA_DIR, f"{name}.jsonl"))
        if smoke:
            rng = random.Random(seed)
            rng.shuffle(rows)
            rows = rows[:SMOKE_N]
        for r in rows:
            r["labels"] = LABEL2ID[r["label"]]
        splits[name] = Dataset.from_list(rows)
    return DatasetDict(splits)


def make_compute_metrics():
    # 이 레포의 평가 스크립트 이름이 evaluate.py라서, 같은 디렉터리에서
    # `import evaluate`를 하면 pip 패키지 대신 evaluate.py 자신이 잡힌다
    # (스크립트 실행 시 그 디렉터리가 sys.path[0]이 되기 때문). 그래서
    # HF `evaluate` 라이브러리 대신 sklearn으로 동일한 지표를 계산한다.
    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        return {"accuracy": acc, "f1": f1}

    return compute_metrics


def plot_curves(log_history, out_path):
    setup_korean_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_epochs, train_losses = [], []
    eval_epochs, eval_losses, eval_f1s = [], [], []
    for entry in log_history:
        if "loss" in entry and "epoch" in entry and "eval_loss" not in entry:
            train_epochs.append(entry["epoch"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_epochs.append(entry["epoch"])
            eval_losses.append(entry["eval_loss"])
            eval_f1s.append(entry.get("eval_f1"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(train_epochs, train_losses, marker="o", label="train loss")
    ax1.plot(eval_epochs, eval_losses, marker="o", label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.set_title("Loss")

    ax2.plot(eval_epochs, eval_f1s, marker="o", color="green", label="val macro-F1")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("macro-F1")
    ax2.legend()
    ax2.set_title("Validation macro-F1")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="klue/roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                         help="32건·1epoch만 돌려 코드가 도는지 확인 (로컬 mps/cpu용)")
    parser.add_argument("--out", default="out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device()
    log_device(device)

    fp16 = args.fp16
    if fp16 and device != "cuda":
        print("fp16은 cuda에서만 지원돼 무시합니다.")
        fp16 = False

    os.makedirs(args.out, exist_ok=True)

    print("데이터 로딩 중...")
    dsd = build_dataset_dict(args.smoke, args.seed)
    for name in ("train", "val", "test"):
        print(f"  {name}: {len(dsd[name])}건")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    dsd = dsd.map(tokenize_fn, batched=True)
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    for name in dsd:
        drop_cols = [c for c in dsd[name].column_names if c not in keep_cols]
        dsd[name] = dsd[name].remove_columns(drop_cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    epochs = 1 if args.smoke else args.epochs
    training_args = TrainingArguments(
        output_dir=os.path.join(args.out, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=fp16,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["val"],
        data_collator=collator,
        compute_metrics=make_compute_metrics(),
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"학습 시간: {elapsed:.1f}초")

    best_dir = os.path.join(args.out, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"best 체크포인트 저장: {best_dir}")

    curves_path = os.path.join(args.out, "curves.png")
    plot_curves(trainer.state.log_history, curves_path)
    print(f"학습 곡선 저장: {curves_path}")

    with open(os.path.join(args.out, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "epochs": epochs,
            "lr": args.lr,
            "batch": args.batch,
            "fp16": fp16,
            "smoke": args.smoke,
            "device": device,
            "train_seconds": elapsed,
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
