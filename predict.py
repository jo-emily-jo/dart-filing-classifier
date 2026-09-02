"""rcept_no 하나를 받아 본문을 크롤링하고 학습된 모델로 라벨을 예측한다."""

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import dart_io

MAX_LENGTH = 512
TEXT_TRUNCATE = 1500


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rcept_no")
    parser.add_argument("--model_dir", default="out/best_model")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    device = pick_device()

    print(f"본문 크롤링 중... ({args.rcept_no})")
    text = dart_io.get_document_text(args.rcept_no)
    if not text:
        print("본문을 가져오지 못했습니다.")
        return
    text = text[:TEXT_TRUNCATE]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = F.softmax(logits, dim=-1).cpu().tolist()

    id2label = model.config.id2label
    ranked = sorted(enumerate(probs), key=lambda x: -x[1])[:args.topk]

    print(f"\n예측 결과 (상위 {args.topk}):")
    for idx, p in ranked:
        label = id2label.get(idx, id2label.get(str(idx), f"id={idx}"))
        print(f"  {label}: {p:.1%}")


if __name__ == "__main__":
    main()
