"""LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct(실패 시 Qwen2.5-1.5B-Instruct 폴백)로
공급계약 공시 5건을 "계약금액·상대방·계약기간 3줄 요약"하고, 로드시간·VRAM·
건당 생성시간을 기록한다. 무거운 생성형 LLM 비교라 Colab T4에서 돌리는
용도다(로컬 스모크 대상 아님 — README 참고)."""

import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import load_jsonl

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PRIMARY_MODEL = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_LABEL = "단일판매ㆍ공급계약체결"
N_SAMPLES = 5

PROMPT_TEMPLATE = """다음은 한국 상장사의 공시 본문이다. 계약금액ㆍ상대방ㆍ계약기간을
3줄로 요약해라. 다른 설명은 붙이지 마라.

=== 공시 본문 ===
{text}
=== 본문 끝 ==="""


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def reset_vram_stats(device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def vram_gb(device):
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1e9
        except Exception:
            return None
    return None


def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, device, text):
    prompt = PROMPT_TEMPLATE.format(text=text)
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=200, do_sample=False)
    gen = out[0][input_ids.shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="out")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = pick_device()
    print(f"device: {device}")

    rows = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
    samples = [r for r in rows if r["label"] == TARGET_LABEL][:N_SAMPLES]
    print(f"{TARGET_LABEL} 샘플 {len(samples)}건 선택")

    model_name = PRIMARY_MODEL
    reset_vram_stats(device)
    t0 = time.time()
    try:
        tokenizer, model = load_model(model_name, device)
    except Exception as e:
        print(f"{PRIMARY_MODEL} 로드 실패({e}) -> {FALLBACK_MODEL}로 폴백")
        model_name = FALLBACK_MODEL
        reset_vram_stats(device)
        t0 = time.time()
        tokenizer, model = load_model(model_name, device)
    load_time = time.time() - t0
    load_vram = vram_gb(device)

    vram_str = f"{load_vram:.2f}GB" if load_vram is not None else "측정 불가"
    print(f"모델: {model_name}, 로드시간 {load_time:.1f}초, VRAM {vram_str}")

    lines = [
        "# LLM 요약 비교",
        "",
        f"- 모델: {model_name}",
        f"- device: {device}",
        f"- 로드 시간: {load_time:.1f}초",
        f"- 로드 후 VRAM: {vram_str}",
        "",
        "| rcept_no | 회사명 | 생성시간(초) |",
        "|---|---|---:|",
    ]

    generations = []
    for r in samples:
        t0 = time.time()
        summary = generate(tokenizer, model, device, r["text"])
        gen_time = time.time() - t0
        lines.append(f"| {r['rcept_no']} | {r['corp_name']} | {gen_time:.1f} |")
        generations.append((r, summary))
        print(f"  {r['corp_name']} 생성 {gen_time:.1f}초")

    lines.append("")
    lines.append("## 생성 결과")
    for r, summary in generations:
        lines.append("")
        lines.append(f"### {r['corp_name']} ({r['rcept_no']})")
        lines.append("")
        lines.append(summary)

    out_path = os.path.join(args.out, "llm_compare.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
