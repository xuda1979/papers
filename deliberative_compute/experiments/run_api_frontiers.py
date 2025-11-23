#!/usr/bin/env python3
"""API-backed evaluation of test-time compute policies.

This script mirrors `run_experiments.py` but uses hosted models via the
OpenAI and Gemini APIs. It measures accuracy, token usage, and latency for
self-consistency (fixed n) and adaptive margin halting, enabling direct
Budget–Performance Frontier comparisons across providers.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from api_clients import GeminiChatModel, OpenAIChatModel

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)


# ---------------- Dataset loaders -----------------
def load_gsm8k(split: str = "test", max_examples: Optional[int] = None):
    ds = load_dataset("gsm8k", "main")[split]

    def to_item(ex):
        return {"id": ex["question"][:64], "question": ex["question"], "answer": ex["answer"]}

    data = [to_item(x) for x in ds]
    return data[:max_examples] if max_examples else data


def load_mmlu(
    subjects: Sequence[str] = ("abstract_algebra", "anatomy"),
    split: str = "validation",
    max_examples: Optional[int] = 200,
):
    ds = load_dataset("hendrycks_test", subject=subjects)
    data = []
    for subj in subjects:
        rows = ds[subj][split]
        for r in rows:
            choices = r["choices"]
            prompt = (
                f"Question: {r['question']}\nChoices: "
                + "; ".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
                + "\nAnswer with the letter only."
            )
            data.append({"id": f"{subj}:{r['question'][:48]}", "question": prompt, "answer": r["answer"]})
    return data[:max_examples] if max_examples else data


# ---------------- Prompting templates -----------------
def prompt_math(q: str) -> str:
    return (
        "You are a careful mathematician. Solve step by step with clear reasoning.\n"
        f"Problem: {q}\n"
        "Reason step by step, then end with: 'Final Answer: <number>'.\n"
    )


def prompt_mc(q: str) -> str:
    return (
        "Answer carefully. Think step by step.\n"
        f"{q}\n"
        "Reason, then end with: 'Final Answer: <A/B/C/D>'.\n"
    )


# ---------------- Extraction helpers -----------------
ANS_PAT = re.compile(r"Final Answer:\s*([^\n]+)", re.IGNORECASE)
LETTER = set(list("ABCD"))


def parse_final_answer(s: str) -> str:
    m = ANS_PAT.search(s)
    if not m:
        m2 = re.findall(r"[-+]?\d+(\.\d+)?", s)
        if m2:
            return m2[-1]
        for ch in reversed(s.strip()):
            if ch.upper() in LETTER:
                return ch.upper()
        return s.strip()[-32:]
    return m.group(1).strip()


def majority_vote(vals: List[str]) -> Tuple[str, float]:
    counts: Dict[str, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.items(), key=lambda x: x[1])
    conf = best[1] / len(vals)
    return best[0], conf


def adaptive_margin_halt(samples: List[str], margin_th: float = 0.6) -> bool:
    _, conf = majority_vote([parse_final_answer(s) for s in samples])
    return conf >= margin_th


# ---------------- Correctness helpers -----------------
def normalize_math_answer(ans: str) -> str:
    try:
        m = re.findall(r"[-+]?\d+(\.\d+)?", ans)
        return m[-1] if m else ans
    except Exception:
        return ans


def is_correct_math(pred: str, gold: str) -> bool:
    return normalize_math_answer(pred) == normalize_math_answer(gold)


def is_correct_mc(pred: str, gold_letter: str) -> bool:
    return pred.strip().upper() == gold_letter.strip().upper()


# ---------------- Provider plumbing -----------------
@dataclass
class Provider:
    name: str
    model: Any

    def generate(self, prompt: str, n: int, temperature: float, max_tokens: int) -> Tuple[List[str], int]:
        raise NotImplementedError


class OpenAIProvider(Provider):
    def generate(self, prompt: str, n: int, temperature: float, max_tokens: int) -> Tuple[List[str], int]:
        res = self.model.generate(prompt, n=n, temperature=temperature, max_tokens=max_tokens)
        return res.texts, res.total_tokens


class GeminiProvider(Provider):
    def generate(self, prompt: str, n: int, temperature: float, max_tokens: int) -> Tuple[List[str], int]:
        res = self.model.generate(prompt, n=n, temperature=temperature, max_tokens=max_tokens)
        return res.texts, res.total_tokens


# ---------------- Experiment loop -----------------
def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    if args.provider == "openai":
        provider = OpenAIProvider("openai", OpenAIChatModel(args.model))
    elif args.provider == "gemini":
        provider = GeminiProvider("gemini", GeminiChatModel(args.model))
    else:
        raise ValueError(f"Unsupported provider: {args.provider}")

    if args.dataset == "gsm8k":
        data = load_gsm8k(split=args.split, max_examples=args.max_examples)
        make_prompt = prompt_math
        correctness = is_correct_math
    elif args.dataset == "mmlu":
        data = load_mmlu(split="validation", max_examples=args.max_examples)
        make_prompt = prompt_mc
        correctness = is_correct_mc
    else:
        raise ValueError("Unsupported dataset")

    policies: List[Tuple[str, Dict[str, Any]]] = []
    if "sc_fixed" in args.policies:
        for n in args.sc_n:
            policies.append(("sc_fixed", {"n": n}))
    if "adaptive_margin" in args.policies:
        for n in args.adaptive_max_n:
            for th in args.margin_th:
                policies.append(("adaptive_margin", {"n": n, "margin_th": th}))

    run_rows: List[Dict[str, Any]] = []
    for pname, pconf in policies:
        print(f"Policy {pname} conf={pconf}")
        acc_count = 0
        token_total = 0
        wall_total = 0.0
        for ex in tqdm(data, ncols=100):
            prompt = make_prompt(ex["question"])
            start = time.time()
            samples: List[str] = []
            tokens_this_ex = 0
            for k in range(1, pconf["n"] + 1):
                texts, tok = provider.generate(
                    prompt=prompt,
                    n=1,
                    temperature=args.temperature,
                    max_tokens=args.max_new_tokens,
                )
                samples += texts
                tokens_this_ex += tok
                if pname == "adaptive_margin" and adaptive_margin_halt(samples, pconf["margin_th"]):
                    break
            elapsed = time.time() - start
            wall_total += elapsed
            token_total += tokens_this_ex

            answers = [parse_final_answer(s) for s in samples]
            pred, conf = majority_vote(answers)
            gold = ex["answer"]
            correct = correctness(pred, gold)
            acc_count += int(correct)

            row = {
                "provider": provider.name,
                "model": args.model,
                "policy": pname,
                "conf": pconf,
                "id": ex["id"],
                "pred": pred,
                "gold": gold,
                "correct": bool(correct),
                "num_samples": len(samples),
                "vote_conf": conf,
                "tokens": tokens_this_ex,
                "wall_time_sec": elapsed,
            }
            run_rows.append(row)

        accuracy = acc_count / max(1, len(data))
        print(
            f"Policy {pname} finished: accuracy={accuracy:.3f}, "
            f"avg_tokens/ex={(token_total/len(data)):.1f}, avg_wall/ex={(wall_total/len(data)):.2f}s"
        )

    out_path = os.path.join(
        args.out_dir,
        f"{args.dataset}_{provider.name}_{args.model.replace('/', '_')}_{int(time.time())}.jsonl",
    )
    with open(out_path, "w", encoding="utf-8") as fw:
        for row in run_rows:
            fw.write(json.dumps(row) + "\n")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "gemini"], required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--dataset", type=str, choices=["gsm8k", "mmlu"], required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_examples", type=int, default=32, help="Limit examples to manage API cost")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--policies", nargs="+", default=["sc_fixed", "adaptive_margin"])
    ap.add_argument("--sc_n", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--adaptive_max_n", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--margin_th", nargs="+", type=float, default=[0.6, 0.7])
    ap.add_argument("--out_dir", type=str, default="results/api_runs")
    args = ap.parse_args()
    run(args)
