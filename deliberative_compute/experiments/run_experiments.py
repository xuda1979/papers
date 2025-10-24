#!/usr/bin/env python
import argparse, os, time, json, math, re, random, uuid, statistics
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

try:
    from .nvml_power import EnergyMeter
except ImportError:  # pragma: no cover - script-style invocation
    from nvml_power import EnergyMeter

SEED = 1234
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# -------- Dataset loaders --------
def load_gsm8k(split="test", max_examples=None):
    ds = load_dataset("gsm8k", "main")[split]
    def to_item(ex):
        return {"id": ex["question"][:64], "question": ex["question"], "answer": ex["answer"]}
    data = [to_item(x) for x in ds]
    return data[:max_examples] if max_examples else data

def load_mmlu(subjects=("abstract_algebra","anatomy"), split="validation", max_examples=200):
    # use hendrycks_test; subselect subjects for speed
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

# -------- Prompting templates --------
def prompt_math(q:str)->str:
    return (
        "You are a careful mathematician. Solve step by step with clear reasoning.\n"
        f"Problem: {q}\n"
        "Reason step by step, then end with: 'Final Answer: <number>'.\n"
    )

def prompt_mc(q:str)->str:
    return (
        "Answer carefully. Think step by step.\n"
        f"{q}\n"
        "Reason, then end with: 'Final Answer: <A/B/C/D>'.\n"
    )

# -------- Extraction helpers --------
ANS_PAT = re.compile(r"Final Answer:\s*([^\n]+)", re.IGNORECASE)
LETTER = set(list("ABCD"))

def parse_final_answer(s:str)->str:
    m = ANS_PAT.search(s)
    if not m:
        # fallback: last number or letter
        m2 = re.findall(r"[-+]?\d+(\.\d+)?", s)
        if m2: return m2[-1]
        for ch in reversed(s.strip()):
            if ch.upper() in LETTER: return ch.upper()
        return s.strip()[-32:]
    return m.group(1).strip()

def majority_vote(vals:List[str])->Tuple[str,float]:
    counts={}
    for v in vals: counts[v]=counts.get(v,0)+1
    best = max(counts.items(), key=lambda x:x[1])
    conf = best[1]/len(vals)
    return best[0], conf

# -------- Policies --------
def sc_fixed_generate(model, tokenizer, prompt, n:int, temperature:float, max_new_tokens:int, stop=None):
    inputs = tokenizer([prompt]*n, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, do_sample=True, temperature=temperature,
            max_new_tokens=max_new_tokens, top_p=0.95, num_return_sequences=1
        )
    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    gens = [t[len(prompt):] if t.startswith(prompt) else t for t in texts]
    return gens

def adaptive_margin_halt(samples:List[str], margin_th=0.6)->bool:
    # stop when vote margin exceeds threshold (entropy-like)
    _, conf = majority_vote([parse_final_answer(s) for s in samples])
    return conf >= margin_th

# -------- Evaluation --------
def normalize_math_answer(ans:str)->str:
    # strip units/commas; keep number
    try:
        # last number
        m = re.findall(r"[-+]?\d+(\.\d+)?", ans)
        return m[-1] if m else ans
    except Exception:
        return ans

def is_correct_math(pred:str, gold:str)->bool:
    return normalize_math_answer(pred)==normalize_math_answer(gold)

def is_correct_mc(pred:str, gold_letter:str)->bool:
    return pred.strip().upper()==gold_letter.strip().upper()

# -------- Main runner --------
def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if args.dataset=="gsm8k":
        data = load_gsm8k(split=args.split, max_examples=args.max_examples)
        make_prompt = prompt_math
        correctness = is_correct_math
    elif args.dataset=="mmlu":
        data = load_mmlu(split="validation", max_examples=args.max_examples)
        make_prompt = prompt_mc
        correctness = is_correct_mc
    else:
        raise ValueError("Unsupported dataset")

    # Prepare policy grid
    policies=[]
    if "sc_fixed" in args.policies:
        for n in args.sc_n:
            policies.append(("sc_fixed", {"n":n}))
    if "adaptive_margin" in args.policies:
        for n in args.adaptive_max_n:
            for th in args.margin_th:
                policies.append(("adaptive_margin", {"n":n, "margin_th":th}))

    run_id = uuid.uuid4().hex[:8]
    out_path = os.path.join(args.out_dir, f"{args.dataset}_{run_id}.jsonl")
    fw = open(out_path, "w", encoding="utf-8")

    # Energy meter
    meter = EnergyMeter(enable=args.measure_energy)

    for pid, (pname, pconf) in enumerate(policies):
        print(f"Policy {pname} conf={pconf}")
        acc_count=0; tok_total=0; wall_total=0.0
        energy_total_j = 0.0

        with meter.session() as m:
            for ex in tqdm(data, ncols=100):
                prompt = make_prompt(ex["question"])
                start = time.time()
                samples=[]; gen_tokens=0
                for k in range(1, pconf["n"]+1):
                    gens = sc_fixed_generate(model, tokenizer, prompt, 1, args.temperature, args.max_new_tokens)
                    samples += gens
                    # token counting (approx)
                    gen_tokens += sum(len(tokenizer.encode(g)) for g in gens)
                    if pname=="adaptive_margin" and adaptive_margin_halt(samples, pconf["margin_th"]):
                        break
                elapsed = time.time()-start
                wall_total += elapsed
                tok_total += gen_tokens

                answers = [parse_final_answer(s) for s in samples]
                pred, conf = majority_vote(answers)
                gold = ex["answer"]
                correct = correctness(pred, gold)
                acc_count += int(correct)

                row = {
                    "policy": pname, "conf": pconf, "id": ex["id"],
                    "pred": pred, "gold": gold, "correct": bool(correct),
                    "num_samples": len(samples), "vote_conf": conf,
                    "gen_tokens": gen_tokens,
                    "wall_time_sec": elapsed,
                }
                if m is not None:
                    row["energy_j"] = m.last_step_energy_j()
                fw.write(json.dumps(row)+"\n")
        fw.flush()
    fw.close()
    print(f"Wrote: {out_path}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--dataset", type=str, choices=["gsm8k","mmlu"], required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_examples", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--policies", nargs="+", default=["sc_fixed","adaptive_margin"])
    ap.add_argument("--sc_n", nargs="+", type=int, default=[1,2,4,8,16])
    ap.add_argument("--adaptive_max_n", nargs="+", type=int, default=[8,16])
    ap.add_argument("--margin_th", nargs="+", type=float, default=[0.6,0.7,0.8])
    ap.add_argument("--measure_energy", action="store_true")
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()
    run(args)
