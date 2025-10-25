#!/usr/bin/env python
import argparse
import csv
import glob
import json
import os


def load_rows(pattern):
    rows = []
    for path in glob.glob(pattern):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def summarize(rows):
    by_key = {}
    for r in rows:
        key = (r["policy"], str(r["conf"]))
        by_key.setdefault(key, []).append(r)
    out = []
    for key, group in by_key.items():
        n = len(group)
        acc = sum(g["correct"] for g in group) / max(1, n)
        toks = sum(g["gen_tokens"] for g in group) / max(1, n)
        wall = sum(g["wall_time_sec"] for g in group) / max(1, n)
        energy = None
        if all("energy_j" in g for g in group):
            energy = sum(g["energy_j"] for g in group) / max(1, n)
        out.append(
            {
                "policy": key[0],
                "conf": key[1],
                "n_examples": n,
                "accuracy": acc,
                "avg_gen_tokens": toks,
                "avg_wall_time_sec": wall,
                "avg_energy_j": energy,
            }
        )
    return sorted(out, key=lambda r: (r["policy"], -r["accuracy"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, required=True, help="e.g., results/gsm8k_r1d8b/*.jsonl")
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()
    rows = load_rows(args.glob)
    summary_rows = summarize(rows)
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        "policy",
        "conf",
        "n_examples",
        "accuracy",
        "avg_gen_tokens",
        "avg_wall_time_sec",
        "avg_energy_j",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            cleaned = {
                key: ("" if row[key] is None else row[key])
                for key in fieldnames
            }
            writer.writerow(cleaned)
    print("Wrote:", args.out_csv)
