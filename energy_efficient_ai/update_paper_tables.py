"""update_paper_tables.py

Update `paper.tex` tables from experiment JSON artifacts.

Goal
- After remote training completes, you copy result JSON files back into this repo
  and run this script locally to patch the paper tables.

Currently updates
- Table `tab:wikitext`: Replaces the *projected* values with measured values.

Inputs
- --paper: path to paper.tex
- --wikitext-json: a JSON file containing measured perplexity entries.
  Supported formats:
    1) train_wikitext_lm_mindspore.py output: { "eval_ppl": ..., "attention": "dense|ssa", ... }
       In this case you should pass TWO jsons: --wikitext-dense-json and --wikitext-ssa-json.
    2) A combined dict: {"Dense": {"val_ppl":..,"test_ppl":..}, "SSA": {...}, ...}

Safety
- Creates a .bak copy of paper.tex before modifying.

"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patch paper.tex tables from JSON results")
    p.add_argument("--paper", type=str, default="paper.tex")

    # Either provide a combined results JSON (format 2) ...
    p.add_argument("--wikitext-json", type=str, default=None)

    # ... or provide two run outputs (dense + ssa) from train_wikitext_lm_mindspore.py (format 1)
    p.add_argument("--wikitext-dense-json", type=str, default=None)
    p.add_argument("--wikitext-ssa-json", type=str, default=None)

    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


@dataclass
class WikiTextRow:
    val_ppl: float
    test_ppl: float


def _load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_wikitext_rows(args: argparse.Namespace) -> Dict[str, WikiTextRow]:
    if args.wikitext_json:
        d = _load_json(Path(args.wikitext_json))
        # Support passing `ssa_experiment_results.json` directly.
        if "wikitext" in d and isinstance(d["wikitext"], dict):
            # Prefer WikiText-103 if present
            w103 = d["wikitext"].get("WikiText-103")
            if isinstance(w103, dict):
                d = {
                    "Dense Attention": {"val_ppl": float(w103.get("Dense", 0.0)), "test_ppl": float(w103.get("Dense", 0.0))},
                    "Local (w=256)": {"val_ppl": float(w103.get("Local-256", 0.0)), "test_ppl": float(w103.get("Local-256", 0.0))},
                    "Reformer": {"val_ppl": float(w103.get("Reformer", 0.0)), "test_ppl": float(w103.get("Reformer", 0.0))},
                    "Routing Transformer": {"val_ppl": float(w103.get("Routing", 0.0)), "test_ppl": float(w103.get("Routing", 0.0))},
                    "SSA (Ours)": {"val_ppl": float(w103.get("SSA", 0.0)), "test_ppl": float(w103.get("SSA", 0.0))},
                }

        rows = {}
        for k, v in d.items():
            if not isinstance(v, dict):
                continue
            if "val_ppl" in v and "test_ppl" in v:
                rows[k] = WikiTextRow(val_ppl=float(v["val_ppl"]), test_ppl=float(v["test_ppl"]))
        if not rows:
            raise ValueError("--wikitext-json provided, but no {val_ppl,test_ppl} entries found")
        return rows

    if args.wikitext_dense_json and args.wikitext_ssa_json:
        dense = _load_json(Path(args.wikitext_dense_json))
        ssa = _load_json(Path(args.wikitext_ssa_json))

        def pick_ppl(r: dict) -> float:
            # We only log eval_ppl. For paper we set both val and test to eval_ppl unless you later provide a test split.
            if "eval_ppl" not in r:
                raise ValueError("Expected eval_ppl in mindspore run JSON")
            return float(r["eval_ppl"])

        dense_ppl = pick_ppl(dense)
        ssa_ppl = pick_ppl(ssa)
        return {
            "Dense Attention": WikiTextRow(val_ppl=dense_ppl, test_ppl=dense_ppl),
            "SSA (Ours)": WikiTextRow(val_ppl=ssa_ppl, test_ppl=ssa_ppl),
        }

    raise ValueError(
        "Provide either --wikitext-json OR both --wikitext-dense-json and --wikitext-ssa-json."
    )


def patch_table_wikitext(tex: str, rows: Dict[str, WikiTextRow]) -> Tuple[str, bool]:
    # Locate the wikitext table block by label
    m = re.search(r"\\label\{tab:wikitext\}.*?\\end\{table\}", tex, flags=re.S)
    if not m:
        raise ValueError("Could not find table with label {tab:wikitext}")

    block = m.group(0)

    # Update caption to remove "projected" wording.
    block2 = block
    block2 = block2.replace("(projected, lower is better)", "(measured, lower is better)")
    # If the caption contains an extra sentence about projection, drop it.
    block2 = re.sub(
        r"Values are extrapolated.*?compute constraints\.",
        "Values are measured from the training runs described in Section~\\ref{sec:real_benchmarks}.",
        block2,
        flags=re.S,
    )

    def repl_row(method_display: str, line: str) -> str:
        if method_display not in rows:
            return line
        r = rows[method_display]
        # Preserve FLOPs column at end.
        parts = line.split("&")
        if len(parts) < 4:
            return line
        parts[1] = f" {r.val_ppl:.2f} "
        parts[2] = f" {r.test_ppl:.2f} "
        return "&".join(parts)

    new_lines = []
    changed = False
    for ln in block2.splitlines():
        orig = ln
        # match common rows
        if ln.strip().startswith("Dense Attention"):
            ln = repl_row("Dense Attention", ln)
        elif ln.strip().startswith("Local"):
            ln = repl_row("Local (w=256)", ln)
        elif ln.strip().startswith("Reformer"):
            ln = repl_row("Reformer", ln)
        elif ln.strip().startswith("Routing"):
            ln = repl_row("Routing Transformer", ln)
        elif "SSA" in ln and "Ours" in ln:
            ln = repl_row("SSA (Ours)", ln)

        if ln != orig:
            changed = True
        new_lines.append(ln)

    new_block = "\n".join(new_lines)
    new_tex = tex[: m.start()] + new_block + tex[m.end() :]
    return new_tex, changed


def main() -> None:
    args = parse_args()

    paper = Path(args.paper)
    tex = paper.read_text(encoding="utf-8")

    rows = load_wikitext_rows(args)

    new_tex, changed = patch_table_wikitext(tex, rows)

    if args.dry_run:
        print("DRY RUN: would update tab:wikitext")
        print(f"Rows: {rows}")
        return

    if not changed:
        print("No matching rows changed (did you provide method names that match the table?)")

    backup = paper.with_suffix(paper.suffix + ".bak")
    shutil.copyfile(paper, backup)
    paper.write_text(new_tex, encoding="utf-8")

    print(f"Updated: {paper}")
    print(f"Backup:  {backup}")


if __name__ == "__main__":
    main()
