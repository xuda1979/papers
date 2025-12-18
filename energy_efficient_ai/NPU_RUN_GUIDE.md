# NPU Run Guide (Ascend 910B)

This repo already contains the full synthetic SSA validation suite (`experiments_final.py`).
The paper still has a few *projected* results that require **real training / device measurements**:

- WikiText-103 perplexity (Table `tab:wikitext`)
- Long Range Arena accuracy (Table `tab:lra`) — full LRA pipeline not included here
- Real device energy/power (Table `tab:energy`)

This guide focuses on what you can run remotely with `--npu`.

> Important: your remote session in the log is running **bash** (Linux), not PowerShell.
> PowerShell line-continuations (backticks) will fail in bash.

## 0) What should work locally vs remotely

- Local laptop/desktop (no MindSpore):
  - `experiments_final.py` (CPU) ✅
  - `train_wikitext_lm.py --quick` (CPU synthetic) ✅

- Remote Ascend (MindSpore installed):
  - `experiments_final.py --npu` ✅
  - `train_wikitext_lm_mindspore.py --npu` ✅ (real perplexity, no external deps)

## 1) Run the synthetic SSA suite on Ascend

### Linux / bash

```bash
python experiments_final.py --npu --output-dir ./npu_results
```

### Windows / PowerShell (optional)

```powershell
python .\experiments_final.py --npu --output-dir .\npu_results
```

Outputs:
- `npu_results\ssa_experiment_results.json`

## 2) WikiText-style perplexity on Ascend (real run)

### Option A: if you already have WikiText-103 text files
Prepare two plain-text files:
- Train text: `wikitext103_train.txt`
- Valid text: `wikitext103_valid.txt`

Then run:

### Linux / bash

```bash
python train_wikitext_lm_mindspore.py --npu \
  --data-file /path/to/wikitext103_train.txt \
  --eval-file /path/to/wikitext103_valid.txt \
  --output-dir ./npu_results \
  --steps 50000 \
  --seq-len 1024 \
  --batch-size 8 \
  --attention ssa
```

### Windows / PowerShell (optional)

```powershell
python .\train_wikitext_lm_mindspore.py --npu `
  --data-file D:\data\wikitext103_train.txt `
  --eval-file D:\data\wikitext103_valid.txt `
  --output-dir .\npu_results `
  --steps 50000 `
  --seq-len 1024 `
  --batch-size 8 `
  --attention ssa
```

For a dense baseline:

### Linux / bash

```bash
python train_wikitext_lm_mindspore.py --npu \
  --data-file /path/to/wikitext103_train.txt \
  --eval-file /path/to/wikitext103_valid.txt \
  --output-dir ./npu_results \
  --steps 50000 \
  --seq-len 1024 \
  --batch-size 8 \
  --attention dense
```

Outputs:
- `npu_results\wikitext_ms_results.json`

### Option B: no dataset yet (sanity run)
If you omit `--data-file`, the script trains on synthetic token streams (still produces a perplexity number):

```bash
python train_wikitext_lm_mindspore.py --npu --output-dir ./npu_results --steps 2000 --seq-len 512
```

## 2.1) If you get: `No module named 'mindspore'`

That means your current remote Python environment doesn't have MindSpore installed.

On many Ascend clusters, the recommended fix is to run inside a prebuilt MindSpore+Ascend container
or load a module provided by your admins.

Because installation is environment-specific, this repo doesn't auto-install MindSpore.
Ask your cluster admin for the correct option (container/module/venv) for Ascend 910B.

## 3) Energy measurement on Ascend

Actual energy measurement is cluster/site-specific.

Recommended approach:
- Run the same workload twice (Dense vs SSA)
- Collect average power (W) from your site tool
- Compute joules: $E = P_{avg} \times t$

If your environment exposes a command that prints current power in watts (example placeholder: `npu-smi --show-power`), you can:
- run that command before and after a run
- compute average power

The helper module `energy_probe.py` is provided as a hook point.

## 4) Updating the paper

Once you have `wikitext_ms_results.json` and (optionally) your power numbers, you can:
- paste the measured perplexity into Table `tab:wikitext`
- paste the measured energy numbers into Table `tab:energy`

If you want, I can also add a small `update_paper_tables.py` script to automatically patch `paper.tex` based on JSON outputs.
