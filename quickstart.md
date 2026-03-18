# Quickstart: From Dataset Generation to Current Stage 1 Results

This guide walks through the exact workflow from creating the random Breakout training set up to the latest Stage 1 artifacts (training run, full grid rerun, latent-dim sweep rerun, and regenerated charts).

## 1) Environment Setup

From the repository root:

```bash
uv venv .venv
uv pip install gymnasium ale-py autorom torch torchvision pillow matplotlib numpy opencv-python imageio
.venv/bin/AutoROM --accept-license
```

Notes:
- If your environment already exists, you can skip this section.
- If OpenCV is unavailable, data collection still works, but debug video generation may be limited.

## 2) Generate the Random Breakout Dataset

Create the random offline transition dataset (50k transitions, shard size 5000):

```bash
.venv/bin/python Breakout/collect_random_data.py \
  --num-steps 50000 \
  --shard-size 5000 \
  --output-dir Breakout/data/random \
  --video-path Breakout/data/random/random_debug.mp4
```

Expected dataset artifacts in Breakout/data/random:
- transitions_00000.npz, transitions_00001.npz, ...
- run_summary.json
- random_debug.mp4

Optional richer sidecars (OCAtari object + RAM decode):

```bash
.venv/bin/python Breakout/collect_random_data.py \
  --num-steps 50000 \
  --shard-size 5000 \
  --output-dir Breakout/data/random \
  --use-ocatari \
  --ocatari-mode ram \
  --ocatari-hud \
  --video-path Breakout/data/random/random_debug.mp4
```

## 3) Stage 1 JEPA Training (Current Run)

Run Stage 1 with the currently documented main settings:
- train_horizon=1
- mask_ratio=0.7
- latent_dim=256 (training run setting used before latest latent sweep recommendation)

```bash
.venv/bin/python Breakout/train_jepa.py \
  --data-dir Breakout/data/random \
  --output-dir Breakout/checkpoints/stage1_run_20260318 \
  --epochs 8 \
  --batch-size 256 \
  --context-length 4 \
  --latent-dim 256 \
  --train-horizon 1 \
  --mask-ratio 0.7 \
  --val-ratio 0.1
```

Expected outputs:
- Breakout/checkpoints/stage1_run_20260318/jepa_epoch_001.pt ... jepa_epoch_008.pt

## 4) Full Horizon x Mask Grid Search (Rerun)

Run the full Stage 1 grid exactly as described in Stage1.md:

```bash
.venv/bin/python Breakout/run_stage1_grid.py \
  --data-dir Breakout/data/random \
  --output-root Breakout/checkpoints/grid_stage1_full_20260318 \
  --horizons 1 2 3 4 5 6 \
  --mask-ratios 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 \
  --epochs 1 \
  --batch-size 256 \
  --context-length 4 \
  --val-ratio 0.1 \
  --diagnostic-rollout-steps 3 \
  --max-train-batches 40 \
  --max-val-batches 12 \
  --seed 7 \
  --python-bin .venv/bin/python
```

Expected outputs:
- Breakout/checkpoints/grid_stage1_full_20260318/grid_results.csv
- Breakout/checkpoints/grid_stage1_full_20260318/h*_m*/jepa_epoch_001.pt

## 5) Regenerate Stage 1 Charts From Current Artifacts

Use the updated visualization script with explicit inputs:

```bash
.venv/bin/python Breakout/generate_stage1_visualizations.py \
  --screenshots-dir screenshots \
  --epoch-ckpt-dir Breakout/checkpoints/stage1_run_20260318 \
  --grid-csv Breakout/checkpoints/grid_stage1_full_20260318/grid_results.csv
```

Expected refreshed images:
- screenshots/stage1_loss_train_val.png
- screenshots/stage1_action_sensitivity.png
- screenshots/stage1_copy_baseline_gap.png
- screenshots/stage1_rollout_drift.png
- screenshots/stage1_grid_val_sensitivity_heatmap.png
- screenshots/stage1_grid_val_loss_heatmap.png
- screenshots/stage1_pareto_loss_vs_sensitivity.png

## 6) Two-Stage Latent-Dimension Sweep (Rerun)

Run the coarse + refine sweep with multithreading:

```bash
.venv/bin/python Breakout/run_latent_dim_sweep.py \
  --data-dir Breakout/data/random \
  --output-root Breakout/checkpoints/latent_dim_sweep_20260318 \
  --train-horizon 1 \
  --mask-ratio 0.7 \
  --context-length 4 \
  --batch-size 256 \
  --epochs 1 \
  --val-ratio 0.1 \
  --max-train-batches 40 \
  --max-val-batches 12 \
  --diagnostic-rollout-steps 3 \
  --seed 7 \
  --python-bin .venv/bin/python \
  --coarse-dims 128 256 384 512 768 1024 \
  --refine-step 64 \
  --refine-radius 2 \
  --workers 2 \
  --screenshots-dir screenshots \
  --pareto-chart-name stage1_latent_dim_pareto_knee.png
```

Expected outputs:
- Breakout/checkpoints/latent_dim_sweep_20260318/coarse_results.csv
- Breakout/checkpoints/latent_dim_sweep_20260318/refine_results.csv
- Breakout/checkpoints/latent_dim_sweep_20260318/all_results.csv
- Breakout/checkpoints/latent_dim_sweep_20260318/summary.txt
- screenshots/stage1_latent_dim_pareto_knee.png

Current rerun recommendation from summary.txt:
- recommended_knee_latent_dim=192
- recommended_val_total_loss=0.005307
- recommended_val_action_sensitivity=0.020172

## 7) Quick Verification Commands

Check that key outputs exist:

```bash
ls Breakout/checkpoints/stage1_run_20260318
ls Breakout/checkpoints/grid_stage1_full_20260318
ls Breakout/checkpoints/latent_dim_sweep_20260318
ls -lt screenshots/stage1_*.png
```

## 8) Where the Report Was Updated

The latest run outputs are documented in:
- Stage1.md

It now references:
- grid results at Breakout/checkpoints/grid_stage1_full_20260318/grid_results.csv
- latent sweep outputs at Breakout/checkpoints/latent_dim_sweep_20260318/*
- updated latent-dim recommendation (192)

## 9) Optional Next Step
## 9) Common Failure Fixes

### A) ALE/ROM errors (Breakout cannot start)

Symptoms:
- "ROM not found"
- env creation fails for ALE/Breakout-v5

Fix:

```bash
.venv/bin/AutoROM --accept-license
```

Then re-run a quick smoke test:

```bash
.venv/bin/python Breakout/random_agent.py --steps 200
```

### B) Missing Python packages

Symptoms:
- ModuleNotFoundError for gymnasium, ale_py, torch, matplotlib, cv2, imageio, or ocatari

Fix base stack:

```bash
uv pip install gymnasium ale-py autorom torch torchvision pillow matplotlib numpy opencv-python imageio
```

Optional OCAtari stack (only if you use --use-ocatari):

```bash
uv pip install ocatari
```

### C) CUDA / GPU not available

Symptoms:
- Torch CUDA errors
- Slow or failing startup on GPU

Fix:
- Force CPU by adding `--device cpu` to training commands.

Example:

```bash
.venv/bin/python Breakout/train_jepa.py \
  --data-dir Breakout/data/random \
  --output-dir Breakout/checkpoints/stage1_run_cpu \
  --epochs 1 \
  --batch-size 64 \
  --context-length 4 \
  --latent-dim 192 \
  --train-horizon 1 \
  --mask-ratio 0.7 \
  --val-ratio 0.1 \
  --device cpu
```

### D) Out-of-memory during training

Symptoms:
- OOM kill
- CUDA out of memory

Fix options:
- Reduce `--batch-size` (256 -> 128 or 64)
- Reduce `--max-train-batches` / `--max-val-batches` for faster debug loops
- Use `--device cpu` if GPU memory is too small

### E) Disk usage grows too much

Symptoms:
- No space left on device

High-volume folders:
- Breakout/checkpoints/*
- Breakout/data/random/*
- screenshots/*

Quick checks:

```bash
du -sh Breakout/checkpoints Breakout/data screenshots
df -h .
```

### F) Reproducibility mismatch (numbers differ from report)

Checklist:
- Use the same seed (`--seed 7`)
- Keep `--frameskip 4` and repeat-action probability 0.0 during data generation
- Use the exact command blocks from sections 3-6
- Verify you are pointing to the intended output roots (dated folders vs legacy folders)
