"""Quick script to print a per-epoch metrics table for a Stage 1 checkpoint directory."""
import sys
from pathlib import Path
import torch

ckpt_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Breakout/checkpoints/stage1_dim256")
checkpoints = sorted(ckpt_dir.glob("jepa_epoch_*.pt"))

header = f"{'Ep':>3}  {'tr_jepa':>10}  {'tr_sens':>10}  {'tr_copy':>10}  {'vl_jepa':>10}  {'vl_sens':>10}  {'vl_total':>10}  {'vl_drift':>9}"
print(header)
print("-" * len(header))
for ckpt_path in checkpoints:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = ckpt["metrics"]
    ep = ckpt["epoch"]
    print(
        f"{ep:>3}  "
        f"{m['train_jepa_loss']:>10.6f}  "
        f"{m['train_action_sensitivity']:>10.6f}  "
        f"{m['train_copy_baseline']:>10.6f}  "
        f"{m['val_jepa_loss']:>10.6f}  "
        f"{m['val_action_sensitivity']:>10.6f}  "
        f"{m['val_total_loss']:>10.6f}  "
        f"{m['val_rollout_drift']:>9.4f}"
    )
