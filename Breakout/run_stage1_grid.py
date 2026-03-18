from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Stage 1 grid over horizon and mask ratio.")
    parser.add_argument("--data-dir", type=Path, default=Path("Breakout/data/random"))
    parser.add_argument("--output-root", type=Path, default=Path("Breakout/checkpoints/grid_stage1"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--mask-ratios", type=float, nargs="+", default=[0.2, 0.35, 0.5, 0.65, 0.8])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--diagnostic-rollout-steps",
        type=int,
        default=3,
        help="Rollout depth for drift diagnostics (independent from train-horizon).",
    )
    parser.add_argument("--max-train-batches", type=int, default=40)
    parser.add_argument("--max-val-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--python-bin", type=str, default=".venv/bin/python")
    return parser.parse_args()


def run_one(args: argparse.Namespace, horizon: int, mask_ratio: float) -> dict[str, float | int | str]:
    run_dir = args.output_root / f"h{horizon}_m{mask_ratio:.2f}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_bin,
        "Breakout/train_jepa.py",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(run_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--context-length",
        str(args.context_length),
        "--val-ratio",
        str(args.val_ratio),
        "--train-horizon",
        str(horizon),
        "--rollout-steps",
        str(args.diagnostic_rollout_steps),
        "--mask-ratio",
        str(mask_ratio),
        "--max-train-batches",
        str(args.max_train_batches),
        "--max-val-batches",
        str(args.max_val_batches),
        "--seed",
        str(args.seed),
    ]

    print(f"\n[grid] running horizon={horizon}, mask={mask_ratio:.2f}")
    subprocess.run(cmd, check=True)

    checkpoint_path = run_dir / f"jepa_epoch_{args.epochs:03d}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("metrics", {})

    result = {
        "horizon": horizon,
        "mask_ratio": mask_ratio,
        "train_total_loss": float(metrics.get("train_total_loss", metrics.get("train_jepa_loss", float("nan")))),
        "train_jepa_loss": float(metrics.get("train_jepa_loss", float("nan"))),
        "train_action_sensitivity": float(metrics.get("train_action_sensitivity", float("nan"))),
        "val_total_loss": float(metrics.get("val_total_loss", metrics.get("val_jepa_loss", float("nan")))),
        "val_jepa_loss": float(metrics.get("val_jepa_loss", float("nan"))),
        "val_action_sensitivity": float(metrics.get("val_action_sensitivity", float("nan"))),
        "val_copy_baseline": float(metrics.get("val_copy_baseline", float("nan"))),
        "val_rollout_drift": float(metrics.get("val_rollout_drift", float("nan"))),
        "checkpoint": str(checkpoint_path),
    }
    return result


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, float | int | str]] = []
    for horizon, mask_ratio in itertools.product(args.horizons, args.mask_ratios):
        result = run_one(args, horizon=horizon, mask_ratio=mask_ratio)
        results.append(result)

    csv_path = args.output_root / "grid_results.csv"
    fieldnames = [
        "horizon",
        "mask_ratio",
        "train_total_loss",
        "train_jepa_loss",
        "train_action_sensitivity",
        "val_total_loss",
        "val_jepa_loss",
        "val_action_sensitivity",
        "val_copy_baseline",
        "val_rollout_drift",
        "checkpoint",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    sorted_results = sorted(results, key=lambda r: float(r["val_action_sensitivity"]), reverse=True)

    print("\n=== Grid Summary (top by val_action_sensitivity) ===")
    for row in sorted_results[:10]:
        print(
            "h={horizon}, mask={mask_ratio:.2f}, val_sens={val_action_sensitivity:.6f}, "
            "val_total={val_total_loss:.6f}, val_jepa={val_jepa_loss:.6f}, "
            "val_copy={val_copy_baseline:.6f}, "
            "drift={val_rollout_drift:.6f}".format(**row)
        )

    print(f"\nSaved grid CSV: {csv_path}")


if __name__ == "__main__":
    main()
