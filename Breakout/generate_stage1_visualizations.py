from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path("/mnt/sdc/ArcadeJepa")
SCREENSHOTS = ROOT / "screenshots"
EPOCH_CKPT_DIR = ROOT / "Breakout/checkpoints/stage1_viz_epochs"
GRID_CSV = ROOT / "Breakout/checkpoints/grid_stage1_full/grid_results.csv"


def load_epoch_metrics(checkpoint_dir: Path) -> list[dict]:
    checkpoints = sorted(checkpoint_dir.glob("jepa_epoch_*.pt"))
    metrics_rows: list[dict] = []
    for ckpt_path in checkpoints:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        epoch = int(checkpoint.get("epoch", 0))
        metrics = checkpoint.get("metrics", {})
        row = {"epoch": epoch}
        row.update(metrics)
        metrics_rows.append(row)
    metrics_rows.sort(key=lambda r: r["epoch"])
    return metrics_rows


def load_grid_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for key, value in row.items():
                if key == "checkpoint":
                    converted[key] = value
                else:
                    converted[key] = float(value)
            rows.append(converted)
    return rows


def save_loss_train_val(rows: list[dict]) -> None:
    epochs = [r["epoch"] for r in rows]
    train_total = [r["train_total_loss"] for r in rows]
    val_total = [r["val_total_loss"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_total, marker="o", label="Train Total Loss")
    plt.plot(epochs, val_total, marker="o", label="Val Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stage 1: Train vs Val Total Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SCREENSHOTS / "stage1_loss_train_val.png", dpi=150)
    plt.close()


def save_loss_components(rows: list[dict]) -> None:
    epochs = [r["epoch"] for r in rows]
    train_jepa = [r["train_jepa_loss"] for r in rows]
    train_reward = [r["train_reward_loss"] for r in rows]
    val_jepa = [r["val_jepa_loss"] for r in rows]
    val_reward = [r["val_reward_loss"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_jepa, marker="o", label="Train JEPA Loss")
    plt.plot(epochs, train_reward, marker="o", label="Train Reward Loss")
    plt.plot(epochs, val_jepa, marker="o", linestyle="--", label="Val JEPA Loss")
    plt.plot(epochs, val_reward, marker="o", linestyle="--", label="Val Reward Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stage 1: Loss Components")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SCREENSHOTS / "stage1_loss_components.png", dpi=150)
    plt.close()


def save_action_sensitivity(rows: list[dict]) -> None:
    epochs = [r["epoch"] for r in rows]
    train_sens = [r["train_action_sensitivity"] for r in rows]
    val_sens = [r["val_action_sensitivity"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_sens, marker="o", label="Train Action Sensitivity")
    plt.plot(epochs, val_sens, marker="o", label="Val Action Sensitivity")
    plt.xlabel("Epoch")
    plt.ylabel("Mean |Δ latent|")
    plt.title("Stage 1: Action Sensitivity")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SCREENSHOTS / "stage1_action_sensitivity.png", dpi=150)
    plt.close()


def save_copy_baseline_gap(rows: list[dict]) -> None:
    epochs = [r["epoch"] for r in rows]
    train_copy = [r["train_copy_baseline"] for r in rows]
    train_jepa = [r["train_jepa_loss"] for r in rows]
    val_copy = [r["val_copy_baseline"] for r in rows]
    val_jepa = [r["val_jepa_loss"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_copy, marker="o", label="Train Copy Baseline")
    plt.plot(epochs, train_jepa, marker="o", label="Train JEPA Loss")
    plt.plot(epochs, val_copy, marker="o", linestyle="--", label="Val Copy Baseline")
    plt.plot(epochs, val_jepa, marker="o", linestyle="--", label="Val JEPA Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stage 1: Copy Baseline vs JEPA Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SCREENSHOTS / "stage1_copy_baseline_gap.png", dpi=150)
    plt.close()


def save_rollout_drift(rows: list[dict]) -> None:
    epochs = [r["epoch"] for r in rows]
    train_drift = [r["train_rollout_drift"] for r in rows]
    val_drift = [r["val_rollout_drift"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_drift, marker="o", label="Train Rollout Drift")
    plt.plot(epochs, val_drift, marker="o", label="Val Rollout Drift")
    plt.xlabel("Epoch")
    plt.ylabel("Mean L2 Drift")
    plt.title("Stage 1: Multi-step Rollout Drift")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SCREENSHOTS / "stage1_rollout_drift.png", dpi=150)
    plt.close()


def _heatmap_matrix(rows: list[dict], metric: str):
    horizons = sorted({int(r["horizon"]) for r in rows})
    masks = sorted({float(r["mask_ratio"]) for r in rows})
    h_idx = {h: i for i, h in enumerate(horizons)}
    m_idx = {m: i for i, m in enumerate(masks)}

    mat = np.full((len(horizons), len(masks)), np.nan, dtype=np.float64)
    for r in rows:
        i = h_idx[int(r["horizon"])]
        j = m_idx[float(r["mask_ratio"])]
        mat[i, j] = float(r[metric])
    return horizons, masks, mat


def save_grid_heatmap(rows: list[dict], metric: str, title: str, out_name: str) -> None:
    horizons, masks, mat = _heatmap_matrix(rows, metric)

    plt.figure(figsize=(9, 5.5))
    im = plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.colorbar(im, label=metric)
    plt.xticks(np.arange(len(masks)), [f"{m:.2f}" for m in masks])
    plt.yticks(np.arange(len(horizons)), [str(h) for h in horizons])
    plt.xlabel("Mask Ratio")
    plt.ylabel("Train Horizon")
    plt.title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                plt.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(SCREENSHOTS / out_name, dpi=150)
    plt.close()


def save_pareto(rows: list[dict]) -> None:
    x = np.array([float(r["val_total_loss"]) for r in rows])
    y = np.array([float(r["val_action_sensitivity"]) for r in rows])
    horizons = np.array([int(r["horizon"]) for r in rows])

    plt.figure(figsize=(8, 5.5))
    scatter = plt.scatter(x, y, c=horizons, cmap="tab10", s=55, alpha=0.85)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Train Horizon")
    plt.xlabel("Validation Total Loss (lower is better)")
    plt.ylabel("Validation Action Sensitivity (higher is better)")
    plt.title("Stage 1 Pareto: Loss vs Action Sensitivity")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(SCREENSHOTS / "stage1_pareto_loss_vs_sensitivity.png", dpi=150)
    plt.close()


def main() -> None:
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)

    epoch_rows = load_epoch_metrics(EPOCH_CKPT_DIR)
    grid_rows = load_grid_rows(GRID_CSV)

    save_loss_train_val(epoch_rows)
    save_loss_components(epoch_rows)
    save_action_sensitivity(epoch_rows)
    save_copy_baseline_gap(epoch_rows)
    save_rollout_drift(epoch_rows)

    save_grid_heatmap(
        grid_rows,
        metric="val_action_sensitivity",
        title="Grid Heatmap: Val Action Sensitivity",
        out_name="stage1_grid_val_sensitivity_heatmap.png",
    )
    save_grid_heatmap(
        grid_rows,
        metric="val_total_loss",
        title="Grid Heatmap: Val Total Loss",
        out_name="stage1_grid_val_loss_heatmap.png",
    )

    save_pareto(grid_rows)

    print("Saved Stage 1 visualizations to:", SCREENSHOTS)


if __name__ == "__main__":
    main()
