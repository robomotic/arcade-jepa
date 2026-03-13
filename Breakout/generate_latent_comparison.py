"""Generate a 3-column conceptual figure:

    Column 1 – Original RGB frame  (210×160, from screenshots/)
    Column 2 – Pre-processed 84×84 grayscale frame (from screenshots/)
    Column 3 – Latent z-vector (ℝ²⁵⁶) visualised as a 16×16 grid of
                colour-coded numbers, mimicking what the ConvEncoder outputs
                after LayerNorm (values roughly in [-2.5, 2.5]).

The z-vector is randomly sampled to illustrate the concept; it is not from
a trained model.  Values are drawn from a standard normal and then layer-norm
rescaled to resemble real encoder output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


ROOT = Path("/mnt/sdc/ArcadeJepa")
SCREENSHOTS = ROOT / "screenshots"

LATENT_DIM = 256          # matches the recommended latent_dim=256
GRID_SIDE   = 16          # 16×16 = 256 cells
FONT_SIZE   = 5.0         # pt – small enough to fit in each cell


def layernorm_like(x: np.ndarray) -> np.ndarray:
    """Mimic PyTorch LayerNorm: zero-mean, unit-variance normalisation."""
    return (x - x.mean()) / (x.std() + 1e-5)


def main() -> None:
    rng = np.random.default_rng(42)

    # ── Load existing frames ────────────────────────────────────────────────
    rgb_img   = plt.imread(SCREENSHOTS / "breakout_original_rgb_210x160.png")
    gray_img  = plt.imread(SCREENSHOTS / "breakout_grayscale_84x84.png")

    # ── Simulated latent vector ─────────────────────────────────────────────
    z_raw  = rng.standard_normal(LATENT_DIM).astype(np.float32)
    z      = layernorm_like(z_raw)                # shape (256,)
    z_grid = z.reshape(GRID_SIDE, GRID_SIDE)      # shape (16, 16)

    # ── Figure layout ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 3,
        figsize=(14, 5),
        gridspec_kw={"wspace": 0.35},
    )
    fig.patch.set_facecolor("#0f0f0f")

    label_kw  = dict(color="white", fontsize=11, fontweight="bold",
                     ha="center", va="bottom")
    annot_kw  = dict(color="#aaaaaa", fontsize=8, ha="center", va="top")

    # ── Column 1: Original RGB ──────────────────────────────────────────────
    ax0 = axes[0]
    ax0.imshow(rgb_img)
    ax0.set_facecolor("#0f0f0f")
    ax0.set_xticks([]); ax0.set_yticks([])
    for spine in ax0.spines.values():
        spine.set_edgecolor("#444444")
    ax0.set_title("Original frame\n210 × 160 · RGB",
                  color="white", fontsize=10, pad=8)
    ax0.set_xlabel("ALE/Breakout-v5  (raw observation)",
                   color="#888888", fontsize=7.5)

    # ── Column 2: 84×84 Grayscale ───────────────────────────────────────────
    ax1 = axes[1]
    ax1.imshow(gray_img, cmap="gray")
    ax1.set_facecolor("#0f0f0f")
    ax1.set_xticks([]); ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_edgecolor("#444444")
    ax1.set_title("Pre-processed frame\n84 × 84 · grayscale",
                  color="white", fontsize=10, pad=8)
    ax1.set_xlabel("Resize + grayscale  →  ConvEncoder input",
                   color="#888888", fontsize=7.5)

    # Arrow between col 1 and col 2
    fig.text(0.365, 0.52, "→", color="#888888", fontsize=22,
             ha="center", va="center")
    fig.text(0.365, 0.43, "resize\n+ grayscale", color="#555555",
             fontsize=7, ha="center", va="center")

    # ── Column 3: Latent z vector ───────────────────────────────────────────
    ax2 = axes[2]

    # Colour map: diverging blue-white-red so negatives are blue, positives red
    cmap  = plt.cm.RdBu_r
    norm  = mcolors.TwoSlopeNorm(vmin=z_grid.min(), vcenter=0.0, vmax=z_grid.max())
    im    = ax2.imshow(z_grid, cmap=cmap, norm=norm, aspect="equal",
                       interpolation="nearest")

    # Overlay the numeric value in every cell
    for row in range(GRID_SIDE):
        for col in range(GRID_SIDE):
            val   = z_grid[row, col]
            # White text on dark cells, dark text on bright cells
            bg_rgba  = cmap(norm(val))
            lum      = 0.299 * bg_rgba[0] + 0.587 * bg_rgba[1] + 0.114 * bg_rgba[2]
            txt_col  = "white" if lum < 0.5 else "#111111"
            ax2.text(
                col, row,
                f"{val:.2f}",
                color=txt_col,
                fontsize=FONT_SIZE,
                ha="center", va="center",
                fontfamily="monospace",
            )

    # Grid lines to separate cells
    ax2.set_xticks(np.arange(-0.5, GRID_SIDE, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, GRID_SIDE, 1), minor=True)
    ax2.grid(which="minor", color="#222222", linewidth=0.4)
    ax2.tick_params(which="both", bottom=False, left=False,
                    labelbottom=False, labelleft=False)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444444")

    ax2.set_title(f"Latent vector  z  ∈  ℝ{LATENT_DIM}\n"
                  f"({GRID_SIDE}×{GRID_SIDE} cells · LayerNorm output)",
                  color="white", fontsize=10, pad=8)
    ax2.set_xlabel("ConvEncoder(frame stack)  →  z",
                   color="#888888", fontsize=7.5)

    # Colour-bar
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white",
                                labelsize=7)
    cb.outline.set_edgecolor("#444444")
    cb.set_label("activation value", color="#888888", fontsize=7.5)

    # Arrow between col 2 and col 3
    fig.text(0.635, 0.52, "→", color="#888888", fontsize=22,
             ha="center", va="center")
    fig.text(0.635, 0.43, "ConvEncoder\n+ LayerNorm", color="#555555",
             fontsize=7, ha="center", va="center")

    # ── Overall title ───────────────────────────────────────────────────────
    fig.suptitle(
        "Breakout  ·  Observation → Pre-processing → Latent Encoding",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )

    out_path = SCREENSHOTS / "breakout_latent_comparison.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
