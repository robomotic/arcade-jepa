"""
Generate comparative charts for Stage 1.5 and Stage 2 training metrics.
Produces: screenshots/stage_metrics.png
"""

from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_s15(ckpt_dir, n=10):
    records = {}
    for ep in range(1, n + 1):
        path = Path(ckpt_dir) / f"q_imagination_epoch_{ep:03d}.pt"
        if not path.exists():
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        for split in ("train", "val"):
            for metric in ("loss", "q_std", "entropy", "reward"):
                key = f"{split}_{metric}"
                records.setdefault(key, []).append(float(ck[split][metric]))
    return records


def load_eval_runs(eval_csv: Path, max_runs: int = 6):
    if not eval_csv.exists():
        return None, None
    with eval_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None

    selected = rows[-max_runs:]
    palette = ["#4c9e6a", "#3a7abf", "#e67e22", "#9b59b6", "#16a085", "#c0392b"]
    labels, means, stds, colors = [], [], [], []
    for i, row in enumerate(selected):
        q_name = Path(row.get("q_checkpoint", "unknown")).stem or "unknown"
        eps_raw = row.get("epsilon", "?")
        label = f"{q_name}\nε={eps_raw}"
        labels.append(label)
        means.append(float(row.get("mean_return", 0.0)))
        stds.append(float(row.get("std_return", 0.0)))
        colors.append(palette[i % len(palette)])
    return (labels, means, stds, colors), selected[-1]

BASE = SCRIPT_DIR / "checkpoints"
s15_base = load_s15(BASE / "q_imagination")
s15_fix1 = load_s15(BASE / "q_imagination_fix1")
s15_fix2 = load_s15(BASE / "q_imagination_fix2")

# Stage 2 — from training logs (q_std/entropy not saved in checkpoints)
s2_short = dict(
    train_loss=[0.004836,0.003851,0.003948,0.003936,0.003901,0.003934,0.003899,0.003875,0.003927,0.003896,0.003982,0.003990,0.004114,0.004072,0.004000],
    val_loss  =[0.004681,0.005826,0.004482,0.004665,0.004608,0.004850,0.004747,0.004664,0.004760,0.005102,0.004828,0.004873,0.005079,0.004899,0.004786],
    q_std     =[0.0295,0.0221,0.0201,0.0208,0.0162,0.0177,0.0150,0.0137,0.0118,0.0119,0.0128,0.0113,0.0120,0.0098,0.0107],
    entropy   =[1.3856,1.3861,1.3861,1.3861,1.3862,1.3862,1.3862,1.3862,1.3862,1.3862,1.3862,1.3862,1.3862,1.3863,1.3862],
)
s2_long = dict(
    train_loss=[0.004570,0.003248,0.003248,0.003256,0.003271,0.003293,0.003302,0.003299,0.003281,0.003275,0.003296,0.003320,0.003355,0.003268,0.003283,0.003270,0.003263,0.003280,0.003278,0.003272,0.003276,0.003273,0.003288,0.003251,0.003282,0.003315,0.003273,0.003282,0.003294,0.003290,0.003296,0.003301,0.003295,0.003293,0.003306,0.003287,0.003293,0.003280,0.003301,0.003285,0.003285,0.003299,0.003287,0.003292,0.003289,0.003329,0.003316,0.003326,0.003323,0.003332],
    q_std     =[0.0208,0.0131,0.0133,0.0136,0.0139,0.0140,0.0150,0.0153,0.0150,0.0131,0.0131,0.0150,0.0177,0.0123,0.0131,0.0128,0.0103,0.0124,0.0124,0.0133,0.0117,0.0136,0.0126,0.0108,0.0104,0.0115,0.0095,0.0090,0.0100,0.0103,0.0097,0.0105,0.0090,0.0091,0.0095,0.0079,0.0092,0.0070,0.0084,0.0074,0.0066,0.0081,0.0061,0.0064,0.0061,0.0068,0.0069,0.0073,0.0050,0.0057],
    entropy   =[1.3858]+[1.3862]*14+[1.3862]*10+[1.3863]*25,
)

# Real-env eval data (appendable artifact source; fallback to static values)
eval_artifacts = BASE / "eval_policy" / "eval_runs.csv"
eval_loaded, latest_eval_row = load_eval_runs(eval_artifacts)
if eval_loaded is not None:
    eval_labels, eval_means, eval_stds, eval_colors = eval_loaded
else:
    eval_labels = ["Random\nagent", "S1.5 Fix2\nε=0.05", "S2 ep1\nε=0.05", "S2 ep15\nε=0.05", "S2 ep15\ngreedy ε=0", "S2 ep7\n(long) ε=0.05"]
    eval_means  = [1.10,   0.65,   0.70,   10.85,  0.00,   10.55]
    eval_stds   = [0.00,   0.79,   0.84,    0.65,  0.00,    1.12]
    eval_colors = ["#888888", "#e06060", "#e07050", "#4c9e6a", "#c0392b", "#3a7abf"]

# ─── Style helpers ────────────────────────────────────────────────────────────
HC = "#2ecc71"
FC = "#e74c3c"
NC = "#3498db"
WC = "#e67e22"

def style_ax(ax, title, xlabel="Epoch", ylabel=""):
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#cccccc", labelsize=8)
    ax.set_title(title, color="white", fontsize=8.5, pad=4)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=7.5)
    if ylabel:
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=7.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.grid(color="#222233", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

def ep(data): return list(range(1, len(data) + 1))

# ─── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(19, 15))
fig.patch.set_facecolor("#0f1117")

gs_top = fig.add_gridspec(2, 4, left=0.05, right=0.97, top=0.91, bottom=0.37,
                          hspace=0.52, wspace=0.36)
gs_bot = fig.add_gridspec(1, 2, left=0.05, right=0.97, top=0.30, bottom=0.04,
                          wspace=0.10)

axes = [[fig.add_subplot(gs_top[r, c]) for c in range(4)] for r in range(2)]
ax_eval = fig.add_subplot(gs_bot[0, 0])
ax_sum  = fig.add_subplot(gs_bot[0, 1])

# ─── Row 0: Stage 1.5 ────────────────────────────────────────────────────────
fig.text(0.01, 0.960, "STAGE 1.5 — Latent Imagination Q-Head Bootstrap",
         color="#ff9944", fontsize=12, fontweight="bold", va="top")
fig.text(0.01, 0.940,
         "All three runs fail: td_loss collapses, q_std decays, entropy stuck at 0.325 (not ln(4)=1.386). "
         "Root cause: encoder latents are reward-blind (ROC-AUC=0.50).",
         color="#aaaaaa", fontsize=8.5, va="top")

# 0-0  TD loss
ax = axes[0][0]
style_ax(ax, "TD Loss  (should stay stable, not → 0)", ylabel="SmoothL1 loss")
ax.semilogy(ep(s15_base["train_loss"]), s15_base["train_loss"], color=FC, lw=1.8, label="Baseline", marker="o", ms=3)
ax.semilogy(ep(s15_fix1["train_loss"]), s15_fix1["train_loss"], color=WC, lw=1.8, label="Fix 1 — clamp rewards", marker="s", ms=3)
ax.semilogy(ep(s15_fix2["train_loss"]), s15_fix2["train_loss"], color=NC, lw=1.8, label="Fix 2 — reward weight ×10", marker="^", ms=3)
ax.legend(fontsize=6.5, labelcolor="white", facecolor="#1a1d27", edgecolor="#333344")
ax.annotate("❌ Instant\ncollapse", xy=(2, s15_fix2["train_loss"][1]),
            xytext=(5.5, s15_fix2["train_loss"][0]*2),
            arrowprops=dict(arrowstyle="->", color="#ff6666", lw=0.9),
            color="#ff6666", fontsize=7)

# 0-1  q_std
ax = axes[0][1]
style_ax(ax, "Q-value std  (should grow above 0.05)", ylabel="q_std")
ax.plot(ep(s15_base["train_q_std"]), s15_base["train_q_std"], color=FC, lw=1.8, label="Baseline", marker="o", ms=3)
ax.plot(ep(s15_fix1["train_q_std"]), s15_fix1["train_q_std"], color=WC, lw=1.8, label="Fix 1", marker="s", ms=3)
ax.plot(ep(s15_fix2["train_q_std"]), s15_fix2["train_q_std"], color=NC, lw=1.8, label="Fix 2", marker="^", ms=3)
ax.axhline(0.05, color=WC, lw=0.9, ls="--", alpha=0.7, label="Target threshold 0.05")
ax.legend(fontsize=6.5, labelcolor="white", facecolor="#1a1d27", edgecolor="#333344")
ax.annotate("❌ All three runs\ndecay to near-zero",
            xy=(10, s15_fix2["train_q_std"][-1]), xytext=(6.5, 0.022),
            arrowprops=dict(arrowstyle="->", color="#ff6666", lw=0.9),
            color="#ff6666", fontsize=7)

# 0-2  entropy
ax = axes[0][2]
style_ax(ax, "Action entropy  (target: ln(4)=1.386 = uniform)", ylabel="entropy (nats)")
ax.plot(ep(s15_base["train_entropy"]), s15_base["train_entropy"], color=FC, lw=1.8, label="Baseline", marker="o", ms=3)
ax.plot(ep(s15_fix1["train_entropy"]), s15_fix1["train_entropy"], color=WC, lw=1.8, label="Fix 1", marker="s", ms=3)
ax.plot(ep(s15_fix2["train_entropy"]), s15_fix2["train_entropy"], color=NC, lw=1.8, label="Fix 2", marker="^", ms=3)
ax.axhline(np.log(4), color=HC, lw=1.1, ls="--", label="ln(4) = uniform policy", alpha=0.9)
ax.axhline(0.325, color=FC, lw=0.9, ls=":", alpha=0.8, label="0.325 = always-same-action")
ax.set_ylim(-0.05, 1.60)
ax.legend(fontsize=6.2, labelcolor="white", facecolor="#1a1d27", edgecolor="#333344")
ax.text(1.3, 1.30, "← healthy", color=HC, fontsize=7.5)
ax.text(1.3, 0.42, "❌ stuck here", color=FC, fontsize=7.5)

# 0-3  imagined reward
ax = axes[0][3]
style_ax(ax, "Imagined reward mean  (must be ≥ 0)", ylabel="mean reward / step")
ax.plot(ep(s15_base["train_reward"]), s15_base["train_reward"], color=FC, lw=1.8, label="Baseline", marker="o", ms=3)
ax.plot(ep(s15_fix1["train_reward"]), s15_fix1["train_reward"], color=WC, lw=1.8, label="Fix 1 — clamp", marker="s", ms=3)
ax.plot(ep(s15_fix2["train_reward"]), s15_fix2["train_reward"], color=NC, lw=1.8, label="Fix 2 — weight ×10", marker="^", ms=3)
ax.axhline(0, color="#555566", lw=1.0)
ax.legend(fontsize=6.5, labelcolor="white", facecolor="#1a1d27", edgecolor="#333344")
ax.annotate("❌ Baseline < 0\n(negative bias)",
            xy=(5, s15_base["train_reward"][4]), xytext=(6, -0.007),
            arrowprops=dict(arrowstyle="->", color="#ff6666", lw=0.9),
            color="#ff6666", fontsize=7)
ax.annotate("✅ Fixed by\nFix 1 + Fix 2",
            xy=(5, s15_fix2["train_reward"][4]), xytext=(6, 0.015),
            arrowprops=dict(arrowstyle="->", color=HC, lw=0.9),
            color=HC, fontsize=7)

# ─── Row 1: Stage 2 ──────────────────────────────────────────────────────────
fig.text(0.01, 0.635, "STAGE 2 — Offline TD Q-Learning on Real Transitions",
         color="#44aaff", fontsize=12, fontweight="bold", va="top")
fig.text(0.01, 0.615,
         "td_loss stable, entropy = ln(4) (healthy). 10.85 return at ε=0.05 (9.9× random). "
         "Greedy=0 — offline random data cannot teach state-conditional Q-values.",
         color="#aaaaaa", fontsize=8.5, va="top")

# 1-0  TD loss S1.5 vs S2
ax = axes[1][0]
style_ax(ax, "TD Loss — S1.5 Fix2 vs Stage 2", ylabel="SmoothL1 loss")
ax.semilogy(ep(s15_fix2["train_loss"]), s15_fix2["train_loss"],
            color=FC, lw=1.8, label="S1.5 Fix2 (collapses)", marker="o", ms=3)
ax.semilogy(ep(s2_short["train_loss"]), s2_short["train_loss"],
            color=HC, lw=1.8, label="S2 train (15 ep)", marker="^", ms=3)
ax.semilogy(ep(s2_short["val_loss"]), s2_short["val_loss"],
            color=HC, lw=1.2, ls="--", alpha=0.6, label="S2 val")
ax.legend(fontsize=6.5, labelcolor="white", facecolor="#1a1d27", edgecolor="#333344")
ax.text(4.5, 0.0038, "✅ Stable ~0.004", color=HC, fontsize=7.5)
ax.text(4.5, 4e-6,   "❌ Collapses", color=FC, fontsize=7.5)

# 1-1  q_std S2 short + long
ax = axes[1][1]
style_ax(ax, "Q-value std — Stage 2 short vs long run", ylabel="q_std")
ax.plot(ep(s2_short["q_std"]), s2_short["q_std"],
        color=HC, lw=1.8, label="S2 short (15 ep)", marker="o", ms=3)
ax.plot(ep(s2_long["q_std"]),  s2_long["q_std"],
        color=NC, lw=1.8, label="S2 long (50 ep)", marker="^", ms=3, alpha=0.85)
ax.axhline(0.05, color=WC, lw=0.9, ls="--", alpha=0.7, label="Target threshold 0.05")
ax.legend(fontsize=6.5, labelcolor="white", facecolor="#1a1d27", edgecolor="#333344")
ax.annotate("⚠️ Plateau — more\nepochs don't help",
            xy=(42, s2_long["q_std"][41]), xytext=(22, 0.018),
            arrowprops=dict(arrowstyle="->", color="#ffaa44", lw=0.9),
            color="#ffaa44", fontsize=7)

# 1-2  entropy S1.5 vs S2
ax = axes[1][2]
style_ax(ax, "Action entropy — S1.5 vs Stage 2", ylabel="entropy (nats)")
ax.plot(ep(s15_fix2["train_entropy"]), s15_fix2["train_entropy"],
        color=FC, lw=1.8, label="S1.5 Fix2 (≈0.325)", marker="o", ms=3)
ax.plot(ep(s2_short["entropy"]), s2_short["entropy"],
        color=HC, lw=1.8, label="S2 short", marker="^", ms=3)
ax.plot(ep(s2_long["entropy"]),  s2_long["entropy"],
        color=NC, lw=1.2, ls="--", alpha=0.75, label="S2 long (50 ep)")
ax.axhline(np.log(4), color=HC, lw=1.0, ls="--", alpha=0.8, label="ln(4) = uniform")
ax.axhline(0.325, color=FC, lw=0.9, ls=":", alpha=0.8, label="0.325 = degenerate")
ax.set_ylim(-0.05, 1.60)
ax.legend(fontsize=6.2, labelcolor="white", facecolor="#1a1d27", edgecolor="#333344")
ax.text(1.3, 1.32, "✅ S2 uniform", color=HC, fontsize=7.5)
ax.text(1.3, 0.23, "❌ S1.5 degenerate", color=FC, fontsize=7.5)

# 1-3  Root-cause diagnostics bar
ax = axes[1][3]
style_ax(ax, "Root-cause diagnostics (Stage 1.5 autopsy)", xlabel="", ylabel="Score")
diag_names  = ["RewardHead\nR²", "Encoder\nROC-AUC", "Val action\nsensitivity"]
measured    = [-0.005, 0.5008, 0.0076]
ideal       = [1.0,    1.0,    0.10]
x = np.arange(3)
ax.bar(x - 0.19, measured, width=0.35, color=[FC, FC, FC], label="Measured", zorder=3)
ax.bar(x + 0.19, ideal,    width=0.35, color=["#2d5a3d"]*3, label="Ideal", zorder=3, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(diag_names, color="#cccccc", fontsize=8)
ax.tick_params(axis="x", bottom=False)
ax.axhline(0, color="#555566", lw=0.8)
ax.set_ylim(-0.15, 1.15)
ax.legend(fontsize=7, labelcolor="white", facecolor="#1a1d27", edgecolor="#333344")
for xi, val in zip(x - 0.19, measured):
    ax.text(xi, max(val, 0) + 0.03, f"{val:+.4f}" if abs(val) < 0.01 else f"{val:.4f}",
            ha="center", va="bottom", color="#ff8888", fontsize=7.5, fontweight="bold")
ax.text(0.5, -0.12, "All three metrics confirm the encoder is reward-blind.",
        transform=ax.transAxes, ha="center", color="#ff9944", fontsize=7, style="italic")

# ─── Row 2: Eval bar + summary ────────────────────────────────────────────────
fig.text(0.01, 0.305, "REAL ENVIRONMENT EVALUATION — ALE/Breakout-v5",
         color="#dddd44", fontsize=12, fontweight="bold", va="top")

ax_eval.set_facecolor("#1a1d27")
x = np.arange(len(eval_labels))
bars = ax_eval.bar(x, eval_means, yerr=eval_stds, capsize=4,
                   color=eval_colors, width=0.55, zorder=3,
                   error_kw=dict(ecolor="#ffffff", lw=1.2, capthick=1.2))
ax_eval.axhline(1.10, color="#888888", lw=1.2, ls="--", label="Random agent baseline (1.10)")
ax_eval.set_xticks(x)
ax_eval.set_xticklabels(eval_labels, color="#cccccc", fontsize=8.5)
ax_eval.tick_params(axis="x", bottom=False, colors="#cccccc")
ax_eval.tick_params(axis="y", colors="#cccccc", labelsize=8)
ax_eval.set_ylabel("Mean episode return", color="#aaaaaa", fontsize=9)
ax_eval.set_title("Real-environment returns by checkpoint & ε setting",
                  color="white", fontsize=9.5, pad=6)
for spine in ax_eval.spines.values():
    spine.set_edgecolor("#333344")
ax_eval.grid(axis="y", color="#222233", lw=0.5, ls="--", alpha=0.7)
ax_eval.legend(fontsize=8, labelcolor="white", facecolor="#1a1d27", edgecolor="#333344", loc="upper left")
for bar, val, std in zip(bars, eval_means, eval_stds):
    ypos = val + std + 0.3
    ax_eval.text(bar.get_x() + bar.get_width()/2, ypos, f"{val:.2f}",
                 ha="center", va="bottom", color="white", fontsize=8, fontweight="bold")
ax_eval.text(3.0, 5.0, "9.9× random\n15.5× random-init", color="white", ha="center", fontsize=7.5, style="italic")
ax_eval.text(4.0, 0.6, "❌\nalways\nLEFT", color="#ff8888", ha="center", fontsize=7.5)
ax_eval.set_ylim(bottom=-1.0)

ax_sum.set_facecolor("#141720")
ax_sum.axis("off")
lines = [
    ("STAGE 1.5 — FAILURE SIGNATURES", "#ff9944", True,  0.97),
    ("td_loss → 0 within epoch 1  (all 3 runs)", "#ff6666", False, 0.90),
    ("q_std decays 0.029 → 0.004  (target: > 0.05)", "#ff6666", False, 0.83),
    ("entropy stuck at 0.325  (not ln(4)=1.386)", "#ff6666", False, 0.76),
    ("imag_reward < 0 in baseline  (fixed by Fix 1+2)", "#ffaa44", False, 0.69),
    ("RewardHead R² = -0.005  (worse than constant)", "#ff8888", False, 0.62),
    ("Encoder ROC-AUC = 0.5008  (reward-blind)", "#ff8888", False, 0.55),
    ("STAGE 2 — PARTIAL SUCCESS", "#44aaff", True,  0.46),
    ("td_loss stable ~0.004 throughout  ✅", "#88dd88", False, 0.39),
    ("entropy = ln(4) = 1.386 (uniform)  ✅", "#88dd88", False, 0.32),
    ("ε=0.05 return: 10.85 ± 0.65  (9.9× random)  ✅", "#88dd88", False, 0.25),
    ("q_std decays slowly — no state signal  ⚠️", "#ffaa44", False, 0.18),
    ("Greedy = 0.00, always LEFT — state-independent  ❌", "#ff6666", False, 0.11),
    ("NEXT: Stage 3 — Online DQN fine-tuning  ⬜", "#dddd44", True,  0.03),
]

if latest_eval_row is not None:
    latest_mean = float(latest_eval_row.get("mean_return", 0.0))
    latest_std = float(latest_eval_row.get("std_return", 0.0))
    latest_brick = int(float(latest_eval_row.get("total_brick_hits_proxy", 0.0)))
    latest_paddle = int(float(latest_eval_row.get("total_paddle_hits_estimate", 0.0)))
    latest_cov = float(latest_eval_row.get("ball_coverage_ratio", 0.0))
    lines.extend([
        ("LATEST APPENDED EVAL", "#dddd44", True, 0.16),
        (f"return={latest_mean:.2f} ± {latest_std:.2f}", "#88dd88", False, 0.13),
        (f"brick_hits(proxy)={latest_brick}  paddle_hits(est)={latest_paddle}", "#88dd88", False, 0.10),
        (f"ball_coverage={latest_cov:.4f}", "#88dd88", False, 0.07),
    ])

for text, color, bold, y in lines:
    prefix = "  " if not bold else ""
    ax_sum.text(0.04, y, prefix + text, transform=ax_sum.transAxes,
                color=color, fontsize=8, fontweight="bold" if bold else "normal", va="bottom")
ax_sum.set_title("Findings summary", color="white", fontsize=9.5, pad=6)

fig.text(0.50, 0.997, "ArcadeJepa — Stage 1.5 & Stage 2 Training Metrics",
         color="white", fontsize=13.5, fontweight="bold", ha="center", va="top")

out_dir = PROJECT_ROOT / "screenshots"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "stage_metrics.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_path.resolve()}")
