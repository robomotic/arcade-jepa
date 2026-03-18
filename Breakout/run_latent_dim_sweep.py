from __future__ import annotations

import argparse
import csv
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage latent_dim sweep with multithreaded execution and Pareto-knee selection."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Breakout/data/random"))
    parser.add_argument("--output-root", type=Path, default=Path("Breakout/checkpoints/latent_dim_sweep"))

    # Fixed JEPA settings during latent-dim sweep
    parser.add_argument("--train-horizon", type=int, default=1)
    parser.add_argument("--mask-ratio", type=float, default=0.7)
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-batches", type=int, default=40)
    parser.add_argument("--max-val-batches", type=int, default=12)
    parser.add_argument("--diagnostic-rollout-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--python-bin", type=str, default=".venv/bin/python")

    # Stage A (coarse) + Stage B (refine)
    parser.add_argument("--coarse-dims", type=int, nargs="+", default=[128, 256, 384, 512, 768, 1024])
    parser.add_argument("--refine-step", type=int, default=64)
    parser.add_argument("--refine-radius", type=int, default=2, help="Number of +/- steps around knee dim.")

    # Parallelism
    parser.add_argument("--workers", type=int, default=2)

    # Charts
    parser.add_argument("--screenshots-dir", type=Path, default=Path("screenshots"))
    parser.add_argument("--pareto-chart-name", type=str, default="stage1_latent_dim_pareto_knee.png")
    return parser.parse_args()


def run_one(args: argparse.Namespace, latent_dim: int, stage: str) -> dict[str, float | int | str]:
    run_dir = args.output_root / stage / f"ld{latent_dim}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_bin,
        "Breakout/train_jepa.py",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(run_dir),
        "--context-length",
        str(args.context_length),
        "--latent-dim",
        str(latent_dim),
        "--train-horizon",
        str(args.train_horizon),
        "--mask-ratio",
        str(args.mask_ratio),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--val-ratio",
        str(args.val_ratio),
        "--rollout-steps",
        str(args.diagnostic_rollout_steps),
        "--max-train-batches",
        str(args.max_train_batches),
        "--max-val-batches",
        str(args.max_val_batches),
        "--seed",
        str(args.seed),
    ]

    print(f"[{stage}] running latent_dim={latent_dim}")
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    ckpt = run_dir / f"jepa_epoch_{args.epochs:03d}.pt"
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = data.get("metrics", {})

    return {
        "stage": stage,
        "latent_dim": latent_dim,
        "train_total_loss": float(m.get("train_total_loss", m.get("train_jepa_loss", float("nan")))),
        "train_jepa_loss": float(m.get("train_jepa_loss", float("nan"))),
        "train_action_sensitivity": float(m.get("train_action_sensitivity", float("nan"))),
        "val_total_loss": float(m.get("val_total_loss", m.get("val_jepa_loss", float("nan")))),
        "val_jepa_loss": float(m.get("val_jepa_loss", float("nan"))),
        "val_action_sensitivity": float(m.get("val_action_sensitivity", float("nan"))),
        "val_copy_baseline": float(m.get("val_copy_baseline", float("nan"))),
        "val_rollout_drift": float(m.get("val_rollout_drift", float("nan"))),
        "checkpoint": str(ckpt),
    }


def run_parallel(args: argparse.Namespace, dims: list[int], stage: str) -> list[dict[str, float | int | str]]:
    dims = sorted(set(dims))
    out: list[dict[str, float | int | str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        fut_to_dim = {pool.submit(run_one, args, d, stage): d for d in dims}
        for fut in as_completed(fut_to_dim):
            dim = fut_to_dim[fut]
            try:
                row = fut.result()
                out.append(row)
                print(
                    f"[{stage}] done latent_dim={dim} "
                    f"val_sens={row['val_action_sensitivity']:.6f} val_loss={row['val_total_loss']:.6f}"
                )
            except Exception as exc:
                print(f"[{stage}] failed latent_dim={dim}: {exc}")
                raise
    out.sort(key=lambda r: int(r["latent_dim"]))
    return out


def pareto_front(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    # Minimize val_total_loss, maximize val_action_sensitivity
    front = []
    for r in rows:
        r_loss = float(r["val_total_loss"])
        r_sens = float(r["val_action_sensitivity"])
        dominated = False
        for q in rows:
            if q is r:
                continue
            q_loss = float(q["val_total_loss"])
            q_sens = float(q["val_action_sensitivity"])
            if (q_loss <= r_loss and q_sens >= r_sens) and (q_loss < r_loss or q_sens > r_sens):
                dominated = True
                break
        if not dominated:
            front.append(r)
    front.sort(key=lambda r: float(r["val_total_loss"]))
    return front


def pick_knee(front: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    losses = [float(r["val_total_loss"]) for r in front]
    sens = [float(r["val_action_sensitivity"]) for r in front]

    l_min, l_max = min(losses), max(losses)
    s_min, s_max = min(sens), max(sens)

    def nloss(v: float) -> float:
        return 0.0 if math.isclose(l_max, l_min) else (v - l_min) / (l_max - l_min)

    def nsens(v: float) -> float:
        return 1.0 if math.isclose(s_max, s_min) else (v - s_min) / (s_max - s_min)

    best = None
    best_dist = float("inf")
    for r in front:
        x = nloss(float(r["val_total_loss"]))
        y = nsens(float(r["val_action_sensitivity"]))
        # distance to ideal point (loss=0, sensitivity=1)
        d = math.sqrt((x - 0.0) ** 2 + (y - 1.0) ** 2)
        if d < best_dist:
            best_dist = d
            best = r
    assert best is not None
    return best


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def plot_pareto(rows: list[dict[str, float | int | str]], front: list[dict[str, float | int | str]], knee: dict[str, float | int | str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = [float(r["val_total_loss"]) for r in rows]
    y = [float(r["val_action_sensitivity"]) for r in rows]
    dims = [int(r["latent_dim"]) for r in rows]

    plt.figure(figsize=(8.5, 5.8))
    sc = plt.scatter(x, y, c=dims, cmap="plasma", s=70, alpha=0.85, edgecolors="k", linewidths=0.4)
    cbar = plt.colorbar(sc)
    cbar.set_label("latent_dim")

    fx = [float(r["val_total_loss"]) for r in front]
    fy = [float(r["val_action_sensitivity"]) for r in front]
    plt.plot(fx, fy, "-o", color="black", linewidth=1.5, markersize=4, label="Pareto front")

    kx = float(knee["val_total_loss"])
    ky = float(knee["val_action_sensitivity"])
    kd = int(knee["latent_dim"])
    plt.scatter([kx], [ky], s=180, marker="*", color="red", edgecolors="black", linewidths=0.8, label=f"Knee (dim={kd})")

    for r in rows:
        plt.annotate(
            str(int(r["latent_dim"])),
            (float(r["val_total_loss"]), float(r["val_action_sensitivity"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    plt.xlabel("Validation Total Loss (lower better)")
    plt.ylabel("Validation Action Sensitivity (higher better)")
    plt.title("Latent Dim Sweep Pareto + Knee")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    # Stage A: coarse sweep
    coarse_rows = run_parallel(args, args.coarse_dims, stage="coarse")
    coarse_csv = args.output_root / "coarse_results.csv"
    write_csv(coarse_rows, coarse_csv)

    coarse_front = pareto_front(coarse_rows)
    coarse_knee = pick_knee(coarse_front)
    knee_dim = int(coarse_knee["latent_dim"])

    # Stage B: refine around coarse knee
    refine_dims = [knee_dim + i * args.refine_step for i in range(-args.refine_radius, args.refine_radius + 1)]
    refine_dims = [d for d in refine_dims if d >= 64]
    # avoid duplicate work
    refine_dims = sorted(set(refine_dims) - set(int(r["latent_dim"]) for r in coarse_rows))

    refine_rows: list[dict[str, float | int | str]] = []
    if refine_dims:
        refine_rows = run_parallel(args, refine_dims, stage="refine")
    refine_csv = args.output_root / "refine_results.csv"
    if refine_rows:
        write_csv(refine_rows, refine_csv)

    all_rows = sorted(coarse_rows + refine_rows, key=lambda r: int(r["latent_dim"]))
    all_csv = args.output_root / "all_results.csv"
    write_csv(all_rows, all_csv)

    front = pareto_front(all_rows)
    knee = pick_knee(front)

    screenshots_dir = args.screenshots_dir
    if not screenshots_dir.is_absolute():
        screenshots_dir = Path("/mnt/sdc/ArcadeJepa") / screenshots_dir
    plot_path = screenshots_dir / args.pareto_chart_name
    plot_pareto(all_rows, front, knee, plot_path)

    summary_path = args.output_root / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"coarse_csv={coarse_csv}",
                f"refine_csv={refine_csv}",
                f"all_csv={all_csv}",
                f"pareto_chart={plot_path}",
                f"recommended_knee_latent_dim={int(knee['latent_dim'])}",
                f"recommended_val_total_loss={float(knee['val_total_loss']):.6f}",
                f"recommended_val_action_sensitivity={float(knee['val_action_sensitivity']):.6f}",
            ]
        ),
        encoding="utf-8",
    )

    print("\n=== Latent-dim sweep complete ===")
    print(f"Coarse results: {coarse_csv}")
    if refine_rows:
        print(f"Refine results: {refine_csv}")
    print(f"Combined results: {all_csv}")
    print(f"Pareto chart: {plot_path}")
    print(
        "Recommended knee latent_dim="
        f"{int(knee['latent_dim'])} "
        f"(val_loss={float(knee['val_total_loss']):.6f}, "
        f"val_sens={float(knee['val_action_sensitivity']):.6f})"
    )


if __name__ == "__main__":
    main()
