"""Evaluate a trained Q-Head in the ALE/Breakout-v5 environment.

Loads a frozen JEPA encoder together with a trained Q-Head (from either Stage
1.5 or Stage 2) and runs N greedy episodes in the real environment, reporting
per-episode returns and a summary table.

Both Stage 1.5 checkpoints (``q_imagination_epoch_*.pt``) and Stage 2
checkpoints (``q_head_epoch_*.pt``) use the same ``{"q_head": ...}`` key, so
this script works with either.

Usage:
    python Breakout/eval_policy.py \\
        --encoder-checkpoint  Breakout/checkpoints/jepa/jepa_epoch_020.pt \\
        --q-checkpoint        Breakout/checkpoints/q_imagination/q_imagination_epoch_010.pt \\
        --episodes            20

Optional render (requires a display):
    python Breakout/eval_policy.py ... --render
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .envs import create_breakout_env
    from .models import ConvEncoder, QHead
except ImportError:
    from envs import create_breakout_env
    from models import ConvEncoder, QHead

try:
    from ocatari.core import OCAtari
except ImportError:
    OCAtari = None

NUM_BREAKOUT_ACTIONS = 4
PIXEL_NORM_SCALE: float = 1.0 / 255.0
FRAME_HEIGHT = 84
FRAME_WIDTH = 84


@dataclass
class EpisodeAnalysis:
    total_return: float
    steps: int
    brick_hits_proxy: int
    paddle_hits_estimate: int
    ball_detections: int
    ball_heatmap: np.ndarray
    paddle_hit_events: list[dict]
    blocks_visible_mean: float


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a JEPA Q-Head in ALE/Breakout-v5."
    )
    parser.add_argument(
        "--encoder-checkpoint", type=Path, required=True,
        help="Path to a Stage 1 JEPA checkpoint (contains 'encoder' key).",
    )
    parser.add_argument(
        "--q-checkpoint", type=Path, required=True,
        help="Path to a Q-Head checkpoint (Stage 1.5 or Stage 2; contains 'q_head' key).",
    )
    parser.add_argument("--context-length", type=int, default=4)
    # Default 256: validated best latent dimensionality from Stage 1.
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.05,
        help="ε-greedy rate during evaluation (small value for near-greedy play).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=10_000,
        help="Hard cap on steps per episode (prevents infinite loops on no-op policies).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--render", action="store_true",
        help="Open a human-render window (requires a display).",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=Path,
        default=None,
        help="Directory for append-only eval artifacts (CSVs, JSONL, heatmaps).",
    )
    parser.add_argument(
        "--save-debug-screenshots",
        action="store_true",
        help="Save debug screenshots for accepted paddle-hit estimates (last N kept).",
    )
    parser.add_argument(
        "--debug-screenshot-dir",
        type=Path,
        default=None,
        help="Directory for paddle-hit debug screenshots.",
    )
    parser.add_argument(
        "--debug-keep-last",
        type=int,
        default=10,
        help="How many latest debug screenshots to keep.",
    )
    parser.add_argument("--ball-min-intensity", type=int, default=170)
    parser.add_argument("--ball-motion-threshold", type=int, default=18)
    parser.add_argument("--ball-max-pixels", type=int, default=12)
    parser.add_argument("--paddle-min-intensity", type=int, default=120)
    parser.add_argument("--paddle-band-top", type=int, default=76)
    parser.add_argument("--paddle-near-tolerance", type=int, default=3)
    parser.add_argument("--paddle-x-margin", type=int, default=1)
    parser.add_argument("--paddle-vy-threshold", type=float, default=1.0)
    parser.add_argument("--paddle-min-gap-steps", type=int, default=2)
    parser.add_argument(
        "--metrics-backend",
        type=str,
        choices=("pixel", "ocatari"),
        default="pixel",
        help="Use legacy pixel heuristic or OCAtari object extractor for analysis metrics.",
    )
    parser.add_argument(
        "--ocatari-mode",
        type=str,
        choices=("ram", "vision"),
        default="ram",
        help="OCAtari extraction mode (used when --metrics-backend ocatari).",
    )
    parser.add_argument(
        "--ocatari-hud",
        action="store_true",
        help="Enable HUD object extraction in OCAtari mode.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(args: argparse.Namespace) -> tuple[ConvEncoder, QHead]:
    """Load and freeze the encoder and Q-Head from their respective checkpoints."""
    # Encoder
    enc_ckpt = torch.load(args.encoder_checkpoint, map_location=args.device, weights_only=False)
    encoder = ConvEncoder(
        input_channels=args.context_length, latent_dim=args.latent_dim
    ).to(args.device)
    encoder.load_state_dict(enc_ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Q-Head (compatible with Stage 1.5 and Stage 2 checkpoint formats)
    q_ckpt = torch.load(args.q_checkpoint, map_location=args.device, weights_only=False)
    q_head = QHead(latent_dim=args.latent_dim, num_actions=NUM_BREAKOUT_ACTIONS).to(args.device)
    q_head.load_state_dict(q_ckpt["q_head"])
    q_head.eval()
    for p in q_head.parameters():
        p.requires_grad_(False)

    return encoder, q_head


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resize_grayscale_to_84(frame_gray: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(frame_gray, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(FRAME_HEIGHT, FRAME_WIDTH), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).clamp(0, 255).to(torch.uint8).cpu().numpy()


def get_ocatari_ale(env):
    inner_env = getattr(env, "_env", None)
    if inner_env is None or not hasattr(inner_env, "unwrapped"):
        return None
    unwrapped = inner_env.unwrapped
    if not hasattr(unwrapped, "ale"):
        return None
    return unwrapped.ale


def extract_policy_observation(env, obs: np.ndarray, metrics_backend: str) -> np.ndarray:
    if metrics_backend != "ocatari":
        return np.asarray(obs, dtype=np.uint8)
    ale = get_ocatari_ale(env)
    if ale is None:
        raise RuntimeError("OCAtari backend requested but ALE handle was not available.")
    gray = ale.getScreenGrayscale()
    return resize_grayscale_to_84(gray)


def extract_ocatari_objects(env) -> tuple[tuple[int, int] | None, tuple[int, int, int, int] | None, int]:
    objects = [obj for obj in getattr(env, "objects", []) if obj is not None]
    ball_obj = next((o for o in objects if type(o).__name__ == "Ball" and getattr(o, "w", 0) > 0), None)
    player_obj = next((o for o in objects if type(o).__name__ == "Player" and getattr(o, "w", 0) > 0), None)
    block_count = sum(
        1 for o in objects if type(o).__name__ == "Block" and getattr(o, "w", 0) > 0 and getattr(o, "h", 0) > 0
    )

    ball_xy = None if ball_obj is None else (int(getattr(ball_obj, "x", 0)), int(getattr(ball_obj, "y", 0)))
    if player_obj is None:
        paddle_bbox = None
    else:
        paddle_bbox = (
            int(getattr(player_obj, "x", 0)),
            int(getattr(player_obj, "y", 0)),
            int(getattr(player_obj, "w", 0)),
            int(getattr(player_obj, "h", 0)),
        )

    return ball_xy, paddle_bbox, int(block_count)


def is_conservative_paddle_hit_ocatari(
    ball_history: collections.deque[tuple[int, int]],
    paddle_bbox: tuple[int, int, int, int] | None,
    vy_threshold: float,
    x_margin: int,
    y_tolerance: int,
) -> bool:
    if paddle_bbox is None or len(ball_history) < 3:
        return False
    (x2, y2), (x1, y1), (_x0, y0) = list(ball_history)[-3:]
    vy_prev = float(y1 - y2)
    vy_curr = float(y0 - y1)

    if vy_prev < vy_threshold:
        return False
    if vy_curr > -vy_threshold:
        return False

    px, py, pw, ph = paddle_bbox
    if not ((px - x_margin) <= x1 <= (px + pw + x_margin)):
        return False
    if not ((py - y_tolerance) <= y1 <= (py + ph + y_tolerance)):
        return False
    return True


def detect_paddle_x_bounds(frame: np.ndarray, min_intensity: int, band_top: int) -> tuple[int, int] | None:
    y0 = max(0, min(int(band_top), frame.shape[0] - 1))
    band = frame[y0:, :]
    mask = band >= min_intensity
    x_cols = np.where(mask.any(axis=0))[0]
    if x_cols.size == 0:
        return None
    return int(x_cols.min()), int(x_cols.max())


def detect_ball_position(
    prev_frame: np.ndarray,
    cur_frame: np.ndarray,
    min_intensity: int,
    motion_threshold: int,
    max_pixels: int,
) -> tuple[int, int] | None:
    prev_i = prev_frame.astype(np.int16)
    cur_i = cur_frame.astype(np.int16)
    motion = np.abs(cur_i - prev_i)
    mask = (cur_i >= int(min_intensity)) & (motion >= int(motion_threshold))

    # Ignore top HUD rows for conservative detection.
    mask[:4, :] = False

    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    if coords.shape[0] > int(max_pixels):
        return None

    y, x = np.round(coords.mean(axis=0)).astype(int)
    if not (0 <= x < cur_frame.shape[1] and 0 <= y < cur_frame.shape[0]):
        return None
    return int(x), int(y)


def is_conservative_paddle_hit(
    ball_history: collections.deque[tuple[int, int]],
    paddle_x_bounds: tuple[int, int] | None,
    paddle_band_top: int,
    near_tolerance: int,
    x_margin: int,
    vy_threshold: float,
) -> bool:
    if paddle_x_bounds is None or len(ball_history) < 3:
        return False

    (x2, y2), (x1, y1), (_x0, y0) = list(ball_history)[-3:]
    vy_prev = float(y1 - y2)
    vy_curr = float(y0 - y1)

    # Conservative: clear downward movement followed by clear upward movement.
    if vy_prev < vy_threshold:
        return False
    if vy_curr > -vy_threshold:
        return False

    # Conservative: bounce must happen near paddle vertical band.
    if y1 < (int(paddle_band_top) - int(near_tolerance)):
        return False

    x_left, x_right = paddle_x_bounds
    return (x_left - int(x_margin)) <= x1 <= (x_right + int(x_margin))


def save_debug_screenshot(
    out_path: Path,
    frame: np.ndarray,
    ball_pos: tuple[int, int] | None,
    paddle_x_bounds: tuple[int, int] | None,
    title: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=130)
    ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=8)
    ax.axis("off")

    if paddle_x_bounds is not None:
        x_left, x_right = paddle_x_bounds
        ax.plot([x_left, x_right], [80, 80], color="lime", linewidth=1.4)
    if ball_pos is not None:
        ax.scatter([ball_pos[0]], [ball_pos[1]], color="red", s=14)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def prune_debug_screenshots(debug_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    pngs = sorted(debug_dir.glob("paddle_hit_*.png"))
    if len(pngs) <= keep_last:
        return
    for path in pngs[: len(pngs) - keep_last]:
        path.unlink(missing_ok=True)


def save_ball_heatmap_png(out_path: Path, heatmap: np.ndarray, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=160)
    ax.imshow(heatmap, cmap="magma")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env,
    encoder: ConvEncoder,
    q_head: QHead,
    context_length: int,
    epsilon: float,
    max_steps: int,
    device: str,
    analysis_cfg: dict,
    run_id: str,
    episode_idx: int,
) -> EpisodeAnalysis:
    """Run one episode and return ``(total_return, steps_taken)``.

    A deque of length ``context_length`` is used as a sliding frame buffer.
    It is pre-filled with the initial observation so the encoder always
    receives a full ``(context_length, 84, 84)`` stack, even on the first step.
    """
    obs, _ = env.reset()
    policy_obs = extract_policy_observation(env, obs, analysis_cfg["metrics_backend"])

    heatmap_height, heatmap_width = analysis_cfg["heatmap_shape"]
    # Pre-fill the context buffer with the initial frame.
    frame_buffer: collections.deque[np.ndarray] = collections.deque(
        [policy_obs] * context_length, maxlen=context_length
    )

    total_return = 0.0
    steps = 0
    brick_hits_proxy = 0
    paddle_hits_estimate = 0
    ball_detections = 0
    ball_heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.int64)
    paddle_hit_events: list[dict] = []
    blocks_visible_sum = 0.0
    blocks_visible_steps = 0

    prev_policy_obs = policy_obs
    ball_history: collections.deque[tuple[int, int]] = collections.deque(maxlen=3)
    last_paddle_hit_step = -10_000

    terminated = truncated = False

    while not (terminated or truncated) and steps < max_steps:
        # Build float32 context tensor: (1, C, H, W)
        context = np.stack(list(frame_buffer), axis=0).astype(np.float32) * PIXEL_NORM_SCALE
        context_tensor = torch.from_numpy(context).unsqueeze(0).to(device)

        with torch.no_grad():
            z = encoder(context_tensor)        # (1, D)
            q_values = q_head(z)               # (1, A)

        # ε-greedy action
        if epsilon > 0.0 and torch.rand(1).item() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(q_values.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_policy_obs = extract_policy_observation(env, next_obs, analysis_cfg["metrics_backend"])
        frame_buffer.append(next_policy_obs)
        total_return += float(reward)
        steps += 1

        if reward > 0.0:
            brick_hits_proxy += 1

        paddle_x_bounds: tuple[int, int] | None = None
        paddle_bbox: tuple[int, int, int, int] | None = None

        if analysis_cfg["metrics_backend"] == "ocatari":
            ball_pos, paddle_bbox, block_count = extract_ocatari_objects(env)
            if paddle_bbox is not None:
                paddle_x_bounds = (int(paddle_bbox[0]), int(paddle_bbox[0] + paddle_bbox[2]))
            blocks_visible_sum += float(block_count)
            blocks_visible_steps += 1
        else:
            paddle_x_bounds = detect_paddle_x_bounds(
                next_policy_obs,
                min_intensity=analysis_cfg["paddle_min_intensity"],
                band_top=analysis_cfg["paddle_band_top"],
            )
            ball_pos = detect_ball_position(
                prev_policy_obs,
                next_policy_obs,
                min_intensity=analysis_cfg["ball_min_intensity"],
                motion_threshold=analysis_cfg["ball_motion_threshold"],
                max_pixels=analysis_cfg["ball_max_pixels"],
            )

        if ball_pos is None:
            ball_history.clear()
        else:
            bx, by = ball_pos
            if 0 <= bx < heatmap_width and 0 <= by < heatmap_height:
                ball_heatmap[by, bx] += 1
                ball_detections += 1
            ball_history.append(ball_pos)

        if analysis_cfg["metrics_backend"] == "ocatari":
            hit_detected = is_conservative_paddle_hit_ocatari(
                ball_history,
                paddle_bbox,
                vy_threshold=analysis_cfg["paddle_vy_threshold"],
                x_margin=analysis_cfg["paddle_x_margin"],
                y_tolerance=analysis_cfg["paddle_near_tolerance"],
            )
        else:
            hit_detected = is_conservative_paddle_hit(
                ball_history,
                paddle_x_bounds,
                paddle_band_top=analysis_cfg["paddle_band_top"],
                near_tolerance=analysis_cfg["paddle_near_tolerance"],
                x_margin=analysis_cfg["paddle_x_margin"],
                vy_threshold=analysis_cfg["paddle_vy_threshold"],
            )

        if hit_detected:
            if (steps - last_paddle_hit_step) >= analysis_cfg["paddle_min_gap_steps"]:
                paddle_hits_estimate += 1
                last_paddle_hit_step = steps
                event = {
                    "run_id": run_id,
                    "episode": int(episode_idx),
                    "step": int(steps),
                    "reward": float(reward),
                    "action": int(action),
                    "ball": list(ball_history[-1]) if len(ball_history) > 0 else None,
                    "paddle_x_bounds": list(paddle_x_bounds) if paddle_x_bounds is not None else None,
                    "paddle_bbox": list(paddle_bbox) if paddle_bbox is not None else None,
                    "metrics_backend": analysis_cfg["metrics_backend"],
                    "timestamp_utc": utc_now_iso(),
                }
                paddle_hit_events.append(event)

                if analysis_cfg["save_debug_screenshots"]:
                    debug_path = analysis_cfg["debug_screenshot_dir"] / (
                        f"paddle_hit_{run_id}_ep{episode_idx:03d}_s{steps:05d}.png"
                    )
                    save_debug_screenshot(
                        out_path=debug_path,
                        frame=next_policy_obs if analysis_cfg["metrics_backend"] != "ocatari" else np.asarray(get_ocatari_ale(env).getScreenGrayscale(), dtype=np.uint8),
                        ball_pos=tuple(ball_history[-1]) if len(ball_history) > 0 else None,
                        paddle_x_bounds=paddle_x_bounds,
                        title=f"run={run_id} ep={episode_idx} step={steps}",
                    )
                    prune_debug_screenshots(
                        analysis_cfg["debug_screenshot_dir"],
                        keep_last=analysis_cfg["debug_keep_last"],
                    )

        prev_policy_obs = next_policy_obs

    blocks_visible_mean = (
        float(blocks_visible_sum / max(1, blocks_visible_steps))
        if analysis_cfg["metrics_backend"] == "ocatari"
        else float("nan")
    )

    return EpisodeAnalysis(
        total_return=float(total_return),
        steps=int(steps),
        brick_hits_proxy=int(brick_hits_proxy),
        paddle_hits_estimate=int(paddle_hits_estimate),
        ball_detections=int(ball_detections),
        ball_heatmap=ball_heatmap,
        paddle_hit_events=paddle_hit_events,
        blocks_visible_mean=blocks_visible_mean,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    analysis_output_dir = (
        args.analysis_output_dir
        if args.analysis_output_dir is not None
        else (script_dir / "checkpoints" / "eval_policy")
    )
    debug_screenshot_dir = (
        args.debug_screenshot_dir
        if args.debug_screenshot_dir is not None
        else (repo_root / "screenshots" / "eval_debug" / "last10")
    )

    analysis_cfg = {
        "ball_min_intensity": int(args.ball_min_intensity),
        "ball_motion_threshold": int(args.ball_motion_threshold),
        "ball_max_pixels": int(args.ball_max_pixels),
        "paddle_min_intensity": int(args.paddle_min_intensity),
        "paddle_band_top": int(args.paddle_band_top),
        "paddle_near_tolerance": int(args.paddle_near_tolerance),
        "paddle_x_margin": int(args.paddle_x_margin),
        "paddle_vy_threshold": float(args.paddle_vy_threshold),
        "paddle_min_gap_steps": int(args.paddle_min_gap_steps),
        "save_debug_screenshots": bool(args.save_debug_screenshots),
        "debug_screenshot_dir": debug_screenshot_dir,
        "debug_keep_last": int(args.debug_keep_last),
        "metrics_backend": str(args.metrics_backend),
    }

    run_id = build_run_id()

    print(f"Device:             {args.device}")
    print(f"Encoder checkpoint: {args.encoder_checkpoint}")
    print(f"Q checkpoint:       {args.q_checkpoint}")
    print(f"Latent dim:         {args.latent_dim}  context: {args.context_length}")
    print(f"Episodes:           {args.episodes}  ε={args.epsilon}  max_steps={args.max_steps}")
    print(f"Metrics backend:    {args.metrics_backend}")
    if args.metrics_backend == "ocatari":
        print(f"OCAtari mode:       {args.ocatari_mode}  hud={bool(args.ocatari_hud)}")
    print(f"Run ID:             {run_id}")
    print(f"Analysis dir:       {analysis_output_dir}")
    if args.save_debug_screenshots:
        print(f"Debug screenshots:  {debug_screenshot_dir} (keep last {args.debug_keep_last})")
    print()

    encoder, q_head = load_models(args)

    if args.metrics_backend == "ocatari":
        if OCAtari is None:
            raise ImportError(
                "metrics-backend=ocatari requires OCAtari. Install with: pip install ocatari"
            )
        env = OCAtari(
            "ALE/Breakout-v5",
            mode=args.ocatari_mode,
            hud=bool(args.ocatari_hud),
            render_mode="human" if args.render else None,
            frameskip=4,
            repeat_action_probability=0.0,
        )
        ale = get_ocatari_ale(env)
        if ale is None:
            raise RuntimeError("Failed to access ALE interface from OCAtari.")
        heatmap_shape = tuple(int(v) for v in ale.getScreenDims())
    else:
        env = create_breakout_env(
            render_mode="human" if args.render else None,
            seed=args.seed,
        )
        heatmap_shape = (FRAME_HEIGHT, FRAME_WIDTH)

    analysis_cfg["heatmap_shape"] = heatmap_shape

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    global_ball_heatmap = np.zeros(heatmap_shape, dtype=np.int64)
    total_brick_hits_proxy = 0
    total_paddle_hits_estimate = 0
    total_ball_detections = 0
    total_blocks_visible_mean = 0.0

    runs_csv_path = analysis_output_dir / "eval_runs.csv"
    episodes_csv_path = analysis_output_dir / "eval_episodes.csv"
    events_jsonl_path = analysis_output_dir / "paddle_hit_events.jsonl"

    run_timestamp = utc_now_iso()

    for ep in range(1, args.episodes + 1):
        ep_analysis = run_episode(
            env, encoder, q_head,
            args.context_length, args.epsilon, args.max_steps, args.device,
            analysis_cfg=analysis_cfg,
            run_id=run_id,
            episode_idx=ep,
        )
        episode_returns.append(ep_analysis.total_return)
        episode_lengths.append(ep_analysis.steps)

        global_ball_heatmap += ep_analysis.ball_heatmap
        total_brick_hits_proxy += ep_analysis.brick_hits_proxy
        total_paddle_hits_estimate += ep_analysis.paddle_hits_estimate
        total_ball_detections += ep_analysis.ball_detections
        if not np.isnan(ep_analysis.blocks_visible_mean):
            total_blocks_visible_mean += ep_analysis.blocks_visible_mean

        print(
            f"  Episode {ep:3d}:  return={ep_analysis.total_return:7.1f}"
            f"  steps={ep_analysis.steps:5d}"
            f"  brick_hits={ep_analysis.brick_hits_proxy:4d}"
            f"  paddle_hits≈{ep_analysis.paddle_hits_estimate:3d}"
            f"  ball_detect={ep_analysis.ball_detections:5d}"
            + (
                f"  blocks_vis≈{ep_analysis.blocks_visible_mean:5.2f}"
                if not np.isnan(ep_analysis.blocks_visible_mean)
                else ""
            )
        )

        append_csv_row(
            episodes_csv_path,
            row={
                "run_id": run_id,
                "timestamp_utc": run_timestamp,
                "episode": int(ep),
                "return": ep_analysis.total_return,
                "steps": ep_analysis.steps,
                "brick_hits_proxy": ep_analysis.brick_hits_proxy,
                "paddle_hits_estimate": ep_analysis.paddle_hits_estimate,
                "ball_detections": ep_analysis.ball_detections,
                "blocks_visible_mean": ep_analysis.blocks_visible_mean,
                "metrics_backend": args.metrics_backend,
            },
            fieldnames=[
                "run_id",
                "timestamp_utc",
                "episode",
                "return",
                "steps",
                "brick_hits_proxy",
                "paddle_hits_estimate",
                "ball_detections",
                "blocks_visible_mean",
                "metrics_backend",
            ],
        )

        for event in ep_analysis.paddle_hit_events:
            append_jsonl_row(events_jsonl_path, event)

    env.close()

    mean_ret = float(np.mean(episode_returns))
    std_ret = float(np.std(episode_returns))
    mean_len = float(np.mean(episode_lengths))
    ball_coverage_ratio = float(np.count_nonzero(global_ball_heatmap) / global_ball_heatmap.size)
    mean_blocks_visible = float(total_blocks_visible_mean / args.episodes) if args.metrics_backend == "ocatari" else float("nan")

    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_npy_path = analysis_output_dir / f"ball_heatmap_{run_id}.npy"
    np.save(heatmap_npy_path, global_ball_heatmap)

    heatmap_png_path = analysis_output_dir / f"ball_heatmap_{run_id}.png"
    save_ball_heatmap_png(
        heatmap_png_path,
        global_ball_heatmap,
        title=f"Ball spatial distribution ({run_id})",
    )

    append_csv_row(
        runs_csv_path,
        row={
            "run_id": run_id,
            "timestamp_utc": run_timestamp,
            "encoder_checkpoint": str(args.encoder_checkpoint),
            "q_checkpoint": str(args.q_checkpoint),
            "episodes": int(args.episodes),
            "epsilon": float(args.epsilon),
            "max_steps": int(args.max_steps),
            "metrics_backend": args.metrics_backend,
            "ocatari_mode": args.ocatari_mode if args.metrics_backend == "ocatari" else "",
            "ocatari_hud": int(bool(args.ocatari_hud)) if args.metrics_backend == "ocatari" else 0,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "min_return": float(min(episode_returns)),
            "max_return": float(max(episode_returns)),
            "mean_episode_len": mean_len,
            "total_brick_hits_proxy": int(total_brick_hits_proxy),
            "total_paddle_hits_estimate": int(total_paddle_hits_estimate),
            "total_ball_detections": int(total_ball_detections),
            "mean_blocks_visible": mean_blocks_visible,
            "ball_coverage_ratio": ball_coverage_ratio,
            "ball_min_intensity": analysis_cfg["ball_min_intensity"],
            "ball_motion_threshold": analysis_cfg["ball_motion_threshold"],
            "ball_max_pixels": analysis_cfg["ball_max_pixels"],
            "paddle_min_intensity": analysis_cfg["paddle_min_intensity"],
            "paddle_band_top": analysis_cfg["paddle_band_top"],
            "paddle_near_tolerance": analysis_cfg["paddle_near_tolerance"],
            "paddle_x_margin": analysis_cfg["paddle_x_margin"],
            "paddle_vy_threshold": analysis_cfg["paddle_vy_threshold"],
            "paddle_min_gap_steps": analysis_cfg["paddle_min_gap_steps"],
            "save_debug_screenshots": int(bool(args.save_debug_screenshots)),
            "heatmap_npy": str(heatmap_npy_path),
            "heatmap_png": str(heatmap_png_path),
        },
        fieldnames=[
            "run_id",
            "timestamp_utc",
            "encoder_checkpoint",
            "q_checkpoint",
            "episodes",
            "epsilon",
            "max_steps",
            "metrics_backend",
            "ocatari_mode",
            "ocatari_hud",
            "mean_return",
            "std_return",
            "min_return",
            "max_return",
            "mean_episode_len",
            "total_brick_hits_proxy",
            "total_paddle_hits_estimate",
            "total_ball_detections",
            "mean_blocks_visible",
            "ball_coverage_ratio",
            "ball_min_intensity",
            "ball_motion_threshold",
            "ball_max_pixels",
            "paddle_min_intensity",
            "paddle_band_top",
            "paddle_near_tolerance",
            "paddle_x_margin",
            "paddle_vy_threshold",
            "paddle_min_gap_steps",
            "save_debug_screenshots",
            "heatmap_npy",
            "heatmap_png",
        ],
    )

    print()
    print("=" * 52)
    print(f"  Episodes:          {args.episodes}")
    print(f"  Mean return:       {mean_ret:.2f} ± {std_ret:.2f}")
    print(f"  Min / Max return:  {min(episode_returns):.1f} / {max(episode_returns):.1f}")
    print(f"  Mean episode len:  {mean_len:.1f} steps")
    print(f"  Brick hits (proxy): {total_brick_hits_proxy}")
    print(f"  Paddle hits (est.): {total_paddle_hits_estimate}")
    print(f"  Ball detections:    {total_ball_detections}")
    if not np.isnan(mean_blocks_visible):
        print(f"  Blocks visible μ:   {mean_blocks_visible:.2f}")
    print(f"  Ball coverage:      {ball_coverage_ratio:.4f}")
    print(f"  Runs CSV:           {runs_csv_path}")
    print(f"  Episodes CSV:       {episodes_csv_path}")
    print(f"  Events JSONL:       {events_jsonl_path}")
    print(f"  Heatmap NPY:        {heatmap_npy_path}")
    print(f"  Heatmap PNG:        {heatmap_png_path}")
    print("=" * 52)


if __name__ == "__main__":
    main()
