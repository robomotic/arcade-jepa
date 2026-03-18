from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

try:
    from .envs import ALE_BREAKOUT_ENV_ID, DEFAULT_OBS_SHAPE, create_breakout_env
except ImportError:
    from envs import ALE_BREAKOUT_ENV_ID, DEFAULT_OBS_SHAPE, create_breakout_env

try:
    from ocatari.core import OCAtari
except ImportError:
    OCAtari = None

try:
    from ocatari.ram._helper_methods import _convert_number as ocatari_convert_number
except ImportError:
    ocatari_convert_number = None

try:
    from ocatari.ram.breakout import _make_block_bitmap as ocatari_make_block_bitmap
except ImportError:
    ocatari_make_block_bitmap = None


@dataclass
class ShardBuffer:
    observations: list[np.ndarray] = field(default_factory=list)
    next_observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    episode_ids: list[int] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def append(
        self,
        observation: np.ndarray,
        next_observation: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        episode_id: int,
        episode_step: int,
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.uint8))
        self.next_observations.append(np.asarray(next_observation, dtype=np.uint8))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.episode_ids.append(int(episode_id))
        self.episode_steps.append(int(episode_step))

    def clear(self) -> None:
        self.observations.clear()
        self.next_observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.terminated.clear()
        self.truncated.clear()
        self.episode_ids.clear()
        self.episode_steps.clear()


@dataclass
class ObjectShardBuffer:
    ball_x: list[float] = field(default_factory=list)
    ball_y: list[float] = field(default_factory=list)
    ball_vx: list[float] = field(default_factory=list)
    ball_vy: list[float] = field(default_factory=list)
    paddle_x: list[float] = field(default_factory=list)
    paddle_y: list[float] = field(default_factory=list)
    paddle_w: list[float] = field(default_factory=list)
    paddle_h: list[float] = field(default_factory=list)
    block_count: list[int] = field(default_factory=list)
    player_score_x: list[float] = field(default_factory=list)
    player_score_y: list[float] = field(default_factory=list)
    player_score_w: list[float] = field(default_factory=list)
    player_score_h: list[float] = field(default_factory=list)
    live_x: list[float] = field(default_factory=list)
    live_y: list[float] = field(default_factory=list)
    live_w: list[float] = field(default_factory=list)
    live_h: list[float] = field(default_factory=list)
    player_number_x: list[float] = field(default_factory=list)
    player_number_y: list[float] = field(default_factory=list)
    player_number_w: list[float] = field(default_factory=list)
    player_number_h: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ball_x)

    def append(self, metrics: dict[str, float | int]) -> None:
        self.ball_x.append(float(metrics["ball_x"]))
        self.ball_y.append(float(metrics["ball_y"]))
        self.ball_vx.append(float(metrics["ball_vx"]))
        self.ball_vy.append(float(metrics["ball_vy"]))
        self.paddle_x.append(float(metrics["paddle_x"]))
        self.paddle_y.append(float(metrics["paddle_y"]))
        self.paddle_w.append(float(metrics["paddle_w"]))
        self.paddle_h.append(float(metrics["paddle_h"]))
        self.block_count.append(int(metrics["block_count"]))
        self.player_score_x.append(float(metrics["player_score_x"]))
        self.player_score_y.append(float(metrics["player_score_y"]))
        self.player_score_w.append(float(metrics["player_score_w"]))
        self.player_score_h.append(float(metrics["player_score_h"]))
        self.live_x.append(float(metrics["live_x"]))
        self.live_y.append(float(metrics["live_y"]))
        self.live_w.append(float(metrics["live_w"]))
        self.live_h.append(float(metrics["live_h"]))
        self.player_number_x.append(float(metrics["player_number_x"]))
        self.player_number_y.append(float(metrics["player_number_y"]))
        self.player_number_w.append(float(metrics["player_number_w"]))
        self.player_number_h.append(float(metrics["player_number_h"]))

    def clear(self) -> None:
        self.ball_x.clear()
        self.ball_y.clear()
        self.ball_vx.clear()
        self.ball_vy.clear()
        self.paddle_x.clear()
        self.paddle_y.clear()
        self.paddle_w.clear()
        self.paddle_h.clear()
        self.block_count.clear()
        self.player_score_x.clear()
        self.player_score_y.clear()
        self.player_score_w.clear()
        self.player_score_h.clear()
        self.live_x.clear()
        self.live_y.clear()
        self.live_w.clear()
        self.live_h.clear()
        self.player_number_x.clear()
        self.player_number_y.clear()
        self.player_number_w.clear()
        self.player_number_h.clear()


@dataclass
class RamDecodeShardBuffer:
    ram_raw: list[np.ndarray] = field(default_factory=list)
    ram_player_x_byte: list[int] = field(default_factory=list)
    ram_ball_x_byte: list[int] = field(default_factory=list)
    ram_ball_y_byte: list[int] = field(default_factory=list)
    ram_lives_byte: list[int] = field(default_factory=list)
    ram_score_hi_byte: list[int] = field(default_factory=list)
    ram_score_lo_byte: list[int] = field(default_factory=list)
    decoded_paddle_x: list[float] = field(default_factory=list)
    decoded_paddle_y: list[float] = field(default_factory=list)
    decoded_ball_x: list[float] = field(default_factory=list)
    decoded_ball_y: list[float] = field(default_factory=list)
    decoded_ball_vx: list[float] = field(default_factory=list)
    decoded_ball_vy: list[float] = field(default_factory=list)
    decoded_block_count: list[int] = field(default_factory=list)
    decoded_score: list[int] = field(default_factory=list)
    decoded_lives: list[int] = field(default_factory=list)
    decoded_player_number: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ram_raw)

    def append(self, row: dict[str, np.ndarray | float | int]) -> None:
        self.ram_raw.append(np.asarray(row["ram_raw"], dtype=np.uint8))
        self.ram_player_x_byte.append(int(row["ram_player_x_byte"]))
        self.ram_ball_x_byte.append(int(row["ram_ball_x_byte"]))
        self.ram_ball_y_byte.append(int(row["ram_ball_y_byte"]))
        self.ram_lives_byte.append(int(row["ram_lives_byte"]))
        self.ram_score_hi_byte.append(int(row["ram_score_hi_byte"]))
        self.ram_score_lo_byte.append(int(row["ram_score_lo_byte"]))
        self.decoded_paddle_x.append(float(row["decoded_paddle_x"]))
        self.decoded_paddle_y.append(float(row["decoded_paddle_y"]))
        self.decoded_ball_x.append(float(row["decoded_ball_x"]))
        self.decoded_ball_y.append(float(row["decoded_ball_y"]))
        self.decoded_ball_vx.append(float(row["decoded_ball_vx"]))
        self.decoded_ball_vy.append(float(row["decoded_ball_vy"]))
        self.decoded_block_count.append(int(row["decoded_block_count"]))
        self.decoded_score.append(int(row["decoded_score"]))
        self.decoded_lives.append(int(row["decoded_lives"]))
        self.decoded_player_number.append(int(row["decoded_player_number"]))

    def clear(self) -> None:
        self.ram_raw.clear()
        self.ram_player_x_byte.clear()
        self.ram_ball_x_byte.clear()
        self.ram_ball_y_byte.clear()
        self.ram_lives_byte.clear()
        self.ram_score_hi_byte.clear()
        self.ram_score_lo_byte.clear()
        self.decoded_paddle_x.clear()
        self.decoded_paddle_y.clear()
        self.decoded_ball_x.clear()
        self.decoded_ball_y.clear()
        self.decoded_ball_vx.clear()
        self.decoded_ball_vy.clear()
        self.decoded_block_count.clear()
        self.decoded_score.clear()
        self.decoded_lives.clear()
        self.decoded_player_number.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect random Breakout transitions into compressed NPZ shards.")
    parser.add_argument("--output-dir", type=Path, default=Path("Breakout/data/random"))
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--repeat-action-probability", type=float, default=0.0)
    parser.add_argument("--use-ocatari", action="store_true", help="Collect with OCAtari object extraction and sidecar NPZ metrics.")
    parser.add_argument("--ocatari-mode", type=str, choices=("ram", "vision"), default="ram")
    parser.add_argument("--ocatari-hud", action="store_true", help="Enable OCAtari HUD object extraction.")
    parser.add_argument("--video-path", type=Path, default=None, help="Optional debug video path (.mp4).")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-max-frames", type=int, default=10000)
    parser.add_argument("--launch-noop-min", type=int, default=1, help="Min no-op warmup steps before launch.")
    parser.add_argument("--launch-noop-max", type=int, default=20, help="Max no-op warmup steps before launch.")
    parser.add_argument("--launch-delay-min", type=int, default=1, help="Min additional no-op delay steps before FIRE.")
    parser.add_argument("--launch-delay-max", type=int, default=20, help="Max additional no-op delay steps before FIRE.")
    parser.add_argument("--launch-move-min", type=int, default=5, help="Min paddle move steps before FIRE to randomize launch angle.")
    parser.add_argument("--launch-move-max", type=int, default=40, help="Max paddle move steps before FIRE to randomize launch angle.")
    parser.add_argument("--noop-action", type=int, default=0, help="Action index used for no-op steps.")
    parser.add_argument("--fire-action", type=int, default=1, help="Action index used for FIRE launch.")
    parser.add_argument("--left-action", type=int, default=3, help="Action index for moving paddle left (Breakout=3).")
    parser.add_argument("--right-action", type=int, default=2, help="Action index for moving paddle right (Breakout=2).")
    parser.add_argument("--skip-fire-action", action="store_true", help="Skip FIRE after random no-op launch delays.")
    parser.add_argument("--sweep-paddle", action="store_true",
                        help="Deterministic paddle sweep: calibrate full range at startup then increment "
                             "paddle position by 1 step each episode (requires --use-ocatari).")
    parser.add_argument("--sweep-noop-delay", type=int, default=2,
                        help="No-op steps after paddle positioning before FIRE in sweep mode.")
    return parser.parse_args()


def flush_shard(buffer: ShardBuffer, output_dir: Path, shard_index: int) -> Path | None:
    if len(buffer) == 0:
        return None

    output_path = output_dir / f"transitions_{shard_index:05d}.npz"
    np.savez_compressed(
        output_path,
        obs=np.stack(buffer.observations).astype(np.uint8),
        next_obs=np.stack(buffer.next_observations).astype(np.uint8),
        action=np.asarray(buffer.actions, dtype=np.int64),
        reward=np.asarray(buffer.rewards, dtype=np.float32),
        terminated=np.asarray(buffer.terminated, dtype=np.bool_),
        truncated=np.asarray(buffer.truncated, dtype=np.bool_),
        episode_id=np.asarray(buffer.episode_ids, dtype=np.int32),
        episode_step=np.asarray(buffer.episode_steps, dtype=np.int32),
    )
    buffer.clear()
    return output_path


def flush_object_shard(buffer: ObjectShardBuffer, output_dir: Path, shard_index: int) -> Path | None:
    if len(buffer) == 0:
        return None

    output_path = output_dir / f"transitions_{shard_index:05d}_objects.npz"
    np.savez_compressed(
        output_path,
        ball_x=np.asarray(buffer.ball_x, dtype=np.float32),
        ball_y=np.asarray(buffer.ball_y, dtype=np.float32),
        ball_vx=np.asarray(buffer.ball_vx, dtype=np.float32),
        ball_vy=np.asarray(buffer.ball_vy, dtype=np.float32),
        paddle_x=np.asarray(buffer.paddle_x, dtype=np.float32),
        paddle_y=np.asarray(buffer.paddle_y, dtype=np.float32),
        paddle_w=np.asarray(buffer.paddle_w, dtype=np.float32),
        paddle_h=np.asarray(buffer.paddle_h, dtype=np.float32),
        block_count=np.asarray(buffer.block_count, dtype=np.int32),
        player_score_x=np.asarray(buffer.player_score_x, dtype=np.float32),
        player_score_y=np.asarray(buffer.player_score_y, dtype=np.float32),
        player_score_w=np.asarray(buffer.player_score_w, dtype=np.float32),
        player_score_h=np.asarray(buffer.player_score_h, dtype=np.float32),
        live_x=np.asarray(buffer.live_x, dtype=np.float32),
        live_y=np.asarray(buffer.live_y, dtype=np.float32),
        live_w=np.asarray(buffer.live_w, dtype=np.float32),
        live_h=np.asarray(buffer.live_h, dtype=np.float32),
        player_number_x=np.asarray(buffer.player_number_x, dtype=np.float32),
        player_number_y=np.asarray(buffer.player_number_y, dtype=np.float32),
        player_number_w=np.asarray(buffer.player_number_w, dtype=np.float32),
        player_number_h=np.asarray(buffer.player_number_h, dtype=np.float32),
    )
    buffer.clear()
    return output_path


def flush_ramdecode_shard(buffer: RamDecodeShardBuffer, output_dir: Path, shard_index: int) -> Path | None:
    if len(buffer) == 0:
        return None

    output_path = output_dir / f"transitions_{shard_index:05d}_ramdecode.npz"
    np.savez_compressed(
        output_path,
        ram_raw=np.stack(buffer.ram_raw).astype(np.uint8),
        ram_player_x_byte=np.asarray(buffer.ram_player_x_byte, dtype=np.uint8),
        ram_ball_x_byte=np.asarray(buffer.ram_ball_x_byte, dtype=np.uint8),
        ram_ball_y_byte=np.asarray(buffer.ram_ball_y_byte, dtype=np.uint8),
        ram_lives_byte=np.asarray(buffer.ram_lives_byte, dtype=np.uint8),
        ram_score_hi_byte=np.asarray(buffer.ram_score_hi_byte, dtype=np.uint8),
        ram_score_lo_byte=np.asarray(buffer.ram_score_lo_byte, dtype=np.uint8),
        decoded_paddle_x=np.asarray(buffer.decoded_paddle_x, dtype=np.float32),
        decoded_paddle_y=np.asarray(buffer.decoded_paddle_y, dtype=np.float32),
        decoded_ball_x=np.asarray(buffer.decoded_ball_x, dtype=np.float32),
        decoded_ball_y=np.asarray(buffer.decoded_ball_y, dtype=np.float32),
        decoded_ball_vx=np.asarray(buffer.decoded_ball_vx, dtype=np.float32),
        decoded_ball_vy=np.asarray(buffer.decoded_ball_vy, dtype=np.float32),
        decoded_block_count=np.asarray(buffer.decoded_block_count, dtype=np.int16),
        decoded_score=np.asarray(buffer.decoded_score, dtype=np.int32),
        decoded_lives=np.asarray(buffer.decoded_lives, dtype=np.int16),
        decoded_player_number=np.asarray(buffer.decoded_player_number, dtype=np.int16),
    )
    buffer.clear()
    return output_path


def resize_grayscale_frame(frame: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(frame, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=out_shape, mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).clamp(0, 255).to(torch.uint8).cpu().numpy()


def get_ale_from_ocatari(env) -> object:
    inner_env = getattr(env, "_env", None)
    if inner_env is None or not hasattr(inner_env, "unwrapped"):
        raise RuntimeError("Could not access wrapped ALE env from OCAtari.")
    unwrapped = inner_env.unwrapped
    if not hasattr(unwrapped, "ale"):
        raise RuntimeError("OCAtari wrapped env does not expose ALE interface.")
    return unwrapped.ale


def read_ocatari_metrics(env, prev_ball_xy: tuple[float, float] | None) -> tuple[dict[str, float | int], tuple[float, float] | None]:
    objects = [obj for obj in getattr(env, "objects", []) if obj is not None]

    def first_obj(class_name: str):
        return next((o for o in objects if type(o).__name__ == class_name and getattr(o, "w", 0) > 0), None)

    def bbox_or_nan(obj) -> tuple[float, float, float, float]:
        if obj is None:
            return (np.nan, np.nan, np.nan, np.nan)
        return (
            float(getattr(obj, "x", np.nan)),
            float(getattr(obj, "y", np.nan)),
            float(getattr(obj, "w", np.nan)),
            float(getattr(obj, "h", np.nan)),
        )

    ball = first_obj("Ball")
    player = first_obj("Player")
    player_score = first_obj("PlayerScore")
    live = first_obj("Live")
    player_number = first_obj("PlayerNumber")
    block_count = sum(
        1
        for o in objects
        if type(o).__name__ == "Block" and getattr(o, "w", 0) > 0 and getattr(o, "h", 0) > 0
    )

    if ball is None:
        ball_xy = None
        ball_x = np.nan
        ball_y = np.nan
        ball_vx = np.nan
        ball_vy = np.nan
    else:
        ball_x = float(getattr(ball, "x", np.nan))
        ball_y = float(getattr(ball, "y", np.nan))
        ball_xy = (ball_x, ball_y)
        if prev_ball_xy is None:
            ball_vx = np.nan
            ball_vy = np.nan
        else:
            ball_vx = float(ball_x - prev_ball_xy[0])
            ball_vy = float(ball_y - prev_ball_xy[1])

    paddle_x, paddle_y, paddle_w, paddle_h = bbox_or_nan(player)
    ps_x, ps_y, ps_w, ps_h = bbox_or_nan(player_score)
    l_x, l_y, l_w, l_h = bbox_or_nan(live)
    pn_x, pn_y, pn_w, pn_h = bbox_or_nan(player_number)

    metrics = {
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_vx": ball_vx,
        "ball_vy": ball_vy,
        "paddle_x": paddle_x,
        "paddle_y": paddle_y,
        "paddle_w": paddle_w,
        "paddle_h": paddle_h,
        "block_count": int(block_count),
        "player_score_x": ps_x,
        "player_score_y": ps_y,
        "player_score_w": ps_w,
        "player_score_h": ps_h,
        "live_x": l_x,
        "live_y": l_y,
        "live_w": l_w,
        "live_h": l_h,
        "player_number_x": pn_x,
        "player_number_y": pn_y,
        "player_number_w": pn_w,
        "player_number_h": pn_h,
    }
    return metrics, ball_xy


def decode_breakout_score(score_hi_byte: int, score_lo_byte: int) -> int:
    if ocatari_convert_number is not None:
        return int(ocatari_convert_number(int(score_hi_byte)) * 100 + ocatari_convert_number(int(score_lo_byte)))
    # Fallback BCD-like decoding.
    def bcd_to_int(v: int) -> int:
        return int((v // 16) * 10 + (v % 16))

    return int(bcd_to_int(int(score_hi_byte)) * 100 + bcd_to_int(int(score_lo_byte)))


def decode_block_count_from_ram(ram_state: np.ndarray) -> int:
    if ocatari_make_block_bitmap is not None:
        bitmap = ocatari_make_block_bitmap(ram_state)
        return int(np.asarray(bitmap, dtype=np.int32).sum())
    # Conservative fallback: population count over bytes 0..35.
    return int(sum(bin(int(v)).count("1") for v in ram_state[:36]))


def read_ramdecode_metrics(
    ram_state: np.ndarray,
    prev_ball_xy: tuple[float, float] | None,
) -> tuple[dict[str, np.ndarray | float | int], tuple[float, float] | None]:
    ram_player_x = int(ram_state[72])
    ram_ball_x = int(ram_state[99])
    ram_ball_y = int(ram_state[101])
    ram_lives = int(ram_state[57])
    ram_score_hi = int(ram_state[76])
    ram_score_lo = int(ram_state[77])

    paddle_x = float(ram_player_x - 47)
    paddle_y = 189.0

    ball_visible = (ram_ball_y + 9 <= 196) and (ram_ball_y != 0)
    if ball_visible:
        ball_x = float(ram_ball_x - 49)
        ball_y = float(ram_ball_y + 9)
        current_ball_xy = (ball_x, ball_y)
        if prev_ball_xy is None:
            ball_vx = np.nan
            ball_vy = np.nan
        else:
            ball_vx = float(ball_x - prev_ball_xy[0])
            ball_vy = float(ball_y - prev_ball_xy[1])
    else:
        ball_x = np.nan
        ball_y = np.nan
        ball_vx = np.nan
        ball_vy = np.nan
        current_ball_xy = None

    row = {
        "ram_raw": np.asarray(ram_state, dtype=np.uint8),
        "ram_player_x_byte": ram_player_x,
        "ram_ball_x_byte": ram_ball_x,
        "ram_ball_y_byte": ram_ball_y,
        "ram_lives_byte": ram_lives,
        "ram_score_hi_byte": ram_score_hi,
        "ram_score_lo_byte": ram_score_lo,
        "decoded_paddle_x": paddle_x,
        "decoded_paddle_y": paddle_y,
        "decoded_ball_x": ball_x,
        "decoded_ball_y": ball_y,
        "decoded_ball_vx": ball_vx,
        "decoded_ball_vy": ball_vy,
        "decoded_block_count": decode_block_count_from_ram(ram_state),
        "decoded_score": decode_breakout_score(ram_score_hi, ram_score_lo),
        "decoded_lives": ram_lives,
        # Breakout random data is collected in single-player mode.
        "decoded_player_number": 1,
    }
    return row, current_ball_xy


def save_heatmap_png(
    heatmap: np.ndarray,
    out_path: Path,
    title: str,
    xlabel: str = "x",
    ylabel: str = "y",
    cmap: str = "magma",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=180)
    ax.imshow(heatmap, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_heatmap_frame(heatmap: np.ndarray, title: str, cmap: str = "magma") -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=150)
    ax.imshow(heatmap, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4)[..., :3].copy()
    plt.close(fig)
    return frame


def create_gif_from_frames(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    if imageio is None or not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(round(1000.0 / max(1, fps))))
    imageio.mimsave(str(out_path), frames, duration=duration_ms / 1000.0, loop=0)


def create_gif_from_video(video_path: Path, gif_path: Path, max_frames: int = 240, sample_every: int = 4, fps: int = 12) -> None:
    if imageio is None or cv2 is None or not video_path.exists():
        return
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    frame_idx = 0
    while capture.isOpened() and len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % max(1, sample_every) == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    capture.release()
    create_gif_from_frames(frames, gif_path, fps=fps)


def compute_paddle_touches(
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    paddle_x: np.ndarray,
    paddle_w: np.ndarray,
    paddle_y: np.ndarray,
    episode_ids: np.ndarray,
    near_tolerance: float = 4.0,
    x_margin: float = 2.0,
    vy_threshold: float = 1.0,
    min_gap_steps: int = 2,
) -> int:
    total = 0
    last_touch_step = -10_000
    for idx in range(2, len(ball_x)):
        if episode_ids[idx] != episode_ids[idx - 1] or episode_ids[idx] != episode_ids[idx - 2]:
            last_touch_step = -10_000
            continue
        window = [ball_x[idx - 2], ball_x[idx - 1], ball_x[idx], ball_y[idx - 2], ball_y[idx - 1], ball_y[idx]]
        if any(np.isnan(v) for v in window):
            continue
        if np.isnan(paddle_x[idx - 1]) or np.isnan(paddle_w[idx - 1]) or np.isnan(paddle_y[idx - 1]):
            continue
        vy_prev = float(ball_y[idx - 1] - ball_y[idx - 2])
        vy_curr = float(ball_y[idx] - ball_y[idx - 1])
        if vy_prev < vy_threshold or vy_curr > -vy_threshold:
            continue
        px = float(paddle_x[idx - 1])
        pw = float(paddle_w[idx - 1])
        py = float(paddle_y[idx - 1])
        bx = float(ball_x[idx - 1])
        by = float(ball_y[idx - 1])
        if not ((px - x_margin) <= bx <= (px + pw + x_margin)):
            continue
        if not ((py - near_tolerance) <= by <= (py + near_tolerance + 4.0)):
            continue
        if idx - last_touch_step < min_gap_steps:
            continue
        total += 1
        last_touch_step = idx
    return total


def build_collection_artifacts(
    output_dir: Path,
    transition_paths: list[Path],
    object_paths: list[Path],
    ramdecode_paths: list[Path],
    video_path: Path | None,
    video_fps: int,
) -> tuple[dict[str, str], dict[str, int | float | str]]:
    rewards: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    for path in transition_paths:
        shard = np.load(path)
        rewards.append(np.asarray(shard["reward"], dtype=np.float32))
        episode_ids.append(np.asarray(shard["episode_id"], dtype=np.int32))

    object_ball_x: list[np.ndarray] = []
    object_ball_y: list[np.ndarray] = []
    object_paddle_x: list[np.ndarray] = []
    object_paddle_y: list[np.ndarray] = []
    object_paddle_w: list[np.ndarray] = []
    object_block_count: list[np.ndarray] = []
    for path in object_paths:
        shard = np.load(path)
        object_ball_x.append(np.asarray(shard["ball_x"], dtype=np.float32))
        object_ball_y.append(np.asarray(shard["ball_y"], dtype=np.float32))
        object_paddle_x.append(np.asarray(shard["paddle_x"], dtype=np.float32))
        object_paddle_y.append(np.asarray(shard["paddle_y"], dtype=np.float32))
        object_paddle_w.append(np.asarray(shard["paddle_w"], dtype=np.float32))
        object_block_count.append(np.asarray(shard["block_count"], dtype=np.int32))

    ram_ball_x: list[np.ndarray] = []
    ram_ball_y: list[np.ndarray] = []
    ram_block_count: list[np.ndarray] = []
    ram_score: list[np.ndarray] = []
    ram_lives: list[np.ndarray] = []
    for path in ramdecode_paths:
        shard = np.load(path)
        ram_ball_x.append(np.asarray(shard["decoded_ball_x"], dtype=np.float32))
        ram_ball_y.append(np.asarray(shard["decoded_ball_y"], dtype=np.float32))
        ram_block_count.append(np.asarray(shard["decoded_block_count"], dtype=np.int32))
        ram_score.append(np.asarray(shard["decoded_score"], dtype=np.int32))
        ram_lives.append(np.asarray(shard["decoded_lives"], dtype=np.int16))

    all_rewards = np.concatenate(rewards) if rewards else np.zeros((0,), dtype=np.float32)
    all_episode_ids = np.concatenate(episode_ids) if episode_ids else np.zeros((0,), dtype=np.int32)
    all_ball_x = np.concatenate(ram_ball_x if ram_ball_x else object_ball_x) if (ram_ball_x or object_ball_x) else np.zeros((0,), dtype=np.float32)
    all_ball_y = np.concatenate(ram_ball_y if ram_ball_y else object_ball_y) if (ram_ball_y or object_ball_y) else np.zeros((0,), dtype=np.float32)
    all_paddle_x = np.concatenate(object_paddle_x) if object_paddle_x else np.zeros((0,), dtype=np.float32)
    all_paddle_y = np.concatenate(object_paddle_y) if object_paddle_y else np.zeros((0,), dtype=np.float32)
    all_paddle_w = np.concatenate(object_paddle_w) if object_paddle_w else np.zeros((0,), dtype=np.float32)
    all_block_count = np.concatenate(ram_block_count if ram_block_count else object_block_count) if (ram_block_count or object_block_count) else np.zeros((0,), dtype=np.int32)
    all_scores = np.concatenate(ram_score) if ram_score else np.zeros((0,), dtype=np.int32)
    all_lives = np.concatenate(ram_lives) if ram_lives else np.zeros((0,), dtype=np.int16)

    ball_heatmap = np.zeros((210, 160), dtype=np.int32)
    for x, y in zip(all_ball_x, all_ball_y):
        if np.isnan(x) or np.isnan(y):
            continue
        xi = int(np.clip(round(float(x)), 0, 159))
        yi = int(np.clip(round(float(y)), 0, 209))
        ball_heatmap[yi, xi] += 1

    paddle_heatmap = np.zeros((210, 160), dtype=np.int32)
    for px, py, pw in zip(all_paddle_x, all_paddle_y, all_paddle_w):
        if np.isnan(px) or np.isnan(py) or np.isnan(pw):
            continue
        y0 = int(np.clip(round(float(py)), 0, 209))
        x0 = int(np.clip(round(float(px)), 0, 159))
        x1 = int(np.clip(round(float(px + max(1.0, pw))), 0, 159))
        paddle_heatmap[y0, x0 : x1 + 1] += 1

    total_rewards = float(np.sum(all_rewards))
    total_bricks_broken = 0
    if len(all_block_count) > 1:
        for idx in range(1, len(all_block_count)):
            if all_episode_ids[idx] != all_episode_ids[idx - 1]:
                continue
            total_bricks_broken += max(0, int(all_block_count[idx - 1]) - int(all_block_count[idx]))
    total_paddle_touches = compute_paddle_touches(
        ball_x=all_ball_x,
        ball_y=all_ball_y,
        paddle_x=all_paddle_x,
        paddle_w=all_paddle_w,
        paddle_y=all_paddle_y,
        episode_ids=all_episode_ids,
    )

    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ball_heatmap_png = artifacts_dir / "ball_heatmap.png"
    paddle_heatmap_png = artifacts_dir / "paddle_heatmap.png"
    ball_heatmap_gif = artifacts_dir / "ball_heatmap.gif"
    debug_gif_path = artifacts_dir / "random_debug.gif"
    stats_json_path = artifacts_dir / "collection_stats.json"

    save_heatmap_png(ball_heatmap, ball_heatmap_png, title="Ball spatial heatmap")
    save_heatmap_png(paddle_heatmap, paddle_heatmap_png, title="Paddle movement heatmap", cmap="viridis")

    if len(all_ball_x) > 0:
        valid_points = [(x, y) for x, y in zip(all_ball_x, all_ball_y) if not (np.isnan(x) or np.isnan(y))]
        if valid_points:
            frame_count = min(36, max(8, len(valid_points) // 200))
            cumulative = np.zeros((210, 160), dtype=np.int32)
            frames: list[np.ndarray] = []
            checkpoints = np.linspace(1, len(valid_points), num=frame_count, dtype=int)
            point_index = 0
            for checkpoint in checkpoints:
                while point_index < checkpoint:
                    x, y = valid_points[point_index]
                    cumulative[int(np.clip(round(float(y)), 0, 209)), int(np.clip(round(float(x)), 0, 159))] += 1
                    point_index += 1
                frames.append(render_heatmap_frame(cumulative, title=f"Ball heatmap ({point_index} samples)"))
            create_gif_from_frames(frames, ball_heatmap_gif, fps=8)

    if video_path is not None:
        create_gif_from_video(video_path, debug_gif_path, fps=min(12, max(4, video_fps // 2)))

    stats = {
        "total_rewards": total_rewards,
        "total_bricks_broken": int(total_bricks_broken),
        "total_paddle_touches": int(total_paddle_touches),
        "num_transitions": int(len(all_rewards)),
        "num_episodes": int(len(np.unique(all_episode_ids))) if len(all_episode_ids) else 0,
        "max_score_seen": int(np.max(all_scores)) if len(all_scores) else 0,
        "min_lives_seen": int(np.min(all_lives)) if len(all_lives) else 0,
    }
    stats_json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    artifact_paths = {
        "artifacts_dir": str(artifacts_dir),
        "ball_heatmap_png": str(ball_heatmap_png),
        "ball_heatmap_gif": str(ball_heatmap_gif),
        "paddle_heatmap_png": str(paddle_heatmap_png),
        "debug_gif_path": str(debug_gif_path),
        "collection_stats_json": str(stats_json_path),
    }
    return artifact_paths, stats


def write_run_summary(
    *,
    output_dir: Path,
    num_steps: int,
    shard_paths: list[Path],
    seed: int,
    frameskip: int,
    repeat_action_probability: float,
    use_ocatari: bool,
    ocatari_mode: str,
    ocatari_hud: bool,
    object_shards: list[str],
    ramdecode_shards: list[str],
    video_path: Path | None,
    randomized_launch: dict[str, int | bool] | None = None,
    artifact_paths: dict[str, str] | None = None,
    collection_stats: dict[str, int | float | str] | None = None,
) -> None:
    summary_path = output_dir / "run_summary.json"
    summary = {
        "env_id": ALE_BREAKOUT_ENV_ID,
        "observation_shape": list(DEFAULT_OBS_SHAPE),
        "num_steps": num_steps,
        "num_shards": len(shard_paths),
        "seed": seed,
        "frameskip": frameskip,
        "repeat_action_probability": repeat_action_probability,
        "use_ocatari": use_ocatari,
        "ocatari_mode": ocatari_mode,
        "ocatari_hud": bool(ocatari_hud),
        "shards": [path.name for path in shard_paths],
        "object_shards": object_shards,
        "ramdecode_shards": ramdecode_shards,
        "video_path": str(video_path) if video_path is not None else "",
        "randomized_launch": randomized_launch or {},
        "artifact_paths": artifact_paths or {},
        "collection_stats": collection_stats or {},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")



def _ale_action(ale, gym_action: int) -> int:
    """Translate a gymnasium action index to the raw ALE action code.

    The gymnasium ALE wrapper maps action indices to game-specific minimal
    action set codes.  ``ale.act()`` requires the raw code, not the index.
    """
    return int(ale.getMinimalActionSet()[gym_action])


def calibrate_paddle_range(
    env,
    ale,
    left_action: int,
    right_action: int,
    max_frames: int = 300,
) -> int:
    """Calibrate the full paddle travel range at single-frame resolution.

    Uses ``ale.act()`` directly (bypassing the frameskip wrapper) so that each
    step is one ALE frame.  Pushes left to the wall, then counts single-frame
    RIGHT steps to reach the right wall.  Returns sweep_width (minimum 1).
    """
    env.reset()
    ale_left = _ale_action(ale, left_action)
    ale_right = _ale_action(ale, right_action)
    # Push to left wall at 1-frame resolution
    prev_x = int(ale.getRAM()[72])
    stuck = 0
    for _ in range(max_frames):
        ale.act(ale_left)
        x = int(ale.getRAM()[72])
        if x == prev_x:
            stuck += 1
            if stuck >= 5:
                break
        else:
            stuck = 0
        prev_x = x

    # Count right-ward frames to reach the right wall
    prev_x = int(ale.getRAM()[72])
    width = 0
    stuck = 0
    for _ in range(max_frames):
        ale.act(ale_right)
        x = int(ale.getRAM()[72])
        if x == prev_x:
            stuck += 1
            if stuck >= 5:
                break
        else:
            stuck = 0
            width += 1
        prev_x = x

    return max(1, width)


def sweep_launch_reset(
    env,
    ale,
    sweep_step: int,
    sweep_width: int,
    left_action: int,
    right_action: int,
    fire_action: int,
    noop_action: int,
    noop_delay: int = 2,
) -> tuple[np.ndarray, dict]:
    """Position the paddle at a deterministic location then launch.

    Uses single-frame ``ale.act()`` steps (bypassing frameskip) so the full
    sweep_width range gives fine-grained paddle positions (~28 unique slots at
    Breakout's native 4.86 px/frame speed).  Each episode increments
    ``sweep_step`` by 1, cycling through 0..sweep_width.
    """
    env.reset()
    ale_left = _ale_action(ale, left_action)
    ale_right = _ale_action(ale, right_action)
    ale_noop = _ale_action(ale, noop_action)
    # Push to left wall with margin using single-frame steps
    for _ in range(sweep_width + 20):
        ale.act(ale_left)
    # Advance to the desired position (one ALE frame per step)
    target_offset = sweep_step % (sweep_width + 1)
    for _ in range(target_offset):
        ale.act(ale_right)
    # Small noop pause so the ball can appear on screen
    for _ in range(noop_delay):
        ale.act(ale_noop)
    # Fire via env.step so OCAtari processes the frame correctly
    obs, _reward, terminated, truncated, info = env.step(fire_action)
    if terminated or truncated:
        obs, info = env.reset()
    return obs, info


def randomized_launch_reset(
    env,
    rng: np.random.Generator,
    *,
    launch_noop_min: int,
    launch_noop_max: int,
    launch_delay_min: int,
    launch_delay_max: int,
    launch_move_min: int,
    launch_move_max: int,
    noop_action: int,
    fire_action: int,
    left_action: int,
    right_action: int,
    skip_fire_action: bool,
) -> tuple[np.ndarray, dict]:
    obs, info = env.reset()

    warmup_steps = int(rng.integers(launch_noop_min, launch_noop_max + 1))
    delay_steps = int(rng.integers(launch_delay_min, launch_delay_max + 1))
    move_steps = int(rng.integers(launch_move_min, launch_move_max + 1))

    for _ in range(warmup_steps):
        obs, _reward, terminated, truncated, info = env.step(noop_action)
        if terminated or truncated:
            obs, info = env.reset()

    # Randomly move the paddle left/right to place it in a varied position.
    # This causes the ball to launch at different angles when FIRE is pressed.
    move_actions = rng.choice([left_action, right_action], size=move_steps)
    for act in move_actions:
        obs, _reward, terminated, truncated, info = env.step(int(act))
        if terminated or truncated:
            obs, info = env.reset()

    for _ in range(delay_steps):
        obs, _reward, terminated, truncated, info = env.step(noop_action)
        if terminated or truncated:
            obs, info = env.reset()

    if not skip_fire_action:
        obs, _reward, terminated, truncated, info = env.step(fire_action)
        if terminated or truncated:
            obs, info = env.reset()

    return obs, info


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.sweep_paddle and not args.use_ocatari:
        raise ValueError("--sweep-paddle requires --use-ocatari for ALE RAM-based paddle detection.")

    if args.use_ocatari:
        if OCAtari is None:
            raise ImportError("--use-ocatari requires package 'ocatari'. Install with: pip install ocatari")
        env = OCAtari(
            ALE_BREAKOUT_ENV_ID,
            mode=args.ocatari_mode,
            hud=bool(args.ocatari_hud),
            render_mode=None,
            frameskip=args.frameskip,
            repeat_action_probability=args.repeat_action_probability,
        )
        ale = get_ale_from_ocatari(env)
        _obs_raw, _ = env.reset(seed=args.seed)
        observation = resize_grayscale_frame(ale.getScreenGrayscale(), tuple(DEFAULT_OBS_SHAPE))
        prev_ball_xy_obj: tuple[float, float] | None = None
        prev_ball_xy_ram: tuple[float, float] | None = None
    else:
        env = create_breakout_env(
            frameskip=args.frameskip,
            repeat_action_probability=args.repeat_action_probability,
            seed=args.seed,
        )
        observation, _ = env.reset(seed=args.seed)
        prev_ball_xy_obj = None
        prev_ball_xy_ram = None

    # Calibrate paddle range for sweep mode.
    sweep_width: int = 0
    sweep_step: int = 0
    if args.sweep_paddle:
        sweep_width = calibrate_paddle_range(
            env, ale,
            left_action=int(args.left_action),
            right_action=int(args.right_action),
        )
        print(f"Paddle sweep calibration: sweep_width={sweep_width} steps ({sweep_width + 1} unique positions)")

    def _do_launch_reset() -> None:
        """Perform the appropriate launch reset and refresh `observation`."""
        nonlocal observation, sweep_step, prev_ball_xy_obj, prev_ball_xy_ram
        if args.sweep_paddle:
            _obs, _ = sweep_launch_reset(
                env, ale,
                sweep_step=sweep_step,
                sweep_width=sweep_width,
                left_action=int(args.left_action),
                right_action=int(args.right_action),
                fire_action=int(args.fire_action),
                noop_action=int(args.noop_action),
                noop_delay=int(args.sweep_noop_delay),
            )
        else:
            _obs, _ = randomized_launch_reset(
                env,
                rng,
                launch_noop_min=int(args.launch_noop_min),
                launch_noop_max=int(args.launch_noop_max),
                launch_delay_min=int(args.launch_delay_min),
                launch_delay_max=int(args.launch_delay_max),
                launch_move_min=int(args.launch_move_min),
                launch_move_max=int(args.launch_move_max),
                noop_action=int(args.noop_action),
                fire_action=int(args.fire_action),
                left_action=int(args.left_action),
                right_action=int(args.right_action),
                skip_fire_action=bool(args.skip_fire_action),
            )
        if args.use_ocatari:
            observation = resize_grayscale_frame(ale.getScreenGrayscale(), tuple(DEFAULT_OBS_SHAPE))
            prev_ball_xy_obj = None
            prev_ball_xy_ram = None
        else:
            observation = np.asarray(_obs, dtype=np.uint8)

    _do_launch_reset()

    env.action_space.seed(args.seed)

    buffer = ShardBuffer()
    object_buffer = ObjectShardBuffer()
    ramdecode_buffer = RamDecodeShardBuffer()
    shard_paths: list[Path] = []
    object_shard_names: list[str] = []
    ramdecode_shard_names: list[str] = []
    episode_id = 0
    episode_step = 0

    video_path = args.video_path
    video_writer = None
    video_frames_written = 0
    if cv2 is not None and video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)

    for global_step in range(args.num_steps):
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, _ = env.step(action)

        if args.use_ocatari:
            ram_state = np.asarray(ale.getRAM(), dtype=np.uint8)
            gray_next = np.asarray(ale.getScreenGrayscale(), dtype=np.uint8)
            processed_next = resize_grayscale_frame(gray_next, tuple(DEFAULT_OBS_SHAPE))
            metrics, prev_ball_xy_obj = read_ocatari_metrics(env, prev_ball_xy_obj)
            object_buffer.append(metrics)
            ram_row, prev_ball_xy_ram = read_ramdecode_metrics(ram_state, prev_ball_xy_ram)
            ramdecode_buffer.append(ram_row)
        else:
            processed_next = np.asarray(next_observation, dtype=np.uint8)

        buffer.append(
            observation=observation,
            next_observation=processed_next,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            episode_id=episode_id,
            episode_step=episode_step,
        )

        if video_path is not None and cv2 is not None and video_frames_written < int(args.video_max_frames):
            if args.use_ocatari:
                video_frame = np.asarray(ale.getScreenRGB(), dtype=np.uint8)
            else:
                rgb_frame = env.render()
                video_frame = None if rgb_frame is None else np.asarray(rgb_frame, dtype=np.uint8)
            if video_frame is not None:
                if video_writer is None:
                    h, w = int(video_frame.shape[0]), int(video_frame.shape[1])
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, float(max(1, int(args.video_fps))), (w, h))
                if video_writer is not None:
                    bgr_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(bgr_frame)
                    video_frames_written += 1

        if len(buffer) >= args.shard_size:
            shard_path = flush_shard(buffer, args.output_dir, len(shard_paths))
            if shard_path is not None:
                shard_paths.append(shard_path)
                if args.use_ocatari:
                    obj_path = flush_object_shard(object_buffer, args.output_dir, len(shard_paths) - 1)
                    if obj_path is not None:
                        object_shard_names.append(obj_path.name)
                    ram_path = flush_ramdecode_shard(ramdecode_buffer, args.output_dir, len(shard_paths) - 1)
                    if ram_path is not None:
                        ramdecode_shard_names.append(ram_path.name)
                print(f"Wrote {shard_path} ({global_step + 1} / {args.num_steps} transitions)")

        if terminated or truncated:
            sweep_step += 1
            _do_launch_reset()
            episode_id += 1
            episode_step = 0
        else:
            observation = processed_next
            episode_step += 1

    final_shard = flush_shard(buffer, args.output_dir, len(shard_paths))
    if final_shard is not None:
        shard_paths.append(final_shard)
        if args.use_ocatari:
            obj_path = flush_object_shard(object_buffer, args.output_dir, len(shard_paths) - 1)
            if obj_path is not None:
                object_shard_names.append(obj_path.name)
            ram_path = flush_ramdecode_shard(ramdecode_buffer, args.output_dir, len(shard_paths) - 1)
            if ram_path is not None:
                ramdecode_shard_names.append(ram_path.name)
        print(f"Wrote {final_shard} (final shard)")

    if video_writer is not None:
        video_writer.release()

    artifact_paths: dict[str, str] | None = None
    collection_stats: dict[str, int | float | str] | None = None
    if args.use_ocatari and object_shard_names:
        artifact_paths, collection_stats = build_collection_artifacts(
            output_dir=args.output_dir,
            transition_paths=shard_paths,
            object_paths=[args.output_dir / name for name in object_shard_names],
            ramdecode_paths=[args.output_dir / name for name in ramdecode_shard_names],
            video_path=video_path,
            video_fps=int(args.video_fps),
        )

    write_run_summary(
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        shard_paths=shard_paths,
        seed=args.seed,
        frameskip=args.frameskip,
        repeat_action_probability=args.repeat_action_probability,
        use_ocatari=bool(args.use_ocatari),
        ocatari_mode=str(args.ocatari_mode),
        ocatari_hud=bool(args.ocatari_hud),
        object_shards=object_shard_names,
        ramdecode_shards=ramdecode_shard_names,
        video_path=video_path,
        randomized_launch={
            "mode": "sweep" if args.sweep_paddle else "random",
            "sweep_paddle": bool(args.sweep_paddle),
            "sweep_width": sweep_width,
            "sweep_noop_delay": int(args.sweep_noop_delay),
            "launch_noop_min": int(args.launch_noop_min),
            "launch_noop_max": int(args.launch_noop_max),
            "launch_delay_min": int(args.launch_delay_min),
            "launch_delay_max": int(args.launch_delay_max),
            "launch_move_min": int(args.launch_move_min),
            "launch_move_max": int(args.launch_move_max),
            "noop_action": int(args.noop_action),
            "fire_action": int(args.fire_action),
            "left_action": int(args.left_action),
            "right_action": int(args.right_action),
            "skip_fire_action": bool(args.skip_fire_action),
        },
        artifact_paths=artifact_paths,
        collection_stats=collection_stats,
    )
    env.close()

    print(f"Collected {args.num_steps} transitions into {len(shard_paths)} shard(s) at {args.output_dir}")
    if args.use_ocatari:
        print(f"Collected {len(object_shard_names)} object sidecar shard(s) with OCAtari metrics")
        print(f"Collected {len(ramdecode_shard_names)} RAM-decoded sidecar shard(s)")
        if collection_stats is not None:
            print(f"Total rewards: {collection_stats['total_rewards']}")
            print(f"Total bricks broken: {collection_stats['total_bricks_broken']}")
            print(f"Total paddle touches: {collection_stats['total_paddle_touches']}")
    if video_path is not None:
        print(f"Debug video: {video_path}")
    if artifact_paths is not None:
        print(f"Artifacts: {artifact_paths['artifacts_dir']}")


if __name__ == "__main__":
    main()
