# ArcadeJepa — JEPA Actor for Atari Breakout

**Goal:** Train an agent that learns to play *Breakout* without ever reconstructing pixels.
Instead of a VAE or a GAN, the agent builds a compact latent model of the world using
**Joint Embedding Predictive Architecture (JEPA)** principles, then trains a lightweight
policy head on top of the frozen latent space.

---

## Why JEPA for Breakout?

Breakout is an ideal JEPA benchmark because the "physics" of the game—ball trajectory,
paddle position, brick collisions—is consistent and predictable, while the visual frame
contains significant redundant information (static bricks, score text, background colour).

| Approach | What it optimises | What it ignores |
|---|---|---|
| **VAE / Autoencoder** | Reconstruct every pixel perfectly | Wastes capacity on irrelevant detail |
| **JEPA (this project)** | Predict the *next latent vector* | Never decodes back to pixels |

A JEPA encoder is forced to learn representations where "ball moving left at speed 2" is
meaningful, because that is exactly the information needed to predict the next embedding.
Static bricks are only encoded when the ball is about to hit them.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         JEPA World Model                        │
│                                                                  │
│  frames (t−3 … t)                      frames (t−2 … t+1)      │
│       │                                       │                 │
│  ┌────▼────┐  online encoder          ┌───────▼───────┐        │
│  │ ConvNet │─────────────────────────▶│ Target Encoder│ (EMA)  │
│  │  +  MLP │  z_t  ∈ ℝ^512           │  (no grad)    │        │
│  └────┬────┘                          └───────┬───────┘        │
│       │                                       │  z̄_{t+1}       │
│       │   action a_t                          │                 │
│  ┌────▼────────────┐                          │                 │
│  │ Predictor MLP   │── ẑ_{t+1} ──── L(ẑ, z̄) ◀┘                │
│  │ (action embed.) │   SmoothL1                                 │
│  └─────────────────┘                                            │
└────────────────────────────────────────────────────────────────┘
```

### Components

| Component | File | Role |
|---|---|---|
| `ConvEncoder` | `models/jepa.py` | CNN + MLP; maps a 4-frame grayscale stack `(4, 84, 84)` → `ℝ^512` |
| Target Encoder | `train_jepa.py` | EMA copy of `ConvEncoder`; provides stable training targets |
| `ActionConditionedPredictor` | `models/jepa.py` | Embeds discrete action, concatenates with latent, predicts next latent |

---

## Environment

| Setting | Value | Reason |
|---|---|---|
| **Env ID** | `ALE/Breakout-v5` | Modern Farama namespace; fully configurable |
| **Frameskip** | `4` | Agent acts on every 4th frame; gives the Predictor a *meaningful* state delta to learn |
| **Repeat action probability** | `0.0` | Deterministic transitions for cleaner dynamics learning |
| **Observation type** | Grayscale | Colour is not semantically meaningful in Breakout |
| **Resolution** | `84 × 84` | Standard Atari benchmark size; ~14× smaller than raw `210 × 160 RGB` |
| **Context window** | 4 frames | Provides implicit velocity / motion information to the encoder |

> **Why `frameskip=4` for JEPA?**
> With `frameskip=1` consecutive frames are nearly identical — the ball moves only a few pixels — so the Predictor
> can achieve near-zero loss by simply copying the previous latent state. This "lazy copy" solution is a
> degenerate local minimum that teaches the encoder nothing useful.
> At `frameskip=4` the ball travels ~16–20 pixels between observations, bricks can disappear, and the paddle
> visibly repositions. The Predictor *must* genuinely model dynamics to minimise the SmoothL1 target, which
> is precisely the representation pressure JEPA is designed to exploit.

### Normalization: two levels

JEPA's loss $\lVert \hat{z}_{t+1} - \bar{z}_{t+1} \rVert$ is sensitive to absolute scale at **both** the input and the latent level:

| Level | Where | What |
|---|---|---|
| **Pixel** | `dataset.py` → `PIXEL_NORM_SCALE` | Stored `uint8 [0, 255]` are multiplied by `1/255` at batch time → `float32 [0.0, 1.0]`. Prevents large initial gradients before the encoder has learned anything. |
| **Latent** | `ConvEncoder` final layer | `nn.LayerNorm(latent_dim)` bounds the magnitude of every latent vector. Without it the encoder can trivially minimise JEPA loss by inflating or collapsing latent norms rather than learning dynamics. |

---

## Data Format

Transitions are stored as compressed NumPy shards (`np.savez_compressed`) so large datasets
(e.g. 1 M transitions ≈ 7 GB) remain manageable without a full database.

Each `.npz` shard contains:

| Field | DType | Shape | Description |
|---|---|---|---|
| `obs` | `uint8` | `(N, 84, 84)` | Current frame (preprocessed) |
| `next_obs` | `uint8` | `(N, 84, 84)` | Next frame |
| `action` | `int64` | `(N,)` | Action taken at time `t` |
| `reward` | `float32` | `(N,)` | Environment reward `r_t` |
| `terminated` | `bool` | `(N,)` | Episode terminated flag |
| `truncated` | `bool` | `(N,)` | Episode truncated flag |
| `episode_id` | `int32` | `(N,)` | Episode identifier |
| `episode_step` | `int32` | `(N,)` | Step index inside episode |

A `run_summary.json` is written alongside the shards with env config metadata.

---

## Three-Stage Training Pipeline
## Training Pipeline

| Stage | Name | Input | Objective | Notes |
|---|---|---|---|---|
| 0 | Data Collection | Environment frames + random actions | Build offline transition shards | Random policy plays ~50k steps |
| 1 | JEPA Pretraining | Masked context frames, actions | Minimise `SmoothL1(ẑ_{t+1}, z̄_{t+1})` | EMA target encoder, object-centric pressure via masking |

### Stage 1 objective

| Component | Formula / Mechanism | Purpose |
|---|---|---|
| JEPA latent loss | `SmoothL1( Predictor(Encoder(masked x_t), a_t), TargetEncoder(x_{t+1}) )` | Learn world dynamics in latent space |
| Masking | Randomly zero ~50% spatial pixels in context | Prevent pixel-copy shortcuts |

---

## Project Layout

| Path | Purpose |
|---|---|
| `.venv/` | Local `uv` virtual environment |
| `Breakout/envs.py` | ALE env factory + torchvision resize wrapper |
| `Breakout/collect_random_data.py` | Random policy collector → compressed NPZ shards |
| `Breakout/dataset.py` | PyTorch dataset, context windows, horizon targets |
| `Breakout/models/jepa.py` | `ConvEncoder`, `ActionConditionedPredictor` |
| `Breakout/train_jepa.py` | Stage 1: masked JEPA pretraining with diagnostics |
| `Breakout/random_agent.py` | Smoke-test runner on `ALE/Breakout-v5` |

---

## Quickstart

| Step | Goal |
|---:|---|
| 1 | Set up environment |
| 2 | Verify installation |
| 3 | Collect random-policy data |
| 4 | Pretrain JEPA world model |

### 1. Set up the environment

```bash
uv venv .venv
uv pip install gymnasium ale-py autorom torch torchvision pillow
.venv/bin/AutoROM --accept-license
```

### 2. Verify the setup

```bash
.venv/bin/python Breakout/random_agent.py --steps 500
# Expected: ALE/Breakout-v5, Discrete(4), obs shape (84, 84)
```

### 3. Collect random-policy data

```bash
.venv/bin/python Breakout/collect_random_data.py \
    --num-steps 50000 \
    --shard-size 5000 \
    --output-dir Breakout/data/random
```

### 4. Pretrain the JEPA world model

```bash
.venv/bin/python Breakout/train_jepa.py \
    --data-dir    Breakout/data/random \
    --output-dir  Breakout/checkpoints/jepa \
    --epochs      20 \
    --batch-size  256 \
    --context-length 4
```
---

## Stack

| Package | Purpose |
|---|---|
| `gymnasium` | Atari environment API |
| `ale-py` | Arcade Learning Environment bindings |
| `torch` / `torchvision` | Neural networks, EMA updates, image resizing |
| `pillow` | Lightweight image I/O fallback |
| `numpy` | Shard storage and batch assembly |
| `autorom` | ROM licence management |

Python ≥ 3.11 · PyTorch ≥ 2.6 · ALE ≥ 0.11

---

## Roadmap

- [x] Action-conditioned masking in Stage 1 (`apply_random_mask`, 50% spatial dropout)
- [ ] Replace offline Q-learning with an online PPO loop using the frozen encoder
- [ ] Extend data collection with a partially-trained policy ("active student")
- [ ] Add TensorBoard / W&B logging to all training scripts
- [ ] Evaluate representations with linear probing (ball position, paddle position)
- [ ] Experiment with a small Vision Transformer (ViT) encoder in place of the CNN
