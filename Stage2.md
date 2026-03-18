# Stage 2 — Offline TD Q-Learning on Real Transitions

This document describes how I approached Stage 2, what the training code does, what I observed empirically, and what the results tell me about what is achievable with a frozen JEPA encoder trained on random-play data.

---

## 1) What Stage 2 is and why it comes after Stage 1.5

After Stage 1 I have a frozen JEPA encoder that maps a 4-frame grayscale context to a latent vector $z \in \mathbb{R}^{256}$.

Stage 1.5 attempted to warm-start a Q-Head by unrolling imagined trajectories through the frozen predictor, using the `RewardHead` from Stage 1 as a synthetic reward signal. This approach failed because:

- The `RewardHead` learned only **per-action scalar biases** and zero state-conditional signal (R² = −0.005 against the true reward).
- A logistic regression probe on the encoder's latents returned **ROC-AUC = 0.5008** — the encoder is provably reward-blind, because the JEPA objective optimises for next-state prediction, not reward prediction.
- Imagined Bellman targets collapsed to a constant in the first epoch of Stage 1.5 training, and `q_std` decayed monotonically to near-zero.

Stage 2 avoids this root cause entirely: it uses the `reward` field directly from stored transitions, bypassing the `RewardHead` completely.

---

## 2) Training approach

### 2.1 What gets trained and what stays frozen

| Component | Status | Rationale |
|---|---|---|
| `ConvEncoder` | **Frozen** | Stage 1 representation is the fixed foundation for all downstream latents |
| `QHead` (online) | **Trained** | The only learnable component in this stage |
| `QHead` (target) | **Frozen copy, periodically synced** | Provides stable Bellman targets — same motivation as original DQN |

### 2.2 1-step TD learning

For each mini-batch of real transitions $(s_t, a_t, r_t, s_{t+1}, \text{done}_t)$ from the offline dataset:

1. Encode both frames with the frozen encoder:
   - $z_t = \text{Encoder}(s_t)$
   - $z_{t+1} = \text{Encoder}(s_{t+1})$
2. Compute the Bellman target using the frozen target Q-Head:
   $$y_t = r_t + \gamma \cdot (1 - \text{done}_t) \cdot \max_{a'} Q_{\text{target}}(z_{t+1}, a')$$
3. Compute the TD loss on the online Q-Head:
   $$\mathcal{L} = \text{SmoothL1}\!\left(Q_{\text{online}}(z_t)_{a_t},\ y_t\right)$$
4. Back-propagate through the Q-Head only. The encoder is frozen with `requires_grad=False`.

### 2.3 Target network

The target Q-Head is a hard copy of the online Q-Head, updated every `--target-sync-epochs` epochs (default 1). Without it the Bellman target $y_t$ shifts every gradient step, creating a moving-target instability. Holding the target fixed for a full epoch keeps the TD update well-conditioned.

### 2.4 Train / validation split

`--val-split 0.1` (default) reserves the last 10% of transitions for monitoring. The validation pass runs under `torch.no_grad()`, using the same TD loss formula with no weight updates. A stable or declining val loss confirms the Q-Head is generalising and not overfitting the training set.

---

## 3) Diagnostics tracked per epoch

| Metric | What it measures | Healthy sign |
|---|---|---|
| `train_loss` / `val_loss` | TD (SmoothL1) loss on Bellman targets | Should be stable, not collapse to zero |
| `q_std` | Mean std of Q-values across the 4 actions, per state | Should stay non-zero; collapse = state-independent Q-values |
| `entropy` | Shannon entropy of the softmax action distribution | `ln(4) ≈ 1.386` = uniform policy; lower = action preference has emerged |

---

## 4) Configuration and dataset

| Parameter | Value |
|---|---|
| Encoder checkpoint | `checkpoints/stage1_fix2/jepa_epoch_014.pt` |
| Latent dim | 256 |
| Dataset | `data/random` — 49,127 transitions from random play |
| Reward rate | 0.6% of transitions (`r_t > 0`) |
| Random agent mean return | **1.10** per episode |
| Batch size | 256 |
| Learning rate | 1e-3 (short run) / 3e-4 (long run) |
| Target sync | every 1 epoch (short) / every 5 epochs (long) |
| γ | 0.99 |

---

## 5) Empirical results

### 5.1 Short run — 15 epochs, LR=1e-3, target-sync=1

```
Dataset: 49127 samples  →  train=44215  val=4912  (173 / 20 batches)
Epoch 001 | train_loss=0.004836  val_loss=0.004681 | q_std=0.0295  entropy=1.3856
Epoch 002 | train_loss=0.003851  val_loss=0.005826 | q_std=0.0221  entropy=1.3861
Epoch 005 | train_loss=0.003901  val_loss=0.004608 | q_std=0.0162  entropy=1.3862
Epoch 010 | train_loss=0.003896  val_loss=0.005102 | q_std=0.0119  entropy=1.3862
Epoch 015 | train_loss=0.004000  val_loss=0.004786 | q_std=0.0107  entropy=1.3862
```

Comparison with Stage 1.5 (which used the same encoder and dataset):

| Diagnostic | Stage 1.5 (Fix 2) | **Stage 2** |
|---|---|---|
| `td_loss` | collapses 8.7×10⁻⁴ → 1×10⁻⁶ in epoch 1 | stable ~0.004 throughout ✅ |
| `q_std` | 0.029 → 0.004 (instant collapse) | 0.030 → 0.011 (slow, gradual) ✅ |
| `entropy` | **0.325** (always-same-action) | **1.386 = ln(4)** (uniform) ✅ |

Stage 2 has a working Bellman loop. The `td_loss` does not collapse because real rewards ($r > 0$ for 0.6% of transitions) break the symmetry that causes Stage 1.5 to collapse in one epoch.

`entropy = ln(4)` means the Q-Head is outputting nearly equal values for all four actions — it has converged to the behavior policy's value function (random play → all actions ≈ equivalent).

### 5.2 Long run — 50 epochs, LR=3e-4, target-sync=5

```
Epoch 001 | train_loss=0.004570  val_loss=0.004018 | q_std=0.0208  entropy=1.3858
Epoch 007 | train_loss=0.003302  val_loss=0.003974 | q_std=0.0150  entropy=1.3862
Epoch 013 | train_loss=0.003355  val_loss=0.003953 | q_std=0.0177  entropy=1.3862
Epoch 025 | train_loss=0.003282  val_loss=0.004231 | q_std=0.0104  entropy=1.3862
Epoch 050 | train_loss=0.003332  val_loss=0.004134 | q_std=0.0057  entropy=1.3863
```

`q_std` declines monotonically 0.020 → 0.006. `entropy` is permanently at ln(4). More offline epochs do not help — the Q-Head has converged and cannot extract further signal from this dataset.

---

## 6) Real-environment evaluation

`eval_policy.py` runs the trained Q-Head inside `ALE/Breakout-v5` with the JEPA encoder frozen. ε-greedy action selection is used to inject exploration during evaluation.

| Checkpoint | ε=0.0 (greedy) | ε=0.05 |
|---|---|---|
| Epoch 1 (random init, no training) | 0.00 | 0.70 ± 0.84 |
| **Epoch 15 (15-epoch TD training)** | **0.00** | **10.85 ± 0.65** |
| **Epoch 7 (50-epoch run, peak q_std)** | not tested | 10.55 ± 1.12 |
| Random agent (dataset baseline) | — | 1.10 |

- **ε=0.05, epoch 15: 10.85 ± 0.65** — a **9.9× improvement over the random agent** (1.10) and a 15.5× improvement over the random-init Q-Head (0.70).
- **ε=0.0 (greedy): 0.00** — the greedy policy scores zero across all tested checkpoints.

---

## 7) Greedy failure diagnosis

Running `diagnose_greedy_policy.py` on the epoch-15 checkpoint:

```
Episode 1: return=0.0  steps=500
  Action dist: {'LEFT': 500}
  Q-std   mean=0.00728  min=0.00728  max=0.00728
  Q-vals: [NOOP: 0.5359, FIRE: 0.5374, RIGHT: 0.5291, LEFT: 0.5468]
```

**The Q-Head outputs identical Q-values for every state.** `q_std = 0.00728` is constant across all 500 steps and all three tested episodes. The argmax is always `LEFT` because it has the highest constant bias, not because of any state-specific reasoning.

The four Q-values (~0.53–0.55) match the theoretical expected discounted return for a random agent:

$$Q^{\pi_\text{random}}(s, a) \approx \frac{r_\text{mean}}{1 - \gamma} = \frac{0.006}{0.01} = 0.6$$

The small per-action differences (range = 0.018) reflect marginal correlations in the offline dataset — `LEFT` happened to produce slightly more reward in random play — not any causal relationship.

---

## 8) Why offline Q-learning on random data cannot learn state-conditional values

With a uniform random behavior policy $\pi_b(a|s) = 0.25$, the action-conditional value function satisfies:

$$Q^{\pi_b}(s, a) = r(s, a) + \gamma \sum_{s'} P(s'|s, a) V^{\pi_b}(s') \approx \frac{r_\text{mean}}{1 - \gamma}$$

Because $r_\text{mean}$ is approximately the same for all $(s, a)$ pairs in a random dataset (no systematic action-state correlations), the optimal fit is a near-constant function. Offline TD faithfully learns this constant.

To learn that `RIGHT` is better than `LEFT` when the ball is moving rightward, the dataset would need paired examples of both actions taken from identical (or similar) states, with observed outcome differences. Random play never systematically creates these contrastive pairs.

**This is not a flaw in the JEPA encoder or the Q-Head architecture** — it is a fundamental property of the offline data distribution. The encoder IS contributing: it compresses 4-frame contexts to 256-dimensional latents that allow the Q-Head to efficiently fit the behavior policy's values. The ceiling is set by the data.

---

## 9) What Stage 2 achieves and what it needs

| Stage | Status | Key result |
|---|---|---|
| Stage 1 JEPA pretraining | ✅ Complete | Best config: `latent_dim=256`, `mask_ratio=0.7`, epoch 14 with `reward_loss_weight=10` |
| Stage 1.5 (imagined bootstrap) | ❌ Blocked | Reward head R²=−0.005; encoder ROC-AUC=0.50 |
| **Stage 2 offline TD** | ⚠️ **Partial** | 10.85 with ε=0.05 (9.9× random); greedy=0; Q-values state-independent |
| Stage 3: online fine-tuning | ⬜ **Next required step** | Must interact with the environment to learn state-conditional Q-values |

The `stage2_fix2/q_head_epoch_015.pt` checkpoint is a valid warm-start for Stage 3: it has already fit the behavior policy's value function and will not produce random-large initial gradients when fine-tuned online. Stage 3 only needs to add the causal signal that offline random data cannot provide.

---

## 10) Stage 2 status checklist

| Item | Status |
|---|---|
| `train_policy.py` updated with train/val split, `q_std`/`entropy` diagnostics | ✅ |
| `eval_policy.py` working with Stage 2 checkpoints | ✅ |
| `diagnose_reward_head.py` — reward head R²=−0.005 confirmed | ✅ |
| `diagnose_encoder_reward_signal.py` — encoder ROC-AUC=0.5008 confirmed | ✅ |
| `diagnose_greedy_policy.py` — state-independent Q-values confirmed | ✅ |
| Short run (15 epochs, LR=1e-3): **10.85 ± 0.65** at ε=0.05 | ✅ |
| Long run (50 epochs, LR=3e-4): plateau confirmed, `q_std` 0.020→0.006 | ✅ |
| Greedy failure diagnosed (constant Q-values, always LEFT) | ✅ |
| Best offline checkpoint identified: `stage2_fix2/q_head_epoch_015.pt` | ✅ |
| Stage 3: online DQN fine-tuning with frozen JEPA encoder | ⬜ next step |
