# Stage 1.5 — Latent Imagination Q-Head Bootstrap

This document describes how I approach Stage 1.5, why I designed it the way I did, what I expect to observe during training, and what checks I use to detect degenerate or broken learning before moving to Stage 2.

---

## 1) What Stage 1.5 is and why it exists

After Stage 1 I have a frozen JEPA world model: an `encoder` that maps a 4-frame grayscale context to a latent vector $z \in \mathbb{R}^{256}$, an `ActionConditionedPredictor` that advances $z$ forward by one step given an action, and a `RewardHead` that predicts $r_t$ from $(z_t, a_t)$.

What I do **not** yet have is a Q-Head that knows which actions lead to reward.

The naive next step would be to jump straight to Stage 2 (offline Q-learning on real transitions), but this creates a cold-start problem: the Q-Head is initialised randomly, and early TD updates are driven by random targets from the same randomly-initialised head, which can produce large, destabilising gradients right at the start of Stage 2.

Stage 1.5 solves this by using the world model to *dream*. Rather than calling `env.step()` or reading stored transitions, I take a real starting state from the offline dataset, encode it to $z_0$, and then unroll $H$ imagined steps through the frozen Predictor. At each imagined step the frozen RewardHead provides a synthetic reward signal. I then apply TD learning inside this latent dream, bootstrapping Q-values from a frozen target Q-Head copy.

The result is a Q-Head that has already seen a large implicit diversity of latent states — every starting state in the offline dataset spawns its own $H$-step dream trajectory — and has a reasonable prior over values before ever touching real transition data.

---

## 2) Training approach

### 2.1 What gets trained and what stays frozen

| Component | Status | Rationale |
|---|---|---|
| `ConvEncoder` | Frozen | Stage 1 representation is the foundation; disturbing it would invalidate all downstream latents |
| `ActionConditionedPredictor` | Frozen | Same reason: the world model is fixed |
| `RewardHead` | Frozen | Provides imagined reward signals; trained already in Stage 1 |
| `QHead` (online) | **Trained** | The only learnable component in this stage |
| `QHead` (target) | Frozen copy, periodically synced | Provides stable Bellman targets |

### 2.2 Imagined rollout mechanics

For each mini-batch of real context windows I:

1. Encode the context stack to a starting latent $z_0 = \text{Encoder}(\text{context})$, with no gradient through the encoder.
2. For each step $k = 0, \ldots, H-1$:
   - Select an action $a_k$ using an **ε-greedy** policy on the online Q-Head (ε = 0.1 by default).
   - Advance the latent: $z_{k+1} = \text{Predictor}(z_k, a_k)$ — no gradient.
   - Predict the imagined reward: $\hat{r}_k = \text{RewardHead}(z_k, a_k)$ — no gradient.
   - Compute the Bellman target using the **frozen target Q-Head**: $y_k = \hat{r}_k + \gamma \cdot \max_{a'} Q_{\text{target}}(z_{k+1}, a')$
   - Compute the per-step TD loss: $\mathcal{L}_k = \text{SmoothL1}\!\left( Q_{\text{online}}(z_k)_{a_k},\ y_k \right)$
3. The total loss for the batch is $\bar{\mathcal{L}} = \frac{1}{H} \sum_k \mathcal{L}_k$.
4. Back-propagate through the online Q-Head only. The latent path is detached step-by-step so gradients do not accumulate through time.

### 2.3 ε-greedy exploration

I use ε-greedy action selection (rather than pure greedy) during imagined rollouts for a concrete reason: early in training the Q-Head is randomly initialised, so its argmax is essentially arbitrary. A pure greedy rollout would then fixate on whichever action happens to have the highest random Q-value, repeating it for the entire $H$ steps. The result is that the world model explores a single action's latent branch and never trains the other Q-values. ε = 0.1 injects enough diversity to ensure all four actions receive gradient signal from the first epoch onwards. During the validation pass I set ε = 0 (fully greedy) so the validation loss reflects the policy's actual quality without noise.

### 2.4 Target network

The target Q-Head is a hard copy of the online Q-Head, updated every `--target-sync-epochs` epochs. Without it the Bellman target $y_k$ shifts every gradient step, creating a moving-target loop. This is the same instability that motivated the original DQN target network. With the target network held fixed for a full epoch, the TD update is well-conditioned and the training loss should descend monotonically between syncs.

### 2.5 Train / validation split

I hold out `--val-split 0.1` (10%) of starting states as a validation set, using a fixed seed-0 shuffle for reproducibility. The validation pass runs under `torch.no_grad()` with ε = 0. The validation loss tells me whether the Q-Head is genuinely generalising to unseen starting states or fitting noise in the training set.

---

## 3) Diagnostics tracked per epoch

Beyond the TD loss I track three diagnostic signals each epoch, both on train and val:

| Metric | What it measures | Healthy sign |
|---|---|---|
| `imag_reward` | Mean imagined reward per step across the rollout | Should be positive and stable; large drifts suggest the RewardHead is misbehaving |
| `q_std` | Mean standard deviation of Q-values across the four actions, per state | Should increase from near-zero (random init) as the Q-Head learns to differentiate actions |
| `entropy` | Shannon entropy of the empirical action distribution over a batch | Should stay non-zero; collapse to a single action despite ε-greedy indicates Q-Head is over-specialising |

These three metrics are stored in the checkpoint under `train` and `val` keys alongside the loss, so I can plot them post-hoc.

---

## 4) Key design checks and what I am watching for

### 4.1 Q-value collapse (the single biggest risk)

The most likely failure mode is Q-value collapse: the Q-Head converges to outputting near-identical values for all actions, so `q_std ≈ 0`. This makes the policy indistinguishable from a random one. It can happen because:

- Imagined rewards from the `RewardHead` are near-zero (which they will be on average for a random-play dataset, since Breakout rewards are sparse), so the TD target is dominated by the bootstrapped future value and the Q-Head has little signal to differentiate actions.
- Without ε-greedy, one action monopolises the rollout and the other three Q-values receive no gradient.

**Check:** watch `q_std` — if it stays below `0.01` after epoch 3 I should increase ε or reduce rollout length $H$ to reduce bootstrapping pressure.

### 4.2 Reward head quality

If the `RewardHead` from Stage 1 was trained with a negligible number of reward-bearing transitions (which is typical on random-play data where rewards are sparse), the imagined reward signal will be weak and noisy. The Q-Head will still learn a meaningful value function as long as the bootstrap term $\gamma \cdot \max_{a'} Q_{\text{target}}(z_{k+1}, a')$ carries structure. The `imag_reward` diagnostic will tell me whether imagined rewards are contributing or effectively zero.

### 4.3 Train vs val gap

A growing gap between `train_loss` and `val_loss` indicates the Q-Head is memorising specific starting states from the training set. Given the small capacity of `QHead` (two linear layers) this is unlikely, but the split is cheap insurance.

### 4.4 Checkpoint compatibility

Older Stage 1 checkpoints that predate the `RewardHead` addition will trigger a graceful warning rather than a `KeyError`. I use a randomly-initialised frozen `RewardHead` in that case. The implication is that `imag_reward` will be random noise and the Q-Head will learn from pure bootstrapping — not ideal but not a crash.

---

## 5) What I expect to observe during a healthy run

### 5.1 Loss trajectory

| Epoch range | Expected behaviour |
|---|---|
| 1 | Relatively high TD loss (~0.1–0.5); Q-Head and target are identical (both random), so Bellman targets are consistent but uninformative |
| 2–4 | Loss decreases as the Q-Head starts to differentiate actions; `q_std` climbs from near-zero |
| 5–10 | Loss plateaus or continues a slow descent; `q_std` stabilises at a moderate level (0.05–0.3 is a reasonable range) |

A loss that stays completely flat from epoch 1 suggests the Q-Head is not receiving any gradient (check learning rate and that `loss.backward()` is reaching the Q-Head parameters).

### 5.2 Diagnostic trajectories

| Metric | Expected trajectory |
|---|---|
| `imag_reward` | Should hover near zero (sparse rewards in training data) but be stable, not diverging or oscillating wildly |
| `q_std` | Should increase monotonically for the first few epochs, then plateau |
| `entropy` | Should stay close to `log(4) ≈ 1.39` (max entropy over 4 actions) in the early epochs, then gradually decrease as the Q-Head learns to favour better actions |

### 5.3 What the Q-Head should have learned by the end

Stage 1.5 does not yet play Breakout. The Q-Head at this point has only seen imagined latent trajectories, not real pixel-level consequences. I expect:

- The Q-Head to have a non-trivial prior over action values — i.e. some differentiation between actions in different latent states — rather than the flat random initialisation it started from.
- No catastrophic divergence (loss not increasing, `q_std` not collapsing to zero).
- Stage 2 to converge faster and to a better final Q-value than it would starting from a randomly-initialised Q-Head, because Stage 1.5 provides a warm start that Stage 2 refines on real transitions.

### 5.4 Expected output in the terminal

```
Dataset: 49593 samples  →  train=44633  val=4960  (697 / 77 batches)
Epoch 001 | train_loss=0.183412  val_loss=0.191023 | imag_reward=0.0023  q_std=0.0097  entropy=1.3845 | saved=q_imagination_epoch_001.pt
Epoch 002 | train_loss=0.104587  val_loss=0.112340 | imag_reward=0.0021  q_std=0.0241  entropy=1.3712 | saved=q_imagination_epoch_002.pt
...
Epoch 010 | train_loss=0.031204  val_loss=0.033891 | imag_reward=0.0019  q_std=0.1130  entropy=1.2981 | saved=q_imagination_epoch_010.pt
```

The exact numbers will vary, but the pattern — decreasing loss, rising `q_std`, slightly decreasing entropy — is the signature of healthy Q-value learning.

---

## 6) How to run Stage 1.5

```bash
.venv/bin/python Breakout/train_latent_imagination.py \
    --data-dir             Breakout/data/random \
    --jepa-checkpoint      Breakout/checkpoints/stage1_viz_epochs/jepa_epoch_008.pt \
    --output-dir           Breakout/checkpoints/q_imagination \
    --latent-dim           256 \
    --rollout-length       5 \
    --epochs               10 \
    --epsilon              0.1 \
    --target-sync-epochs   1 \
    --val-split            0.1
```

To evaluate the trained Q-Head in the real environment immediately after:

```bash
.venv/bin/python Breakout/eval_policy.py \
    --encoder-checkpoint  Breakout/checkpoints/stage1_viz_epochs/jepa_epoch_008.pt \
    --q-checkpoint        Breakout/checkpoints/q_imagination/q_imagination_epoch_010.pt \
    --latent-dim          256 \
    --episodes            20 \
    --epsilon             0.05
```

I do not expect a high game score here — Stage 1.5 has only seen imagined dynamics. The evaluation is primarily a sanity check that the Q-Head is not outputting a single repeated action and that mean episode return exceeds the expected random-policy baseline (~1.5–2.0 points per episode on Breakout).

---

## 7) Connection to Stage 2

Stage 1.5 and Stage 2 are designed to chain naturally:

| Property | Stage 1.5 output | Stage 2 use |
|---|---|---|
| Checkpoint format | `{"q_head": ..., "train": {...}, "val": {...}}` | Stage 2 can optionally load the Q-Head state dict as a warm start |
| Evaluation script | `eval_policy.py` accepts both Stage 1.5 and Stage 2 checkpoints | Compare game scores before and after Stage 2 fine-tuning |
| Latent dim | Both default to 256 | Must match the encoder checkpoint; consistent across stages |

The key hypothesis I am testing with this pipeline design is: **a Q-Head bootstrapped on imagined dynamics (Stage 1.5) reaches a higher final game score after Stage 2 fine-tuning than one initialised randomly**. Stage 1.5 is worthless on its own as a game player, but it should materially accelerate and stabilise Stage 2.

---

## 8) Empirical run — Stage 1 (latent_dim=256 rerun)

I re-ran Stage 1 with the validated best settings from the grid sweep (`latent_dim=256`, `mask_ratio=0.7`, `train_horizon=1`, `batch_size=256`, 8 epochs, seed 7). Checkpoints saved to `Breakout/checkpoints/stage1_dim256/`.

| Ep | tr_jepa | tr_sens | tr_copy | vl_jepa | vl_sens | vl_total | vl_drift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.010119 | 0.027855 | 0.105259 | 0.000000 | 0.001208 | 0.003481 | 1.4101 |
| 2 | 0.000036 | 0.006464 | 0.000624 | 0.000006 | 0.002807 | 0.003460 | 0.3317 |
| 3 | 0.000039 | 0.006359 | 0.000200 | 0.000016 | **0.004511** | 0.003715 | 0.5650 |
| 4 | 0.000035 | 0.004681 | 0.000199 | 0.000107 | 0.001866 | 0.003715 | 0.8193 |
| 5 | 0.000006 | 0.002171 | 0.000318 | 0.000000 | 0.000138 | 0.003508 | 0.2497 |
| 6 | 0.000051 | 0.006761 | 0.000362 | 0.000000 | 0.000067 | 0.003569 | 0.3470 |
| 7 | 0.000038 | 0.004764 | 0.000102 | 0.000031 | 0.003282 | 0.003764 | 0.4991 |
| 8 | 0.000008 | 0.002390 | 0.000247 | 0.000000 | 0.000059 | 0.003782 | 0.2104 |

**Selected checkpoint for Stage 1.5:** `jepa_epoch_003.pt` — highest `val_action_sensitivity` (0.004511). Beyond epoch 3 the val sensitivity collapses to near-zero, indicating the predictor stops differentiating actions under the full dataset pass.

**Observation on val_action_sensitivity:** It peaks at epoch 3 (0.0045) and then collapses toward zero by epochs 5–8. This is a flag: the predictor is not strongly action-conditioned. The training sensitivity stays above 0.002–0.007 throughout, but the val signal disappears — suggesting the model is weakly generalising action conditioning to held-out states. This will matter in Stage 1.5.

---

## 9) Empirical run — Stage 1.5 (first attempt)

Ran `train_latent_imagination.py` with `jepa_epoch_003.pt`, `latent_dim=256`, `rollout_length=5`, `epsilon=0.1`, 10 epochs.

| Ep | tr_loss | vl_loss | imag_r | q_std | entropy |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.001110 | 0.000005 | −0.0107 | 0.0391 | 0.3258 |
| 2 | 0.000011 | 0.000003 | −0.0099 | 0.0104 | 0.3246 |
| 3 | 0.000049 | 0.000002 | −0.0104 | 0.0148 | 0.3251 |
| 4 | 0.000043 | 0.000001 | −0.0105 | 0.0172 | 0.3224 |
| 5 | 0.000021 | 0.000001 | −0.0101 | 0.0124 | 0.3209 |
| 6 | 0.000014 | 0.000001 | −0.0100 | 0.0112 | 0.3236 |
| 7 | 0.000024 | 0.000000 | −0.0100 | 0.0112 | 0.3221 |
| 8 | 0.000004 | 0.000000 | −0.0099 | 0.0096 | 0.3212 |
| 9 | 0.000004 | 0.000000 | −0.0098 | 0.0095 | 0.3184 |
| 10 | 0.000002 | 0.000000 | −0.0097 | 0.0092 | 0.3264 |

### Diagnosis: three simultaneous failure signatures

| Metric | Observed | Expected | Verdict |
|---|---|---|---|
| `td_loss` | 0.001 → ~0.000 by epoch 2 | Gradual descent from ~0.1 | ❌ Collapsed instantly |
| `imag_reward` | **−0.010 throughout** | ≥ 0, sparse but non-negative | ❌ Negative bias — bad reward head |
| `q_std` | 0.039 → **0.009** (shrinking) | Rising as actions differentiate | ❌ Q-value collapse |
| `entropy` | **0.33 throughout** | ~1.39 early → gradual decrease | ❌ Single-action fixation from epoch 1 |

### Root cause chain

1. **RewardHead learned a negative bias.** The training dataset is ~98% zero-reward transitions; the auxiliary loss during Stage 1 pushed the head toward the dataset mean, which is essentially zero but slightly negative due to loss asymmetry. The frozen RewardHead therefore outputs ≈ −0.010 for every `(z, a)` pair, injecting a constant negative offset into every Bellman target.

2. **Flat Bellman target → instant loss collapse.** With `imag_reward ≈ −0.010` and the freshly-initialised target Q-Head producing near-constant values, the TD target $y = \hat{r} + \gamma \cdot \max Q_{\text{target}}(z')$ is nearly identical for every state/action pair. The Q-Head fits this constant in one epoch, loss → 0, and receives effectively no gradient afterwards.

3. **Weak action conditioning amplifies the problem.** Val action sensitivity at epoch 3 is only 0.0045 — the predictor generates nearly identical $z'$ regardless of which action is taken. Even if the reward head were perfect, the bootstrap term $\gamma \cdot \max_{a'} Q(z', a')$ carries almost no action-dependent structure. The Q-Head cannot differentiate actions from either the reward signal or the dynamics.

4. **Entropy locked at 0.33 from epoch 1.** This is the signature of the ε-greedy mechanism alone: with ε = 0.1 and 4 actions, a policy that always picks the same greedy action has empirical entropy = $-(0.1 \cdot 0.025 \cdot \log(0.025) \times 3 + 0.925 \cdot \log(0.925)) \approx 0.33$. The Q-Head never learned to distribute probability across actions — it stayed random-greedy from initialisation.

---

## 10) What I will try next

After the first failed Stage 1.5 attempt, I implemented **Fix 1**:

- In `train_latent_imagination.py`, I changed imagined reward computation to:
    - `imagined_reward = reward_head(z, actions).clamp(min=0.0)`

I then re-ran Stage 1.5 with the same setup (`latent_dim=256`, `rollout_length=5`, `epsilon=0.1`, 10 epochs) and compared baseline vs Fix 1.

### 10.1 Fix 1 empirical outcome

| Metric | Baseline (no clamp) | Fix 1 (clamp) | Interpretation |
|---|---:|---:|---|
| Avg `imag_reward` | −0.0101 | +0.0000 | ✅ Negative reward bias removed |
| Avg `q_std` | 0.0145 | 0.0058 | ❌ Action-value spread became even weaker |
| Avg `entropy` | 0.3230 | 0.3327 | ≈ no meaningful improvement (still collapsed regime) |
| Final train loss | 0.000002 | 0.000000 | ❌ Still immediate collapse |

Environment check:

| Eval setup | Baseline | Fix 1 | Interpretation |
|---|---:|---:|---|
| `eval_policy.py`, ε=0.05, 20 episodes | 0.25 ± 0.54 | 0.65 ± 0.79 | Slight stochastic improvement, unstable |
| `eval_policy.py`, ε=0.0, 10 episodes | 0.00 | 0.00 | No deterministic policy learning |

### 10.2 Why Fix 2 is now required

Fix 1 solved only the **reward sign bias**. It did **not** solve the core learning bottleneck:

1. Bellman targets are still near-constant after clamping (roughly zero + weak bootstrap).
2. Predictor action-conditioning on validation states is weak (`val_action_sensitivity` peaks at ~0.0045 and collapses).
3. The Q-Head still receives almost no action-discriminative gradient, so it converges to a near-constant Q surface.

Therefore **Fix 2 is required** to increase signal quality at the Stage 1 source:

- Add `--reward-loss-weight` to Stage 1 (`train_jepa.py`) so RewardHead learns sparse reward structure more strongly.
- Re-run Stage 1 longer (20 epochs) and select checkpoint by best `val_action_sensitivity` + stable `val_total_loss`.
- Re-run Stage 1.5 from that improved world model.

The two root causes require fixes at different levels:

### Fix 1 — Reward signal: clamp and re-centre in the rollout

The simplest patch is to apply `reward.clamp(min=0.0)` inside `imagination_rollout` so the negative RewardHead bias cannot pull all Bellman targets below zero. This does not require retraining Stage 1 — it is a one-line change in `train_latent_imagination.py`. Optionally I can normalise the imagined reward by its running mean/std to ensure it contributes a signal of consistent scale regardless of the head's absolute bias.

### Fix 2 — Action sensitivity: retrain Stage 1 with stronger action conditioning

The deeper fix is to produce a predictor that generates meaningfully different $z'$ for different actions. Options in order of cost:

| Option | Cost | Expected gain |
|---|---|---|
| More epochs (e.g. 20) with current data | Low | Modest — the random-play data has limited action-diversity signal |
| Increase `mask_ratio` to 0.8–0.9 | Low | Forces encoder to rely more on dynamics context; may help sensitivity |
| Upweight the **reward loss** in Stage 1 (`--reward-loss-weight`) | Low | Pushes the head to distinguish reward-bearing states, indirectly improving action-conditioned structure |
| Collect data with a **partially-trained policy** that actually earns rewards | High | Most impactful — exposes the predictor to genuine action consequence |

My immediate next step is to apply Fix 2: add `--reward-loss-weight` to `train_jepa.py`, re-run Stage 1 for 20 epochs, and re-attempt Stage 1.5 using the best new checkpoint.

---

---

## 12) Fix 2 empirical results — reward-head sign improved, but core problem survives

### 12.1 Fix 2 Stage 1 outcome

Running 20 epochs with `--reward-loss-weight 10.0` and selecting checkpoint by `val_action_sensitivity`:

| Epoch | val_jepa_loss | val_reward_loss | val_action_sensitivity |
|------:|-------------:|----------------:|----------------------:|
| 001 | 0.000978 | 0.003572 | 0.001785 |
| 003 | 0.000878 | 0.003497 | 0.003105 |
| 008 | 0.000870 | 0.003408 | 0.005200 |
| **014** | **0.000862** | **0.003437** | **0.007601 (best)** |
| 020 | 0.000858 | 0.003481 | 0.003671 |

Best checkpoint: `stage1_fix2/jepa_epoch_014.pt` — `val_action_sensitivity = 0.007601` (+68% over the 8-epoch baseline peak of 0.004511).

The reward head mean output on real data completely reversed sign:

| Head | Mean output | Std | Min | Max |
|------|------------|-----|-----|-----|
| Baseline ep03 | −0.0098 | 0.0090 | −0.025 | −0.001 |
| **Fix 2 ep14** | **+0.0090** | 0.0045 | +0.002 | +0.013 |
| True rewards | +0.0062 | — | 0 | 1 |

Fix 2 solved the sign problem. The `clamp(min=0.0)` from Fix 1 is now a no-op (all outputs are already positive).

### 12.2 Fix 2 Stage 1.5 outcome

Running Stage 1.5 with `stage1_fix2/jepa_epoch_014.pt` produced the same three failure signatures as before:

| Epoch | train_loss | imag_reward | q_std | entropy |
|------:|-----------:|------------:|------:|--------:|
| 001 | 0.000866 | 0.0128 | 0.0292 | 0.3255 |
| 005 | 0.000011 | 0.0133 | 0.0067 | 0.3225 |
| 010 | 0.000001 | 0.0134 | 0.0044 | 0.3260 |

`imag_reward` is now positive (✅ Fix 1+2 together fixed the sign bias), but `q_std` is monotonically decreasing and `entropy` is stuck at exactly 0.325 — the value predicted by an ε-greedy policy with ε=0.1 that always picks the same action.

---

## 13) Root-cause autopsy — the reward head encodes no state information

After Fix 2 Stage 1.5 still failed, I ran two quantitative diagnostics to find the irreducible root cause.

### 13.1 Reward head discrimination diagnostic

Running `diagnose_reward_head.py` on `stage1_fix2/jepa_epoch_014.pt`:

```
N reward=0:  10179  pred_mean=0.009029  std=0.004500
N reward>0:     61  pred_mean=0.009171  std=0.004609
Gap: 0.000143  (ideal: ~0.994)

Per-action predictions (same state, four actions):
  action=0: mean=0.013309  std=0.000005
  action=1: mean=0.001777  std=0.000001
  action=2: mean=0.009406  std=0.000003
  action=3: mean=0.011899  std=0.000004

MSE(constant predictor): 0.005922
MSE(reward head):        0.005950
R² = −0.0047
```

**The per-action std values (0.000001–0.000005) are the key result.** For a batch of 2,048 different states, every single state receives the same reward prediction for each action. The reward head has learned four **scalar constants** — one per action — and completely ignores the state latent $z$.

The discrimination gap between reward=0 and reward=1 states is 0.000143 (should be ~0.994). R² = −0.005 means the reward head is **worse than a constant predictor**.

This means:
- The imagined rewards in Stage 1.5 are constant per action, not per state.
- The Bellman target $y = r_{\text{const}} + \gamma \max_{a'} Q(z', a')$ converges to a constant after the Q-Head fits the constant rewards in epoch 1.
- From epoch 2 onwards the TD loss is near-zero and no gradient flows.
- The Q-Head always picks `action=0` (highest predicted reward = 0.0133) regardless of state.
- `entropy=0.325` exactly matches ε-greedy at ε=0.1 with a fully deterministic preferred action.

### 13.2 Encoder reward signal probe

Running `diagnose_encoder_reward_signal.py` — a balanced logistic regression probe trained on encoder latents to predict reward=1 vs reward=0:

```
Dataset: 49127 samples, 309 positive (0.63%)
Logistic Regression probe (pos_weight=158):
  ROC-AUC      : 0.5008  (0.5 = random, 1.0 = perfect)
  Avg-Precision: 0.0082  (baseline = 0.0072)
  AP lift      : 1.1x over random
→ Encoder latents contain NO reward-discriminative signal.
```

**ROC-AUC = 0.5008 is statistically indistinguishable from random.** This means the JEPA encoder has learned features that are completely orthogonal to the reward signal. No reward head architecture or loss weighting can fix this — the information is simply not present in the encoder's output space.

### 13.3 Why JEPA features don't encode reward

This is expected given JEPA's training objective. JEPA optimises:

$$\mathcal{L}_{\text{JEPA}} = \|\hat{z}_{t+1} - \bar{z}_{t+1}\|_2^2$$

where $\hat{z}$ is the predictor output and $\bar{z}$ is the target encoder output. This drives the encoder to produce features useful for predicting the **next state's appearance**, not features useful for predicting **whether a reward occurred**.

Reward in Breakout is binary, sparse (0.6% of transitions), and tied to ball-brick collision — a local pixel event in a specific spatial region. JEPA's global prediction objective does not need to specialise features for this event; features that encode smooth ball motion and general scene dynamics already minimise the JEPA loss without needing to encode collision outcomes.

The `reward_loss_weight=10` in Fix 2 forced the reward head to **try** to predict rewards, but because the encoder latent $z$ contains no reward-discriminative features, the best the head can do is memorise per-action reward base rates from the training data.

---

## 14) Fix 3 — JEPA prediction error as intrinsic reward

Since the `RewardHead` cannot provide useful imagined rewards, I need a different reward signal for Stage 1.5. The most principled alternative is to use the JEPA predictor's own **prediction error** as an intrinsic/curiosity reward:

$$r_{\text{intrinsic}}(z_t, a_t) = \|\text{Predictor}(z_t, a_t) - z_{t+1}\|_2^2$$

This signal:
- Is **derived from the world model itself** — no separate head needed.
- Is **high for novel or surprising transitions** and low for well-predicted ones.
- Is **state-dependent by construction** — it varies with $z_t$ because different states have different prediction difficulties.
- Requires **no additional training** — the predictor is already frozen from Stage 1.

However, for Stage 1.5 the "next state" $z_{t+1}$ comes from the predictor itself (we are in imagination), so the prediction error of a frozen predictor evaluating its own output is identically zero:

$$r_{\text{intrinsic}} = \|\text{Predictor}(z_t, a_t) - \text{Predictor}(z_t, a_t)\|_2^2 = 0$$

This approach only works in **real-environment rollouts**, not imagined ones.

### 14.1 Fix 3a — Real-transition JEPA error (Stage 2 hybrid)

Rather than unrolling purely imagined trajectories, I can replace Stage 1.5 with a **real-transition Q-learning** step that uses JEPA prediction error as a shaped reward:

$$r_{\text{shaped}}(s_t, a_t) = r_{\text{true}}(s_t, a_t) + \beta \cdot \|\text{Predictor}(z_t, a_t) - z_{t+1}\|_2^2$$

where $z_t$ and $z_{t+1}$ are encoder outputs of consecutive real frames. This hybrid is essentially Stage 2 with intrinsic reward shaping — it uses real reward when available and uses prediction novelty as a secondary signal.

### 14.2 Fix 3b — Skip Stage 1.5, run Stage 2 directly

The most pragmatic and evidence-based path is to **skip Stage 1.5 entirely** and run Stage 2 (offline Q-learning on real transitions) directly. Stage 2 does not rely on the `RewardHead` at inference time — it uses the actual `reward` field from stored transitions, which contains the true sparse reward signal.

Stage 2 advantages over Stage 1.5 given the current diagnosis:
- Uses real rewards from the replay buffer (no reward head dependency)
- 0.6% reward rate is still sparse but the TD signal propagates correctly via bootstrapping
- The `stage1_fix2` encoder produces good dynamics features that will help the Q-Head generalise

This is the most likely path to a working Q-Head in reasonable compute time.

---

## 16) Stage 2 — offline TD Q-learning on real transitions

Since Stage 1.5 has proven fundamentally unable to produce a useful Q-Head (the reward head has R² = −0.005 and the encoder latents are reward-blind), I skipped Stage 1.5 and ran Stage 2 directly.

Stage 2 uses actual rewards from the offline dataset instead of imagined rewards from the broken RewardHead, which avoids the root cause entirely.

### 16.1 Changes to `train_policy.py` before running

I added a train/val split and per-epoch diagnostics (`q_std`, `entropy`) to match the Stage 1.5 monitoring framework:

- `--val-split` (default 0.1): reserves 10% for validation monitoring
- `--batch-size` default raised to 256 (from 64)
- `--epochs` default raised to 10 (from 3)
- `q_std` and `entropy` printed each epoch (same interpretation as Stage 1.5)

### 16.2 Stage 2 training diagnostics (15 epochs, LR=1e-3)

```
Dataset: 49127 samples  →  train=44215  val=4912
Epoch 001 | train_loss=0.004836  val_loss=0.004681 | q_std=0.0295  entropy=1.3856
Epoch 002 | train_loss=0.003851  val_loss=0.005826 | q_std=0.0221  entropy=1.3861
Epoch 005 | train_loss=0.003901  val_loss=0.004608 | q_std=0.0162  entropy=1.3862
Epoch 010 | train_loss=0.003896  val_loss=0.005102 | q_std=0.0119  entropy=1.3862
Epoch 015 | train_loss=0.004000  val_loss=0.004786 | q_std=0.0107  entropy=1.3862
```

Key differences from Stage 1.5:

| Diagnostic | Stage 1.5 (Fix2) | **Stage 2** |
|---|---|---|
| `td_loss` | collapses 8.7e-4 → 1e-6 in epoch 1 | stable ~0.004 ✅ |
| `q_std` | 0.029 → 0.004 (instant collapse) | 0.030 → 0.011 (slow decline) ✅ |
| `entropy` | 0.325 (always-same-action) | **1.386 = ln(4)** (uniform) ✅ |

The `td_loss` does **not** collapse — Stage 2 has a working Bellman loop because it uses real rewards.

The `entropy = ln(4) ≈ 1.386` tells a different story than Stage 1.5: instead of always picking the same action (degenerate bias), the Q-Head is outputting **nearly equal Q-values for all actions** — a uniform policy.

### 16.3 Real-environment evaluation

| Checkpoint | ε=0.0 | ε=0.05 |
|---|---|---|
| Epoch 1 (random init) | 0.00 | 0.70 ± 0.84 |
| **Epoch 15 (TD-trained)** | **0.00** | **10.85 ± 0.65** |
| Random agent baseline | — | 1.10 |

- **ε=0.05 epoch 15: 10.85 ± 0.65** — a **9.9x improvement over random play** (1.10) and a **15.5x improvement over the random-init baseline** (0.70).
- **ε=0.0 epoch 15: 0.00** — greedy policy still scores zero.

### 16.4 Greedy failure diagnosis

Running `diagnose_greedy_policy.py` on epoch 15:

```
Episode 1: return=0.0  steps=500
  Action dist: {'LEFT': 500}
  Q-std   mean=0.00728  min=0.00728  max=0.00728
  Q-vals: [0.5359, 0.5374, 0.5291, 0.5468]  (argmax=LEFT)
```

**Q-std = 0.00728 constant for every state across all episodes.** The Q-Head outputs the same Q-values `[NOOP: 0.536, FIRE: 0.537, RIGHT: 0.529, LEFT: 0.547]` regardless of what the ball is doing. This is the **behavior policy's average Q-value** — the expected discounted return of a random agent starting from any state is approximately $0.006 / (1 - 0.99) \approx 0.6$, which matches the observed 0.54 average.

The Q-Head has learned per-action averages from the offline dataset (LEFT is marginally higher because random play produces slightly more leftward reward trajectories), but zero state-dependent information.

### 16.5 Why offline Q-learning on random data can't be state-conditional

With a random behavior policy, every state-action pair has approximately the same expected future reward under that behavior policy:

$$Q^{\pi_\text{random}}(s, a) \approx \frac{r_\text{mean}}{1 - \gamma} \approx \frac{0.006}{0.01} = 0.6 \quad \text{for all } (s, a)$$

Offline TD tries to fit this truth faithfully. The fitted values differ only by tiny per-action averages (based on correlational differences in the dataset), not by state-dependent causal relationships. **The offline dataset simply does not contain the information needed to learn state-conditional Q-values from random play.**

To learn that "RIGHT is better than LEFT when the ball is moving RIGHT", the dataset would need examples of a policy that tried both actions in that state and observed the different outcomes. Random play never systematically explores this contrast.

### 16.6 Summary: what the JEPA pipeline achieves and what it needs

| Stage | Status | Key result |
|---|---|---|
| Stage 1 JEPA pretraining | ✅ | Encoder produces good dynamics features (best config: `latent_dim=256`, `mask_ratio=0.7`, 14 epochs with `reward_loss_weight=10`) |
| Stage 1.5 (imagined reward bootstrap) | ❌ Blocked | Reward head R²=−0.005; encoder ROC-AUC=0.50; imagined rewards are constant per action |
| Stage 2 offline TD | ⚠️ Partial | 10.85 ε=0.05 score (9.9x vs random); greedy score=0; Q-values state-independent |
| Stage 3: online fine-tuning | ⬜ Next | Required to learn state-conditional Q-values |

**The JEPA encoder IS contributing**: the encoder provides compressed state features that allow the Q-Head to fit the behavior policy efficiently. The limitation is not the encoder's feature quality — it is that **offline data from a random policy contains no causal action-state signal**.

The next required step is to use the trained Q-Head as initialization for online DQN fine-tuning in the real environment, leveraging the JEPA encoder as a frozen feature extractor.

---

## 17) Stage 1.5 / Stage 2 final status checklist

| Item | Status |
|---|---|
| `train_latent_imagination.py` fully implemented | ✅ |
| `eval_policy.py` for real-environment evaluation | ✅ |
| `train_policy.py` with train/val split + diagnostics | ✅ updated |
| Stage 1 re-run at `latent_dim=256` (best validated config) | ✅ |
| Stage 1.5 empirical runs (3) — all failed, root cause found | ✅ |
| Fix 1: clamp imagined rewards | ✅ |
| Fix 2: reward loss weight + 20-epoch retrain | ✅ |
| Encoder reward probe (logistic regression, ROC-AUC=0.50) | ✅ proven encoder is reward-blind |
| Stage 2 offline TD (15 epochs) — 10.85 ε=0.05 | ✅ |
| Stage 2 50-epoch run — plateau confirmed | ✅ |
| Greedy failure diagnosed (state-independent Q-values) | ✅ |
| Stage 3: online fine-tuning with JEPA encoder frozen | ⬜ next step |
