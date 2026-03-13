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

My immediate next step is to apply Fix 1 (clamp) and also add a `--reward-loss-weight` argument to `train_jepa.py` so the reward head is trained more strongly relative to the JEPA loss, then re-run Stage 1 for 20 epochs and re-attempt Stage 1.5.

---

## 11) Stage 1.5 status checklist

| Item | Status |
|---|---|
| `train_latent_imagination.py` fully implemented with ε-greedy, train/val split, target network, diagnostics | ✅ |
| `eval_policy.py` for real-environment evaluation of any Q-Head checkpoint | ✅ |
| `train_policy.py` (Stage 2) updated with matching latent-dim default and target network | ✅ |
| Graceful fallback for legacy Stage 1 checkpoints missing `reward_head` key | ✅ |
| Stage 1 re-run at `latent_dim=256` (best validated config) | ✅ |
| First Stage 1.5 empirical run | ✅ (completed — failure diagnosed) |
| Fix 1: clamp imagined rewards to `[0, ∞)` in rollout | ⬜ |
| Fix 2: `--reward-loss-weight` in Stage 1 + 20-epoch retrain | ⬜ |
| Fix 3 (optional): collect policy-guided data for better action diversity | ⬜ |
| Clean Stage 1.5 run with healthy diagnostics | ⬜ |
| Stage 2 warm-start comparison (with vs without Stage 1.5 init) | ⬜ (after clean Stage 1.5) |
