# CONSIDERATIONS

## Rewards: Realism vs. Benchmarking in a JEPA Actor

I treat reward realism and benchmark comparability as two different goals.

The short version is: **assuming reward exists directly in observation is a shortcut**. In realistic settings (for example, robotics), no explicit game-like score is available.

### 1. Human-in-the-loop framing

In practical systems, reward often behaves like an **internal cost function**.

- **Hardwired costs:** JEPA-style planning can include a cost module (analogous to constraints like safety, energy, or damage avoidance).
- **Goal-conditioned JEPA:** Instead of scalar reward, I can define a target observation/latent and optimize distance from current latent $z_t$ to goal latent $z_{goal}$.

### 2. Why Atari reward is still useful

Using Atari `reward` remains a reasonable benchmarking choice.

- The score is observable in-frame, so a sufficiently capable model can infer it.
- Training `PolicyHead` with reward provides a direct optimization signal and speeds evaluation of representation quality.

### 3. No-reward alternative: curiosity

For stricter realism, I can replace extrinsic reward with **prediction-error-driven exploration**.

- Objective: seek states where the predictor error is high.
- Result: the agent explores novel transitions and can still discover high-value behaviors while improving world-model fidelity.

### Updated realistic Stage 2 (goal-seeking)

| Stage | Input | Target | Reality Check |
| --- | --- | --- | --- |
| **Stage 1 (JEPA)** | Pixels | Next Latent | **Realistic:** observation-only training |
| **Stage 2 (Actor)** | Latent $z_t$ | **Distance to $z_{goal}$** | **Realistic:** goal representation rather than numeric score |

### Realism summary

In this Atari pipeline, reward-based training is useful for benchmarking and comparability.
For a more general JEPA actor, I would prefer a goal/state-cost objective over explicit game score.

---

## Continual Learning: From Staged to Lifelong JEPA

I currently use **staged learning** (offline pretraining followed by downstream policy training). This is stable, but not fully continual.

### 1. Why staged learning is limited

- **Stage 1 (JEPA):** learns from a fixed data distribution.
- **Stage 2 (Actor):** optimizes policy on frozen or mostly fixed representations.
- **Gap:** once dynamics shift, a frozen world model cannot adapt to new regimes.

### 2. What continual learning requires

To make the system continual, I need concurrent adaptation:

- The JEPA encoder/predictor keeps updating online.
- The actor keeps updating against the evolving latent space.
- Data collection, representation learning, and control become a closed loop.

### 3. Main obstacles and mitigations

| Problem | Description | Mitigation |
| --- | --- | --- |
| **Catastrophic Forgetting** | New regimes overwrite older dynamics | Replay buffer with mixed old/new data |
| **Non-Stationarity** | Policy trains on shifting latent semantics | EMA/target networks and slower representation updates |

### Continual-learning summary

The current 3-stage design is a strong and stable baseline.
To claim true continual learning, I need an additional online co-adaptation phase where both JEPA and actor update during active play.
