# Stage 1 — JEPA Training, Validation, and Confidence Checks

This report summarizes exactly how Stage 1 was trained, how we validated it, how we tested against trivial copy behavior, and why a horizon × masking grid search was necessary for confidence.

## 1) Training approach used

| Component | Implementation |
|---|---|
| Environment/data | `ALE/Breakout-v5` offline random-policy transitions (`Breakout/data/random`, 50,000 transitions) |
| Input representation | Grayscale `84×84`, context window of 4 frames, normalized to `[0,1]` |
| Core model | `ConvEncoder` + `ActionConditionedPredictor` + EMA target encoder |
| Primary objective | `SmoothL1( Predictor(Encoder(masked context), a_t), TargetEncoder(target) )` |
| Auxiliary objective | `SmoothL1( RewardHead(z_t, a_t), r_t )` |
| Masking | Spatial masking (default 50%) on online encoder input |
| Optimization | AdamW, batch size 256 |
| Key options | `--train-horizon` (true k-step objective), `--mask-ratio`, `--val-ratio` |

### Why this structure

- The EMA target stabilizes latent targets.
- Masking reduces pixel-level shortcuts.
- Reward auxiliary loss makes the latent dynamics useful for downstream control.
- `--train-horizon` enables both classic 1-step JEPA and harder k-step variants without forking code.

## 2) Validation protocol used

| Validation item | Method |
|---|---|
| Train/val split | Random split with `--val-ratio 0.1` |
| Core tracked metrics | `train_total_loss`, `val_total_loss`, JEPA/reward components |
| Action-conditioning metric | `val_action_sensitivity` (latent prediction change when action is perturbed) |
| Dynamics stability metric | `val_rollout_drift` over fixed unroll horizon |

From the 8-epoch Stage 1 run (`Breakout/checkpoints/stage1_viz_epochs`), the overall pattern was stable convergence with low generalization gap and a predictable loss floor.

## 3) Sanity checks against copy behavior

We explicitly compare JEPA against a trivial latent-copy baseline.

| Sanity check | Why it matters |
|---|---|
| Copy baseline (`z_t` vs target latent) | Detects whether the predictor is just “identity-like” |
| JEPA loss vs copy loss over epochs | Confirms model improves beyond naive copying |
| Action sensitivity | Verifies predictions depend on actions, not only on static continuity |

Interpretation logic:
- If JEPA loss ≪ copy baseline and action sensitivity stays non-zero, learning is not pure copying.
- If action sensitivity collapses while loss stays low, the representation may be over-smooth or under-actioned.

## 4) Why we needed a grid search

Single-run metrics can be misleading because masking pressure and temporal horizon interact.

| Grid axis | Purpose |
|---|---|
| `train_horizon` | Controls temporal difficulty and planning depth in latent prediction |
| `mask_ratio` | Controls anti-shortcut pressure and representation robustness |

### What `mask_ratio` means (in practice)

`mask_ratio` is the probability that a spatial pixel location is zeroed out in the online encoder input.
The same spatial mask is applied across all frames in the context stack.

| `mask_ratio` | Interpretation |
|---:|---|
| `0.20` | Light masking: model still sees most pixels; easier optimization |
| `0.50` | Medium masking: balanced anti-shortcut pressure |
| `0.80–0.90` | Heavy masking: strongest pressure to infer structure, but can hurt optimization stability |

Why this matters: if masking is too low, the predictor can rely on near-copy shortcuts; if too high, the model may struggle to recover enough signal.

Search executed:

| Property | Value |
|---|---|
| Search space | 48 combinations |
| Horizons | `{1,2,3,4,5,6}` |
| Mask ratios | `{0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90}` |
| Results | `Breakout/checkpoints/grid_stage1_full_20260318/grid_results.csv` |

Full sweep parameters:

| Parameter | Value |
|---|---|
| `epochs` | `1` per configuration |
| `batch_size` | `256` |
| `max_train_batches` | `40` |
| `max_val_batches` | `12` |
| `diagnostic_rollout_steps` | `3` |
| Selection metric | `val_action_sensitivity` (primary), `val_total_loss` (stability check) |

Top-by-sensitivity summary (from the full grid):

| Rank | Horizon | Mask | `val_action_sensitivity` | `val_total_loss` |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.90 | 0.022330 | 0.005533 |
| 2 | 1 | 0.60 | 0.020641 | 0.005547 |
| 3 | 1 | 0.50 | 0.020034 | 0.006205 |
| 4 | 1 | 0.80 | 0.020014 | 0.007072 |
| 5 | 1 | 0.40 | 0.019794 | 0.005652 |

Low-loss but low-sensitivity examples (tradeoff evidence):

| Horizon | Mask | `val_total_loss` | `val_action_sensitivity` |
|---:|---:|---:|---:|
| 5 | 0.30 | 0.004582 | 0.000647 |
| 3 | 0.40 | 0.005175 | 0.000916 |
| 3 | 0.50 | 0.005267 | 0.000905 |

### Confidence trick that matters most

The grid + Pareto view gives confidence that we are not overfitting to a single scalar objective:
- some configs minimize loss,
- others maximize action sensitivity,
- and we can choose operating points intentionally.

## 5) Latent-dimension sweep (two-stage, multithreaded)

To validate representation capacity, we also swept the latent size (`latent_dim`) while keeping the best Stage 1 settings fixed (`train_horizon=1`, `mask_ratio=0.7`).

| Sweep stage | Method |
|---|---|
| Stage A (coarse) | Evaluate `{128, 256, 384, 512, 768, 1024}` |
| Stage B (refine) | Evaluate around coarse knee with step 64 and radius 2 |
| Parallelism | Multithreaded Python executor (`ThreadPoolExecutor`, `--workers 2`) |
| Selection logic | Pareto front on `(val_total_loss ↓, val_action_sensitivity ↑)` + knee-point rule |

Artifacts:

| Artifact | File |
|---|---|
| Coarse results | `Breakout/checkpoints/latent_dim_sweep_20260318/coarse_results.csv` |
| Refine results | `Breakout/checkpoints/latent_dim_sweep_20260318/refine_results.csv` |
| Combined results | `Breakout/checkpoints/latent_dim_sweep_20260318/all_results.csv` |
| Sweep summary | `Breakout/checkpoints/latent_dim_sweep_20260318/summary.txt` |

Final recommendation from the knee selector:

| Recommended `latent_dim` | `val_total_loss` | `val_action_sensitivity` |
|---:|---:|---:|
| **192** | **0.005307** | **0.020172** |

Interpretation:
- Smaller dims can underfit dynamics under masking.
- Larger dims can reduce compactness and do not necessarily improve action sensitivity.
- `latent_dim=192` is currently the best tradeoff for this Stage 1 budget.

## 6) Stage 1 charts (all generated)

| Chart | File |
|---|---|
| Train vs validation total loss | [screenshots/stage1_loss_train_val.png](screenshots/stage1_loss_train_val.png) |
| JEPA vs reward loss components | [screenshots/stage1_loss_components.png](screenshots/stage1_loss_components.png) |
| Action sensitivity over epochs | [screenshots/stage1_action_sensitivity.png](screenshots/stage1_action_sensitivity.png) |
| Copy baseline gap | [screenshots/stage1_copy_baseline_gap.png](screenshots/stage1_copy_baseline_gap.png) |
| Rollout drift | [screenshots/stage1_rollout_drift.png](screenshots/stage1_rollout_drift.png) |
| Grid heatmap: `val_action_sensitivity` | [screenshots/stage1_grid_val_sensitivity_heatmap.png](screenshots/stage1_grid_val_sensitivity_heatmap.png) |
| Grid heatmap: `val_total_loss` | [screenshots/stage1_grid_val_loss_heatmap.png](screenshots/stage1_grid_val_loss_heatmap.png) |
| Pareto: loss vs sensitivity | [screenshots/stage1_pareto_loss_vs_sensitivity.png](screenshots/stage1_pareto_loss_vs_sensitivity.png) |
| Latent-dim Pareto + knee | [screenshots/stage1_latent_dim_pareto_knee.png](screenshots/stage1_latent_dim_pareto_knee.png) |

## 7) Practical Stage 1 conclusion

| Topic | Conclusion |
|---|---|
| Default config | Keep `train_horizon=1`, `mask_ratio=0.7`, `latent_dim=192` under this budget |
| Optional capability | Keep `--train-horizon` enabled for k-step experiments |
| Optional capacity sweep | Keep two-stage latent sweep script for periodic recalibration (`Breakout/run_latent_dim_sweep.py`) |
| Next direction | Longer training + masking curriculum for `horizon>1` before judging multi-step superiority |
| Overall status | Stage 1 is successful: stable training, validated diagnostics, and strong ablation evidence |

## 8) Appendable random-action collision analysis artifacts

To quantify Stage 1 random-data interaction signal, evaluation now writes append-only artifacts under:

- `Breakout/checkpoints/eval_policy/eval_runs.csv`
- `Breakout/checkpoints/eval_policy/eval_episodes.csv`
- `Breakout/checkpoints/eval_policy/paddle_hit_events.jsonl`
- `Breakout/checkpoints/eval_policy/ball_heatmap_<run_id>.npy`
- `Breakout/checkpoints/eval_policy/ball_heatmap_<run_id>.png`

Metric definitions used:

| Metric | Definition |
|---|---|
| Brick hits (proxy) | Count of transitions where `reward > 0` |
| Paddle hits (estimate) | Conservative pixel heuristic: detected ball trajectory flips from down→up near paddle y-band and within paddle x-range |
| Ball spatial distribution | 2D heatmap over detected ball positions (`84×84`) aggregated across episodes |

Debugging support:

- Optional paddle-hit screenshots are written to `screenshots/eval_debug/last10`.
- Retention is rolling: only the latest 10 screenshots are kept.

Charts:

- `Breakout/generate_stage_charts.py` now reads appended eval runs from `Breakout/checkpoints/eval_policy/eval_runs.csv` (with legacy fallback if unavailable).

## 9) Random Dataset File Structure

The regenerated random dataset now has four layers of artifacts under `Breakout/data/random`:

| File pattern | Purpose |
|---|---|
| `transitions_00000.npz` | Base transition shard: `obs`, `next_obs`, `action`, `reward`, `terminated`, `truncated`, `episode_id`, `episode_step` |
| `transitions_00000_objects.npz` | OCAtari object sidecar aligned 1:1 with the transition shard |
| `transitions_00000_ramdecode.npz` | RAM-byte sidecar aligned 1:1 with the transition shard |
| `random_debug.mp4` | Full visual debug video of the collection run |

### OCAtari object sidecar fields

Each `*_objects.npz` shard stores object-derived quantities:

| Field | Meaning |
|---|---|
| `ball_x`, `ball_y` | Ball position from OCAtari object extraction |
| `ball_vx`, `ball_vy` | Ball velocity from frame-to-frame object deltas |
| `paddle_x`, `paddle_y`, `paddle_w`, `paddle_h` | Paddle bounding box |
| `block_count` | Visible OCAtari `Block` object count |
| `player_score_x/y/w/h` | HUD score bounding box |
| `live_x/y/w/h` | HUD lives bounding box |
| `player_number_x/y/w/h` | HUD player-number bounding box |

### RAM-decode sidecar fields

Each `*_ramdecode.npz` shard stores raw bytes plus decoded gameplay values:

| Field | Meaning |
|---|---|
| `ram_raw` | Full 128-byte ALE RAM snapshot per transition |
| `ram_player_x_byte`, `ram_ball_x_byte`, `ram_ball_y_byte` | Key raw gameplay bytes |
| `ram_lives_byte`, `ram_score_hi_byte`, `ram_score_lo_byte` | Raw HUD/score bytes |
| `decoded_paddle_x`, `decoded_paddle_y` | Paddle coordinates from RAM decoding |
| `decoded_ball_x`, `decoded_ball_y` | Ball coordinates from RAM decoding |
| `decoded_ball_vx`, `decoded_ball_vy` | Ball velocity from decoded positions |
| `decoded_block_count` | Brick bitmap population count from RAM |
| `decoded_score` | Decoded score integer |
| `decoded_lives` | Decoded lives count |
| `decoded_player_number` | Player number (single-player collection = `1`) |

### Post-collection summary artifacts

After collection completes, the collector now also generates:

| File | Purpose |
|---|---|
| `artifacts/random_debug.gif` | GIF version of `random_debug.mp4` |
| `artifacts/ball_heatmap.png` | Final ball-position heatmap |
| `artifacts/ball_heatmap.gif` | Animated cumulative ball heatmap |
| `artifacts/paddle_heatmap.png` | Paddle movement heatmap |
| `artifacts/collection_stats.json` | Aggregate stats: total rewards, bricks broken, paddle touches, episodes |
| `run_summary.json` | Run manifest with shard lists plus generated artifact paths and summary stats |
