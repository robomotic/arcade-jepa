"""
Quick probe: what actions does the greedy Q-Head select,
and what are the Q-value spreads per episode?
"""
import sys
import collections
import torch
import numpy as np
from envs import create_breakout_env, register_atari_envs
from models import ConvEncoder, QHead

register_atari_envs()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load encoder
enc_ck = torch.load('checkpoints/stage1_fix2/jepa_epoch_014.pt',
                    map_location=device, weights_only=False)
encoder = ConvEncoder(input_channels=4, latent_dim=256).to(device)
encoder.load_state_dict(enc_ck['encoder']); encoder.eval()

# Load Q-Head
q_ck = torch.load('checkpoints/stage2_fix2/q_head_epoch_015.pt',
                  map_location=device, weights_only=False)
q_head = QHead(latent_dim=256, num_actions=4).to(device)
q_head.load_state_dict(q_ck['q_head']); q_head.eval()

env = create_breakout_env(render_mode=None)

context_length = 4
n_episodes = 3
action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']

for ep in range(n_episodes):
    obs, _ = env.reset()
    obs = np.array(obs, dtype=np.float32) / 255.0
    frame_buffer = collections.deque([obs] * context_length, maxlen=context_length)
    action_counts = collections.Counter()
    q_stds = []
    total_reward = 0
    steps = 0

    for step in range(500):   # just first 500 steps
        ctx = np.stack(list(frame_buffer), axis=0)[None]  # (1, 4, 84, 84)
        ctx_t = torch.from_numpy(ctx).to(device)
        with torch.no_grad():
            z = encoder(ctx_t)
            q = q_head(z)[0]   # (4,)
        q_stds.append(float(q.std()))
        action = int(q.argmax())
        action_counts[action_names[action]] += 1

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        obs = np.array(obs, dtype=np.float32) / 255.0
        frame_buffer.append(obs)
        steps += 1
        if terminated or truncated:
            break

    print(f"Episode {ep+1}: return={total_reward:.1f}  steps={steps}")
    print(f"  Action dist: {dict(sorted(action_counts.items()))}")
    print(f"  Q-std   mean={np.mean(q_stds):.5f}  min={np.min(q_stds):.5f}  max={np.max(q_stds):.5f}")
    q_vals_sample = None
    # Print a few Q-value vectors
    ctx_t_last = torch.from_numpy(np.stack(list(frame_buffer), axis=0)[None]).to(device)
    with torch.no_grad():
        z_last = encoder(ctx_t_last)
        q_last = q_head(z_last)[0]
    print(f"  Final Q-vals: {[f'{v:.4f}' for v in q_last.tolist()]}  "
          f"(argmax={action_names[int(q_last.argmax())]})")

env.close()
