"""Diagnose reward head discrimination ability."""
import sys
import torch
import numpy as np
from dataset import BreakoutTransitionDataset
from models import ConvEncoder, RewardHead
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Fix2 checkpoint
ck = torch.load('checkpoints/stage1_fix2/jepa_epoch_014.pt',
                map_location=device, weights_only=False)
encoder = ConvEncoder(input_channels=4, latent_dim=256).to(device)
encoder.load_state_dict(ck['encoder']); encoder.eval()
rhead = RewardHead(latent_dim=256, num_actions=4).to(device)
rhead.load_state_dict(ck['reward_head']); rhead.eval()

ds = BreakoutTransitionDataset('data/random', context_length=4)
loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

all_pred, all_true = [], []
with torch.no_grad():
    for i, batch in enumerate(loader):
        if i >= 20: break
        ctx = batch['context'].to(device)
        act = batch['action'].to(device)
        rew = batch['reward'].to(device)
        z = encoder(ctx)
        pred = rhead(z, act)
        all_pred.append(pred.cpu())
        all_true.append(rew.cpu())

all_pred = torch.cat(all_pred).numpy()
all_true = torch.cat(all_true).numpy()

rew0 = all_pred[all_true == 0]
rew1 = all_pred[all_true > 0]
print(f"N reward=0:  {len(rew0):5d}  pred_mean={rew0.mean():.6f}  std={rew0.std():.6f}")
print(f"N reward>0:  {len(rew1):5d}  pred_mean={rew1.mean():.6f}  std={rew1.std():.6f}")
print(f"Gap: {rew1.mean()-rew0.mean():.6f}  (ideal: ~0.994)")
print()

# Per-action reward head output on a single batch
with torch.no_grad():
    batch0 = next(iter(loader))
    ctx = batch0['context'][:512].to(device)
    z = encoder(ctx)
    preds = []
    for a in range(4):
        at = torch.full((z.size(0),), a, dtype=torch.long, device=device)
        preds.append(rhead(z, at).cpu().numpy())

pm = np.stack(preds, axis=1)  # (N, 4)
rng = pm.max(axis=1) - pm.min(axis=1)
print(f"Action sensitivity — mean per-state range: {rng.mean():.6f}  max: {rng.max():.6f}")
for a in range(4):
    print(f"  action={a}: mean={preds[a].mean():.6f}  std={preds[a].std():.6f}  "
          f"min={preds[a].min():.6f}  max={preds[a].max():.6f}")

print()
# MSE vs constant predictor
true_mean = all_true.mean()
mse_constant = ((all_true - true_mean)**2).mean()
mse_reward_head = ((all_true - all_pred)**2).mean()
print(f"MSE(constant={true_mean:.4f}): {mse_constant:.6f}")
print(f"MSE(reward_head):            {mse_reward_head:.6f}")
print(f"R² = 1 - MSE_head/MSE_const: {1 - mse_reward_head/mse_constant:.6f}  (0=useless, 1=perfect)")
