"""
Check if encoder latents contain linearly-separable reward signal.
Uses a tiny PyTorch logistic-regression head trained from scratch,
with pos_weight balancing for the class imbalance.

Reports:
  - ROC-AUC  (0.5 = random, 1.0 = perfect)
  - Average Precision  (baseline = class prior)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import BreakoutTransitionDataset
from models import ConvEncoder
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ck = torch.load('checkpoints/stage1_fix2/jepa_epoch_014.pt',
                map_location=device, weights_only=False)
encoder = ConvEncoder(input_channels=4, latent_dim=256).to(device)
encoder.load_state_dict(ck['encoder']); encoder.eval()

ds = BreakoutTransitionDataset('data/random', context_length=4)
loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

feats, labels = [], []
with torch.no_grad():
    for batch in loader:
        ctx = batch['context'].to(device)
        z = encoder(ctx)
        feats.append(z.cpu())
        labels.append(batch['reward'])

feats = torch.cat(feats, dim=0)                    # (N, 256)
labels = (torch.cat(labels, dim=0) > 0).float()   # (N,) binary

n = len(labels)
n_pos = labels.sum().item()
n_neg = n - n_pos
print(f"Dataset: {n} samples, {int(n_pos)} positive ({100*n_pos/n:.2f}%)")

# ── Train/val split ───────────────────────────────────────────────────
split = int(0.8 * n)
feats_tr, feats_val = feats[:split].to(device), feats[split:].to(device)
y_tr, y_val = labels[:split].to(device), labels[split:].to(device)

pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=device)

# ── Tiny logistic regression head ─────────────────────────────────────
lin = nn.Linear(256, 1).to(device)
opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(50):
    lin.train()
    logits = lin(feats_tr).squeeze(1)
    loss = F.binary_cross_entropy_with_logits(logits, y_tr, pos_weight=pos_weight)
    opt.zero_grad(); loss.backward(); opt.step()

lin.eval()
with torch.no_grad():
    logits_val = lin(feats_val).squeeze(1)
    proba = torch.sigmoid(logits_val).cpu().numpy()

y_np = y_val.cpu().numpy()

# ── ROC-AUC (manual) ──────────────────────────────────────────────────
def roc_auc(y_true, y_score):
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    pos_scores = y_score[pos_mask]
    neg_scores = y_score[neg_mask]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    # U-statistic
    n_p, n_n = len(pos_scores), len(neg_scores)
    u = sum((p > n).sum() + 0.5 * (p == n).sum()
            for p in pos_scores for n in [neg_scores])
    # vectorised version
    u = (pos_scores[:, None] > neg_scores[None, :]).sum() + \
        0.5 * (pos_scores[:, None] == neg_scores[None, :]).sum()
    return float(u) / (n_p * n_n)

def avg_precision(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    precision = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1)
    recall_delta = y_sorted / y_sorted.sum()
    return float((precision * recall_delta).sum())

roc = roc_auc(y_np, proba)
ap  = avg_precision(y_np, proba)
baseline_ap = y_np.mean()

print(f"\nLogistic Regression probe on encoder latents (pos_weight={pos_weight.item():.0f}):")
print(f"  ROC-AUC      : {roc:.4f}  (0.5 = random, 1.0 = perfect)")
print(f"  Avg-Precision: {ap:.4f}  (baseline = class prior = {baseline_ap:.4f})")
print(f"  AP lift      : {ap/baseline_ap:.1f}x over random  (>3x → useful signal)")
print()
print("Interpretation:")
if roc > 0.7:
    print("  ✅ Encoder latents DO contain reward-predictive information.")
elif roc > 0.55:
    print("  ⚠️  Weak reward signal in encoder latents — Fix 3 may help but won't be dramatic.")
else:
    print("  ❌ Encoder latents contain NO reward-discriminative signal.")
    print("     Fix 3 (BCE reward head) will not help unless the encoder is jointly retrained.")
