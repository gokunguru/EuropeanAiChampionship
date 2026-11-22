"""Cible combinée : moyenne ARITHMÉTIQUE à T=2, poids kNN adaptatif par image.

  p = (1 - a(x)) * mean_surr p^T2 + a(x) * p_kNN^T2,   a(x) = amax * sigmoid((sbar(x) - median(sbar)) / ds)
  a(x) = 0 si argmax(orig) est une classe absente d'ext.
CPU, quelques secondes. Usage :
  python combine_gold.py --knn gold_knn2/gold_knn2_g0.5_K15_t0.02.pt --surr gold_k6sh50.pt gold_k8sh30.pt --amax 0.5 --out gold_combo/gold_combo_a50.pt
"""
import argparse
import os

import torch
import torch.nn.functional as F

from task_template import load_forget_images, load_model, predict

ap = argparse.ArgumentParser()
ap.add_argument("--knn", required=True)
ap.add_argument("--surr", nargs="+", required=True)
ap.add_argument("--amax", type=float, default=0.5)
ap.add_argument("--ds", type=float, default=0.05)
ap.add_argument("--out", required=True)
ap.add_argument("--y", default="pseudo_y.pt", help="cache des pseudo-labels (argmax orig)")
args = ap.parse_args()

if os.path.exists(args.y):
    y = torch.load(args.y)
else:
    y = predict(load_model().eval(), load_forget_images()).argmax(1).cpu()
    torch.save(y, args.y)

ps = torch.stack([F.softmax(torch.load(f, map_location="cpu")["logits"].float() / 2, 1) for f in args.surr]).mean(0)
k = torch.load(args.knn, map_location="cpu")
pk = F.softmax(k["logits"].float() / 2, 1)
sbar = k["sbar"].float()
a = args.amax * torch.sigmoid((sbar - sbar.median()) / args.ds)
a[k["missing"]] = 0.0
p = (1 - a[:, None]) * ps + a[:, None] * pk

for name, q in (("surrogates", ps), ("kNN", pk), ("combiné", p)):
    q1 = q ** 2; q1 = q1 / q1.sum(1, keepdim=True)          # p^T1 ∝ (p^T2)^2
    ce = -q1.clamp_min(1e-8).log().gather(1, y[:, None]).squeeze(1)
    print(f"  {name:<11} acc={(q.argmax(1) == y).float().mean():.3f}  CE(T=1) {ce.mean():.3f} ± {ce.std():.3f}")
print(f"  poids kNN a(x) : moy {a.mean():.3f}  min {a.min():.3f}  max {a.max():.3f}")
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
torch.save({"logits": 2 * p.clamp_min(1e-8).log()}, args.out)
print("écrit", args.out)
