"""Interpolation de poids entre le modèle original et une soumission (pas de GPU, quelques secondes sur CPU).

  w = orig + alpha * (sub - orig)
  alpha < 1 : plus proche de l'original (meilleur retain, moins d'oubli)
  alpha > 1 : extrapole (plus d'oubli, plus de dérive)

Usage : python interp.py sub_v6_none_r0.3.pt 0.8 1.2
"""
import sys

import torch

src = sys.argv[1]
alphas = [float(a) for a in sys.argv[2:]]
orig = torch.load("model.pt", map_location="cpu", weights_only=True)
sub = torch.load(src, map_location="cpu", weights_only=True)
for a in alphas:
    out = {k: (orig[k].float() + a * (sub[k].float() - orig[k].float())).contiguous() for k in orig}
    assert all(torch.isfinite(v).all() for v in out.values())
    name = src.replace(".pt", f"_a{a}.pt")
    torch.save(out, name)
    print("écrit", name)
