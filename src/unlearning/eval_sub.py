"""Évalue des soumissions hors-ligne pour choisir l'alpha d'extrapolation sans griller de créneau.

Pour chaque .pt : CE (T=1) mean/std sur Df propre + Df MILD, S_pred estimé vs la cible gold, retain externe.
Usage (GPU) : python -u eval_sub.py --gold gold_combo/gold_combo_a50.pt sub_v12_*.pt
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

import unlearn as U
from task_template import build_model, load_forget_images, load_model, predict

ap = argparse.ArgumentParser()
ap.add_argument("--gold", required=True)
ap.add_argument("--ext", default="ext_retain.npy")
ap.add_argument("subs", nargs="+")
args = ap.parse_args()

torch.manual_seed(0); np.random.seed(0)
orig = load_model().eval()
imgs = load_forget_images()
U.EXT = np.load(args.ext, mmap_mode="r")
z0 = predict(orig, imgs)
y = z0.argmax(1)
g = F.softmax(torch.load(args.gold, map_location="cpu")["logits"].float() / 2, 1).to(z0.device)
g1 = g ** 2; g1 = g1 / g1.sum(1, keepdim=True)
ceg = -g1.clamp_min(1e-8).log().gather(1, y[:, None]).squeeze(1)
den = F.kl_div(F.log_softmax(z0 / 2, 1), g, reduction="batchmean").item()
print(f"cible {args.gold} : CE {ceg.mean():.3f} ± {ceg.std():.3f}")

# vues MILD fixes (même tirage pour toutes les soumissions)
xm = torch.stack([U.MILD(torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)) for im in imgs])
xm = xm.permute(0, 2, 3, 1).contiguous().numpy()


def stats(z):
    ce = F.cross_entropy(z, y, reduction="none")
    return ce.mean().item(), ce.std().item()


for f in args.subs:
    m = build_model(); m.load_state_dict(torch.load(f, map_location="cpu"), strict=True); m = m.to(z0.device).eval()
    z, zm = predict(m, imgs), predict(m, xm)
    (mu, sd), (mum, sdm) = stats(z), stats(zm)
    num = F.kl_div(F.log_softmax(z / 2, 1), g, reduction="batchmean").item()
    d = abs(mu - ceg.mean().item()) + abs(sd - ceg.std().item())
    kl_e, ag = U.retain_proxy_ext(orig, m)
    print(f"{f:<38} acc={(z.argmax(1) == y).float().mean():.3f}  CE {mu:.3f}±{sd:.3f} (mild {mum:.3f}±{sdm:.3f})"
          f"  |Δμ|+|Δσ|={d:.3f}  S_pred≈{max(0, 1 - num / den):.3f}  ext KL={kl_e:.4f} accord={ag:.3f}")
