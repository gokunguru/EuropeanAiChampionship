"""kNN p_gold v2 : features [0.4*h10, 0.6*h11] dé-collapsées, requête moyennée sur (x, hflip x),
et sauvegarde de sbar (similarité moyenne des K voisins) pour la pondération adaptative dans combine_gold.py.

De-collapse : h <- normalize(h - gamma * mu_c), mu_c = centroïde (features L2) des images ext prédites en classe c ;
c = argmax(orig) pour la requête Df comme pour chaque image ext (symétrique si --decol_ext 1).
Réutilise ext_clean_idx.npy (dédup faite par gold_knn.py).

Sorties : gold_knn2/gold_knn2_g{gamma}_K{K}_t{tau}.pt  (logits = 2*log p_T2, sbar [3500])
Usage (1 GPU, ~3 min) : python -u gold_knn2.py --ext ext_retain.npy
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from task_template import load_forget_images, load_model, preprocess


@torch.no_grad()
def run(model, u8, layers=(10, 11), bs=128, flip=False):
    store, hs = {}, []
    for L in layers:
        hs.append(model.blocks[L].register_forward_hook(lambda m, i, o, L=L: store.__setitem__(L, o[:, 0])))
    feats = {L: [] for L in layers}
    logits = []
    for i in range(0, len(u8), bs):
        x = preprocess(torch.from_numpy(np.ascontiguousarray(u8[i:i + bs])))
        if flip:
            x = x.flip(3)
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(x).float()
        for L in layers:
            feats[L].append(F.normalize(store[L].float(), dim=1))
        logits.append(z.cpu())
    for h in hs:
        h.remove()
    return {L: torch.cat(v) for L, v in feats.items()}, torch.cat(logits)


def centroids(f, c, n=100):
    mu = torch.zeros(n, f.shape[1], device=f.device)
    mu.index_add_(0, c, f)
    cnt = torch.bincount(c, minlength=n).clamp_min(1).float()[:, None]
    return mu / cnt


def build(feats, cls, mus, gamma, w=(0.4, 0.6)):
    parts = []
    for (L, f), wl in zip(sorted(feats.items()), w):
        g = f - gamma * mus[L][cls] if gamma > 0 else f
        parts.append(wl * F.normalize(g, dim=1))
    return F.normalize(torch.cat(parts, 1), dim=1).half()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default="ext_retain.npy")
    ap.add_argument("--gammas", default="0,0.5")
    ap.add_argument("--Ks", default="15,20")
    ap.add_argument("--taus", default="0.02,0.03")
    ap.add_argument("--decol_ext", type=int, default=0, help="1 = retire aussi le centroïde côté ext")
    args = ap.parse_args()
    print(vars(args))
    os.makedirs("gold_knn2", exist_ok=True)
    model = load_model().eval()
    forget = load_forget_images()
    ext = np.load(args.ext, mmap_mode="r")
    clean = torch.from_numpy(np.load("ext_clean_idx.npy"))

    ff, zf = run(model, forget)
    ffl, _ = run(model, forget, flip=True)
    fe, ze = run(model, ext)
    y = zf.argmax(1).to(ff[11].device)
    ce_cls = ze.argmax(1).to(ff[11].device)
    fe = {L: v[clean.to(v.device)] for L, v in fe.items()}
    ce_cls = ce_cls[clean.to(ce_cls.device)]
    ze = ze[clean]
    p2 = F.softmax(ze / 2, 1)
    p1 = F.softmax(ze, 1)
    mus = {L: centroids(v, ce_cls) for L, v in fe.items()}
    missing = set(range(100)) - set(ce_cls.unique().tolist())
    print(f"ext propre : {len(clean)} images ; classes absentes d'ext : {sorted(missing)}")
    yc = y.cpu()

    Kmax = max(int(k) for k in args.Ks.split(","))
    for gamma in (float(g) for g in args.gammas.split(",")):
        qe = build(fe, ce_cls, mus, gamma if args.decol_ext else 0.0)
        res = []
        for fq in (ff, ffl):  # x et hflip(x)
            q = build(fq, y, mus, gamma)
            sims, idx = [], []
            for i in range(0, len(q), 512):
                v, ix = (q[i:i + 512] @ qe.T).float().topk(Kmax, dim=1)
                sims.append(v.cpu()); idx.append(ix.cpu())
            res.append((torch.cat(sims), torch.cat(idx)))
        print(f"gamma={gamma} : sim 1er voisin {res[0][0][:, 0].mean():.3f}, {Kmax}e {res[0][0][:, -1].mean():.3f}")
        for K in (int(k) for k in args.Ks.split(",")):
            for tau in (float(t) for t in args.taus.split(",")):
                q2 = q1 = sbar = 0
                for sims, idx in res:
                    w = F.softmax(sims[:, :K] / tau, 1)
                    q2 = q2 + 0.5 * (w[..., None] * p2[idx[:, :K]]).sum(1)
                    q1 = q1 + 0.5 * (w[..., None] * p1[idx[:, :K]]).sum(1)
                    sbar = sbar + 0.5 * sims[:, :K].mean(1)
                ce = -q1.clamp_min(1e-8).log().gather(1, yc[:, None]).squeeze(1)
                acc = (q1.argmax(1) == yc).float().mean().item()
                miss = torch.tensor([int(c) in missing for c in yc])
                name = f"gold_knn2/gold_knn2_g{gamma}_K{K}_t{tau}.pt"
                print(f"  g={gamma} K={K:<3} tau={tau:<5} acc={acc:.3f}  CE(T=1) mean={ce.mean():.3f} std={ce.std():.3f}"
                      f"  sbar med={sbar.median():.3f}  -> {name}")
                torch.save({"logits": 2 * q2.clamp_min(1e-8).log(), "sbar": sbar, "missing": miss,
                            "acc_forget": acc, "ce_mean": ce.mean().item(), "ce_std": ce.std().item()}, name)


if __name__ == "__main__":
    main()
