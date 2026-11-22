"""p_gold par kNN (sans entraînement) + déduplication Df <-> ext.

Sur ext, l'original n'a rien mémorisé : il s'y comporte comme un gold. Pour chaque image forget x, on prend ses K
plus proches voisins dans ext (features CLS du bloc --layer, cosinus) et on moyenne les prédictions de l'original
sur ces voisins (softmax T=2), pondérées par softmax(sim / tau). Les quasi-doublons de Df dans ext sont exclus.

Sorties :
  gold_knn_K{K}_t{tau}.pt   ("logits" = 2*log p_T2, pour que softmax(logits/2) = p_T2 exactement ; lu par unlearn.py --gold)
  ext_clean_idx.npy         indices de ext sans quasi-doublon de Df (pour les surrogates)
Usage (GPU, ~3-5 min) : python -u gold_knn.py --ext ext_retain.npy --layer 8
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from task_template import DEVICE, load_forget_images, load_model, preprocess


@torch.no_grad()
def run(model, u8, layer, bs=128, flip=False):
    feats, logits = [], []
    store = {}
    h = model.blocks[layer].register_forward_hook(lambda m, i, o: store.__setitem__("f", o[:, 0]))
    for i in range(0, len(u8), bs):
        x = preprocess(torch.from_numpy(np.ascontiguousarray(u8[i:i + bs])))
        if flip:
            x = x.flip(3)
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(x).float()
        feats.append(F.normalize(store["f"].float(), dim=1).half())
        logits.append(z.cpu())
    h.remove()
    return torch.cat(feats), torch.cat(logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default="ext_retain.npy")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--Ks", default="15,30")
    ap.add_argument("--taus", default="0.02,0.05,0.1")
    ap.add_argument("--dup_thr", type=float, default=0.0, help="0 = auto (mu + 3 sigma)")
    args = ap.parse_args()
    print(vars(args))
    model = load_model().eval()
    forget = load_forget_images()
    ext = np.load(args.ext, mmap_mode="r")

    ff, zf = run(model, forget, args.layer)
    ffl, _ = run(model, forget, args.layer, flip=True)
    y = zf.argmax(1)
    fe, ze = run(model, ext, args.layer)
    print(f"features : Df {tuple(ff.shape)}  ext {tuple(fe.shape)}")

    # --- déduplication : pour chaque image ext, similarité max avec Df (et Df miroir)
    mx_e = torch.full((len(fe),), -1.0, device=fe.device)
    mx_f = torch.full((len(ff),), -1.0, device=fe.device)
    for j in range(0, len(fe), 4096):
        blk = fe[j:j + 4096]
        s = torch.maximum(ff @ blk.T, ffl @ blk.T).float()          # [Nf, b]
        mx_e[j:j + 4096] = s.max(0).values
        mx_f = torch.maximum(mx_f, s.max(1).values)
    mu, sd = mx_f.mean().item(), mx_f.std().item()
    thr = args.dup_thr or (mu + 3 * sd)
    print(f"sim max Df->ext : moy {mu:.3f}  std {sd:.3f}  p99 {mx_f.quantile(0.99):.3f}  max {mx_f.max():.3f}")
    print(f"seuil doublon = {thr:.3f} : {(mx_f > thr).sum().item()} images Df ont un quasi-doublon ;"
          f" {(mx_e > thr).sum().item()} images ext exclues")
    clean = torch.nonzero(mx_e <= thr).squeeze(1).cpu().numpy()
    np.save("ext_clean_idx.npy", clean)
    print(f"ext_clean_idx.npy : {len(clean)} images")

    # --- kNN
    fec = fe[torch.from_numpy(clean).to(fe.device)]
    p2 = F.softmax(ze[clean] / 2, 1)
    p1 = F.softmax(ze[clean], 1)
    Kmax = max(int(k) for k in args.Ks.split(","))
    sims, idx = [], []
    for i in range(0, len(ff), 512):
        s = (ff[i:i + 512] @ fec.T).float()
        v, ix = s.topk(Kmax, dim=1)
        sims.append(v.cpu()); idx.append(ix.cpu())
    sims, idx = torch.cat(sims), torch.cat(idx)
    print(f"sim moyenne au 1er voisin : {sims[:, 0].mean():.3f}, au {Kmax}e : {sims[:, -1].mean():.3f}")
    for K in (int(k) for k in args.Ks.split(",")):
        for tau in (float(t) for t in args.taus.split(",")):
            w = F.softmax(sims[:, :K] / tau, 1)                     # [Nf, K]
            q2 = (w[..., None] * p2[idx[:, :K]]).sum(1)
            q1 = (w[..., None] * p1[idx[:, :K]]).sum(1)
            ce = -q1.clamp_min(1e-8).log().gather(1, y[:, None]).squeeze(1)
            acc = (q1.argmax(1) == y).float().mean().item()
            print(f"  K={K:<3} tau={tau:<5} acc={acc:.3f}  CE(T=1) mean={ce.mean():.3f} std={ce.std():.3f}")
            torch.save({"logits": 2 * q2.clamp_min(1e-8).log(), "acc_forget": acc,
                        "ce_mean": ce.mean().item(), "ce_std": ce.std().item(), "K": K, "tau": tau, "layer": args.layer},
                       f"gold_knn_K{K}_t{tau}.pt")
    print("Fichiers gold_knn_*.pt écrits.")


if __name__ == "__main__":
    main()
