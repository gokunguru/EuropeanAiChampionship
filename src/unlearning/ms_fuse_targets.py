"""Cible fusionnée pour l'oubli : substituts « sans fuite » + lissage local dans l'espace des features.

Notations
    x_i, i = 1..n         images d'oubli (n = 3500)
    z_i^(s)               logits du substitut s sur x_i (fichiers gold_v3/*.pt, clé "logits")
    q_sur(x_i)            (1/S) Σ_s softmax(z_i^(s) / 2)           moyenne des substituts à T = 2 (comme p_gold du serveur)
    h_i                   feature CLS du modèle original à la sortie du bloc k (avant la zone réinitialisée des
                          substituts : partagée par original et substituts, donc peu marquée par la mémorisation de la tête)
    N_K(i)                K plus proches voisins de i parmi les images d'oubli (cosinus sur h), i exclu
    w_ij                  softmax_j( cos(h_i, h_j) / tau_knn ),  j ∈ N_K(i)
    q_loc(x_i)            Σ_j w_ij q_sur(x_j)                       « professeur local » (idée CISPA / REGUN)
    q(x_i)                (1 - γ) q_sur(x_i) + γ q_loc(x_i)         cible fusionnée (mélange convexe : reste une loi)
Sortie : un fichier au format des substituts de Kamil, logits = 2 log q, donc softmax(logits / 2) = q exactement ;
unlearn.py --gold <fichier> l'utilise tel quel (q2 = q ; sa cible T = 1 vaut softmax(2 log q) ∝ q²).
Heuristique, non prouvé : le lissage réduit le bruit propre à chaque substitut et respecte la structure locale.

Usage (nœud GPU) :
    python ms_fuse_targets.py --golds "gold_v3/*.pt" --model model.pt --forget forget_images.npz \
        --block 6 --K 10 --gammas 0,0.3,0.6 --outdir gold_ms_fuse
"""
import argparse
import glob
import os

import numpy as np
import timm
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def q_surrogates(files: list) -> torch.Tensor:
    """q_sur = moyenne des softmax(z/2) des substituts."""
    return torch.stack([F.softmax(torch.load(f, map_location="cpu")["logits"].float() / 2, 1) for f in files]).mean(0)


@torch.no_grad()
def cls_features(model, X: np.ndarray, block: int, bs: int = 256) -> torch.Tensor:
    """h_i : token CLS à la sortie du bloc `block` (1-indexé), normalisé L2."""
    store = {}
    hook = model.blocks[block - 1].register_forward_hook(lambda m, i, o: store.__setitem__("h", o[:, 0]))
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(DEV).permute(0, 3, 1, 2).float().div(255).sub(0.5).div(0.5)
        model(x)
        out.append(F.normalize(store["h"].float(), dim=1).cpu())
    hook.remove()
    return torch.cat(out)


def local_teacher(h: torch.Tensor, q: torch.Tensor, K: int, tau: float) -> torch.Tensor:
    """q_loc(x_i) = Σ_{j ∈ N_K(i)} w_ij q(x_j), i exclu, poids softmax(cos / tau)."""
    S = h @ h.T
    S.fill_diagonal_(-2.0)
    val, idx = S.topk(K, dim=1)
    w = F.softmax(val / tau, dim=1)
    return (w.unsqueeze(-1) * q[idx]).sum(1)


def local_teacher_cc(h: torch.Tensor, p: torch.Tensor, y: torch.Tensor, K: int, tau: float) -> torch.Tensor:
    """Professeur local CONDITIONNÉ À LA CLASSE : voisins parmi les images de même pseudo-étiquette y (i exclu).
    p_loc(x_i) = Σ_{j ∈ N_K(i), y_j = y_i} w_ij p(x_j) ; sans voisin de même classe, p_loc(x_i) = p(x_i)."""
    S = h @ h.T
    S.fill_diagonal_(-2.0)
    S = S.masked_fill(y[:, None] != y[None, :], -2.0)
    val, idx = S.topk(K, dim=1)
    valid = val > -1.5
    w = F.softmax((val / tau).masked_fill(~valid, -1e9), dim=1) * valid
    out = (w.unsqueeze(-1) * p[idx]).sum(1)
    none = ~valid.any(1)
    out[none] = p[none]
    return out / out.sum(1, keepdim=True)


def fuse(q_sur: torch.Tensor, q_loc: torch.Tensor, gamma: float) -> torch.Tensor:
    return (1 - gamma) * q_sur + gamma * q_loc


def ce_stats(q: torch.Tensor, y: torch.Tensor) -> tuple:
    """Perte CE à T = 1 d'un modèle dont softmax(logits / 2) = q, i.e. logits = 2 log q (étiquette y)."""
    z = 2 * q.clamp_min(1e-12).log()
    ce = F.cross_entropy(z, y, reduction="none")
    return ce.mean().item(), ce.std().item(), (z.argmax(1) == y).float().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golds", default="gold_v3/*.pt"); ap.add_argument("--model", default="model.pt")
    ap.add_argument("--forget", default="forget_images.npz"); ap.add_argument("--block", type=int, default=6)
    ap.add_argument("--K", type=int, default=10); ap.add_argument("--tau_knn", type=float, default=0.05)
    ap.add_argument("--gammas", default="0,0.3,0.6"); ap.add_argument("--outdir", default="gold_ms_fuse")
    ap.add_argument("--arch", default="vit_base_patch16_224")
    ap.add_argument("--per_file", action="store_true",
                    help="un fichier lissé par substitut, à T=1 et voisins de même classe (γ=0 redonne les fichiers d'origine)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    files = sorted(glob.glob(a.golds))
    print(f"{len(files)} substituts : {files}")
    qs = q_surrogates(files)
    m = timm.create_model(a.arch, pretrained=False, num_classes=100, init_values=1e-4)
    m.load_state_dict(torch.load(a.model, map_location="cpu"), strict=True)
    m = m.to(DEV).eval()
    X = np.load(a.forget)["images"]
    h = cls_features(m, X, a.block)
    with torch.no_grad():
        y = torch.cat([m(torch.from_numpy(np.ascontiguousarray(X[i:i + 256])).to(DEV).permute(0, 3, 1, 2).float()
                         .div(255).sub(0.5).div(0.5)).argmax(1).cpu() for i in range(0, len(X), 256)])
    if a.per_file:
        for g in [float(x) for x in a.gammas.split(",")]:
            d = f"{a.outdir}/g{g}"
            os.makedirs(d, exist_ok=True)
            for f in files:
                p = F.softmax(torch.load(f, map_location="cpu")["logits"].float(), 1)
                pf = (1 - g) * p + g * local_teacher_cc(h, p, y, a.K, a.tau_knn)
                keep = (pf.argmax(1) == p.argmax(1)).float().mean().item()
                ce = F.cross_entropy(pf.clamp_min(1e-12).log(), y, reduction="none")
                torch.save({"logits": pf.clamp_min(1e-12).log(), "logits_noflip": pf.clamp_min(1e-12).log(), "gamma": g,
                            "K": a.K, "block": a.block, "source": f}, f"{d}/{os.path.basename(f)}")
                print(f"  γ = {g:.2f}  {os.path.basename(f)} : CE(T=1) = {ce.mean():.3f} ± {ce.std():.3f}, argmax conservé {keep:.3f}")
        return
    ql = local_teacher(h, qs, a.K, a.tau_knn)
    agree = (ql.argmax(1) == qs.argmax(1)).float().mean().item()
    print(f"feature bloc {a.block}, K = {a.K} : le professeur local garde l'argmax des substituts sur {agree:.3f} des images")
    for g in [float(x) for x in a.gammas.split(",")]:
        q = fuse(qs, ql, g)
        mu, sd, acc = ce_stats(q, y)
        out = f"{a.outdir}/gold_fuse_g{g}.pt"
        torch.save({"logits": 2 * q.clamp_min(1e-12).log(), "logits_noflip": 2 * q.clamp_min(1e-12).log(),
                    "gamma": g, "K": a.K, "block": a.block, "sources": files}, out)
        print(f"  γ = {g:.2f} : CE = {mu:.3f} ± {sd:.3f}, précision {acc:.3f} (étiquette = argmax original)  -> {out}")


if __name__ == "__main__":
    main()
