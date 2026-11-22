"""Associe les 100 classes du modèle à des classes ImageNet, à partir du val d'ImageNet (images jamais vues).

Notations
    f_orig            modèle original (ViT-B/16, 100 classes), prétraitement serveur x = (pixels/255 - 0.5)/0.5
    V                 val d'ImageNet-1k : 50 000 images, 50 par classe c ∈ {0..999}
    N[c, k]           nombre d'images de la classe ImageNet c que f_orig classe en k
    π(k)              classe ImageNet associée à k : argmax_c N[c, k] ; pureté(k) = N[π(k), k] / 50
Hypothèse (à vérifier par la sortie) : le jeu d'entraînement est un sous-ensemble à 100 classes d'ImageNet, tiré du
train ; alors V est jamais vu et, pour les classes associées, f_orig y imite un « gold » (modèle qui n'a pas vu l'image).

Sorties (dossier --out) :
    val_logits.npy [50000,100] float32, val_labels.npy, mapping.json (k -> c, pureté),
    sel_images.npy uint8 [M,224,224,3] et sel_targets.npy (classe modèle k) pour les classes associées,
    rapport : pureté, injectivité, statistiques de perte (estimation de gold_mean / gold_std),
              recouvrement oubli / val (si des images d'oubli SONT dans le val, le val n'est pas « jamais vu »).
Usage (nœud GPU) :
    python ms_val_map.py --val ../imagenet_val/data --model model.pt --forget forget_images.npz --out val_map
"""
import argparse
import glob
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import timm
import torch
import torch.nn.functional as F
from PIL import Image


def to224(b: bytes) -> np.ndarray:
    """Côté court à 256 (bicubique) puis recadrage central 224 : prétraitement val ImageNet standard."""
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((l, t, l + 224, t + 224)), dtype=np.uint8)


def load_val(folder: str, workers: int):
    imgs, labels = [], []
    with ProcessPoolExecutor(workers) as ex:
        for f in sorted(glob.glob(os.path.join(folder, "*.parquet"))):
            t = pq.read_table(f, columns=["image", "label"]).to_pydict()
            imgs += list(ex.map(to224, [d["bytes"] for d in t["image"]], chunksize=64))
            labels += t["label"]
            print(f"  {os.path.basename(f)} : {len(labels)} images", flush=True)
    return np.stack(imgs), np.array(labels)


def model_from(path: str):
    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    return m.cuda().eval()


@torch.no_grad()
def logits_of(m, X: np.ndarray, bs: int = 500) -> np.ndarray:
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).cuda().permute(0, 3, 1, 2).float().div(255).sub(0.5).div(0.5)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(m(x).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def small(X: np.ndarray, bs: int = 1000) -> torch.Tensor:
    """Empreinte 32x32 niveaux de gris centrée-normée (robuste au recadrage léger)."""
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).cuda().float().mean(-1, keepdim=True).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, 32).flatten(1)
        out.append(F.normalize(x - x.mean(1, keepdim=True), dim=1))
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True); ap.add_argument("--model", default="model.pt")
    ap.add_argument("--forget", default="forget_images.npz"); ap.add_argument("--out", default="val_map")
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--min_purity", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("1) décodage du val", flush=True)
    V, y = load_val(a.val, a.workers)
    m = model_from(a.model)
    print("2) logits du modèle original sur le val", flush=True)
    L = logits_of(m, V)
    np.save(f"{a.out}/val_logits.npy", L); np.save(f"{a.out}/val_labels.npy", y)
    pred = L.argmax(1)

    N = np.zeros((1000, 100), dtype=int)
    np.add.at(N, (y, pred), 1)
    pi = N.argmax(0); pur = N.max(0) / 50.0
    ok = [k for k in range(100) if pur[k] >= a.min_purity]
    inj = len({pi[k] for k in ok}) == len(ok)
    json.dump({int(k): {"imagenet": int(pi[k]), "purete": float(pur[k])} for k in range(100)},
              open(f"{a.out}/mapping.json", "w"), indent=1)
    print(f"\n3) association : {len(ok)} classes modèle sur 100 ont une pureté >= {a.min_purity} ; injective : {inj}")
    print("   quantiles de pureté (0, 25, 50, 75, 100 %) :", np.round(np.quantile(pur, [0, .25, .5, .75, 1]), 2))

    inv = {int(pi[k]): k for k in ok}
    sel = np.array([i for i in range(len(y)) if int(y[i]) in inv])
    tgt = np.array([inv[int(y[i])] for i in sel])
    np.save(f"{a.out}/sel_images.npy", V[sel]); np.save(f"{a.out}/sel_targets.npy", tgt)
    Ls = torch.from_numpy(L[sel])
    ce = F.cross_entropy(Ls, torch.from_numpy(tgt), reduction="none").numpy()
    acc = float((L[sel].argmax(1) == tgt).mean())
    print(f"\n4) images sélectionnées : {len(sel)} ; précision du modèle original dessus : {acc:.3f}")
    print(f"   perte CE (T=1) sur ces images JAMAIS VUES : moyenne {ce.mean():.3f}, écart-type {ce.std():.3f}"
          "   <- estimation de (gold_mean, gold_std) si l'hypothèse tient")

    Fo = np.load(a.forget)["images"]
    Lf = logits_of(m, Fo)
    pf = torch.from_numpy(Lf)
    cef = F.cross_entropy(pf, pf.argmax(1), reduction="none").numpy()
    print(f"   perte CE du modèle original sur l'oubli (étiquette = sa propre prédiction) : moyenne {cef.mean():.3f},"
          f" écart-type {cef.std():.3f} ; confiance max moyenne {F.softmax(pf, 1).max(1).values.mean():.3f}")
    print(f"   classes prédites sur l'oubli hors classes associées : {np.mean([k not in ok for k in Lf.argmax(1)]):.3f}")

    print("\n5) recouvrement oubli / val (corrélation 32x32 du plus proche voisin)")
    sf, sv = small(Fo), small(V[sel])
    best = torch.cat([(sf[i:i + 500] @ sv.T).max(1).values for i in range(0, len(sf), 500)]).cpu()
    for t in (0.99, 0.97, 0.95):
        print(f"   corr >= {t} : {(best >= t).sum().item()} sur {len(sf)}")
    print("   médiane", round(best.median().item(), 3), " (si beaucoup >= 0,97 : l'oubli vient du val, qui n'est donc PAS jamais vu)")


if __name__ == "__main__":
    main()
