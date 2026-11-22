"""Récupère les VRAIS labels mini-ImageNet des 30k images de ext_retain.npy (même ordre, même filtrage que prep_ext.py).
NŒUD DE LOGIN (Internet). Relit les parquets (cache HF si encore présent, sinon re-téléchargement ~4 Go, quelques min).
Vérifie l'alignement en re-décodant quelques images et en les comparant à ext_retain.npy.
  python fetch_labels.py 30000   -> ext_labels.npy (int64 [N])
"""
import io
import sys

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
REPO = "timm/mini-imagenet"
allf = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
files = [f for f in allf if "validation" in f] + [f for f in allf if "train" in f]


def to224(b):
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((l, t, l + 224, t + 224)), dtype=np.uint8)


ext = np.load("ext_retain.npy", mmap_mode="r")
rng = np.random.default_rng(0)
check = set(rng.choice(N, 20, replace=False).tolist())
labels = np.full(N, -1, dtype=np.int64)
n, bad = 0, 0
for f in files:
    path = hf_hub_download(REPO, f, repo_type="dataset")
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    lab_col = "label" if "label" in names else [c for c in names if c != "image"][0]
    if n == 0:
        print("colonnes parquet :", names, "-> label =", lab_col)
    t = pq.read_table(path, columns=["image", lab_col])
    imgs, labs = t.column("image").to_pylist(), t.column(lab_col).to_pylist()
    for cell, lab in zip(imgs, labs):
        b = cell["bytes"] if isinstance(cell, dict) else cell
        try:
            # même filtrage que prep_ext : on ne garde que les images décodables
            if n in check:
                if not np.array_equal(to224(b), ext[n]):
                    bad += 1
            else:
                Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:
            continue
        labels[n] = int(lab)
        n += 1
        if n >= N:
            break
    print(f"{f} -> {n}", flush=True)
    if n >= N:
        break
assert n == N, n
print(f"alignement : {len(check) - bad}/{len(check)} images identiques à ext_retain.npy")
assert bad == 0, "désalignement : ne pas utiliser ces labels"
np.save("ext_labels.npy", labels)
print("ext_labels.npy écrit,", len(np.unique(labels)), "synsets distincts")
