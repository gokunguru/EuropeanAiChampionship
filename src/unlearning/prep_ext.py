"""Prépare un jeu de rétention externe (autorisé par les orgas) : images ImageNet (timm/mini-imagenet, split validation).
À lancer sur le NŒUD DE LOGIN (les nœuds de calcul n'ont pas Internet). ~2-5 min.
  uv pip install pyarrow
  python prep_ext.py 30000
-> ext_retain.npy : uint8 [N, 224, 224, 3] (resize côté court 256 + center crop 224, comme un val ImageNet)
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
# validation d'abord, puis train (pas le test) : assez d'images pour distiller des surrogates
files = [f for f in allf if "validation" in f] + [f for f in allf if "train" in f]
print(len(files), "fichiers parquet, ex :", files[:2])


def to224(b):
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((l, t, l + 224, t + 224)), dtype=np.uint8)


out = np.lib.format.open_memmap("ext_retain.npy", mode="w+", dtype=np.uint8, shape=(N, 224, 224, 3))
n = 0
for f in files:
    path = hf_hub_download(REPO, f, repo_type="dataset")
    col = pq.read_table(path, columns=["image"]).column("image").to_pylist()
    for cell in col:
        b = cell["bytes"] if isinstance(cell, dict) else cell
        try:
            out[n] = to224(b)
        except Exception:
            continue
        n += 1
        if n >= N:
            break
    print(f"{f} -> {n} images", flush=True)
    if n >= N:
        break
out.flush()
assert n == N, f"seulement {n} images trouvées, relancer avec N={n}"
print("ext_retain.npy", out.shape, f"{out.nbytes / 1e9:.2f} Go")
