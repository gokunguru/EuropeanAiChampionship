"""Découpage intra-famille : matrice d, p-valeurs de permutation, groupes par liaison complète.

Notations : d(e,f) et le test de permutation stratifié par paquet sont définis dans ms_batch_analyze.py.
Règle de regroupement (heuristique, non prouvée) : liaison COMPLÈTE sur d, coupée au seuil t. Deux endpoints ne sont
dans le même groupe que si TOUTES les paires du groupe ont d <= t. On évite ainsi l'effet de chaîne qui a produit les
gros groupes de la v1 (MCC modèle ≈ 0 observé sur le score public).

Usage :
    python ms_intra.py ms_ep_batch.jsonl [t1,t2,...]      # matrices + groupes pour plusieurs seuils (x100)
"""
import sys
from itertools import combinations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ms_batch_analyze import d_pair, load, perm_p

FAM = {"A": "01 02 04 12 14 17 18 23 24 25 33".split(),
       "B": "03 06 07 08 20 21 22 26 28 30 32".split(),
       "C": "05 09 10 11 13 15 16 19 27 29 31".split()}


def matrix(U: dict, G: list) -> np.ndarray:
    d = np.zeros((len(G), len(G)))
    for i, j in combinations(range(len(G)), 2):
        d[i, j] = d[j, i] = max(d_pair(U[G[i]], U[G[j]]), 0.0)
    return d


def groups(d: np.ndarray, G: list, t: float) -> list:
    lab = fcluster(linkage(squareform(d, checks=False), "complete"), t, criterion="distance")
    out = {}
    for e, l in zip(G, lab):
        out.setdefault(l, []).append(e)
    return sorted(out.values(), key=lambda g: (-len(g), g))


def main() -> None:
    U = load(sys.argv[1])
    ts = [float(x) / 100 for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3,5").split(",")]
    reps = {e: min(len(v) for v in U[e].values()) for e in U}
    print("répétitions complètes par endpoint :", dict(sorted(__import__("collections").Counter(reps.values()).items())))
    for f, G in FAM.items():
        d = matrix(U, G)
        print(f"\n===== Famille {f} : d x100")
        print("     " + " ".join(f"{e:>4}" for e in G))
        for i, a in enumerate(G):
            print(f"{a}   " + " ".join("   ." if i == j else f"{100 * d[i, j]:4.1f}" for j in range(len(G))))
        for t in ts:
            print(f"  seuil t = {100 * t:.0f} : " + " | ".join(",".join(g) for g in groups(d, G, t)))
        iu = np.triu_indices(len(G), 1)
        order = np.argsort(d[iu])
        print("  paires les plus proches (d x100, p permutation) :")
        for k in order[:5]:
            i, j = iu[0][k], iu[1][k]
            print(f"    {G[i]}-{G[j]} : {100 * d[i, j]:4.1f}  p = {perm_p(U[G[i]], U[G[j]], n=200):.2f}")
        print("  paires les plus éloignées :")
        for k in order[-3:]:
            i, j = iu[0][k], iu[1][k]
            print(f"    {G[i]}-{G[j]} : {100 * d[i, j]:4.1f}  p = {perm_p(U[G[i]], U[G[j]], n=200):.2f}")


if __name__ == "__main__":
    main()
