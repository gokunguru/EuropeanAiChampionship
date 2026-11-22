"""Test de permutation rapide (M tirages) pour toutes les paires intra-famille, puis correction de Holm.

Notations : d(e,f) et l'unité échangeable (réponse de paquet) sont ceux de ms_batch_analyze.py.
Accélération (exacte, pas une approximation du test) : dans un paquet b, avec R_e + R_f réponses, une permutation
stratifiée n'est qu'un choix de R_e lignes parmi R_e + R_f (252 choix pour 5 + 5). On précalcule, pour chaque choix s,
la somme S[b,s] des termes d_q et leur nombre C[b,s] ; un tirage de permutation est alors un choix s_b par paquet, et
d* = sum_b S[b,s_b] / sum_b C[b,s_b]. C'est la même loi que celle du test de ms_batch_analyze.perm_p.

Sortie : p-valeurs (M tirages), rejets de Holm au niveau alpha par famille, et la plus grande clique d'endpoints deux à
deux significativement différents = borne inférieure PROUVÉE (au niveau alpha, Holm) sur le nombre de modèles.

Usage :
    python ms_perm_exact.py ms_ep_batch.jsonl [M=20000] [alpha=0.05]
"""
import sys
from itertools import combinations

import numpy as np

from ms_batch_analyze import BAD, d_pair, load
from ms_intra import FAM


def terms(rows_e: list, rows_f: list, qs: list):
    """Somme des d_q et nombre de questions valides pour une répartition donnée (même règle que d_pair)."""
    s, c = 0.0, 0
    for q in qs:
        ae = [u[q] for u in rows_e if u.get(q) not in BAD]
        af = [u[q] for u in rows_f if u.get(q) not in BAD]
        if len(ae) < 2 or len(af) < 2:
            continue
        X = sum(x != y for x in ae for y in af) / (len(ae) * len(af))
        We = sum(ae[i] != ae[j] for i in range(len(ae)) for j in range(len(ae)) if i != j) / (len(ae) * (len(ae) - 1))
        Wf = sum(af[i] != af[j] for i in range(len(af)) for j in range(len(af)) if i != j) / (len(af) * (len(af) - 1))
        s += X - (We + Wf) / 2
        c += 1
    return s, c


def perm_p_fast(Ue: dict, Uf: dict, M: int, rng) -> tuple:
    """(d observé, p-valeur) ; p = (1 + #{d* >= d_obs}) / (1 + M)."""
    S, C = [], []
    obs_s, obs_c = 0.0, 0
    for b in sorted(set(Ue) & set(Uf)):
        pool = Ue[b] + Uf[b]
        qs = list(Ue[b][0])
        ne = len(Ue[b])
        sb, cb = [], []
        for idx in combinations(range(len(pool)), ne):
            rest = [pool[i] for i in range(len(pool)) if i not in idx]
            s, c = terms([pool[i] for i in idx], rest, qs)
            sb.append(s); cb.append(c)
        S.append(np.array(sb)); C.append(np.array(cb))
        obs_s += sb[0]; obs_c += cb[0]          # idx = (0..ne-1) : la répartition observée
    d_obs = obs_s / obs_c
    pick = [rng.integers(0, len(s), M) for s in S]
    num = sum(s[p] for s, p in zip(S, pick))
    den = sum(c[p] for c, p in zip(C, pick))
    d_star = num / np.maximum(den, 1)
    return d_obs, (1 + np.sum(d_star >= d_obs - 1e-12)) / (1 + M)


def holm(p: dict, alpha: float) -> set:
    """Ensemble des paires rejetées par la procédure de Holm (contrôle du FWER au niveau alpha)."""
    rej, m = set(), len(p)
    for k, (pair, pv) in enumerate(sorted(p.items(), key=lambda x: x[1])):
        if pv > alpha / (m - k):
            break
        rej.add(pair)
    return rej


def max_clique(G: list, edges: set) -> list:
    for k in range(len(G), 0, -1):
        for c in combinations(G, k):
            if all(frozenset(pq) in edges for pq in combinations(c, 2)):
                return list(c)
    return []


def main() -> None:
    U = load(sys.argv[1])
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    rng = np.random.default_rng(0)
    e0, f0 = FAM["A"][:2]
    assert abs(perm_p_fast(U[e0], U[f0], 10, rng)[0] - d_pair(U[e0], U[f0])) < 1e-9, "d observé != d_pair"
    print(f"M = {M} tirages, p minimal = {1 / (M + 1):.1e} ; Holm au niveau {alpha} par famille (55 paires)")
    for f, G in FAM.items():
        p = {}
        for a, b in combinations(G, 2):
            p[frozenset((a, b))] = perm_p_fast(U[a], U[b], M, rng)[1]
        rej = holm(p, alpha)
        print(f"\n===== Famille {f} : {len(rej)} paires séparées (Holm) sur 55")
        print("     " + " ".join(f"{e:>3}" for e in G))
        for a in G:
            print(f"{a}   " + " ".join("  ." if a == b else (" **" if frozenset((a, b)) in rej else
                                       f"{min(99, round(100 * p[frozenset((a, b))])):3d}") for b in G))
        c = max_clique(G, rej)
        print(f"  (** = séparé après Holm ; sinon p x100)\n  clique séparée maximale : {','.join(c)}"
              f"  =>  au moins {len(c)} modèles distincts (prouvé au niveau {alpha})")


if __name__ == "__main__":
    main()
