"""Statistiques invariantes à la température (cours 1, section « Une statistique invariante à la température »).

Notations
    z(a)             logit de la réponse a (même modèle, même question)
    p_T = softmax(z / T)   loi des réponses à la température T
    c_e(a)           nombre de fois où l'endpoint e a répondu a

Proposition 3.4 : si e et e' servent le même modèle aux températures T1, T2, alors
    log p_e(a) = (T2 / T1) log p_e'(a) + cste      pour tout a (relation affine).
=> statistique 1 : écart à la droite (résidu d'une régression pondérée des log-fréquences).
Proposition 3.5 : x -> x^(1/T) est croissante, donc l'ordre des réponses est conservé.
=> statistique 2 : 1 - corrélation de Spearman entre les fréquences.

Usage :
    python ms_temperature.py ms_ref_answers_v5.jsonl practice_answers.csv
"""
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations

from ms_analyze import cluster, mcc, normalize


def loglinear_residual(c1: Counter, c2: Counter, smooth: float = 0.5, min_count: int = 2) -> float:
    """Écart à la relation affine log p1 = alpha log p2 + beta (proposition 3.4), dans [0, 1].

    On garde les réponses vues au moins `min_count` fois au total, on lisse les comptages (+ smooth) pour éviter
    log 0, puis on fait une régression pondérée (poids = c1 + c2). Renvoie 1 - R^2 (0 = droite parfaite)."""
    support = [a for a in set(c1) | set(c2) if c1[a] + c2[a] >= min_count]
    if len(support) < 3:
        return 0.0
    n1 = sum(c1[a] for a in support) + smooth * len(support)
    n2 = sum(c2[a] for a in support) + smooth * len(support)
    y = [math.log((c1[a] + smooth) / n1) for a in support]
    x = [math.log((c2[a] + smooth) / n2) for a in support]
    w = [c1[a] + c2[a] for a in support]
    sw = sum(w)
    mx = sum(wi * xi for wi, xi in zip(w, x)) / sw
    my = sum(wi * yi for wi, yi in zip(w, y)) / sw
    sxx = sum(wi * (xi - mx) ** 2 for wi, xi in zip(w, x))
    syy = sum(wi * (yi - my) ** 2 for wi, yi in zip(w, y))
    sxy = sum(wi * (xi - mx) * (yi - my) for wi, xi, yi in zip(w, x, y))
    if sxx == 0 or syy == 0:
        return 0.0
    return 1.0 - sxy * sxy / (sxx * syy)


def _ranks(values: list) -> list:
    """Rangs moyens (gestion des ex aequo), base 1."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman_distance(c1: Counter, c2: Counter, min_count: int = 2) -> float:
    """1 - corrélation de Spearman entre les fréquences des deux endpoints (proposition 3.5), dans [0, 2]."""
    support = [a for a in set(c1) | set(c2) if c1[a] + c2[a] >= min_count]
    if len(support) < 3:
        return 0.0
    r1 = _ranks([c1[a] / sum(c1.values()) for a in support])
    r2 = _ranks([c2[a] / sum(c2.values()) for a in support])
    m1, m2 = sum(r1) / len(r1), sum(r2) / len(r2)
    cov = sum((a - m1) * (b - m2) for a, b in zip(r1, r2))
    v1 = math.sqrt(sum((a - m1) ** 2 for a in r1))
    v2 = math.sqrt(sum((b - m2) ** 2 for b in r2))
    return 1.0 - cov / (v1 * v2) if v1 and v2 else 0.0


def load(path: str) -> dict:
    """counts[e][q] = Counter des réponses normalisées (q = identifiant de la question)."""
    counts = defaultdict(lambda: defaultdict(Counter))
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            qtype = "num10" if r["id"].startswith("tnum") else ("letter" if r["id"].startswith("tlet") else "")
            counts[r["endpoint"]][r["id"]][normalize(r["prompt"], r["response"], qtype)] += 1
    return counts


def auc(d: dict, truth: dict) -> float:
    """Probabilité qu'une paire « même modèle » soit plus proche qu'une paire « modèles différents »."""
    pos = [v for p, v in d.items() if len({truth[e] for e in p}) == 1]
    neg = [v for p, v in d.items() if len({truth[e] for e in p}) == 2]
    return sum((a < b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def main() -> None:
    counts = load(sys.argv[1])
    eps = sorted(counts)
    truth = {r["endpoint"]: r["model"] for r in csv.DictReader(open(sys.argv[2]))} if len(sys.argv) > 2 else None
    for name, f in [("log-linéaire (1 - R^2)", loglinear_residual), ("Spearman (1 - rho)", spearman_distance)]:
        d = {}
        for a, b in combinations(eps, 2):
            qs = set(counts[a]) & set(counts[b])
            d[frozenset((a, b))] = sum(f(counts[a][q], counts[b][q]) for q in qs) / len(qs)
        print(f"\n== {name}, x 1000")
        print("       " + " ".join(f"{e[-5:]:>6}" for e in eps))
        for a in eps:
            print(f"{a[-6:]:>6} " + " ".join("   -  " if a == b else f"{1000 * d[frozenset((a, b))]:6.0f}" for b in eps))
        if truth:
            cuts = sorted(set(d.values()))
            ths = [(x + y) / 2 for x, y in zip(cuts, cuts[1:])] or [cuts[0] + 1e-9]
            best = max((mcc(cluster(eps, d, t), truth), t) for t in ths)
            print(f"AUC paires = {auc(d, truth):.3f}   meilleur MCC = {best[0]:.3f} (seuil {best[1]:.3f})")


if __name__ == "__main__":
    main()
