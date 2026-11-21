"""Analyse des réponses : distance entre endpoints, regroupement, score MCC sur les endpoints de référence.

Notations
    E                 ensemble des endpoints
    Q                 ensemble des questions (probes)
    P_{e,q}           distribution empirique des réponses normalisées de e à la question q
    TV(P, P')         distance en variation totale = 1/2 * sum_a |P(a) - P'(a)|
    D(P, P')          = ||P - P'||_2^2, estimé sans biais (MMD^2 à noyau one-hot, Gao et al. 2024)
    d(e, e')          = moyenne sur q in Q de D(P_{e,q}, P_{e',q})           (distance entre endpoints)

Méthode (heuristique, à valider sur les références) : deux endpoints servent le même modèle si d est petite.
Regroupement agglomératif à liaison moyenne, coupé au seuil t ; t est choisi pour maximiser le MCC sur les références.

Usage :
    python ms_analyze.py ms_ref_answers.jsonl practice_answers.csv
"""
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations


def normalize(prompt: str, response: str) -> str:
    """Réduit une réponse à un symbole comparable (ignore la politesse et le style du persona)."""
    text = response.strip().lower()
    if "integer" in prompt or "number" in prompt:
        m = re.search(r"-?\d+", text)
        return m.group(0) if m else "<none>"
    if "letter" in prompt:
        letters = re.findall(r"\b([a-z])\b", text)
        return letters[-1] if letters else "<none>"
    words = re.findall(r"[a-zà-ÿ]+", text)
    return words[-1] if words else "<none>"


def load(path: str) -> dict:
    """dist[e][q] = Counter des réponses normalisées."""
    dist = defaultdict(lambda: defaultdict(Counter))
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                dist[r["endpoint"]][r["id"]][normalize(r["prompt"], r["response"])] += 1
    return dist


def tv(p: Counter, q: Counter) -> float:
    """Distance en variation totale entre deux distributions empiriques (dans [0, 1])."""
    np_, nq = sum(p.values()), sum(q.values())
    return 0.5 * sum(abs(p[a] / np_ - q[a] / nq) for a in set(p) | set(q))


def l2_unbiased(p: Counter, q: Counter) -> float:
    """Estimateur sans biais de ||P - Q||_2^2 = sum_a (P(a) - Q(a))^2 (MMD^2 à noyau one-hot).

    sum_a P(a)^2 est estimé par sum_a c_a (c_a - 1) / (n (n - 1)) (U-statistique), le terme croisé par
    sum_a c_a c'_a / (n n'). Espérance exacte = ||P - Q||^2, donc ≈ 0 si P = Q quel que soit n (peut être < 0)."""
    n, m = sum(p.values()), sum(q.values())
    if n < 2 or m < 2:
        return 0.0
    pp = sum(c * (c - 1) for c in p.values()) / (n * (n - 1))
    qq = sum(c * (c - 1) for c in q.values()) / (m * (m - 1))
    pq = sum(p[a] * q[a] for a in p) / (n * m)
    return pp + qq - 2 * pq


def distance(dist: dict, e1: str, e2: str) -> float:
    """d(e1, e2) : moyenne sur les questions communes de l'estimateur sans biais de ||P - Q||_2^2."""
    qs = set(dist[e1]) & set(dist[e2])
    return sum(l2_unbiased(dist[e1][q], dist[e2][q]) for q in qs) / len(qs)


def cluster(endpoints: list, d: dict, t: float) -> dict:
    """Agglomératif, liaison moyenne, fusion tant que la distance moyenne entre groupes < t. Renvoie e -> id de groupe."""
    groups = [[e] for e in endpoints]
    while len(groups) > 1:
        best, pair = None, None
        for i, j in combinations(range(len(groups)), 2):
            dd = sum(d[frozenset((a, b))] for a in groups[i] for b in groups[j]) / (len(groups[i]) * len(groups[j]))
            if best is None or dd < best:
                best, pair = dd, (i, j)
        if best >= t:
            break
        i, j = pair
        groups[i] += groups.pop(j)
    return {e: k for k, g in enumerate(groups) for e in g}


def mcc(pred: dict, truth: dict) -> float:
    """MCC sur les paires : positif = même groupe. Valeur < 0 ramenée à 0, comme pour le score officiel."""
    tp = tn = fp = fn = 0
    for a, b in combinations(sorted(truth), 2):
        p, t = pred[a] == pred[b], truth[a] == truth[b]
        tp += p and t
        tn += (not p) and (not t)
        fp += p and not t
        fn += (not p) and t
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / den) if den else 0.0


def main() -> None:
    answers, truth_path = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None
    dist = load(answers)
    endpoints = sorted(dist)
    d = {frozenset((a, b)): distance(dist, a, b) for a, b in combinations(endpoints, 2)}

    print("Distances d(e, e') :")
    print("       " + " ".join(f"{e[-5:]:>6}" for e in endpoints))
    for a in endpoints:
        print(f"{a[-6:]:>6} " + " ".join("   -  " if a == b else f"{d[frozenset((a, b))]:6.3f}" for b in endpoints))

    if truth_path:
        with open(truth_path, newline="") as f:
            truth = {r["endpoint"]: r["model"] for r in csv.DictReader(f)}
        cuts = sorted(set(d.values()))
        thresholds = [(a + b) / 2 for a, b in zip(cuts, cuts[1:])] + [cuts[-1] + 1e-9]
        best = max((mcc(cluster(endpoints, d, t), truth), -t) for t in thresholds)
        best = (best[0], -best[1])
        print(f"\nMeilleur seuil t = {best[1]:.2f} -> MCC (même modèle) = {best[0]:.3f}")
        groups = cluster(endpoints, d, best[1])
        for e in endpoints:
            print(f"  {e}: groupe {groups[e]}  (vérité : {truth.get(e)})")


if __name__ == "__main__":
    main()
