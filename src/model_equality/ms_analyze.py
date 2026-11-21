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


def normalize(prompt: str, response: str, qtype: str = "") -> str:
    """Réduit une réponse à un symbole comparable (ignore la politesse et le style du persona).
    qtype (type de question, v2) prime sur les mots-clés du prompt."""
    text = response.strip().lower()
    if qtype.startswith("num") or (not qtype and ("integer" in prompt or "number" in prompt)):
        m = re.search(r"-?\d+", text)
        return m.group(0) if m else "<none>"
    if qtype == "letter" or (not qtype and "letter" in prompt):
        letters = re.findall(r"\b([a-z])\b", text)
        return letters[-1] if letters else "<none>"
    words = re.findall(r"[a-zà-ÿ]+", text)
    return words[-1] if words else "<none>"


def symbols(qtype: str, response: str) -> list:
    """v3 : suite de symboles extraite d'une réponse de type "seq..." (ordre conservé)."""
    t = response.lower()
    if qtype == "seqnum":
        return re.findall(r"\d+", t)
    if qtype == "seqcoin":
        return [w[0] for w in re.findall(r"\b(?:heads|tails|h|t)\b", t)]
    if qtype == "seqletter":
        return re.findall(r"\b([a-z])\b", t)
    items = [i.strip() for i in re.split(r"[,\n;]", t) if i.strip()]
    return [w[-1] for w in (re.findall(r"[a-zà-ÿ]+", i) for i in items) if w]


def load_by_formulation(path: str) -> dict:
    """form[e][t][k] = Counter des réponses normalisées de e à la formulation k de la question de type t (v2)."""
    form = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "_" in r["id"]:
                    t, k = r["id"].split("_", 1)
                    if t.startswith("seq"):
                        # v3 : unigrammes et bigrammes, chacun comme une "distribution" séparée (t/u et t/b)
                        seq = symbols(t, r["response"])
                        form[r["endpoint"]][t + "/u"][k].update(seq)
                        if t != "seqanimal":
                            form[r["endpoint"]][t + "/b"][k].update(f"{a}>{b}" for a, b in zip(seq, seq[1:]))
                    else:
                        form[r["endpoint"]][t][k][normalize(r["prompt"], r["response"], t)] += 1
    return form


def cross_formulation(fe: dict, fe2: dict) -> float:
    """Estimateur D_x(e, e') = 1/(K(K-1)) * sum_{k != l} <delta_k, delta_l>, delta_k = p_hat_{e,k} - p_hat_{e',k}.

    Proposition : si e et e' ont la même loi de réponse pour chaque formulation k, E[D_x] = 0 (même avec 1 réponse par
    formulation). Preuve : pour k != l, delta_k et delta_l sont indépendants (requêtes distinctes) et d'espérance nulle,
    donc E<delta_k, delta_l> = <E delta_k, E delta_l> = 0.
    Calcul en O(K) : sum_{k != l} <d_k, d_l> = ||sum_k d_k||^2 - sum_k ||d_k||^2."""
    ks = sorted(k for k in set(fe) & set(fe2) if sum(fe[k].values()) and sum(fe2[k].values()))
    if len(ks) < 2:
        return 0.0
    total, diag = Counter(), 0.0
    for k in ks:
        n, m = sum(fe[k].values()), sum(fe2[k].values())
        dk = Counter()
        for a in set(fe[k]) | set(fe2[k]):
            dk[a] = fe[k][a] / n - fe2[k][a] / m
        diag += sum(v * v for v in dk.values())
        total.update(dk)
    return (sum(v * v for v in total.values()) - diag) / (len(ks) * (len(ks) - 1))


def load(path: str) -> dict:
    """dist[e][q] = Counter des réponses normalisées. En v2 (id "<type>_<k>"), q = <type> : les formulations d'une
    même question sont regroupées."""
    dist = defaultdict(lambda: defaultdict(Counter))
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                qtype = r["id"].split("_")[0] if "_" in r["id"] else ""
                dist[r["endpoint"]][qtype or r["id"]][normalize(r["prompt"], r["response"], qtype)] += 1
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
    form = load_by_formulation(answers)
    if form:   # v2 : estimateur croisé entre formulations (robuste au watermark), moyenné sur les types de question
        def dx(a, b):
            ts = set(form[a]) & set(form[b])
            return sum(cross_formulation(form[a][t], form[b][t]) for t in ts) / len(ts)
        d = {frozenset((a, b)): dx(a, b) for a, b in combinations(endpoints, 2)}
        print("Estimateur : croisé entre formulations (v2)")
    else:      # v1 : une seule formulation par question
        d = {frozenset((a, b)): distance(dist, a, b) for a, b in combinations(endpoints, 2)}
        print("Estimateur : L2 sans biais (v1)")

    print("Distances d(e, e') x 1000 :")
    print("       " + " ".join(f"{e[-5:]:>6}" for e in endpoints))
    for a in endpoints:
        print(f"{a[-6:]:>6} " + " ".join("   -  " if a == b else f"{1000 * d[frozenset((a, b))]:6.1f}" for b in endpoints))

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
