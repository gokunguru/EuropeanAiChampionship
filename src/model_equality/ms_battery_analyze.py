"""Analyse de la phase 1 (cours 1, chapitre « Profil d'erreurs et test de permutation »).

Notations
    a_{e,q,r}       réponse extraite de l'endpoint e à la question q, répétition r in {0, 1}
    G               groupe de questions (mul, ari, kno, wrd = « poids » ; rnd = « hasard »)
    X_G(e,e')       désaccord croisé moyen : moyenne sur q in G et r, s de 1[a_{e,q,r} != a_{e',q,s}]
    W_G(e)          désaccord interne : moyenne sur q in G de 1[a_{e,q,0} != a_{e,q,1}]
    d_G(e,e')       X_G(e,e') - (W_G(e) + W_G(e')) / 2   (MMD^2 sans biais à noyau one-hot, ~0 si même modèle)
    p_G(e,e')       p-valeur du test de permutation stratifié par question (les 4 réponses d'une question sont
                    échangeables sous H0 « même modèle »)
    K(e,e')         nombre de groupes « poids » avec p_G < alpha
    D(q)            pouvoir discriminant : désaccord entre endpoints / désaccord interne (+ epsilon)

Usage :
    python ms_battery_analyze.py ms_ep_battery.jsonl [ms_ep_peek.jsonl]
"""
import json
import random
import re
import sys
from collections import defaultdict
from itertools import combinations

from ms_analyze import cluster
from ms_battery import QUESTIONS

KIND = {qid: kind for qid, kind, _ in QUESTIONS}
GROUP = {qid: qid[:3] for qid, _, _ in QUESTIONS}
WEIGHT_GROUPS = ["mul", "ari", "kno", "wrd"]
STOP = {"the", "is", "a", "an", "answer", "of", "it", "s", "word", "letter", "number", "reversed", "backwards"}


def extract(kind: str, response: str) -> str:
    """Réponse extraite : dernier entier (virgules de milliers retirées) ou dernier mot significatif."""
    t = response.lower()
    if kind == "num":
        t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
        nums = re.findall(r"-?\d+", t)
        return nums[-1] if nums else "<none>"
    words = [w for w in re.findall(r"[a-zà-ÿ]+", t) if w not in STOP]
    return words[-1] if words else "<none>"


def load(path: str) -> dict:
    """A[e][q] = [réponse extraite de la répétition 0, de la répétition 1] (seules les questions complètes)."""
    raw = defaultdict(lambda: defaultdict(dict))
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r["id"] in KIND:
                raw[r["endpoint"]][r["id"]][r["repeat"]] = extract(KIND[r["id"]], r["response"])
    return {e: {q: [v[0], v[1]] for q, v in qs.items() if 0 in v and 1 in v} for e, qs in raw.items()}


def cross_within(ans_e: list, ans_f: list) -> tuple:
    """(désaccord croisé moyen, désaccord interne de e, désaccord interne de f) pour une question."""
    x = sum(a != b for a in ans_e for b in ans_f) / (len(ans_e) * len(ans_f))
    return x, float(ans_e[0] != ans_e[1]), float(ans_f[0] != ans_f[1])


def stat(A: dict, e: str, f: str, qs: list) -> float:
    """d_G(e,f) sur la liste de questions qs."""
    vals = [cross_within(A[e][q], A[f][q]) for q in qs]
    return sum(x - (we + wf) / 2 for x, we, wf in vals) / len(vals)


def perm_pvalue(A: dict, e: str, f: str, qs: list, n_perm: int = 2000, rng=random.Random(0)) -> float:
    """Test de permutation stratifié : pour chaque question, on redistribue au hasard les 4 réponses en 2 + 2."""
    obs = stat(A, e, f, qs)
    pooled = {q: A[e][q] + A[f][q] for q in qs}
    hits = 0
    for _ in range(n_perm):
        Ae, Af = {}, {}
        for q in qs:
            v = pooled[q][:]
            rng.shuffle(v)
            Ae[q], Af[q] = v[:2], v[2:]
        if stat({e: Ae, f: Af}, e, f, qs) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def discriminative_power(A: dict) -> dict:
    """D(q) = désaccord moyen entre endpoints distincts / (désaccord interne moyen + 0,05)."""
    eps = sorted(A)
    out = {}
    for q in sorted({q for e in eps for q in A[e]}):
        es = [e for e in eps if q in A[e]]
        between = [sum(a != b for a in A[e][q] for b in A[f][q]) / 4 for e, f in combinations(es, 2)]
        within = [float(A[e][q][0] != A[e][q][1]) for e in es]
        out[q] = (sum(between) / len(between)) / (sum(within) / len(within) + 0.05)
    return out


def family_hint(peek_path: str) -> dict:
    """Famille annoncée par le modèle lui-même dans le coup d'œil (qwen / llama / claude / ?)."""
    hint = {}
    for line in open(peek_path, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] == "who":
            t = r["response"].lower()
            hint[r["endpoint"]] = ("qwen" if "qwen" in t or "alibaba" in t else
                                   "llama" if "llama" in t or "meta" in t else
                                   "claude" if "anthropic" in t or "claude" in t else "?")
    return hint


def main() -> None:
    A = load(sys.argv[1])
    hint = family_hint(sys.argv[2]) if len(sys.argv) > 2 else {}
    eps = sorted(A, key=lambda e: (hint.get(e, "?"), e))
    wq = [q for q, _, _ in QUESTIONS if GROUP[q] in WEIGHT_GROUPS]
    complete = [e for e in eps if all(q in A[e] for q in wq)]
    print(f"{len(complete)}/{len(eps)} endpoints avec les {len(wq)} questions « poids » complètes")

    # pouvoir discriminant des questions
    D = discriminative_power({e: A[e] for e in complete})
    print("\nD(q), du plus discriminant au moins discriminant :")
    print("  " + "  ".join(f"{q}={v:.1f}" for q, v in sorted(D.items(), key=lambda kv: -kv[1])))

    # distances et tests
    d = {frozenset((e, f)): stat(A, e, f, wq) for e, f in combinations(complete, 2)}
    print("\nd(e,e') x 100 sur les questions « poids » (≈ 0 : même modèle) :")
    print("        " + " ".join(f"{e[-2:]:>4}" for e in complete))
    for e in complete:
        print(f"{e} {hint.get(e, '?')[0]} " + " ".join("   ." if e == f else f"{100 * d[frozenset((e, f))]:4.0f}"
                                                      for f in complete))

    # regroupement : seuil = plus petite valeur où les paires fusionnées ne sont pas significativement différentes
    for t in (0.05, 0.10, 0.15, 0.20):
        groups = cluster(complete, d, t)
        sizes = defaultdict(list)
        for e, g in groups.items():
            sizes[g].append(e[-2:])
        print(f"\nseuil {t:.2f} : " + " | ".join(",".join(v) for v in sorted(sizes.values(), key=len, reverse=True)))


if __name__ == "__main__":
    main()
