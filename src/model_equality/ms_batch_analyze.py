"""Analyse des paquets (ms_batch.py) : distance corrigée, test de permutation, groupes.

Notations (plan, section 2 ; R répétitions par paquet)
    a_{e,q,r}     réponse extraite de e à la question q, répétition r
    X_q(e,f)      moyenne sur r, s de 1[a_{e,q,r} != a_{f,q,s}]                 (désaccord croisé)
    W_q(e)        moyenne sur r != s de 1[a_{e,q,r} != a_{e,q,s}]                (désaccord interne)
    d_q(e,f)      X_q - (W_q(e) + W_q(f)) / 2 = ½ MMD² sans biais, noyau one-hot ; E[d_q] = ½ ||π_e - π_f||²
    d(e,f)        moyenne de d_q sur les questions renseignées des deux côtés
Test de permutation : l'unité échangeable est une réponse de paquet (endpoint, paquet, répétition). Sous H0, pour un
paquet donné, les 2R réponses de e et f sont échangeables ; on redistribue les étiquettes à l'intérieur de chaque paquet.

Usage :
    python ms_batch_analyze.py ms_ep_batch.jsonl
"""
import json
import random
import sys
from collections import defaultdict
from itertools import combinations

BAD = {None, "<absent>", "<none>"}
QWEN = "01 02 03 04 06 07 08 12 14 17 18 20 21 22 23 24 25 26 28 30 32 33".split()
LLAMA = "05 09 10 11 13 15 16 19 27 29 31".split()


def load(path: str) -> dict:
    """U[e][b] = liste des réponses (dict question -> réponse) aux répétitions du paquet b."""
    U = defaultdict(lambda: defaultdict(list))
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["status"] == "ok":
            U[r["endpoint"][-2:]][r["batch"]].append(r["answers"])
    return U


def d_pair(Ue: dict, Uf: dict) -> float:
    """d(e,f) moyen sur les questions où chaque endpoint a au moins 2 réponses valides."""
    vals = []
    for b in set(Ue) & set(Uf):
        for q in Ue[b][0]:
            ae = [u[q] for u in Ue[b] if u.get(q) not in BAD]
            af = [u[q] for u in Uf[b] if u.get(q) not in BAD]
            if len(ae) < 2 or len(af) < 2:
                continue
            X = sum(x != y for x in ae for y in af) / (len(ae) * len(af))
            We = sum(ae[i] != ae[j] for i in range(len(ae)) for j in range(len(ae)) if i != j) / (len(ae) * (len(ae) - 1))
            Wf = sum(af[i] != af[j] for i in range(len(af)) for j in range(len(af)) if i != j) / (len(af) * (len(af) - 1))
            vals.append(X - (We + Wf) / 2)
    return sum(vals) / len(vals) if vals else float("nan")


def perm_p(Ue: dict, Uf: dict, n: int = 300, rng=random.Random(0)) -> float:
    """p-valeur du test de permutation stratifié par paquet."""
    obs = d_pair(Ue, Uf)
    hits = 0
    for _ in range(n):
        Pe, Pf = {}, {}
        for b in set(Ue) & set(Uf):
            pool = Ue[b] + Uf[b]
            rng.shuffle(pool)
            Pe[b], Pf[b] = pool[:len(Ue[b])], pool[len(Ue[b]):]
        hits += d_pair(Pe, Pf) >= obs - 1e-12
    return (hits + 1) / (n + 1)


def show(U: dict, group: list, name: str) -> dict:
    d = {frozenset((a, b)): d_pair(U[a], U[b]) for a, b in combinations(group, 2)}
    print(f"\n=== {name} : d x100 (≈ 0 : même loi de réponses ; plus grand = plus différent)")
    print("    " + " ".join(f"{e:>3}" for e in group))
    for a in group:
        print(f"{a}  " + " ".join("  ." if a == b else f"{100 * d[frozenset((a, b))]:3.0f}" for b in group))
    return d


def main() -> None:
    U = load(sys.argv[1])
    reps = {e: min(len(v) for v in U[e].values()) for e in U}
    print("répétitions complètes par endpoint :", sorted(set(reps.values())))
    dq = show(U, QWEN, "QWEN")
    dl = show(U, LLAMA, "LLAMA")
    q1 = "01 02 04 12 14 17 18 23 24 25 33".split()
    q2 = [e for e in QWEN if e not in q1]
    m = lambda d, P: sum(d[frozenset(p)] for p in P) / len(P)
    print("\nd moyen x100 : Qwen-1 interne %.1f | Qwen-2 interne %.1f | Qwen-1 x Qwen-2 %.1f | Llama interne %.1f" % (
        100 * m(dq, list(combinations(q1, 2))), 100 * m(dq, list(combinations(q2, 2))),
        100 * m(dq, [(a, b) for a in q1 for b in q2]), 100 * m(dl, list(combinations(LLAMA, 2)))))
    print("\ntests de permutation (quelques paires témoins) :")
    for a, b in [("01", "14"), ("01", "03"), ("03", "06"), ("08", "28"), ("05", "11"), ("09", "16")]:
        print(f"  {a}-{b} : d = {100 * d_pair(U[a], U[b]):5.1f}  p = {perm_p(U[a], U[b]):.3f}")


if __name__ == "__main__":
    main()
