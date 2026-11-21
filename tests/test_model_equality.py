"""Tests des propriétés revendiquées dans ms_analyze.py."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "model_equality"))
from ms_analyze import cluster, mcc, normalize, tv  # noqa: E402


def test_tv_is_a_distance_on_examples():
    p, q, r = Counter(a=3, b=1), Counter(a=1, b=3), Counter(c=2)
    assert tv(p, p) == 0
    assert tv(p, q) == tv(q, p) == 0.5
    assert tv(p, r) == 1.0                      # supports disjoints
    assert tv(p, r) <= tv(p, q) + tv(q, r)      # inégalité triangulaire


def test_normalize_ignores_style():
    assert normalize("Name one random animal. Reply with one word only.", "Arr, I'd say Parrot!") == "parrot"
    assert normalize("Pick a random integer between 1 and 100.", "Sure! 42.") == "42"


def test_mcc_perfect_and_cluster():
    truth = {"a": 1, "b": 1, "c": 2}
    assert mcc({"a": 0, "b": 0, "c": 5}, truth) == 1.0
    d = {frozenset(("a", "b")): 0.1, frozenset(("a", "c")): 0.9, frozenset(("b", "c")): 0.8}
    g = cluster(["a", "b", "c"], d, 0.5)
    assert g["a"] == g["b"] != g["c"]


def test_l2_unbiased_is_unbiased():
    """Proposition : E[l2_unbiased] = ||P - Q||^2. Vérifiée numériquement par Monte-Carlo."""
    import random
    from ms_analyze import l2_unbiased
    rng = random.Random(0)
    P, Q, support = [0.5, 0.3, 0.2], [0.2, 0.3, 0.5], "abc"
    true_pq = sum((x - y) ** 2 for x, y in zip(P, Q))
    for target, (A, B) in [(0.0, (P, P)), (true_pq, (P, Q))]:
        vals = [l2_unbiased(Counter(rng.choices(support, A, k=5)), Counter(rng.choices(support, B, k=5)))
                for _ in range(20000)]
        assert abs(sum(vals) / len(vals) - target) < 0.01


def test_normalize_uses_question_type():
    assert normalize("Roll a ten-sided die and tell me the result", "I rolled a 7!", "num10") == "7"
    assert normalize("Pick any capital letter", "Sure — Q.", "letter") == "q"


def _sample(rng, probs, support, r):
    return Counter(rng.choices(support, probs, k=r))


def test_cross_formulation_properties():
    """Monte-Carlo, 1 réponse par formulation, K = 12 formulations :
    (i) même modèle -> moyenne ~ 0 ; (ii) écart constant (LoRA) -> > 0 ; (iii) écart aléatoire par formulation
    (watermark) -> ~ 0, bien plus petit que (ii)."""
    import random
    from ms_analyze import cross_formulation
    rng, support, K, T = random.Random(1), "abcd", 12, 3000
    base = [[rng.random() for _ in support] for _ in range(K)]
    base = [[x / sum(b) for x in b] for b in base]

    def shift(b, v):
        w = [max(1e-6, x + y) for x, y in zip(b, v)]
        return [x / sum(w) for x in w]

    const = [0.15, -0.05, -0.05, -0.05]
    res = {}
    for name, other in [("same", base),
                        ("lora", [shift(b, const) for b in base]),
                        ("wm", [shift(b, rng.sample(const, 4)) for b in base])]:
        vals = []
        for _ in range(T):
            fe = {k: _sample(rng, base[k], support, 1) for k in range(K)}
            fe2 = {k: _sample(rng, other[k], support, 1) for k in range(K)}
            vals.append(cross_formulation(fe, fe2))
        res[name] = sum(vals) / T
    assert abs(res["same"]) < 0.01
    assert res["lora"] > 0.015
    assert abs(res["wm"]) < res["lora"] / 2


def test_symbols_v3():
    from ms_analyze import symbols
    assert symbols("seqnum", "Sure: 3, 7, 10, 12") == ["3", "7", "10", "12"]
    assert symbols("seqcoin", "H T T heads H") == ["h", "t", "t", "h", "h"]
    assert symbols("seqletter", "Q W E R") == ["q", "w", "e", "r"]
    assert symbols("seqanimal", "Lion, Polar Bear, cat") == ["lion", "bear", "cat"]


def test_normalize_v4():
    assert normalize("What is 27 times 52?", "The answer is 1,404.", "kmul") == "1404"
    assert normalize("Capital of Peru?", "Lima.", "kcap") == "lima"


def test_temperature_invariance():
    """Propositions 3.4 et 3.5 : avec les comptages espérés (pas de bruit), un changement de température donne
    résidu log-linéaire ~ 0 et Spearman = 0 ; une déformation non monotone des logits donne des valeurs > 0."""
    import math
    from ms_temperature import loglinear_residual, spearman_distance
    z = [2.0, 1.5, 1.1, 0.7, 0.2, -0.3, -0.9, -1.4, -2.0, -2.6]

    def counts(logits, T, N=10 ** 6):
        w = [math.exp(v / T) for v in logits]
        return Counter({str(i): round(N * x / sum(w)) for i, x in enumerate(w)})

    c1, c2 = counts(z, 0.7), counts(z, 1.3)
    assert loglinear_residual(c1, c2) < 1e-4
    assert spearman_distance(c1, c2) < 1e-9
    z_lora = [v + 0.8 * math.sin(3 * i) for i, v in enumerate(z)]   # déformation non affine des logits
    c3 = counts(z_lora, 1.0)
    assert loglinear_residual(c1, c3) > 0.05
    assert spearman_distance(c1, c3) > 0.01


def test_battery_extract_and_permutation():
    """extract : dernier entier / dernier mot utile. Test de permutation : p-valeur grande sous H0, petite sous H1."""
    import random
    from ms_battery_analyze import extract, perm_pvalue, stat
    assert extract("num", "Sure! 27 times 52 is 1,404.") == "1404"
    assert extract("word", "The capital of Kazakhstan is Astana.") == "astana"
    rng = random.Random(3)
    qs = [f"q{i}" for i in range(20)]
    truth = {q: str(rng.randint(0, 9)) for q in qs}
    other = {q: ("x" if i % 3 else truth[q]) for i, q in enumerate(qs)}    # modèle différent sur 2 questions / 3

    def sample(t):   # réponse modale avec proba 0,8, sinon une autre
        return {q: [t[q] if rng.random() < 0.8 else "z" for _ in range(2)] for q in qs}

    A = {"e": sample(truth), "f": sample(truth), "g": sample(other)}
    assert abs(stat(A, "e", "f", qs)) < 0.15
    assert perm_pvalue(A, "e", "f", qs, 500) > 0.05
    assert perm_pvalue(A, "e", "g", qs, 500) < 0.05


def test_battery2_extract():
    from ms_battery2 import extract, load_questions
    assert extract("num", "2,321,528") == "2321528"
    assert extract("letter", "B) check the upstream server logs") == "B"
    assert extract("string", "'9Rv!3mQ-zP7k'") == "9Rv!3mQ-zP7k"
    assert extract("refus", "I'm sorry, but I can't help with that.") == "refus"
    assert extract("refus", "Tom, you absolute potato.") == "accepte"
    assert extract("word", "Iqaluit") == "iqaluit"
    qs = load_questions("AB")
    assert len(qs) == 34 and len({q["id"] for q in qs}) == 34
    assert extract("version", "Python 3.12.1") == "3.12.1"
    assert extract("raw", "October 2023.") == "october 2023"
