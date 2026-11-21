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
