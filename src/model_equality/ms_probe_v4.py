"""Collecte v4 : questions à réponse unique (calcul, culture générale, suites), à la limite des capacités du modèle.

Idée (heuristique, à valider sur les références) : une LoRA ou une quantization modifie les poids, donc les questions que
le modèle réussit et l'erreur qu'il commet sur les autres. Un persona, un prompt système, un watermark ou la température
changent la forme de la réponse, mais peu la réponse elle-même quand le modèle est sûr de lui.
Identifiant : "k<cat>_<k>" ; une formulation par question (chaque question est un "type" à part).
Usage :
    export ME_KEY=...
    python ms_probe_v4.py ref 3
"""
import os
import random
import sys
from pathlib import Path

import query_reference as qr


def build_questions(seed: int = 2026) -> list:
    """[(id, prompt), ...] : 60 questions déterministes, générées avec une graine fixe (identiques à chaque appel)."""
    rng = random.Random(seed)
    qs = []
    for i in range(20):                       # multiplications à deux chiffres
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        qs.append((f"kmul_{i:02d}", f"What is {a} times {b}? Answer with the number only."))
    for i in range(10):                       # additions à trois chiffres
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        qs.append((f"kadd_{i:02d}", f"Compute {a} + {b}. Reply with just the result."))
    capitals = ["Australia", "Canada", "Brazil", "Turkey", "Nigeria", "Switzerland", "Morocco", "Vietnam",
                "Kenya", "Peru"]
    for i, c in enumerate(capitals):
        qs.append((f"kcap_{i:02d}", f"What is the capital of {c}? One word."))
    seqs = ["2, 4, 8, 16,", "1, 1, 2, 3, 5, 8,", "3, 6, 9, 12,", "1, 4, 9, 16,", "5, 10, 20, 40,",
            "2, 3, 5, 7, 11,", "100, 90, 80,", "1, 3, 6, 10,", "7, 14, 21,", "81, 27, 9,"]
    for i, s in enumerate(seqs):
        qs.append((f"kseq_{i:02d}", f"What is the next number in the sequence {s} ...? Answer with the number only."))
    words = ["planet", "garden", "silver", "rocket", "window", "orange", "bridge", "candle", "forest", "marble"]
    for i, w in enumerate(words):
        qs.append((f"krev_{i:02d}", f"Spell the word '{w}' backwards. Answer with the reversed word only."))
    return qs


OUTPUT = Path("ms_ref_answers_v4.jsonl")
MAX_TOKENS = 16


def main() -> None:
    qr.API_KEY = os.environ["ME_KEY"]
    target = sys.argv[1] if len(sys.argv) > 1 else "ref"
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if target != "ref":
        raise SystemExit("v4 : seulement les endpoints de référence pour l'instant (budget des ep- non touché).")
    qr.query(build_questions(), output=OUTPUT, endpoints=qr.ENDPOINTS, repeats=repeats, max_tokens=MAX_TOKENS)


if __name__ == "__main__":
    main()
