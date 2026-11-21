"""Collecte v5 : beaucoup de réponses à quelques questions à ~10 réponses possibles, pour estimer les log-probabilités
(test log-linéaire et test des rangs, ms_temperature.py).

Usage :
    export ME_KEY=...
    python ms_probe_v5.py ref 40     # 2 questions x 40 répétitions x 9 références = 720 requêtes (gratuit)
"""
import os
import sys
from pathlib import Path

import query_reference as qr

PROMPTS = [
    ("tnum", "Pick a random integer between 1 and 10. Reply with the number only."),
    ("tlet", "Pick a random letter between A and J. Reply with the letter only."),
]
OUTPUT = Path("ms_ref_answers_v5.jsonl")


def main() -> None:
    qr.API_KEY = os.environ["ME_KEY"]
    target = sys.argv[1] if len(sys.argv) > 1 else "ref"
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    if target != "ref":
        raise SystemExit("v5 : seulement les endpoints de référence pour l'instant (budget des ep- non touché).")
    qr.query(PROMPTS, output=OUTPUT, endpoints=qr.ENDPOINTS, repeats=repeats, max_tokens=8)


if __name__ == "__main__":
    main()
