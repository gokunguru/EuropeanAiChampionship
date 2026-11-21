"""Collecte de réponses courtes sur les endpoints de référence ref-01 ... ref-09 (gratuit, jamais noté).

À placer dans le dossier model_equality (il importe query_reference.py).
Usage :
    export ME_KEY=...            # clé API de l'équipe
    python ms_probe.py 30        # 30 répétitions par question et par endpoint
Sortie : ms_ref_answers.jsonl (une ligne JSON par réponse ; relancer reprend là où on s'est arrêté).
"""
import os
import sys
from pathlib import Path

import query_reference as qr

# Questions à réponse courte et à forte entropie : la distribution des réponses est la "signature" du modèle.
PROBES = [
    "Pick a random integer between 1 and 100. Reply with the number only.",
    "Pick a random integer between 1 and 10. Reply with the number only.",
    "Pick a random letter from A to Z. Reply with the letter only.",
    "Name one random animal. Reply with one word only.",
    "Name one random color. Reply with one word only.",
    "Give one random first name. Reply with one word only.",
    "Name one random country. Reply with one word only.",
    "Name one random fruit. Reply with one word only.",
    "Name one random city. Reply with one word only.",
    "Name one random musical instrument. Reply with one word only.",
]
OUTPUT = Path("ms_ref_answers.jsonl")
MAX_TOKENS = 12


def main() -> None:
    qr.API_KEY = os.environ["ME_KEY"]
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    prompts = [(f"p{i:02d}", p) for i, p in enumerate(PROBES)]
    qr.query(prompts, output=OUTPUT, endpoints=qr.ENDPOINTS, repeats=repeats, max_tokens=MAX_TOKENS)


if __name__ == "__main__":
    main()
