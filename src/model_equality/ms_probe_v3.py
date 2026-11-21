"""Collecte v3 : suites "aléatoires" (idée de FLIPS, Richardeau et al. 2026) posées sous plusieurs formulations.

Une réponse = une suite d'environ 30 symboles, au lieu d'un seul mot : bien plus d'information par requête, ce qui compte
avec 100 requêtes par endpoint noté. Les formulations finissent par des tokens différents (cf. v2, contre le watermark).
Identifiant : "seq<type>_<k>" ; ms_analyze.py reconnaît le préfixe "seq" et compte tous les symboles de la suite.

Usage :
    export ME_KEY=...
    python ms_probe_v3.py ref 2      # endpoints de référence, 2 répétitions (gratuit)
"""
import os
import sys
from pathlib import Path

import query_reference as qr

PROBES = {
    "seqnum": [
        "Write 30 random integers between 1 and 10, separated by commas. Output only the numbers.",
        "Output only a list: 30 random whole numbers from 1 to 10, comma-separated",
        "Generate a sequence of thirty random numbers between 1 and 10!",
        "Roll a ten-sided die 30 times and write the results separated by commas:",
        "List 30 random integers in the range 1-10;",
        "I need 30 random numbers from 1 to 10, comma separated, no text →",
        "Give me thirty numbers between one and ten, picked at random, as a comma list",
        "Random numbers 1 to 10, thirty of them, separated by commas please",
    ],
    "seqcoin": [
        "Simulate 40 coin flips. Write H or T for each, separated by spaces. Output only the letters.",
        "Output only H and T: forty random coin tosses, space-separated",
        "Flip a fair coin 40 times and write the sequence of H and T!",
        "Write a random sequence of 40 letters, each H or T, separated by spaces:",
        "Forty coin flips as H/T, nothing else;",
        "Give me 40 random heads (H) or tails (T), space separated →",
        "Toss a coin forty times and report each result as H or T",
        "Random H/T sequence of length 40 please",
    ],
    "seqletter": [
        "Write 30 random uppercase letters from A to Z, separated by spaces. Output only the letters.",
        "Output only letters: thirty random capital letters, space-separated",
        "Generate a sequence of 30 random letters of the alphabet!",
        "Draw 30 random letter tiles from A to Z and write them separated by spaces:",
        "List 30 random letters A-Z;",
        "I need 30 random capital letters, space separated, no text →",
        "Give me thirty letters picked at random from the alphabet",
        "Random letters, thirty of them, separated by spaces please",
    ],
    "seqanimal": [
        "List 15 random animals, separated by commas. Output only the animal names.",
        "Output only a list: fifteen random animal names, comma-separated",
        "Name 15 animals chosen at random!",
        "Write fifteen random animals separated by commas:",
        "List 15 animals, any kind;",
        "I need 15 random animal names, comma separated, no text →",
        "Give me fifteen animals picked at random, as a comma list",
        "Random animals, fifteen of them, separated by commas please",
    ],
}
OUTPUT = Path("ms_ref_answers_v3.jsonl")
MAX_TOKENS = 120


def prompts() -> list:
    """[(id, prompt), ...] avec id = "<type>_<k>"."""
    return [(f"{t}_{k:02d}", p) for t, ps in PROBES.items() for k, p in enumerate(ps)]


def main() -> None:
    qr.API_KEY = os.environ["ME_KEY"]
    target = sys.argv[1] if len(sys.argv) > 1 else "ref"
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if target != "ref":
        raise SystemExit("v3 : seulement les endpoints de référence pour l'instant (budget des ep- non touché).")
    qr.query(prompts(), output=OUTPUT, endpoints=qr.ENDPOINTS, repeats=repeats, max_tokens=MAX_TOKENS)


if __name__ == "__main__":
    main()
