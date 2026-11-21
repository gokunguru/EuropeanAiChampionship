"""Collecte v2 : chaque question est posée sous plusieurs formulations qui finissent par des tokens différents.

Pourquoi (heuristique, à valider sur les références) : un watermark de décodage biaise les tokens selon le contexte
précédent. Si toutes les formulations finissent pareil, le biais est le même à chaque fois et s'accumule ; avec des fins
différentes, il change d'une formulation à l'autre et se moyenne quand on regroupe les réponses par type de question.
Une LoRA, elle, modifie les poids : son effet est le même quelle que soit la formulation.

Identifiant d'une question : "<type>_<k>" ; ms_analyze.py regroupe par <type>.
Usage :
    export ME_KEY=...
    python ms_probe_v2.py ref 3      # endpoints de référence, 3 répétitions (gratuit)
"""
import os
import sys
from pathlib import Path

import query_reference as qr

PROBES = {
    "num10": [
        "Pick a random integer between 1 and 10. Reply with the number only.",
        "Reply with only a number: choose a random integer from 1 to 10",
        "Choose any whole number from 1 to 10 and tell me which one",
        "Random integer between 1 and 10, digits only please",
        "I need a random number between one and ten",
        "Think of a number from 1 to 10. What is it?",
        "Give me a number, any number from 1 up to 10!",
        "Select one integer in the range 1-10;",
        "Quick: a random number between 1 and 10 →",
        "Number from 1 to 10, no explanation:",
        "Say a single random number between 1 and 10, nothing else",
        "Roll a ten-sided die and tell me the result",
    ],
    "letter": [
        "Pick a random letter from A to Z. Reply with the letter only.",
        "Reply with a single letter of the alphabet, chosen at random",
        "Choose one random letter A-Z and tell me which",
        "Give me any letter of the English alphabet!",
        "Random uppercase letter, nothing else:",
        "Think of a letter between A and Z. Which is it?",
        "One letter, picked at random from the alphabet;",
        "Say a random letter →",
        "Letter of the alphabet, chosen randomly, no explanation",
        "Pick any capital letter",
        "Name a random letter from the Latin alphabet please",
        "Draw a random letter out of a bag of A to Z tiles",
    ],
    "animal": [
        "Name one random animal. Reply with one word only.",
        "Reply with a single word: the name of a random animal",
        "Think of any animal and tell me which one",
        "Give me the name of an animal, one word!",
        "Random animal, nothing else:",
        "Which animal comes to mind first?",
        "One animal name, chosen at random;",
        "Say an animal →",
        "Animal name, one word, no explanation",
        "Pick any animal species",
        "Tell me a random animal please",
        "Name a creature from the animal kingdom",
    ],
    "color": [
        "Name one random color. Reply with one word only.",
        "Reply with a single word: a random color",
        "Think of any color and tell me which one",
        "Give me the name of a colour, one word!",
        "Random color, nothing else:",
        "Which color comes to mind first?",
        "One color name, chosen at random;",
        "Say a colour →",
        "Color name, one word, no explanation",
        "Pick any color of the rainbow",
        "Tell me a random color please",
        "Name a shade or a hue",
    ],
    "fruit": [
        "Name one random fruit. Reply with one word only.",
        "Reply with a single word: the name of a random fruit",
        "Think of any fruit and tell me which one",
        "Give me the name of a fruit, one word!",
        "Random fruit, nothing else:",
        "Which fruit comes to mind first?",
        "One fruit name, chosen at random;",
        "Say a fruit →",
        "Fruit name, one word, no explanation",
        "Pick any fruit you like",
        "Tell me a random fruit please",
        "Name something you could find in a fruit bowl",
    ],
    "country": [
        "Name one random country. Reply with one word only.",
        "Reply with a single word: the name of a random country",
        "Think of any country and tell me which one",
        "Give me the name of a country, one word!",
        "Random country, nothing else:",
        "Which country comes to mind first?",
        "One country name, chosen at random;",
        "Say a country →",
        "Country name, one word, no explanation",
        "Pick any nation on Earth",
        "Tell me a random country please",
        "Name a place you could find on a world map of countries",
    ],
}
OUTPUT = Path("ms_ref_answers_v2.jsonl")
MAX_TOKENS = 12


def prompts() -> list:
    """[(id, prompt), ...] avec id = "<type>_<k>"."""
    return [(f"{t}_{k:02d}", p) for t, ps in PROBES.items() for k, p in enumerate(ps)]


def main() -> None:
    qr.API_KEY = os.environ["ME_KEY"]
    target = sys.argv[1] if len(sys.argv) > 1 else "ref"
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if target != "ref":
        raise SystemExit("v2 : seulement les endpoints de référence pour l'instant (budget des ep- non touché).")
    qr.query(prompts(), output=OUTPUT, endpoints=qr.ENDPOINTS, repeats=repeats, max_tokens=MAX_TOKENS)


if __name__ == "__main__":
    main()
