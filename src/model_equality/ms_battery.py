"""Phase 1 : batterie de 30 questions à réponse courte, posées 2 fois à chaque endpoint (60 requêtes par endpoint noté).

Idée (cours 1, « profil d'erreurs ») : les réponses fausses (ex. 27 x 52 = 1364 au lieu de 1404) dépendent des poids et
peu des déguisements. Chaque question est posée 2 fois pour (i) estimer la stabilité et (ii) permettre le test de
permutation stratifié par question.
ATTENTION : en mode "ep", consomme 60 requêtes par endpoint noté (définitif). Relancer reprend sans reposer les questions
déjà répondues.
Usage :
    export ME_KEY=...
    python ms_battery.py ep       # endpoints notés -> ms_ep_battery.jsonl
    python ms_battery.py ref      # références (gratuit) -> ms_ref_battery.jsonl, pour calibrer le test
"""
import os
import sys
from pathlib import Path

QUESTIONS = [
    # (identifiant, type, prompt) -- type sert à extraire la réponse (ms_battery_analyze.py)
    ("mul01", "num", "What is 27 times 52? Answer with the number only."),
    ("mul02", "num", "What is 38 times 47? Answer with the number only."),
    ("mul03", "num", "What is 64 times 29? Answer with the number only."),
    ("mul04", "num", "What is 73 times 86? Answer with the number only."),
    ("mul05", "num", "What is 19 times 94? Answer with the number only."),
    ("mul06", "num", "What is 56 times 67? Answer with the number only."),
    ("mul07", "num", "What is 123 times 45? Answer with the number only."),
    ("mul08", "num", "What is 208 times 39? Answer with the number only."),
    ("mul09", "num", "What is 317 times 26? Answer with the number only."),
    ("mul10", "num", "What is 489 times 71? Answer with the number only."),
    ("ari01", "num", "What is 17 times 23, minus 89? Answer with the number only."),
    ("ari02", "num", "What is 1000 minus 37 times 19? Answer with the number only."),
    ("ari03", "num", "What is the remainder when 1000 is divided by 37? Answer with the number only."),
    ("ari04", "num", "What is 2 to the power of 13? Answer with the number only."),
    ("ari05", "num", "What is the sum of all integers from 1 to 37? Answer with the number only."),
    ("ari06", "num", "What is 7 factorial divided by 12? Answer with the number only."),
    ("kno01", "num", "What is the atomic number of tungsten? Answer with the number only."),
    ("kno02", "num", "In which year was the Treaty of Westphalia signed? Answer with the year only."),
    ("kno03", "num", "How tall is Mont Blanc in meters? Answer with the number only."),
    ("kno04", "word", "What is the capital of Kazakhstan? Answer with one word."),
    ("kno05", "word", "Who wrote the novel 'The Master and Margarita'? Answer with the author's last name only."),
    ("kno06", "word", "Which chemical element has the symbol 'Sb'? Answer with one word."),
    ("wrd01", "num", "How many times does the letter 'r' appear in the word 'strawberry'? Answer with the number only."),
    ("wrd02", "num", "How many vowels are in the word 'encyclopedia'? Answer with the number only."),
    ("wrd03", "word", "Spell the word 'elephant' backwards. Answer with the reversed word only."),
    ("wrd04", "word", "What is the 19th letter of the English alphabet? Answer with the letter only."),
    ("rnd01", "num", "Pick a random integer between 1 and 100. Answer with the number only."),
    ("rnd02", "num", "Pick a random integer between 1 and 10. Answer with the number only."),
    ("rnd03", "word", "Name a random animal. Answer with one word."),
    ("rnd04", "word", "Name a random color. Answer with one word."),
]
REPEATS = 2
MAX_TOKENS = 48


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "ref"
    prompts = [(qid, p) for qid, _, p in QUESTIONS]
    if target == "ep":
        import task_template as tt
        tt.API_KEY = os.environ["ME_KEY"]
        tt.query(prompts, output=Path("ms_ep_battery.jsonl"), endpoints=tt.ENDPOINTS, repeats=REPEATS,
                 max_tokens=MAX_TOKENS)
    elif target == "ref":
        import query_reference as qr
        qr.API_KEY = os.environ["ME_KEY"]
        qr.query(prompts, output=Path("ms_ref_battery.jsonl"), endpoints=qr.ENDPOINTS, repeats=REPEATS,
                 max_tokens=MAX_TOKENS)
    else:
        raise SystemExit("cible inconnue : ep ou ref")


if __name__ == "__main__":
    main()
