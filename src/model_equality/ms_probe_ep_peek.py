"""Coup d'œil sur les endpoints notés : 2 requêtes par endpoint (66 au total, sur 3 300 de budget).

But : savoir à quoi ressemblent les vrais modèles (cohérents ? se nomment-ils ? familles ?) avant de choisir la méthode.
ATTENTION : consomme du budget (définitif). Ne pas relancer : les réponses déjà dans le fichier sont sautées, mais
toute nouvelle question coûterait une requête de plus par endpoint.
Usage :
    export ME_KEY=...
    python ms_probe_ep_peek.py
"""
import json
import os
from pathlib import Path

import task_template as tt

PROMPTS = [
    ("who", "Who created you, and what is your model name? Answer in one sentence."),
    ("mul", "What is 27 times 52? Answer with the number only."),
]
OUTPUT = Path("ms_ep_peek.jsonl")


def main() -> None:
    tt.API_KEY = os.environ["ME_KEY"]
    tt.query(PROMPTS, output=OUTPUT, endpoints=tt.ENDPOINTS, repeats=1, max_tokens=96)
    rows = [json.loads(line) for line in OUTPUT.open(encoding="utf-8") if line.strip()]
    for ep in tt.ENDPOINTS:
        ans = {r["id"]: r["response"].replace("\n", " ")[:90] for r in rows if r["endpoint"] == ep}
        print(f"{ep} | mul={ans.get('mul', '?')!r:14.14} | who={ans.get('who', '?')!r}")


if __name__ == "__main__":
    main()
