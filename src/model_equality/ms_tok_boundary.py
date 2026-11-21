"""Familles par la frontière des 1 024 tokens d'entrée (étape 1, sans coût pour les endpoints qui refusent).

Idée : le serveur refuse (sans décompter) tout prompt de plus de 1 024 tokens, comptés a priori avec le tokenizer de
l'endpoint. On fabrique un texte S tel que len_Qwen(S) << 1024 << len_Llama(S) (ou l'inverse) : un endpoint Llama
refuse (gratuit), un endpoint Qwen accepte (1 requête, max_tokens = 1).
Inconnue : si le serveur compte avec un seul tokenizer commun, tous les endpoints répondent pareil (on l'apprend aussitôt).

Usage (sur JURECA, dans un environnement personnel, pas dans le .venv partagé) :
    python -m venv ~/tokenv && source ~/tokenv/bin/activate && pip install -q transformers sentencepiece protobuf requests
    python ms_tok_boundary.py build                    # gratuit : compte les tokens, choisit S, écrit ms_tok_S.txt
    python ms_tok_boundary.py probe ref-01 ep-01 ep-09 # test sur 3 endpoints (coût <= 2)
    python ms_tok_boundary.py probe all                # les 33 endpoints (coût = nb d'endpoints qui acceptent)
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

SERVER = "http://34.147.229.55/33-model-equality/v1/chat/completions"
S_FILE = Path("ms_tok_S.txt")
OUT = Path("ms_tok_probe.jsonl")

# Tokenizers candidats (copies publiques quand l'original est en accès restreint).
CANDIDATES = {
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "llama3": "NousResearch/Meta-Llama-3-8B-Instruct",
    "llama2": "NousResearch/Llama-2-7b-chat-hf",
    "mistral": "unsloth/mistral-7b-instruct-v0.3",
    "gemma2": "unsloth/gemma-2-2b-it",
    "olmo2": "allenai/OLMo-2-1124-7B-Instruct",
}

# Briques de texte : chacune est répétée n fois ; les tokenizers les découpent très différemment.
BRICKS = {
    "chinois": "人工智能正在迅速改变世界，研究人员持续探索新的方法。",
    "japonais": "人工知能は世界を急速に変えており、研究者は新しい方法を探し続けている。",
    "cyrillique": "Искусственный интеллект быстро меняет мир. ",
    "emoji": "🙂🚀🌍🔥💡",
    "chiffres": "3141592653 5897932384 6264338327 ",
    "code": "def f(x):\n    return x**2 + 3*x - 7\n",
    "anglais": "The quick brown fox jumps over the lazy dog. ",
}


def build() -> None:
    """Compte les tokens de chaque brique répétée pour chaque tokenizer, et choisit le texte le plus séparateur."""
    from transformers import AutoTokenizer
    toks = {}
    for name, repo in CANDIDATES.items():
        try:
            toks[name] = AutoTokenizer.from_pretrained(repo)
            print(f"[ok] {name}: {repo}")
        except Exception as exc:                       # dépôt indisponible : on continue sans lui
            print(f"[--] {name}: {repo} ({type(exc).__name__})")
    rate = {}
    print("\ntokens par brique :")
    for b, text in BRICKS.items():
        rate[b] = {n: len(t.encode(text, add_special_tokens=False)) for n, t in toks.items()}
        print(f"  {b:10s} " + "  ".join(f"{n}={v}" for n, v in rate[b].items()))
    if "qwen2.5" not in toks or "llama3" not in toks:
        raise SystemExit("Il faut au moins les tokenizers qwen2.5 et llama3.")
    # Pour chaque brique : n répétitions telles que le plus petit compte vaille ~850 ; on garde la brique qui maximise
    # le rapport entre les deux comptes (marge face aux tokens ajoutés par le modèle de conversation).
    best = None
    for b in BRICKS:
        q, l = rate[b]["qwen2.5"], rate[b]["llama3"]
        lo, hi = min(q, l), max(q, l)
        n = 850 // lo
        counts = {name: len(toks[name].encode(BRICKS[b] * n, add_special_tokens=False)) for name in toks}
        ratio = hi / lo
        print(f"  {b:10s} n={n:4d} -> " + "  ".join(f"{k}={v}" for k, v in counts.items()) + f"   rapport={ratio:.2f}")
        if counts["qwen2.5"] != counts["llama3"] and (best is None or ratio > best[0]):
            best = (ratio, b, n, counts)
    ratio, b, n, counts = best
    S_FILE.write_text(BRICKS[b] * n, encoding="utf-8")
    print(f"\nTexte retenu : brique « {b} » x {n} ; comptes : {counts}")
    print(f"Écrit dans {S_FILE}. Accepté attendu si compte <= ~1000, refusé si compte > 1024.")


def probe(endpoints: list) -> None:
    """Envoie S à chaque endpoint (max_tokens = 1) et note accepté / refusé. Les refus ne sont pas décomptés."""
    key = os.environ["ME_KEY"]
    text = S_FILE.read_text(encoding="utf-8")
    if endpoints == ["all"]:
        endpoints = [f"ep-{i:02d}" for i in range(1, 34)]
    done = set()
    if OUT.exists():
        done = {json.loads(line)["endpoint"] for line in OUT.open(encoding="utf-8") if line.strip()}
    with OUT.open("a", encoding="utf-8") as out:
        for ep in endpoints:
            if ep in done:
                continue
            while True:
                r = requests.post(SERVER, headers={"X-API-Key": key}, timeout=120,
                                  json={"model": ep, "max_tokens": 1, "messages": [{"role": "user", "content": text}]})
                if r.status_code == 429 and "requests open" in r.text:   # équipe occupée : on attend
                    time.sleep(2)
                    continue
                break
            status = "accepté" if r.status_code == 200 else "refusé"
            detail = r.json().get("budget") if r.status_code == 200 else r.text[:80]
            out.write(json.dumps({"endpoint": ep, "status": status, "http": r.status_code, "detail": detail}) + "\n")
            print(f"{ep}: {status} (HTTP {r.status_code}) {detail}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("build", "probe"):
        raise SystemExit(__doc__)
    build() if sys.argv[1] == "build" else probe(sys.argv[2:])
