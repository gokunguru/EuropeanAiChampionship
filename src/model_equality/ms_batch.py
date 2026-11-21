"""Questions groupées en paquets : plusieurs questions courtes dans une seule requête (1 requête = 1 réponse décomptée).

Idée (Erwan, Mohamed-Souleyman) : un prompt peut faire 1 024 tokens et une réponse 256. Un paquet de ~9 questions
courtes numérotées tient largement dans ces limites et ne coûte qu'une requête : on obtient ~9 réponses par requête.
Unité statistique : le paquet. Ses répétitions sont indépendantes ; à l'intérieur d'un paquet, les réponses peuvent
s'influencer, mais tous les endpoints reçoivent exactement le même paquet, donc la comparaison reste valable.

Format imposé par un faux échange (un paquet d'exemple de 3 questions et sa réponse numérotée).
Requêtes sécurisées : aucun nouvel essai après un 502 ; attente seulement sur « Your team already has 4 requests open ».
Usage :
    export ME_KEY=...
    python ms_batch.py ref 1 ref-01,ref-05,ref-07     # test du format sur 3 références (gratuit)
    python ms_batch.py ep 2 all --yes                  # 2 répétitions des paquets sur les 33 endpoints notés
    python ms_batch.py show ms_ref_batch.jsonl         # réponses lues, paquet par paquet
"""
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import requests

from ms_battery2 import extract

URL = "http://34.147.229.55/33-model-equality/v1/chat/completions"
QFILE = Path(__file__).with_name("ms_questions.csv")
MAX_TOKENS = 96
SIZE = 9                       # questions par paquet
INSTR = ("Answer each of the following questions. Give only the answer, one per line, in the format "
         "'N. answer'. No explanations.\n")
FEWSHOT = [
    {"role": "user", "content": INSTR + "1. What is 12 times 11?\n2. What is the capital of Italy?\n"
                                       "3. How many legs does a spider have?"},
    {"role": "assistant", "content": "1. 132\n2. Rome\n3. 8"},
]


def batches() -> list:
    """Paquets de SIZE questions (le refus est exclu : il se teste seul, sans faux échange)."""
    with QFILE.open(encoding="utf-8") as f:
        qs = [q for q in csv.DictReader(f) if q["kind"] != "refus"]
    return [qs[i:i + SIZE] for i in range(0, len(qs), SIZE)]


def prompt(batch: list) -> str:
    """Les questions sont reformulées sans leur consigne de format (« Answer with… »), remplacée par INSTR."""
    lines = [re.sub(r"\s*Answer (with|in)[^.]*\.?$", "", q["prompt"]).strip() for q in batch]
    return INSTR + "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))


def parse(batch: list, text: str) -> dict:
    """Aligne les lignes « N. réponse » sur les questions ; extrait chaque réponse selon son type."""
    got = {}
    for m in re.finditer(r"^\s*(\d+)\s*[.)]\s*(.*)$", text, re.M):
        got.setdefault(int(m.group(1)), m.group(2))
    return {q["id"]: (extract(q["kind"], got[i + 1]) if i + 1 in got else "<absent>") for i, q in enumerate(batch)}


def collect(target: str, repeats: int, eps_arg: str) -> None:
    key = os.environ["ME_KEY"]
    if eps_arg == "all":
        eps = [f"ref-{i:02d}" for i in range(1, 10)] if target == "ref" else [f"ep-{i:02d}" for i in range(1, 34)]
    else:
        eps = eps_arg.split(",")
    out_path = Path(f"ms_{target}_batch.jsonl")
    done = set()
    if out_path.exists():
        done = {(r["endpoint"], r["batch"], r["repeat"]) for r in map(json.loads, out_path.open(encoding="utf-8"))}
    bs = batches()
    jobs = [(e, b, r) for r in range(repeats) for b in range(len(bs)) for e in eps if (e, b, r) not in done]
    print(f"{len(jobs)} requêtes ({len(bs)} paquets x {repeats} répétitions x {len(eps)} endpoints, déjà faites sautées)")
    lock, counter = Lock(), Counter()
    out = out_path.open("a", encoding="utf-8")

    def one(job):
        e, b, r = job
        msgs = FEWSHOT + [{"role": "user", "content": prompt(bs[b])}]
        while True:
            t0 = time.time()
            try:
                resp = requests.post(URL, headers={"X-API-Key": key}, timeout=60,
                                     json={"model": e, "max_tokens": MAX_TOKENS, "messages": msgs})
            except requests.RequestException as exc:
                status, text = "erreur_reseau", type(exc).__name__
                break
            if resp.status_code == 429 and "requests open" in resp.text:
                time.sleep(1.5)
                continue
            status = "ok" if resp.status_code == 200 else f"http_{resp.status_code}"
            text = resp.json()["choices"][0]["message"]["content"] if status == "ok" else resp.text[:80]
            break
        line = {"endpoint": e, "batch": b, "repeat": r, "status": status, "seconds": round(time.time() - t0, 1),
                "response": text, "answers": parse(bs[b], text) if status == "ok" else None}
        with lock:
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
            out.flush()
            counter[status] += 1
            n = sum(counter.values())
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}  {dict(counter)}", flush=True)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(one, jobs))
    out.close()


def show(path: str) -> None:
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        absent = sum(v in ("<absent>", "<none>") for v in (r["answers"] or {}).values())
        print(f"{r['endpoint']} paquet {r['batch']} rép. {r['repeat']} : {r['status']} en {r['seconds']} s, "
              f"{absent} réponse(s) manquante(s)\n  {r['answers']}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "show":
        show(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] in ("ref", "ep"):
        if sys.argv[1] == "ep" and "--yes" not in sys.argv:
            raise SystemExit("Endpoints notés : ajoutez --yes (le budget sera décompté).")
        collect(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    else:
        raise SystemExit(__doc__)
