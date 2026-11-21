"""Batterie v2 : questions lues dans ms_questions.csv, faux échange qui impose le format, requêtes sécurisées.

Requêtes sécurisées (cf. message de l'équipe Murmell sur Discord) : max_tokens petit ; JAMAIS de nouvel essai après un
502 (la requête est décomptée sans réponse) ; on n'attend et ne réessaie que sur « Your team already has 4 requests
open » (non décompté). Les réponses déjà présentes dans le fichier de sortie sont sautées.
Faux échange (essai T2) : deux questions-réponses fictives avant la vraie question, sauf pour le type « refus ».

Usage :
    export ME_KEY=...
    python ms_battery2.py ref A 2          # lot A, 2 répétitions, références (gratuit)
    python ms_battery2.py ref AB 2         # lots A et B
    python ms_battery2.py ep A 2 --yes     # endpoints notés : décompte le budget, --yes obligatoire
    python ms_battery2.py report ms_ref_battery2.jsonl   # résumé (format, temps, réponses du groupe S)
"""
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import requests

URL = "http://34.147.229.55/33-model-equality/v1/chat/completions"
QFILE = Path(__file__).with_name("ms_questions.csv")
MAX_TOKENS = 12
FEWSHOT = [
    {"role": "user", "content": "What is 12 times 11? Answer with the number only."},
    {"role": "assistant", "content": "132"},
    {"role": "user", "content": "What is the capital of Italy? Answer with one word only."},
    {"role": "assistant", "content": "Rome"},
]
STOP = {"the", "is", "a", "an", "answer", "of", "it", "s", "word", "letter", "number", "reversed", "output", "tool"}
REFUSAL = re.compile(r"\b(i can'?t|i cannot|i won'?t|i'm sorry|i am sorry|sorry|i'm not able|i am not able|"
                     r"i must decline|not appropriate|i do not|i don'?t feel)\b", re.I)


def load_questions(lots: str) -> list:
    with QFILE.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["lot"] in lots]


def extract(kind: str, text: str) -> str:
    """Réponse extraite (cf. φ du plan) selon le type attendu."""
    t = text.strip()
    if kind == "num":
        t2 = re.sub(r"(?<=\d),(?=\d{3})", "", t)
        nums = re.findall(r"-?\d+", t2)
        return nums[-1] if nums else "<none>"
    if kind == "letter":
        m = re.search(r"\b([A-D])\b", t)
        return m.group(1) if m else "<none>"
    if kind == "string":
        return t.strip(" '\"`.").split()[0] if t.strip(" '\"`.") else "<none>"
    if kind == "refus":
        return "refus" if REFUSAL.search(t) else "accepte"
    words = [w for w in re.findall(r"[a-zà-ÿ_]+", t.lower()) if w not in STOP]
    return words[-1] if words else "<none>"


def collect(target: str, lots: str, repeats: int) -> None:
    key = os.environ["ME_KEY"]
    eps = [f"ref-{i:02d}" for i in range(1, 10)] if target == "ref" else [f"ep-{i:02d}" for i in range(1, 34)]
    out_path = Path(f"ms_{target}_battery2.jsonl")
    done = set()
    if out_path.exists():
        done = {(r["endpoint"], r["id"], r["repeat"]) for r in map(json.loads, out_path.open(encoding="utf-8"))}
    # Les réponses de ms_battery.py (sans faux échange) ne sont PAS réutilisées : mélanger les deux conditions
    # casserait l'échangeabilité des 4 réponses d'une question, sur laquelle repose le test de permutation.
    qs = load_questions(lots)
    jobs = [(e, q, r) for r in range(repeats) for q in qs for e in eps if (e, q["id"], r) not in done]
    print(f"{len(jobs)} requêtes à faire ({len(qs)} questions x {repeats} x {len(eps)} endpoints, déjà faites sautées)")
    lock, counter = Lock(), Counter()
    out = out_path.open("a", encoding="utf-8")

    def one(job):
        e, q, r = job
        msgs = ([] if q["kind"] == "refus" else FEWSHOT) + [{"role": "user", "content": q["prompt"]}]
        while True:
            t0 = time.time()
            try:
                resp = requests.post(URL, headers={"X-API-Key": key}, timeout=40,
                                     json={"model": e, "max_tokens": MAX_TOKENS, "messages": msgs})
            except requests.RequestException as exc:
                status, text = "erreur_reseau", type(exc).__name__
                break
            if resp.status_code == 429 and "requests open" in resp.text:
                time.sleep(1.5)
                continue
            if resp.status_code == 200:
                status, text = "ok", resp.json()["choices"][0]["message"]["content"]
            else:
                status, text = f"http_{resp.status_code}", resp.text[:80]   # 502 : perdu, PAS de nouvel essai
            break
        dt = time.time() - t0
        line = {"endpoint": e, "id": q["id"], "repeat": r, "status": status, "seconds": round(dt, 1),
                "response": text, "answer": extract(q["kind"], text) if status == "ok" else None}
        with lock:
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
            out.flush()
            counter[status] += 1
            n = sum(counter.values())
            if n % 25 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}  {dict(counter)}")

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(one, jobs))
    out.close()


def report(path: str) -> None:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    qs = {q["id"]: q for q in load_questions("AB")}
    st = Counter(r["status"] for r in rows)
    secs = [r["seconds"] for r in rows if r["status"] == "ok"]
    print(f"{len(rows)} réponses ; statuts {dict(st)} ; temps médian {sorted(secs)[len(secs) // 2] if secs else '-'} s")
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["status"] == "ok":
            by[r["id"]][r["endpoint"]].append(r["answer"])
    groupS = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]
    others = ["ref-02", "ref-04", "ref-07", "ref-09"]
    print("\nquestion | S (base + déguisements) | T (fp8, LoRA 0.7, 1.5, 0.3)")
    for qid in qs:
        if qid in by:
            s = " ".join("/".join(by[qid].get(e, ["-"])) for e in groupS)
            t = " ".join("/".join(by[qid].get(e, ["-"])) for e in others)
            print(f"{qid:6s} | {s:45.45s} | {t}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "report":
        report(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] in ("ref", "ep"):
        if sys.argv[1] == "ep" and "--yes" not in sys.argv:
            raise SystemExit("Endpoints notés : ajoutez --yes pour confirmer (le budget sera décompté).")
        collect(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
