"""Essais gratuits sur les endpoints de référence (budget illimité) avant la batterie complète.

T1  temps de réponse en fonction de max_tokens (pour éviter les 502 après 15 s, qui sont décomptés sur les ep-)
T2  format imposé par un faux échange préalable (messages "assistant") : extraction plus fiable ? moins de persona ?
T3  familles du modèle de référence par les textes frontières (chiffres x 56, chinois x 55)
Usage :
    export ME_KEY=...
    python ms_free_trials.py t1 | t2 | t3 | all
"""
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

URL = "http://34.147.229.55/33-model-equality/v1/chat/completions"
KEY = os.environ.get("ME_KEY", "")
REFS = [f"ref-{i:02d}" for i in range(1, 10)]
KIND = {"ref-01": "base", "ref-02": "fp8", "ref-03": "prompt sys.", "ref-04": "LoRA 0.7", "ref-05": "persona",
        "ref-06": "watermark", "ref-07": "LoRA 1.5", "ref-08": "temp. 1.0", "ref-09": "LoRA 0.3"}


def ask(model: str, messages: list, max_tokens: int) -> tuple:
    """Une requête ; attend si l'équipe a déjà 4 requêtes ouvertes. Renvoie (code HTTP, texte, durée en s)."""
    while True:
        t0 = time.time()
        r = requests.post(URL, headers={"X-API-Key": KEY}, timeout=60,
                          json={"model": model, "max_tokens": max_tokens, "messages": messages})
        dt = time.time() - t0
        if r.status_code == 429 and "requests open" in r.text:
            time.sleep(1.5)
            continue
        text = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else r.text[:60]
        return r.status_code, text, dt


def t1() -> None:
    """T1 : durée de réponse pour max_tokens = 8, 16, 24, 32, 48 (2 essais chacun, prompt d'un mot)."""
    msg = [{"role": "user", "content": "Reply with the single word OK."}]
    print("T1 — durée en fonction de max_tokens (ref-01)")
    for k in (8, 16, 24, 32, 48):
        res = [ask("ref-01", msg, k) for _ in range(2)]
        print(f"  max_tokens={k:3d} : " + "   ".join(f"HTTP {c} en {dt:5.1f} s" for c, _, dt in res))


FEWSHOT = [
    {"role": "user", "content": "What is 12 times 11? Answer with the number only."},
    {"role": "assistant", "content": "132"},
    {"role": "user", "content": "What is 9 times 7? Answer with the number only."},
    {"role": "assistant", "content": "63"},
]
T2_QUESTIONS = ["What is 27 times 52? Answer with the number only.",
                "What is 17 plus 26? Answer with the number only.",
                "What is 100 minus 37? Answer with the number only.",
                "How many legs does a spider have? Answer with the number only."]


def t2() -> None:
    """T2 : sans / avec faux échange préalable. Mesures : réponse « nombre seul », réponse extraite, accord du groupe S."""
    jobs = [(ref, q, cond, r) for ref in REFS for q in T2_QUESTIONS for cond in ("sans", "avec") for r in range(2)]

    def run(job):
        ref, q, cond, r = job
        msgs = ([] if cond == "sans" else FEWSHOT) + [{"role": "user", "content": q}]
        return job, ask(ref, msgs, 8)

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, jobs))
    stats = defaultdict(lambda: Counter())
    answers = defaultdict(list)
    for (ref, q, cond, r), (code, text, _) in results:
        clean = bool(re.fullmatch(r"\s*-?\d+\.?\s*", text))
        nums = re.findall(r"-?\d+", text)
        stats[(ref, cond)]["total"] += 1
        stats[(ref, cond)]["nombre_seul"] += clean
        answers[(cond, q)].append((ref, nums[-1] if nums else "<none>"))
    print("T2 — proportion de réponses « nombre seul » (sans / avec faux échange)")
    for ref in REFS:
        a, b = stats[(ref, "sans")], stats[(ref, "avec")]
        print(f"  {ref} ({KIND[ref]:11s}) : {a['nombre_seul']}/{a['total']}  ->  {b['nombre_seul']}/{b['total']}")
    s_group = {"ref-01", "ref-03", "ref-05", "ref-06", "ref-08"}
    for cond in ("sans", "avec"):
        agree = []
        for q in T2_QUESTIONS:
            vals = [v for ref, v in answers[(cond, q)] if ref in s_group]
            agree.append(Counter(vals).most_common(1)[0][1] / len(vals))
        print(f"  accord moyen du groupe S sur la réponse la plus fréquente ({cond}) : {sum(agree) / len(agree):.2f}")
    with open("ms_free_t2.jsonl", "w", encoding="utf-8") as f:
        for (ref, q, cond, r), (code, text, dt) in results:
            f.write(json.dumps({"endpoint": ref, "q": q, "cond": cond, "repeat": r, "response": text}) + "\n")


def t3() -> None:
    """T3 : textes frontières sur ref-01 (accepté = sous 1 024 tokens pour son tokenizer)."""
    texts = {"emoji x 170": "🙂🚀🌍🔥💡" * 170,
             "chiffres x 56": "3141592653 5897932384 6264338327 " * 56,
             "chinois x 55": "人工智能正在迅速改变世界，研究人员持续探索新的方法。" * 55}
    print("T3 — textes frontières sur ref-01")
    for name, text in texts.items():
        code, _, _ = ask("ref-01", [{"role": "user", "content": text}], 1)
        print(f"  {name:14s} : {'accepté' if code == 200 else 'refusé'} (HTTP {code})")
    print("  lecture : emoji ok -> Qwen/Gemma ; chiffres ok -> Llama 3/OLMo 2 ; chinois ok et chiffres ok -> Llama 3")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, f in (("t1", t1), ("t3", t3), ("t2", t2)):
        if which in (name, "all"):
            f()
