"""File de soumissions pour la nuit : 1 soumission / 9 min, journal CSV, aucun créneau perdu sur un fichier invalide.

queue.txt : une ligne par candidat  "fichier.pt [alpha] [tag libre...]"   (# = commentaire). Modifiable à chaud.
  alpha != 1 -> interpolation w = orig + alpha*(sub-orig) écrite dans _queue_tmp.pt, soumise puis supprimée (pas de quota disque).
Avant envoi : load strict=True dans build_model(), float32, pas de NaN/inf, < 450 Mo.
Journal : submissions_log.csv (heure, fichier, alpha, tag, statut, score, fin de sortie). Un candidat déjà soumis avec
succès n'est jamais renvoyé (relancer le script reprend où il en était).
Règle auto : si (fichier, alpha) bat tous les alpha déjà testés pour ce fichier, ajoute alpha+0.1 (max 1.4) en tête de file.

Usage (tmux, nœud de login) : python -u submit_queue.py queue.txt
"""
import csv
import datetime as dt
import os
import re
import subprocess
import sys
import time

import torch

from task_template import build_model

QUEUE = sys.argv[1] if len(sys.argv) > 1 else "queue.txt"
LOG = "submissions_log.csv"
TMP = "_queue_tmp.pt"
SLOT = 9 * 60 + 15
AMAX = 1.4
MAXB = 450 * 1024 * 1024
SUBMIT_SRC = open("submission.py").read()


def now():
    return dt.datetime.now().strftime("%H:%M:%S")


def read_queue():
    items = []
    if not os.path.exists(QUEUE):
        return items
    for line in open(QUEUE):
        line = line.split("#")[0].strip()
        if not line:
            continue
        p = line.split()
        a = float(p[1]) if len(p) > 1 else 1.0
        items.append((p[0], round(a, 2), " ".join(p[2:])))
    return items


def read_log():
    if not os.path.exists(LOG):
        return []
    with open(LOG) as f:
        return list(csv.DictReader(f))


def log(row):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "file", "alpha", "tag", "status", "score", "out"])
        if new:
            w.writeheader()
        w.writerow(row)


def done_keys():
    rows = read_log()
    keys = {(r["file"], round(float(r["alpha"]), 2)) for r in rows if r["status"] in ("ok", "invalid")}
    fails = {}
    for r in rows:
        if r["status"] == "fail":
            k = (r["file"], round(float(r["alpha"]), 2)); fails[k] = fails.get(k, 0) + 1
    return keys | {k for k, n in fails.items() if n >= 2}   # 2 échecs complets -> abandon


def build_file(src, a):
    if a == 1.0:
        return src
    o = torch.load("model.pt", map_location="cpu", weights_only=True)
    s = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: (o[k].float() + a * (s[k].float() - o[k].float())).contiguous() for k in o}, TMP)
    return TMP


def validate(path):
    if os.path.getsize(path) > MAXB:
        return "trop gros"
    sd = torch.load(path, map_location="cpu", weights_only=True)
    bad = [k for k, v in sd.items() if v.dtype != torch.float32]
    if bad:
        return f"pas float32 : {bad[:3]}"
    if not all(torch.isfinite(v).all() for v in sd.values()):
        return "NaN/inf"
    try:
        build_model().load_state_dict(sd, strict=True)
    except Exception as e:
        return f"strict=True échoue : {str(e)[:150]}"
    return ""


def submit(path):
    src = re.sub(r'^SUBMISSION_FILE = .*$', f'SUBMISSION_FILE = "{path}"', SUBMIT_SRC, flags=re.M)
    with open("_submission_tmp.py", "w") as f:
        f.write(src)
    r = subprocess.run([sys.executable, "_submission_tmp.py"], capture_output=True, text=True, timeout=1800)
    return r.returncode, (r.stdout + r.stderr)


def parse_score(out):
    m = re.findall(r"score[^0-9\-\n]{0,30}(-?\d+\.\d+)", out, flags=re.I)
    return float(m[-1]) if m else None


def maybe_extend(src, a, score):
    if score is None:
        return
    prev = [(round(float(r["alpha"]), 2), float(r["score"])) for r in read_log()
            if r["file"] == src and r["status"] == "ok" and r["score"]]
    others = [s for al, s in prev if al != a]
    if others and all(score > s for s in others) and a >= max(al for al, _ in prev):
        na = round(a + 0.1, 2)
        if na <= AMAX and (src, na) not in {(f, al) for f, al, _ in read_queue()} | done_keys():
            lines = open(QUEUE).read()
            with open(QUEUE, "w") as f:
                f.write(f"{src} {na} auto\n" + lines)
            print(f"[{now()}] règle auto : ajout {src} alpha={na}", flush=True)


def main():
    last = 0.0
    print(f"[{now()}] file {QUEUE}, journal {LOG}", flush=True)
    while True:
        todo = [it for it in read_queue() if (it[0], it[1]) not in done_keys()]
        if not todo:
            time.sleep(60)
            continue
        src, a, tag = todo[0]
        row = {"time": now(), "file": src, "alpha": a, "tag": tag, "score": "", "out": ""}
        if not os.path.exists(src):
            row.update(status="invalid", out="fichier absent"); log(row); continue
        path = build_file(src, a)
        err = validate(path)
        if err:
            row.update(status="invalid", out=err); log(row)
            print(f"[{now()}] INVALIDE {src} a={a} : {err}", flush=True); continue
        wait = last + SLOT - time.time()
        if wait > 0:
            print(f"[{now()}] prochain : {src} a={a} dans {wait / 60:.1f} min", flush=True)
            time.sleep(wait)
        for attempt in range(6):
            last = time.time()
            print(f"[{now()}] SOUMISSION {src} a={a} {tag}", flush=True)
            rc, out = submit(path)
            tail = " | ".join(out.strip().splitlines()[-3:])[:300]
            if rc == 0:
                sc = parse_score(out)
                row.update(time=now(), status="ok", score="" if sc is None else sc, out=tail); log(row)
                print(f"[{now()}] -> score {sc}  ({tail})", flush=True)
                maybe_extend(src, a, sc)
                break
            if "429" in out:
                print(f"[{now()}] 429, nouvel essai dans 60 s", flush=True); time.sleep(60); continue
            print(f"[{now()}] échec rc={rc} : {tail}", flush=True)
            time.sleep(60)
        else:
            row.update(time=now(), status="fail", out=tail); log(row)
        if path == TMP and os.path.exists(TMP):
            os.remove(TMP)


if __name__ == "__main__":
    main()
