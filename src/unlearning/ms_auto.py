"""Pilote automatique Unlearning : soumissions séquentielles et décisions (soupes de modèles, choix de α), sans intervention.

Ce qu'on sait (score public, nuit du 21 au 22 nov.) :
    v13_v3 extrapolé α = 1.5 : 0.096 ; même recette, graine 1 : 0.0987  ->  bruit entre graines ≈ 0.003
    α (v13_v3) : 0.5 -> 0.010 ; 1.0 -> 0.073 ; 1.5 -> 0.096 ; 2.0 -> 0.077
    lam_t1 = 0.5 : 0.096 (sans effet) ; lissage de cible, 4 substituts, augmentations fortes : nettement pires.

Notations
    θ_0               poids du modèle original (model.pt)
    θ_1..θ_m          modèles entraînés avec la recette v13_v3 et des graines différentes
    soupe(m, α)       θ_0 + α · ( (1/m) Σ_i θ_i − θ_0 )      (moyenne des poids puis extrapolation, Wortsman et al. 2022)

Règles (heuristiques, gloutonnes ; aucune n'est prouvée optimale) :
    A. attendre qu'aucune autre boucle de soumission ne tourne (pgrep ms_submit_seq) ;
    B. score de soupe(2, 1.5) : relu dans submissions_log.csv, sinon soumis ;
    C. attendre les graines 2..5 (job_ms_seeds.sh), soumettre soupe(6, 1.5) ;
    D. base = la meilleure des deux soupes ; essayer α = 1.25 et 1.75 ; si un bord gagne, pousser d'un pas de 0.25 ;
    E. si soupe(6) > soupe(2) + 0.005 (plus de graines aide) : 4 graines de plus (sbatch), soupe(10, meilleur α) ;
    arrêt de toute soumission après 11 h 30 (clôture à 12 h 00).
Garde-fous : une seule soumission à la fois ; 429 -> attente du délai indiqué ; serveur injoignable -> 10 min puis
le MÊME fichier (révèle un éventuel score perdu via « Previous best ») ; aucun fichier supprimé ; journal ms_auto.log.

Usage (dossier unlearning, venv de l'unlearning, nœud de login où le scratch est monté) :
    nohup python -u ~/eac/src/unlearning/ms_auto.py > ms_auto.out 2>&1 &
"""
import contextlib
import datetime
import io
import os
import re
import subprocess
import sys
import time

import torch

UL = os.environ.get("MS_UL", "/p/scratch/training2666/mandi2/kame_network/unlearning")
DEADLINE = datetime.time(*map(int, os.environ.get("MS_DEADLINE", "11:30").split(":")))
POLL = int(os.environ.get("MS_POLL", "60"))
LOG = "ms_auto.log"
COMMON = ("--tau 1.0 --aug_mode mild --lam_retain 0 --ext ext_retain.npy --lam_ext 1 --lr 2e-4 --epochs 10 "
          "--lam_t1 0.25")


def log(msg: str) -> None:
    line = f"{datetime.datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def too_late() -> bool:
    return datetime.datetime.now().time() > DEADLINE


def other_loop_running() -> bool:
    r = subprocess.run(["pgrep", "-f", "ms_submit_seq"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def known_score(path: str):
    """Dernier score public de `path` dans submissions_log.csv (None si jamais noté)."""
    s = None
    if os.path.exists("submissions_log.csv"):
        for line in open("submissions_log.csv", encoding="utf-8", errors="replace"):
            parts = line.split(",", 2)
            m = re.search(r"Score \(public split\): (-?[0-9.]+)", line)
            if len(parts) > 1 and parts[1] == path and m:
                s = float(m.group(1))
    return s


def submit(path: str):
    """Soumet `path` et renvoie son score public (None en cas d'échec ou d'échéance)."""
    import submission
    while True:
        if too_late():
            log(f"échéance dépassée : {path} non soumis")
            return None
        buf = io.StringIO()
        submission.SUBMISSION_FILE = path
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                submission.main()
            except SystemExit:
                pass
            except Exception as e:  # noqa: BLE001 - on journalise tout et on continue
                print(f"EXCEPTION {e!r}")
        out = buf.getvalue().replace("\n", " ").replace('"', "'")
        with open("submissions_log.csv", "a", encoding="utf-8") as f:
            f.write(f'{datetime.datetime.now():%H:%M:%S},{path},,ms_auto,ok,,"{out}"\n')
        m = re.search(r"Score \(public split\): (-?[0-9.]+)", out)
        if m:
            log(f"SCORE {path} = {m.group(1)}")
            return float(m.group(1))
        w = re.search(r"Wait (\d+) seconds", out)
        if w:
            log(f"429 sur {path} : attente {w.group(1)} s")
            time.sleep(int(w.group(1)) + 20)
            continue
        if "Could not reach" in out:
            log(f"serveur injoignable pour {path} : attente 600 s puis même fichier")
            time.sleep(600)
            continue
        log(f"ÉCHEC {path} : {out[:300]}")
        return None


def stable(path: str, age: int = 90) -> bool:
    """Le fichier existe et n'a pas été modifié depuis `age` secondes (écriture terminée)."""
    return os.path.exists(path) and time.time() - os.path.getmtime(path) > age


def wait_files(paths: list, timeout: int) -> bool:
    t0 = time.time()
    while not all(stable(p) for p in paths):
        if time.time() - t0 > timeout or too_late():
            log(f"délai dépassé en attendant {[p for p in paths if not stable(p)]}")
            return False
        time.sleep(POLL)
    return True


def soup(files: list, alpha: float, out: str) -> str:
    """soupe(m, α) = θ_0 + α ((1/m) Σ θ_i − θ_0), poids float32 finis (contraintes du serveur)."""
    o = torch.load("model.pt", map_location="cpu")
    acc = {k: torch.zeros_like(o[k], dtype=torch.float32) for k in o}
    for f in files:
        v = torch.load(f, map_location="cpu")
        assert v.keys() == o.keys(), f"clés différentes : {f}"
        for k in o:
            acc[k] += v[k].float() / len(files)
    s = {k: (o[k].float() + alpha * (acc[k] - o[k].float())).contiguous() for k in o}
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in s.values())
    torch.save(s, out)
    log(f"soupe de {len(files)} modèles, α = {alpha} -> {out}")
    return out


def scored(files: list, alpha: float, tag: str, cache: dict):
    """Score de soupe(files, α), avec cache (aucune soumission en double)."""
    key = (tag, alpha)
    if key not in cache:
        out = f"sub_ms_auto_{tag}_a{alpha}.pt"
        cache[key] = known_score(out)
        if cache[key] is None:
            cache[key] = submit(soup(files, alpha, out))
    return cache[key]


def alpha_search(files: list, tag: str, s15, cache: dict):
    """Recherche locale de α autour de 1.5 (pas 0.25), poussée dans le sens d'un bord gagnant."""
    cache[(tag, 1.5)] = s15
    for a in (1.25, 1.75):
        scored(files, a, tag, cache)
    res = {a: v for (t, a), v in cache.items() if t == tag and v is not None}
    best = max(res, key=res.get)
    while best in (min(res), max(res)) and best != 1.5 and not too_late():
        nxt = round(best + (0.25 if best == max(res) else -0.25), 2)
        if not 0.75 <= nxt <= 2.5:
            break
        v = scored(files, nxt, tag, cache)
        if v is None or v <= res[best]:
            break
        res[nxt] = v
        best = nxt
    log(f"α retenu pour {tag} : {best} (score {res[best]}) ; essais : {dict(sorted(res.items()))}")
    return best, res[best]


def launch_seeds(seeds: list) -> list:
    """Soumet un job Slurm de 4 graines de la recette v13_v3 ; renvoie les fichiers attendus."""
    os.makedirs("output", exist_ok=True)
    lines = ["#!/bin/bash", "#SBATCH --account=training2666", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
             "#SBATCH --gres=gpu:4", "#SBATCH --cpus-per-task=128", "#SBATCH --partition=dc-gpu",
             "#SBATCH --reservation=cispahack", "#SBATCH --time=00:12:00", "#SBATCH --output=output/%j_ms_auto_seeds.out",
             f"cd {UL}", "source .venv/bin/activate"]
    for i, s in enumerate(seeds):
        lines.append(f'CUDA_VISIBLE_DEVICES={i} python -u unlearn.py --gold "gold_v13a/*.pt" {COMMON} --seed {s} '
                     f"--out sub_ms_v13_s{s}.pt > output/ms_auto_v13_s{s}.log 2>&1 &")
    lines.append("wait")
    path = "output/job_ms_auto_seeds.sh"
    open(path, "w").write("\n".join(lines) + "\n")
    r = subprocess.run(["sbatch", path], capture_output=True, text=True)
    log(f"sbatch graines {seeds} : {r.stdout.strip()} {r.stderr.strip()}")
    return [f"sub_ms_v13_s{s}.pt" for s in seeds]


def other_pilot_running() -> bool:
    """Verrou : un autre processus ms_auto.py tourne déjà (le nôtre exclu)."""
    r = subprocess.run(["pgrep", "-f", "ms_auto.py"], capture_output=True, text=True)
    return any(int(p) != os.getpid() for p in r.stdout.split())


def main() -> None:
    os.chdir(UL)
    sys.path.insert(0, UL)
    if other_pilot_running():
        log("un autre pilote ms_auto.py tourne déjà : arrêt de ce lancement")
        return
    log("démarrage du pilote automatique")
    while other_loop_running():
        log("une autre boucle de soumission tourne : attente")
        time.sleep(POLL)
    cache = {}
    two = ["sub_v13_v3.pt", "sub_ms_f2_v13_s1.pt"]
    s2 = known_score("sub_ms_soup_v13_s01_a1.5.pt")
    cache[("soup2", 1.5)] = s2 if s2 is not None else submit("sub_ms_soup_v13_s01_a1.5.pt")
    s2 = cache[("soup2", 1.5)]
    log(f"soupe à 2 (α = 1.5) : {s2}")

    six = two + [f"sub_ms_v13_s{s}.pt" for s in (2, 3, 4, 5)]
    s6 = None
    if wait_files(six[2:], timeout=5400):
        s6 = scored(six, 1.5, "soup6", cache)
    log(f"soupe à 6 (α = 1.5) : {s6}")

    cands = [(v, t, f) for v, t, f in ((s2, "soup2", two), (s6, "soup6", six)) if v is not None]
    if not cands:
        log("aucune soupe notée : arrêt")
        return
    base_s, tag, files = max(cands)
    a_best, s_best = alpha_search(files, tag, base_s, cache)

    if s6 is not None and s2 is not None and s6 > s2 + 0.005 and not too_late():
        more = launch_seeds([6, 7, 8, 9])
        if wait_files(more, timeout=5400):
            s10 = scored(six + more, a_best, "soup10", cache)
            log(f"soupe à 10 (α = {a_best}) : {s10}")
    best = max(((v, k) for k, v in cache.items() if v is not None), default=None)
    log(f"FIN. Meilleur essai du pilote : {best}. Tous les essais : {cache}")


if __name__ == "__main__":
    main()
