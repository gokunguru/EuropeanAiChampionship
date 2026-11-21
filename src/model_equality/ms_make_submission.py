"""Fabrique un CSV de soumission endpoint,family,model et vérifie les règles du format avant envoi.

Règles (énoncé + précision Discord du 21/11, 18 h 53) :
  - une ligne par endpoint ep-01 ... ep-33, trois colonnes, pas de virgule dans les étiquettes ;
  - une étiquette de modèle n'appartient qu'à une seule famille (sinon le fichier est refusé, avec le délai de 30 min).
On nomme toujours les modèles « lettre de la famille + numéro » (A1, A2, B1…) : la seconde règle est alors vraie par
construction, et elle est revérifiée quand même.

Usage :
    python ms_make_submission.py familles out.csv     # familles d'après ms_tok_probe_*.jsonl, un modèle par endpoint
"""
import csv
import json
import sys
from pathlib import Path

ENDPOINTS = [f"ep-{i:02d}" for i in range(1, 34)]


def families_from_tokenizer(folder: Path = Path(".")) -> dict:
    """Famille d'après les textes frontières : emoji accepté -> Qwen (A) ; chiffres accepté -> Llama 3 (B) ; sinon C."""
    status = {}
    for name in ("emoji", "chiffres"):
        for line in (folder / f"ms_tok_probe_{name}.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            status[(r["endpoint"], name)] = r["status"]
    fam = {}
    for e in ENDPOINTS:
        emo, chi = status.get((e, "emoji")), status.get((e, "chiffres"))
        fam[e] = "A" if emo == "accepté" else ("B" if chi == "accepté" else "C")
    return fam


def check(rows: list) -> None:
    """Vérifie le format ; lève une erreur au lieu de laisser le serveur refuser (et déclencher 30 min d'attente)."""
    eps = [r[0] for r in rows]
    assert sorted(eps) == ENDPOINTS, "il faut exactement une ligne par endpoint ep-01 ... ep-33"
    owner = {}
    for e, f, m in rows:
        assert f and m and "," not in f + m, f"étiquette vide ou avec virgule : {e}"
        assert owner.setdefault(m, f) == f, f"le modèle {m} apparaît dans deux familles"


def write(rows: list, path: str) -> None:
    check(rows)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model"])
        w.writerows(rows)
    fams = {}
    for _, fa, m in rows:
        fams.setdefault(fa, set()).add(m)
    print(f"{path} écrit : " + ", ".join(f"famille {k} = {sum(1 for r in rows if r[1] == k)} endpoints, "
                                         f"{len(v)} modèles" for k, v in sorted(fams.items())))


def main() -> None:
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "familles":
        fam = families_from_tokenizer()
        count = {}
        rows = []
        for e in ENDPOINTS:
            count[fam[e]] = count.get(fam[e], 0) + 1
            rows.append((e, fam[e], f"{fam[e]}{count[fam[e]]}"))
        write(rows, out)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
