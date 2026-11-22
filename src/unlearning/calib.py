"""Étapes 1+2+3 du plan, sans soumission (1 GPU, ~5 min).

(2) Calibration gold : l'original n'a JAMAIS vu ext. On mappe chaque classe de l'original vers le synset mini-ImageNet
    majoritaire (pureté >= --purity, bijectif), puis on mesure la CE de l'original avec les VRAIS labels sur ext
    -> (mu*, sigma*, acc*) = comportement d'un modèle face à des images non vues = estimation de gold sur Df.
(1) Décomposition des soumissions : sur ext[:1000] (jamais utilisé en entraînement ni en rétention) :
    accord top-1, KL(orig||sub) T=2, acc vraie (labels mappés) -> rétention estimée ; si score public connu,
    oubli estimé = score / rétention.
(3) Classement des candidats : chaque .pt évalué pour plusieurs alpha (interpolation EN MÉMOIRE, aucun fichier écrit) :
    CE Df mean/std vs (mu*, sigma*) -> mean_std estimé ; x rétention estimée ; contrôle de la barre des 70 %.
    Les cibles gold (*.pt avec 'logits') sont aussi comparées à (mu*, sigma*).

Usage : python -u calib.py --scores scores.txt --alphas 1.0,1.15,1.3 --subs sub_*.pt --golds gold_*.pt gold_v3/*.pt gold_combo/*.pt
  scores.txt : une ligne "fichier.pt score_public" (facultatif)
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from task_template import build_model, load_forget_images, load_model, predict


def pred_chunks(model, arr, bs=2000):
    return torch.cat([predict(model, torch.from_numpy(np.ascontiguousarray(arr[i:i + bs]))).float().cpu() for i in range(0, len(arr), bs)])


def ce_stats(z, y):
    ce = F.cross_entropy(z, y, reduction="none")
    return ce.mean().item(), ce.std().item(), (z.argmax(1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default="ext_retain.npy")
    ap.add_argument("--labels", default="ext_labels.npy")
    ap.add_argument("--purity", type=float, default=0.5)
    ap.add_argument("--ho", type=int, default=1000)
    ap.add_argument("--alphas", default="1.0,1.15,1.3")
    ap.add_argument("--scores", default="")
    ap.add_argument("--subs", nargs="*", default=[])
    ap.add_argument("--golds", nargs="*", default=[])
    args = ap.parse_args()
    print(vars(args))

    orig = load_model().eval()
    ext = np.load(args.ext, mmap_mode="r")
    lab = np.load(args.labels)
    forget = load_forget_images()
    y_f = predict(orig, forget).float().cpu().argmax(1)

    # ---------- (2) calibration ----------
    if os.path.exists("ext_orig_logits.pt"):
        ze = torch.load("ext_orig_logits.pt")
    else:
        ze = pred_chunks(orig, ext); torch.save(ze, "ext_orig_logits.pt")
    pe = ze.argmax(1).numpy()
    mp = np.arange(len(ext)) >= args.ho          # mapping appris hors holdout
    cand = []
    for c in range(100):
        s = lab[mp & (pe == c)]
        if len(s) < 20:
            continue
        v, cnt = np.unique(s, return_counts=True)
        cand.append((cnt.max() / len(s), c, int(v[cnt.argmax()]), len(s)))
    cls2syn, used = {}, set()
    for pur, c, syn, nn in sorted(cand, reverse=True):
        if pur >= args.purity and syn not in used:
            cls2syn[c] = syn; used.add(syn)
    syn2cls = {s: c for c, s in cls2syn.items()}
    ytrue = torch.tensor([syn2cls.get(int(s), -1) for s in lab])
    purs = [p for p, c, _, _ in cand if c in cls2syn]
    print(f"\n=== (2) CALIBRATION GOLD : {len(cls2syn)}/100 classes mappées (pureté moy {np.mean(purs):.3f}, min {np.min(purs):.3f})")
    keep = ytrue >= 0
    for name, m in (("ext complet", keep), ("ext holdout", keep & (torch.arange(len(ext)) < args.ho))):
        mu, sd, acc = ce_stats(ze[m], ytrue[m])
        ce = F.cross_entropy(ze[m], ytrue[m], reduction="none")
        q = torch.quantile(ce, torch.tensor([0.5, 0.75, 0.9, 0.95]))
        print(f"  {name:<12} n={int(m.sum()):>5}  acc*={acc:.3f}  CE mu*={mu:.3f}  sigma*={sd:.3f}"
              f"  quantiles 50/75/90/95 = {', '.join(f'{x:.2f}' for x in q)}")
    MU, SD, ACC = ce_stats(ze[keep], ytrue[keep])
    print(f"  conf max moyenne (correct / faux) : {F.softmax(ze[keep],1).max(1).values[ze[keep].argmax(1)==ytrue[keep]].mean():.3f}"
          f" / {F.softmax(ze[keep],1).max(1).values[ze[keep].argmax(1)!=ytrue[keep]].mean():.3f}")
    print(f"  -> CIBLE (ordre de grandeur) : CE Df ≈ {MU:.2f} ± {SD:.2f}, acc ≈ {ACC:.2f}")

    def ms_score(mu, sd):
        return max(0.0, 1 - (abs(mu - MU) + abs(sd - SD)) / (MU + SD))

    if args.golds:
        print("\n=== cibles gold sur Df (CE vs argmax orig) — plus proche de (mu*, sigma*) = mieux")
        rows = []
        for f in args.golds:
            d = torch.load(f, map_location="cpu")
            if not isinstance(d, dict) or "logits" not in d:
                continue
            mu, sd, acc = ce_stats(d["logits"].float().cpu(), y_f)
            rows.append((ms_score(mu, sd), f, mu, sd, acc))
        for s, f, mu, sd, acc in sorted(rows, reverse=True):
            print(f"  {f:<45} acc={acc:.3f}  CE {mu:.3f}±{sd:.3f}  mean_std estimé={s:.3f}")

    # ---------- (1) + (3) ----------
    scores = {}
    if args.scores and os.path.exists(args.scores):
        for line in open(args.scores):
            p = line.split()
            if len(p) >= 2:
                scores[p[0]] = float(p[1])
    hoi = np.arange(args.ho)
    xh = ext[:args.ho]
    zo_h = ze[:args.ho]
    yh = ytrue[:args.ho]; mh = yh >= 0
    acc_o = (zo_h[mh].argmax(1) == yh[mh]).float().mean().item()
    osd = torch.load("model.pt", map_location="cpu", weights_only=True)
    alphas = [float(a) for a in args.alphas.split(",")]
    m = build_model().cuda().eval()
    rows = []
    print(f"\n=== (1)+(3) SOUMISSIONS (holdout ext n={args.ho}, acc vraie orig={acc_o:.3f})")
    for f in args.subs:
        sd_ = torch.load(f, map_location="cpu", weights_only=True)
        alist = [1.0] if "_a" in os.path.basename(f) else alphas   # fichier déjà extrapolé : tel quel
        for a in alist:
            w = {k: (osd[k].float() + a * (sd_[k].float() - osd[k].float())) for k in osd}
            m.load_state_dict(w, strict=True)
            zs = pred_chunks(m, xh)
            agree = (zs.argmax(1) == zo_h.argmax(1)).float().mean().item()
            kl = F.kl_div(F.log_softmax(zs / 2, 1), F.softmax(zo_h / 2, 1), reduction="batchmean").item()
            acc_s = (zs[mh].argmax(1) == yh[mh]).float().mean().item()
            ret_agree = agree / (1 + kl)
            ret_true = min(1.0, acc_s / acc_o) / (1 + kl)
            mu, sd, accf = ce_stats(predict(m, forget).float().cpu(), y_f)
            msv = ms_score(mu, sd)
            pub = scores.get(f) if a == 1.0 else None
            rows.append((msv * ret_true, f, a, agree, kl, acc_s, ret_agree, ret_true, mu, sd, accf, msv, pub))
    print(f"{'fichier':<40}{'alpha':>6}{'accord':>8}{'KL':>7}{'accV':>7}{'ret(acc)':>9}{'retV':>7}"
          f"{'CE Df':>14}{'accDf':>7}{'ms_est':>7}{'ms×ret':>8}{'public':>8}{'oubli=pub/ret':>14}")
    for r in sorted(rows, key=lambda r: -r[0]):
        est, f, a, agree, kl, acc_s, ra, rt, mu, sd, accf, msv, pub = r
        gate = "" if acc_s / acc_o >= 0.85 else "  <- BARRE 70 % ?"
        pubs = f"{pub:8.4f}{pub / rt:14.4f}" if pub is not None else f"{'':8}{'':14}"
        print(f"{f:<40}{a:6.2f}{agree:8.3f}{kl:7.3f}{acc_s:7.3f}{ra:9.3f}{rt:7.3f}{mu:8.3f}±{sd:.3f}{accf:7.3f}"
              f"{msv:7.3f}{est:8.3f}{pubs}{gate}")
    with open("candidates_ranked.txt", "w") as fo:
        for r in sorted(rows, key=lambda r: -r[0]):
            fo.write(f"{r[1]} {r[2]:.2f} calib_est={r[0]:.3f}\n")
    print("\n-> candidates_ranked.txt (format directement utilisable comme file de soumission)")


if __name__ == "__main__":
    main()
