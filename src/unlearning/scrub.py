"""SCRUB (Kurmanji et al., NeurIPS 2023) adapté au challenge : pas de retain set, pas de labels.

Ce que le papier fait                         -> adaptation ici
  teacher = original, student initialisé dessus -> idem
  max-step : epoch sur Df, maximise KL(T||S)    -> --max_mode scrub  (KL ascent, à T = --kd_T)
                                                  --max_mode target (distille vers la cible "jamais vue", nos v4-v6)
  min-step : epoch sur Dr, a*KL(T||S) + g*CE    -> retain set indisponible : proxy = vues fortement modifiées
                                                  et/ou mixups des images forget (--retain_src), CE sur argmax teacher (--gamma)
  alternance max/min pendant MAX epochs, puis   -> --max_epochs, --epochs (total) ; lr décroît x0.1 à la fin des max
  min-steps supplémentaires
  SCRUB+R : rewind au checkpoint dont l'erreur  -> référence = stats de la cible calculée sur vues fortes (proxy de
  forget ~ erreur sur un val set iid             l'erreur "jamais vue") ; on mesure exactement le critère du serveur :
                                                  d = |mean CE - ref_mean| + |std CE - ref_std| sur les images propres
                                                  et on garde le checkpoint qui minimise d.

Usage :
  python -u scrub.py --max_mode scrub  --out sub_scrub.pt
  python -u scrub.py --max_mode target --out sub_scrubT.pt
"""
import argparse
import copy
import time

import torch
import torch.nn.functional as F

import unlearn as U
from task_template import DEVICE, load_forget_images, load_model, predict


def ce_vec(logits, y):
    return F.cross_entropy(logits, y, reduction="none")


@torch.no_grad()
def eval_ckpt(student, teacher, imgs, y, q, ref_mean, ref_std):
    student.eval()
    logits = predict(student, imgs)
    ce = ce_vec(logits, y)
    d = abs(ce.mean().item() - ref_mean) + abs(ce.std().item() - ref_std)
    kl_q = F.kl_div(F.log_softmax(logits, 1), q, reduction="batchmean").item()
    acc = (logits.argmax(1) == y).float().mean().item()
    rp = U.retain_proxy(teacher, student, imgs, n=512)
    student.train()
    return dict(acc=acc, mean=ce.mean().item(), std=ce.std().item(), d=d, kl_q=kl_q, rp=rp)


def retain_batch(x, src, strong_aug, synth=None):
    """Proxy du retain set (images forget modifiées, mixups, ou images synthétiques sans contenu forget)."""
    parts = []
    if src in ("synth", "synth+aug"):
        sidx = torch.randint(0, len(synth), (len(x),))
        parts.append(U.norm(torch.stack([U.AUG(xi) for xi in synth[sidx]])))
    if src == "synth+aug":
        parts.append(U.norm(torch.stack([strong_aug(xi) for xi in x])))
    if src in ("aug", "both"):
        parts.append(U.norm(torch.stack([strong_aug(xi) for xi in x])))
    if src in ("mix", "both"):
        parts.append(U.norm(U.mix_batch(x)))
    return torch.cat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_mode", default="scrub", choices=["scrub", "target"])
    ap.add_argument("--kd_T", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_epochs", type=int, default=4, help="nb d'epochs avec max-step")
    ap.add_argument("--epochs", type=int, default=8, help="nb total d'epochs (les dernières = min-steps seuls)")
    ap.add_argument("--bs_f", type=int, default=32, help="batch forget (max-step)")
    ap.add_argument("--bs_r", type=int, default=64, help="batch retain proxy (min-step)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0, help="CE sur argmax teacher dans le min-step")
    ap.add_argument("--retain_src", default="both", choices=["aug", "mix", "both", "synth", "synth+aug"])
    ap.add_argument("--synth", type=int, default=10, help="images synthétiques par classe (retain_src synth)")
    ap.add_argument("--rewind_lam", type=float, default=0.0, help="rewind : minimise d + lam * retain_proxy")
    ap.add_argument("--t_smin", type=float, default=0.35)
    ap.add_argument("--t_smax", type=float, default=0.6)
    ap.add_argument("--t_blur", type=float, default=0.0)
    ap.add_argument("--views", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=40, help="pas entre deux checkpoints pour le rewind")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sub_scrub.pt")
    args = ap.parse_args()
    print(vars(args))
    torch.manual_seed(args.seed)
    t0 = time.time()

    teacher = load_model().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    imgs = load_forget_images()
    N = len(imgs)
    y = predict(teacher, imgs).argmax(1)

    # référence "jamais vue" (proxy du val set iid de SCRUB+R)
    U.TEACHER_AUG = U.make_teacher_aug(args.t_smin, args.t_smax, args.t_blur)
    q = U.teacher_targets(teacher, imgs, args.views)
    ce_q = -q.clamp_min(1e-8).log().gather(1, y[:, None]).squeeze(1)
    ref_mean, ref_std = ce_q.mean().item(), ce_q.std().item()
    print(f"référence : acc={(q.argmax(1) == y).float().mean():.3f}  CE mean={ref_mean:.3f}  std={ref_std:.3f}"
          f"  [{time.time() - t0:.0f}s]")
    strong_aug = U.TEACHER_AUG
    synth = U.synth_impressions(teacher, args.synth) if args.retain_src.startswith("synth") else None

    student = copy.deepcopy(teacher).train()
    for p in student.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)

    ckpts = []  # (stats, state_dict cpu)
    step = 0

    def maybe_ckpt(tag):
        st = eval_ckpt(student, teacher, imgs, y, q, ref_mean, ref_std)
        st["tag"] = tag
        ckpts.append((st, {k: v.detach().cpu().float().clone() for k, v in student.state_dict().items()}))
        print(f"  [{tag}] acc={st['acc']:.3f} CE={st['mean']:.3f}±{st['std']:.3f}  d={st['d']:.3f}"
              f"  KL(q||s)={st['kl_q']:.3f}  retain_proxy={st['rp']:.3f}  [{time.time() - t0:.0f}s]")

    maybe_ckpt("init")
    T = args.kd_T
    for ep in range(args.epochs):
        if ep == args.max_epochs:
            for g in opt.param_groups:
                g["lr"] = args.lr * 0.1
        # ---- max-step epoch (forget) ----
        if ep < args.max_epochs:
            perm = torch.randperm(N)
            for i in range(0, N, args.bs_f):
                idx = perm[i:i + args.bs_f]
                x = U.to_float(imgs[idx])
                xin = torch.cat([U.norm(x), U.norm(U.aug_batch(x))])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s = student(xin).float()
                if args.max_mode == "scrub":
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        t = teacher(xin).float()
                    loss = -F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1), reduction="batchmean") * T * T
                else:
                    qt = q[idx].to(DEVICE).repeat(2, 1)
                    loss = F.kl_div(F.log_softmax(s, 1), qt, reduction="batchmean")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step(); step += 1
                if step % args.eval_every == 0:
                    maybe_ckpt(f"ep{ep} max s{step}")
        # ---- min-step epoch (retain proxy) ----
        perm = torch.randperm(N)
        for i in range(0, N, args.bs_r):
            idx = perm[i:i + args.bs_r]
            x = U.to_float(imgs[idx])
            xr = retain_batch(x, args.retain_src, strong_aug, synth)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t = teacher(xr).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = student(xr).float()
            loss = args.alpha * F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1), reduction="batchmean") * T * T
            if args.gamma > 0:
                loss = loss + args.gamma * F.cross_entropy(s, t.argmax(1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); step += 1
            if step % args.eval_every == 0:
                maybe_ckpt(f"ep{ep} min s{step}")
        maybe_ckpt(f"fin ep{ep}")

    # ---- SCRUB+R : rewind ----
    best_st, best_sd = min(ckpts, key=lambda c: c[0]["d"] + args.rewind_lam * c[0]["rp"])
    print(f"REWIND -> {best_st['tag']}  acc={best_st['acc']:.3f} CE={best_st['mean']:.3f}±{best_st['std']:.3f}"
          f"  d={best_st['d']:.3f} (init d={ckpts[0][0]['d']:.3f})  retain_proxy={best_st['rp']:.3f}")
    assert all(torch.isfinite(v).all() for v in best_sd.values()), "NaN/Inf"
    torch.save(best_sd, args.out)
    print(f"Sauvé -> {args.out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
