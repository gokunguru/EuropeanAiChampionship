"""Surrogate-gold : des modèles qui n'ont JAMAIS vu les images forget, pour estimer p_gold sur ces images.

Idée : sur des images externes, l'original se comporte déjà comme un gold (il ne les a pas mémorisées).
On clone l'original, on RÉINITIALISE ses derniers blocs + la tête (là où se loge l'essentiel de la mémorisation),
puis on les réentraîne par distillation de l'original (KL à T=2) UNIQUEMENT sur les images externes.
Le modèle obtenu retrouve les frontières de classes de l'original mais n'a jamais vu D_f : ses prédictions
sur D_f imitent celles d'un modèle réentraîné sans D_f, y compris QUELLE mauvaise classe il choisit.

Sortie : gold_<tag>.pt = logits [3500, 100] sur les images forget propres (+ stats).
Usage : python -u surrogate.py --ext ext_retain.npy --reinit_from 8 --seed 0 --tag k8s0
"""
import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import init_weights_vit_timm

import unlearn as U
from task_template import DEVICE, load_forget_images, load_model, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default="ext_retain.npy")
    ap.add_argument("--reinit_from", type=int, default=8, help="réinitialise blocks[k:] + head")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--shrink", type=float, default=0.0,
                    help=">0 : au lieu de réinitialiser, 'shrink & perturb' (Ash & Adams) : w = shrink*w + (1-shrink)*w_init."
                         " Garde une partie des connaissances (convergence bien plus rapide) tout en effaçant le détail mémorisé")
    ap.add_argument("--shrink_low", type=float, default=0.0,
                    help=">0 : S&P doux aussi sur patch_embed + blocks[:k] (w = a_low*w + (1-a_low)*w_init), tout le réseau est réentraîné")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="s0")
    args = ap.parse_args()
    print(vars(args))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    t0 = time.time()

    teacher = load_model().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    ext = np.load(args.ext, mmap_mode="r")
    ho = min(1000, len(ext) // 5)
    print(f"ext : {ext.shape}, holdout {ho}")
    forget = load_forget_images()
    y = predict(teacher, forget).argmax(1)

    s = copy.deepcopy(teacher)
    for p in s.parameters():
        p.requires_grad_(False)
    mods = list(s.blocks[args.reinit_from:]) + [s.head]
    keep = [copy.deepcopy(m.state_dict()) for m in mods]
    for m in mods:
        m.apply(init_weights_vit_timm)
    for blk in s.blocks[args.reinit_from:]:
        blk.ls1.gamma.data.fill_(1e-4); blk.ls2.gamma.data.fill_(1e-4)
    if args.shrink > 0:
        a = args.shrink
        for m, sd in zip(mods, keep):
            for k, v in m.state_dict().items():
                v.copy_(a * sd[k] + (1 - a) * v)
        print(f"shrink & perturb : {a:.2f} * original + {1 - a:.2f} * init aléatoire")
    if args.shrink_low > 0:
        a = args.shrink_low
        low = [s.patch_embed] + list(s.blocks[:args.reinit_from])
        for m in low:
            sd = copy.deepcopy(m.state_dict())
            m.apply(init_weights_vit_timm)
            if hasattr(m, "ls1"):
                m.ls1.gamma.data.fill_(1e-4); m.ls2.gamma.data.fill_(1e-4)
            for k, v in m.state_dict().items():
                v.copy_(a * sd[k] + (1 - a) * v)
        for m in low:
            for p in m.parameters():
                p.requires_grad_(True)
        print(f"S&P bas : {a:.2f} * original sur patch_embed + blocks[:{args.reinit_from}]")
    for blk in s.blocks[args.reinit_from:]:
        for p in blk.parameters():
            p.requires_grad_(True)
    for m in (s.norm, s.head):
        for p in m.parameters():
            p.requires_grad_(True)
    params = [p for p in s.parameters() if p.requires_grad]
    print(f"paramètres réinitialisés/entraînés : {sum(p.numel() for p in params) / 1e6:.1f}M")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    steps_per_ep = (len(ext) - ho) // args.bs
    total = args.epochs * steps_per_ep
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(U.AMP_DTYPE == torch.float16))

    def batch(idx):
        x = U.to_float(torch.from_numpy(np.ascontiguousarray(ext[np.sort(idx)])))
        return U.norm(U.aug_batch(x))

    step = 0
    s.train()
    for ep in range(args.epochs):
        perm = np.random.permutation(np.arange(ho, len(ext)))
        for i in range(steps_per_ep):
            x = batch(perm[i * args.bs:(i + 1) * args.bs])
            with torch.no_grad(), torch.autocast("cuda", dtype=U.AMP_DTYPE):
                t = teacher(x).float()
            with torch.autocast("cuda", dtype=U.AMP_DTYPE):
                o = s(x).float()
            loss = F.kl_div(F.log_softmax(o / 2, 1), F.softmax(t / 2, 1), reduction="batchmean") * 4
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % 50 == 0:
                print(f"  ep {ep} step {step}/{total}  kd={loss.item():.4f}  [{time.time() - t0:.0f}s]", flush=True)

    s.eval()
    # fidélité à l'original sur images externes jamais vues (doit être élevée)
    xh = torch.from_numpy(np.ascontiguousarray(ext[:ho]))
    lo, ls_ = predict(teacher, xh), predict(s, xh)
    agree = (lo.argmax(1) == ls_.argmax(1)).float().mean().item()
    kl = F.kl_div(F.log_softmax(ls_ / 2, 1), F.softmax(lo / 2, 1), reduction="batchmean").item()
    print(f"ext holdout : accord avec l'original = {agree:.3f}  KL T=2 = {kl:.4f}")

    # prédictions du surrogate sur les images forget (il ne les a jamais vues), moyennées sur (x, hflip x) à T=2
    z = predict(s, forget)
    zf = predict(s, torch.from_numpy(np.ascontiguousarray(np.asarray(forget)[:, :, ::-1])))
    p2 = 0.5 * (F.softmax(z / 2, 1) + F.softmax(zf / 2, 1))
    zavg = 2 * p2.clamp_min(1e-8).log()
    ce = F.cross_entropy(z, y, reduction="none")
    acc = (z.argmax(1) == y).float().mean().item()
    print(f"forget (jamais vues par ce surrogate) : acc={acc:.3f}  CE mean={ce.mean():.3f}  std={ce.std():.3f}")

    # critère de fuite : Df vs sous-ensemble ext holdout apparié par classe (classe = argmax orig)
    ye = lo.argmax(1).to(y.device)
    ce_e = F.cross_entropy(ls_.to(y.device), ye, reduction="none")
    fd = torch.bincount(y, minlength=100).float(); fe = torch.bincount(ye, minlength=100).float()
    w = (fd / fd.sum())[ye] / (fe / fe.sum())[ye].clamp_min(1e-8)
    g = torch.Generator(device="cpu").manual_seed(0)
    ref = ce_e.cpu()[torch.multinomial(w.cpu(), len(y), replacement=True, generator=g)]
    a_, b_ = ce.cpu().sort().values, ref.sort().values
    w1 = (a_ - b_).abs().mean().item()
    leak = abs(ce.mean().item() - ref.mean().item()) + abs(ce.std().item() - ref.std().item()) + w1
    mf = (z.topk(2, 1).values[:, 0] - z.topk(2, 1).values[:, 1])
    me = (ls_.topk(2, 1).values[:, 0] - ls_.topk(2, 1).values[:, 1])
    print(f"fuite : CE Df {ce.mean():.3f}±{ce.std():.3f} vs ext apparié {ref.mean():.3f}±{ref.std():.3f}  W1={w1:.3f}"
          f"  DELTA_LEAK={leak:.3f}  | marges>8 : Df {(mf > 8).float().mean():.3f} ext {(me > 8).float().mean():.3f}")
    out = f"{args.outdir}/gold_{args.tag}.pt"
    torch.save({"logits": zavg, "logits_noflip": z, "agree_ext": agree, "kl_ext": kl, "acc_forget": acc, "leak": leak,
                "ce_mean": ce.mean().item(), "ce_std": ce.std().item(), "args": vars(args)}, out)
    print(f"Sauvé -> {out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
