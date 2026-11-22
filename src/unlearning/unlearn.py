"""Unlearning sans retain set : distillation vers une cible "jamais vue" + préservation via mixup.

Idée
  - gold n'a jamais vu les images forget -> sur ces images il se comporte comme sur des données de test :
    moins confiant, et parfois il se trompe (vers une classe plausible).
  - Proxy de cette cible : moyenne des prédictions du modèle original sur des vues fortement augmentées
    (la mémorisation est fragile aux perturbations), puis aplatie par --tau.
  - Préservation (retain) : KL(original || élève) à T=2 sur des mixups/augmentations d'images, qui ne sont
    pas des échantillons mémorisés -> l'original y "généralise", on veut garder ça.
  - Les vues augmentées pendant l'entraînement couvrent aussi les "hidden altered copies" du scoring.

Usage (depuis le dossier unlearning/) :
  python -u unlearn.py --diag                       # stats seulement, pas d'entraînement
  python -u unlearn.py --tau 1.5 --epochs 3 --out sub_tau1.5.pt
"""
import argparse
import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from task_template import NUM_CLASSES, DEVICE, load_forget_images, load_model, predict

# bf16 sur A100 (JURECA) ; fp16 + GradScaler sur T4/P100 (Kaggle), qui n'ont pas de bf16 natif
AMP_DTYPE = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)

MILD = v2.Compose([  # "slightly altered versions" : petites perturbations
    v2.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(0.1, 0.1, 0.1, 0.02),
])

AUG = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.35, 1.0), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(0.3, 0.3, 0.3, 0.05),
])


def make_teacher_aug(smin, smax, blur):
    t = [v2.RandomResizedCrop(224, scale=(smin, smax), antialias=True),
         v2.RandomHorizontalFlip(),
         v2.ColorJitter(0.4, 0.4, 0.4, 0.1),
         v2.RandomGrayscale(0.2)]
    if blur > 0:
        t.append(v2.RandomApply([v2.GaussianBlur(23, sigma=(0.1, blur))], p=0.5))
    return v2.Compose(t)


TEACHER_AUG = AUG


def to_float(u8):
    """uint8 [B,H,W,3] -> float [B,3,H,W] in [0,1] on GPU."""
    return u8.to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0


def norm(x):
    return (x - 0.5) / 0.5  # same as server preprocess()


def aug_batch(x):
    return torch.stack([AUG(xi) for xi in x])


def mix_batch(x):
    j = torch.randperm(len(x), device=x.device)
    lam = torch.distributions.Beta(1.0, 1.0).sample((len(x), 1, 1, 1)).to(x.device)
    return aug_batch(lam * x + (1 - lam) * x[j])


@torch.no_grad()
def teacher_targets(model, imgs, views, bs=256):
    model.eval()
    out = []
    for i in range(0, len(imgs), bs):
        x = to_float(imgs[i:i + bs])
        acc = torch.zeros(len(x), NUM_CLASSES, device=DEVICE)
        for _ in range(views):
            with torch.autocast("cuda", dtype=torch.float16):
                acc += model(norm(torch.stack([TEACHER_AUG(xi) for xi in x]))).float().softmax(1)
        out.append((acc / views).cpu())
    return torch.cat(out)


def synth_impressions(teacher, n_per_class, steps=100, lr=0.1, bs=100):
    """Zero-Shot Unlearning (Chundawat et al.) : images synthétiques que le teacher classe dans chaque classe.
    Servent de proxy de retain set SANS contenu des images forget."""
    teacher.eval()
    out = []
    labels = torch.arange(NUM_CLASSES).repeat(n_per_class)
    for i in range(0, len(labels), bs):
        c = labels[i:i + bs].to(DEVICE)
        z = (torch.randn(len(c), 3, 224, 224, device=DEVICE) * 0.5).requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(steps):
            x = torch.sigmoid(z)
            if torch.rand(1).item() < 0.5:
                x = x.flip(3)
            dx, dy = torch.randint(-8, 9, (2,)).tolist()
            x = torch.roll(x, shifts=(dx, dy), dims=(2, 3))
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                logits = teacher(norm(x)).float()
            tv = (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
            loss = F.cross_entropy(logits, c) + 0.5 * tv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        out.append(torch.sigmoid(z).detach())
    synth = torch.cat(out)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        pred = torch.cat([teacher(norm(synth[j:j + 128])).float().argmax(1) for j in range(0, len(synth), 128)])
    print(f"  synth : {len(synth)} images, teacher les classe comme visé : {(pred.cpu() == labels).float().mean():.3f}")
    return synth


def salun_mask(model, params, imgs, q, frac, bs=64):
    """SalUn (Fan et al.) : on ne modifie que les poids les plus 'saillants' pour l'oubli (top frac en |grad|)."""
    model.zero_grad(set_to_none=True)
    for i in range(0, len(imgs), bs):
        x = norm(to_float(imgs[i:i + bs]))
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            out = model(x).float()
        loss = F.kl_div(F.log_softmax(out, 1), q[i:i + bs].to(DEVICE), reduction="sum")
        loss.backward()
    grads = [p.grad.detach().abs().flatten() for p in params]
    allg = torch.cat(grads)
    k = max(1, int(len(allg) * (1 - frac)))
    thr = allg.kthvalue(k).values
    masks = [(p.grad.detach().abs() >= thr).float() for p in params]
    model.zero_grad(set_to_none=True)
    kept = sum(m.sum().item() for m in masks) / len(allg)
    print(f"  SalUn : {kept:.1%} des poids modifiables")
    return masks


def flatten(q, tau):
    q = q.clamp_min(1e-8) ** (1.0 / tau)
    return q / q.sum(1, keepdim=True)


def ce_stats(name, probs_or_logits, y, is_logits=True):
    logp = F.log_softmax(probs_or_logits, 1) if is_logits else probs_or_logits.clamp_min(1e-8).log()
    ce = -logp.gather(1, y[:, None]).squeeze(1)
    acc = (logp.argmax(1) == y).float().mean().item()
    print(f"  {name:<28} acc={acc:.3f}  CE mean={ce.mean():.3f}  std={ce.std():.3f}")


EXT = None  # images externes uint8 [N,224,224,3] (np.memmap), si --ext


@torch.no_grad()
def retain_proxy_ext(orig, sub, n=1024, bs=128):
    """KL(orig || sub) à T=2 sur de VRAIES images non vues (proxy honnête du retain/control)."""
    orig.eval(); sub.eval()
    ho = min(1000, len(EXT) // 5)  # les ho premières images ne servent jamais à l'entraînement
    idx = np.random.choice(ho, size=min(n, ho), replace=False)
    tot, agree = 0.0, 0.0
    for i in range(0, len(idx), bs):
        x = norm(to_float(torch.from_numpy(np.ascontiguousarray(EXT[np.sort(idx[i:i + bs])]))))
        with torch.autocast("cuda", dtype=torch.float16):
            lo, ls = orig(x).float(), sub(x).float()
        tot += F.kl_div(F.log_softmax(ls / 2, 1), F.softmax(lo / 2, 1), reduction="sum").item()
        agree += (lo.argmax(1) == ls.argmax(1)).sum().item()
    return tot / len(idx), agree / len(idx)


@torch.no_grad()
def retain_proxy(orig, sub, imgs, n=1024, bs=128):
    """Mean KL(orig || sub) at T=2 on fresh mixup images (proxy of the retain KL)."""
    orig.eval(); sub.eval()
    tot = 0.0
    idx = torch.randperm(len(imgs))[:n]
    for i in range(0, n, bs):
        xm = norm(mix_batch(to_float(imgs[idx[i:i + bs]])))
        with torch.autocast("cuda", dtype=torch.float16):
            po = F.softmax(orig(xm).float() / 2, 1)
            ls = F.log_softmax(sub(xm).float() / 2, 1)
        tot += F.kl_div(ls, po, reduction="sum").item()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=1.5, help=">1 aplatit la cible (moins confiante)")
    ap.add_argument("--views", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lam_retain", type=float, default=1.0)
    ap.add_argument("--lam_l2", type=float, default=0.0)
    ap.add_argument("--train_last", type=int, default=0, help="n'entraîne que les N derniers blocs + tête (0 = tout)")
    ap.add_argument("--t_smin", type=float, default=0.35, help="crop min des vues de la cible (plus petit = plus d'erreurs)")
    ap.add_argument("--t_smax", type=float, default=1.0)
    ap.add_argument("--t_blur", type=float, default=0.0, help="sigma max du flou sur les vues de la cible")
    ap.add_argument("--aug_mode", default="target", choices=["target", "orig", "none", "mild"],
                    help="vues modifiées : target = pousser vers la cible ; orig = rester comme l'original ; none = ignorer")
    ap.add_argument("--ext", default="", help="fichier .npy d'images externes pour la rétention")
    ap.add_argument("--lam_ext", type=float, default=1.0)
    ap.add_argument("--gold", default="", help="glob des cibles surrogate-gold (ex. 'gold_*.pt') : remplace la cible par augmentation")
    ap.add_argument("--lam_t1", type=float, default=0.0, help="terme auxiliaire KL à T=1 sur la cible gold (creuse la dispersion de CE)")
    ap.add_argument("--eps", type=float, default=0.0, help="mélange la cible avec l'uniforme (queue de distribution)")
    ap.add_argument("--salun", type=float, default=0.0, help="fraction de poids modifiables (SalUn), 0 = tous")
    ap.add_argument("--synth", type=int, default=0, help="images synthétiques par classe pour le retain (0 = aucune)")
    ap.add_argument("--lam_synth", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="submission.pt")
    ap.add_argument("--diag", action="store_true")
    args = ap.parse_args()
    print(vars(args))
    torch.manual_seed(args.seed)
    global TEACHER_AUG, EXT
    TEACHER_AUG = make_teacher_aug(args.t_smin, args.t_smax, args.t_blur)
    if args.ext:
        EXT = np.load(args.ext, mmap_mode="r")
        print(f"images externes : {EXT.shape}")
    t0 = time.time()

    model = load_model()
    imgs = load_forget_images()
    N = len(imgs)

    logits0 = predict(model, imgs)
    y = logits0.argmax(1)  # pseudo-labels (l'original a mémorisé -> très probablement les vrais labels)
    print("Original sur forget (pseudo-labels = argmax original) :")
    ce_stats("original, images propres", logits0, y)

    q2 = None
    if args.gold:
        import glob
        files = sorted(glob.glob(args.gold))
        zs = [torch.load(f, map_location="cpu")["logits"].float() for f in files]
        print(f"surrogate-gold : {len(zs)} modèles {files}")
        for f, z in zip(files, zs):
            ce = F.cross_entropy(z, y, reduction="none")
            print(f"  {f}: acc={(z.argmax(1) == y).float().mean():.3f} CE={ce.mean():.3f}±{ce.std():.3f}")
        ces = [F.cross_entropy(z, y, reduction="none") for z in zs]
        print(f"  référence gold (moy. des modèles) : CE mean={np.mean([c.mean().item() for c in ces]):.3f}"
              f"  std={np.mean([c.std().item() for c in ces]):.3f}")
        q2 = torch.stack([F.softmax(z / 2, 1) for z in zs]).mean(0)   # exactement p_gold du serveur (T=2)
        q_raw = torch.stack([F.softmax(z, 1) for z in zs]).mean(0)
    else:
        q_raw = teacher_targets(model, imgs, args.views)
    ce_stats("original, moyenne vues aug", q_raw, y, is_logits=False)
    q = flatten(q_raw, args.tau)
    if args.eps > 0:
        q = (1 - args.eps) * q + args.eps / NUM_CLASSES
    ce_stats(f"cible (tau={args.tau})", q, y, is_logits=False)
    print(f"  argmax cible != pseudo-label : {(q.argmax(1) != y).float().mean():.3f}")
    print(f"[{time.time() - t0:.0f}s] cibles prêtes")
    if args.diag:
        return

    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = model.train()
    if args.train_last > 0:
        for p in student.parameters():
            p.requires_grad_(False)
        for m in list(student.blocks[-args.train_last:]) + [student.norm, student.head]:
            for p in m.parameters():
                p.requires_grad_(True)
    params = [p for p in student.parameters() if p.requires_grad]
    print(f"paramètres entraînés : {sum(p.numel() for p in params) / 1e6:.1f}M")
    theta0 = [p.detach().clone() for p in params]

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))
    print(f"précision mixte : {AMP_DTYPE}")
    masks = None
    if args.salun > 0:
        masks = salun_mask(student, params, imgs, q, args.salun)
        student.train()
    synth = None
    if args.synth > 0:
        synth = synth_impressions(teacher, args.synth)
        print(f"[{time.time() - t0:.0f}s] synth prêtes")
    steps_per_epoch = math.ceil(N / args.bs)
    total = args.epochs * steps_per_epoch
    warm = max(1, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    step = 0
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        for i in range(0, N, args.bs):
            idx = perm[i:i + args.bs]
            x = to_float(imgs[idx])
            qt = q[idx].to(DEVICE)
            xc = norm(x)                      # images propres : c'est la moitié du score forget
            if args.aug_mode == "mild":
                xa = norm(torch.stack([MILD(xi) for xi in x]))
            else:
                xa = norm(aug_batch(x))           # vues modifiées : l'autre moitié (hidden altered copies)
            xm = norm(mix_batch(x))
            B = len(x)
            if synth is not None:
                sidx = torch.randint(0, len(synth), (B,))
                xs = norm(torch.stack([AUG(xi) for xi in synth[sidx]]))
            else:
                xs = xm[:0]
            if EXT is not None:
                ho = min(1000, len(EXT) // 5)
                eidx = np.sort(np.random.choice(np.arange(ho, len(EXT)), size=B, replace=False))
                xs = torch.cat([xs, norm(aug_batch(to_float(torch.from_numpy(np.ascontiguousarray(EXT[eidx])))))])
            S = len(xs)
            with torch.no_grad(), torch.autocast("cuda", dtype=AMP_DTYPE):
                tt = teacher(torch.cat([xa, xm, xs])).float()
            ta, tm, tsy = tt[:B], tt[B:2 * B], tt[2 * B:]
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                out = student(torch.cat([xc, xa, xm, xs])).float()
            lc, ls, lm, lsy = out[:B], out[B:2 * B], out[2 * B:3 * B], out[3 * B:]
            if q2 is not None:   # cible surrogate-gold : on aligne directement la métrique (KL à T=2)
                q2t = q2[idx].to(DEVICE)
                loss_c = F.kl_div(F.log_softmax(lc / 2, 1), q2t, reduction="batchmean") * 4
                if args.lam_t1 > 0:   # p^T1 ∝ (p^T2)^2
                    q1t = q2t ** 2; q1t = q1t / q1t.sum(1, keepdim=True)
                    loss_c = loss_c + args.lam_t1 * F.kl_div(F.log_softmax(lc, 1), q1t, reduction="batchmean")
            else:
                loss_c = F.kl_div(F.log_softmax(lc, 1), qt, reduction="batchmean")
            if args.aug_mode == "mild":
                tgt = q2t if q2 is not None else qt
                T_ = 2.0 if q2 is not None else 1.0
                loss_f = 0.5 * (loss_c + F.kl_div(F.log_softmax(ls / T_, 1), tgt, reduction="batchmean") * T_ * T_)
            elif args.aug_mode == "target":
                loss_f = 0.5 * (loss_c + F.kl_div(F.log_softmax(ls, 1), qt, reduction="batchmean"))
            elif args.aug_mode == "orig":
                loss_f = loss_c + F.kl_div(F.log_softmax(ls / 2, 1), F.softmax(ta / 2, 1), reduction="batchmean") * 4
            else:
                loss_f = loss_c
            loss_r = F.kl_div(F.log_softmax(lm / 2, 1), F.softmax(tm / 2, 1), reduction="batchmean") * 4
            loss = loss_f + args.lam_retain * loss_r
            if S > 0:
                w = args.lam_ext if (EXT is not None and synth is None) else args.lam_synth
                loss = loss + w * F.kl_div(F.log_softmax(lsy / 2, 1), F.softmax(tsy / 2, 1),
                                           reduction="batchmean") * 4
            if args.lam_l2 > 0:
                loss = loss + args.lam_l2 * sum(((p - p0) ** 2).sum() for p, p0 in zip(params, theta0))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if masks is not None:
                for p, m in zip(params, masks):
                    if p.grad is not None:
                        p.grad.mul_(m)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % 20 == 0 or step == total:
                print(f"  ep {ep} step {step}/{total}  forget_kl={loss_f.item():.4f}  retain_kl={loss_r.item():.4f}"
                      f"  [{time.time() - t0:.0f}s]")

    student.eval()
    logits1 = predict(student, imgs)
    print("Après unlearning :")
    ce_stats("soumission, images propres", logits1, y)
    kl_q = F.kl_div(F.log_softmax(logits1, 1), q, reduction="batchmean").item()
    print(f"  KL(cible || soumission) = {kl_q:.4f}")
    if q2 is not None:
        num = F.kl_div(F.log_softmax(logits1 / 2, 1), q2, reduction="batchmean").item()
        den = F.kl_div(F.log_softmax(logits0 / 2, 1), q2, reduction="batchmean").item()
        print(f"  ratio prédiction KL(gold||sub)/KL(gold||orig) à T=2 = {num:.4f}/{den:.4f} = {num / den:.3f}"
              f"  -> S_pred estimé ≈ {max(0.0, 1 - num / den):.3f}")
    print(f"  retain proxy KL(orig || sub) T=2 sur mixups = {retain_proxy(teacher, student, imgs):.4f}")
    if EXT is not None:
        kl_e, ag_e = retain_proxy_ext(teacher, student)
        print(f"  retain EXTERNE KL(orig || sub) T=2 = {kl_e:.4f}  accord top-1 = {ag_e:.3f}"
              f"  -> retain_kl_sim ~ {1 / (1 + kl_e):.3f}")

    sd = {k: v.detach().cpu().float() for k, v in student.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in sd.values()), "NaN/Inf dans les poids"
    torch.save(sd, args.out)
    print(f"Sauvé -> {args.out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
