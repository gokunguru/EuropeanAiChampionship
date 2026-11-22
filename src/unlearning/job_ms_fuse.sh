#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:15:00
#SBATCH --output=output/%j_ms_fuse.out
# Méthode fusionnée : cible = substituts sans fuite (gold_v3) lissés localement (γ), distillation T=2 + T=1,
# rétention sur ext, masque de saillance (SalUn) en option, puis extrapolation α = 1.5.
# Témoin γ = 0 : sub_ms_v14_g4_t025 (même recette, soumis à part).
cd "${UL:?définir UL, le dossier de travail (modèle, images, ext_retain.npy)}"
source .venv/bin/activate
python -u ~/eac/src/unlearning/ms_fuse_targets.py --golds "gold_v3/*.pt" --model model.pt --forget forget_images.npz \
    --block 6 --K 10 --gammas 0.3,0.6 --outdir gold_ms_fuse || exit 1
COMMON="--tau 1.0 --lam_retain 0 --ext ext_retain.npy --lam_ext 1 --lr 2e-4 --epochs 10 --lam_t1 0.25"
CUDA_VISIBLE_DEVICES=0 python -u unlearn.py --gold gold_ms_fuse/gold_fuse_g0.3.pt $COMMON --aug_mode mild              --out sub_ms_fuse_g03.pt       > output/${SLURM_JOB_ID}_fuse_g03.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u unlearn.py --gold gold_ms_fuse/gold_fuse_g0.6.pt $COMMON --aug_mode mild              --out sub_ms_fuse_g06.pt       > output/${SLURM_JOB_ID}_fuse_g06.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u unlearn.py --gold gold_ms_fuse/gold_fuse_g0.3.pt $COMMON --aug_mode mild --salun 0.5  --out sub_ms_fuse_g03_sal.pt   > output/${SLURM_JOB_ID}_fuse_g03_sal.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u unlearn.py --gold gold_ms_fuse/gold_fuse_g0.3.pt $COMMON --aug_mode target            --out sub_ms_fuse_g03_aug.pt   > output/${SLURM_JOB_ID}_fuse_g03_aug.log 2>&1 &
wait
python - <<'PY'
import glob, torch
o = torch.load('model.pt', map_location='cpu')
for f in sorted(glob.glob('sub_ms_fuse_g0*.pt')):
    if '_a1.5' in f: continue
    v = torch.load(f, map_location='cpu')
    s = {k: (o[k].float() + 1.5 * (v[k].float() - o[k].float())).contiguous() for k in o}
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in s.values())
    torch.save(s, f.replace('.pt', '_a1.5.pt')); print('extrapolé', f)
PY
for f in output/${SLURM_JOB_ID}_fuse_*.log; do echo "=== $f"; grep -h "ratio prédiction\|retain EXTERNE\|référence gold\|Error\|error" "$f"; done
