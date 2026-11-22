#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:15:00
#SBATCH --output=output/%j_ms_fuse2.out
# Fusion v2 : base = les 2 substituts de v13_v3 (gold_v13a, meilleur public), un fichier lissé par substitut à T=1,
# voisins de même classe ; γ = 0 redonne exactement v13_v3. GPU3 = v13_v3 avec une autre graine (variance entre runs).
cd "${UL:?définir UL, le dossier de travail (modèle, images, ext_retain.npy)}"
source .venv/bin/activate
python -u ~/eac/src/unlearning/ms_fuse_targets.py --golds "gold_v13a/*.pt" --model model.pt --forget forget_images.npz \
    --block 6 --K 10 --gammas 0.3,0.6 --per_file --outdir gold_ms_fuse2 || exit 1
COMMON="--tau 1.0 --aug_mode mild --lam_retain 0 --ext ext_retain.npy --lam_ext 1 --lr 2e-4 --epochs 10 --lam_t1 0.25"
CUDA_VISIBLE_DEVICES=0 python -u unlearn.py --gold "gold_ms_fuse2/g0.3/*.pt" $COMMON              --out sub_ms_f2_g03.pt     > output/${SLURM_JOB_ID}_f2_g03.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u unlearn.py --gold "gold_ms_fuse2/g0.6/*.pt" $COMMON              --out sub_ms_f2_g06.pt     > output/${SLURM_JOB_ID}_f2_g06.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u unlearn.py --gold "gold_ms_fuse2/g0.3/*.pt" $COMMON --salun 0.5  --out sub_ms_f2_g03_sal.pt > output/${SLURM_JOB_ID}_f2_g03_sal.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u unlearn.py --gold "gold_v13a/*.pt"          $COMMON --seed 1     --out sub_ms_f2_v13_s1.pt  > output/${SLURM_JOB_ID}_f2_v13_s1.log 2>&1 &
wait
python - <<'PY'
import glob, torch
o = torch.load('model.pt', map_location='cpu')
for f in sorted(glob.glob('sub_ms_f2_*.pt')):
    if '_a1.5' in f: continue
    v = torch.load(f, map_location='cpu')
    s = {k: (o[k].float() + 1.5 * (v[k].float() - o[k].float())).contiguous() for k in o}
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in s.values())
    torch.save(s, f.replace('.pt', '_a1.5.pt')); print('extrapolé', f)
PY
for f in output/${SLURM_JOB_ID}_f2_*.log; do echo "=== $f"; grep -h "référence gold\|ratio prédiction\|retain EXTERNE\|Error" "$f"; done
