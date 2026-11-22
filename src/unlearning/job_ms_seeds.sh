#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:12:00
#SBATCH --output=output/%j_ms_seeds.out
# Recette v13_v3 (meilleure) avec 4 graines de plus, pour une soupe de modèles (moyenne des poids) puis α = 1.5.
cd /p/scratch/training2666/mandi2/kame_network/unlearning
source .venv/bin/activate
COMMON="--tau 1.0 --aug_mode mild --lam_retain 0 --ext ext_retain.npy --lam_ext 1 --lr 2e-4 --epochs 10 --lam_t1 0.25"
for i in 0 1 2 3; do
  s=$((i + 2))
  CUDA_VISIBLE_DEVICES=$i python -u unlearn.py --gold "gold_v13a/*.pt" $COMMON --seed $s --out sub_ms_v13_s$s.pt > output/${SLURM_JOB_ID}_v13_s$s.log 2>&1 &
done
wait
python - <<'PY'
import torch
o = torch.load('model.pt', map_location='cpu')
fs = ['sub_v13_v3.pt', 'sub_ms_f2_v13_s1.pt'] + [f'sub_ms_v13_s{s}.pt' for s in (2, 3, 4, 5)]
acc = {k: torch.zeros_like(o[k], dtype=torch.float32) for k in o}
for f in fs:
    v = torch.load(f, map_location='cpu')
    for k in o: acc[k] += v[k].float() / len(fs)
s = {k: (o[k].float() + 1.5 * (acc[k] - o[k].float())).contiguous() for k in o}
assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in s.values())
torch.save(s, 'sub_ms_soup6_v13_a1.5.pt'); print('soupe de', len(fs), 'modèles -> sub_ms_soup6_v13_a1.5.pt')
PY
