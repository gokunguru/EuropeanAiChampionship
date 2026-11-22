#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:15:00
#SBATCH --output=output/%j.out

cd "${UL:?définir UL, le dossier de travail (modèle, images, ext_retain.npy)}"
source .venv/bin/activate

# unlearning vers la cible surrogate-gold + rétention sur images externes
# lr lam_ext
i=0
for cfg in "1e-4 1" "2e-4 1" "1e-4 3" "2e-4 3"; do
  set -- $cfg
  name=v10_lr$1_ext$2
  CUDA_VISIBLE_DEVICES=$i python -u unlearn.py --gold "gold_*.pt" --tau 1.0 --aug_mode mild --lam_retain 0 \
    --ext ext_retain.npy --lam_ext $2 --lr $1 --epochs 10 --out sub_$name.pt \
    > output/${SLURM_JOB_ID}_$name.log 2>&1 &
  i=$((i+1))
done
wait
for f in output/${SLURM_JOB_ID}_v10_*.log; do
  echo "=== $f"
  grep -h "référence gold\|soumission, images\|ratio prédiction\|retain EXTERNE\|Error" "$f"
done
