#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:30:00
#SBATCH --output=output/%j.out

cd "${UL:?définir UL, le dossier de travail (modèle, images, ext_retain.npy)}"
source .venv/bin/activate
mkdir -p gold_v3

# S&P en deux étages : reinit_from shrink_haut shrink_bas epochs lr tag
i=0
for cfg in "6 0.45 0.85 10 2e-4 k6h45l85" "6 0.45 0.9 10 2e-4 k6h45l90" \
           "6 0.35 0.85 10 2e-4 k6h35l85" "8 0.3 0.85 10 2e-4 k8h30l85"; do
  set -- $cfg
  CUDA_VISIBLE_DEVICES=$i python -u surrogate.py --ext ext_retain.npy --reinit_from $1 --shrink $2 --shrink_low $3 \
    --epochs $4 --lr $5 --seed $i --tag $6 --outdir gold_v3 > output/${SLURM_JOB_ID}_gold3_$6.log 2>&1 &
  i=$((i+1))
done
wait
for f in output/${SLURM_JOB_ID}_gold3_*.log; do
  echo "=== $f"; grep -h "holdout\|forget (jamais\|fuite\|Error" "$f"
done
