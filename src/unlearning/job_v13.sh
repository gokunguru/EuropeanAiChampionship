#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:40:00
#SBATCH --output=output/%j.out

cd "${UL:?définir UL, le dossier de travail (modèle, images, ext_retain.npy)}"
source .venv/bin/activate
mkdir -p gold_v13a gold_v13b
cp gold_v3/gold_k6h35l85.pt gold_v3/gold_k8h30l85.pt gold_v13a/
cp gold_v3/gold_k6h35l85.pt gold_v3/gold_k8h30l85.pt gold_k6sh50.pt gold_k8sh30.pt gold_v13b/
COMMON="--tau 1.0 --aug_mode mild --lam_retain 0 --ext ext_retain.npy --lam_ext 1 --lr 2e-4 --epochs 10"

# GPU0 : refait la meilleure publique (v6 none r0.3, puis x1.3)
CUDA_VISIBLE_DEVICES=0 python -u unlearn.py --tau 1.0 --t_smin 0.35 --t_smax 0.6 --aug_mode none \
  --lr 2e-4 --epochs 10 --lam_retain 0.3 --out sub_v6_none_r0.3.pt > output/${SLURM_JOB_ID}_v6.log 2>&1 &
# GPU1 : refait v10 lr2e-4 ext1 (cible = 2 surrogates v2)
CUDA_VISIBLE_DEVICES=1 python -u unlearn.py --gold "gold_k*sh*.pt" $COMMON \
  --out sub_v10_lr2e-4_ext1.pt > output/${SLURM_JOB_ID}_v10.log 2>&1 &
# GPU2 : v13a = 2 surrogates v3 sans fuite (CE ~1.0 ± 1.4) + terme T=1
CUDA_VISIBLE_DEVICES=2 python -u unlearn.py --gold "gold_v13a/*.pt" --lam_t1 0.25 $COMMON \
  --out sub_v13_v3.pt > output/${SLURM_JOB_ID}_v13_v3.log 2>&1 &
# GPU3 : v13b = 4 surrogates (v2 + v3) + terme T=1
CUDA_VISIBLE_DEVICES=3 python -u unlearn.py --gold "gold_v13b/*.pt" --lam_t1 0.25 $COMMON \
  --out sub_v13_mix4.pt > output/${SLURM_JOB_ID}_v13_mix4.log 2>&1 &
wait

python interp.py sub_v6_none_r0.3.pt 1.3
cp sub_*.pt/   # copie de sécurité hors du dossier partagé
for f in output/${SLURM_JOB_ID}_*.log; do
  echo "=== $f"; grep -h "surrogate-gold :\|référence gold\|soumission, images\|ratio prédiction\|retain EXTERNE\|Sauvé\|Error" "$f"
done

# calibration + décomposition + classement
python -u calib.py --scores scores.txt --alphas 1.0,1.15,1.3 --subs sub_*.pt \
  --golds gold_*.pt gold_v3/*.pt gold_combo/*.pt gold_knn2/*.pt
