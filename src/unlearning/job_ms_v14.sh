#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:30:00
#SBATCH --output=output/%j_ms_v14.out
# v14 : même recette que v13_v3 (meilleur public, 0.073), une seule chose change par GPU.
#   Hypothèse : le serveur compare à la MOYENNE de plusieurs golds ; moyenner 4 substituts « sans fuite » (gold_v3)
#   au lieu de 2 rapproche la cible de p_gold. Contrôles : effet de lam_t1, à cible fixée.
cd /p/scratch/training2666/mandi2/kame_network/unlearning
source .venv/bin/activate
COMMON="--tau 1.0 --aug_mode mild --lam_retain 0 --ext ext_retain.npy --lam_ext 1 --lr 2e-4 --epochs 10"
CUDA_VISIBLE_DEVICES=0 python -u unlearn.py --gold "gold_v3/*.pt"   --lam_t1 0.25 $COMMON --out sub_ms_v14_g4_t025.pt > output/${SLURM_JOB_ID}_g4_t025.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u unlearn.py --gold "gold_v3/*.pt"   --lam_t1 0.5  $COMMON --out sub_ms_v14_g4_t05.pt  > output/${SLURM_JOB_ID}_g4_t05.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u unlearn.py --gold "gold_v13a/*.pt" --lam_t1 0.5  $COMMON --out sub_ms_v14_g2_t05.pt  > output/${SLURM_JOB_ID}_g2_t05.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u unlearn.py --gold "gold_v3/*.pt"   --lam_t1 0    $COMMON --out sub_ms_v14_g4_t0.pt   > output/${SLURM_JOB_ID}_g4_t0.log 2>&1 &
wait
ls -la sub_ms_v14_*.pt
