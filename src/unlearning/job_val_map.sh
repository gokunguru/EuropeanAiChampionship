#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:20:00
#SBATCH --output=output/%j_valmap.out
# Lancer depuis le dossier unlearning : sbatch ~/eac/src/unlearning/job_val_map.sh
cd /p/scratch/training2666/mandi2/kame_network/unlearning
source .venv/bin/activate
python -u ~/eac/src/unlearning/ms_val_map.py --val ../imagenet_val/data --model model.pt \
    --forget forget_images.npz --out val_map --workers 64
