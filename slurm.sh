#!/bin/sh
#SBATCH --job-name=char_recognition
#SBATCH --time=180
#SBATCH --mail-type=END,FAIL
#SBATCH --mem-per-gpu=80G
#SBATCH --gpus=a100-80:1
#SBATCH --mail-user=email@u.nus.edu
###SBATCH --nodelist=xgph1

echo "SBATCH_INFO: Printing diagnostics (visible devices and nvidia-smi)..."
echo $CUDA_VISIBLE_DEVICES
srun nvidia-smi
#srun nvcc --version
#srun accelerate config default
#srun accelerate env

echo "SBATCH_INFO: Running character recognition training..."
srun python train_slurm.py

echo "SBATCH_INFO: Running character recognition evaluation..."
srun python evaluate_slurm.py