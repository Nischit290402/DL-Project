#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=00:30:00
#SBATCH --job-name=selfrag_smoke
#SBATCH --output=logs/smoke_%j.log
#SBATCH --error=logs/smoke_%j.err
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
python cluster/smoke_test.py
