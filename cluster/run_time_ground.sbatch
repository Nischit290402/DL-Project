#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=time_ground
#SBATCH --output=logs/time_ground_%j.log
#SBATCH --error=logs/time_ground_%j.err
# D2 timing: extra seconds per question for DeBERTa grounding of 5 candidates on Set C (no scores written).
#   sbatch cluster/run_time_ground.sbatch
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) on $(hostname)"
python grounding/time_ground.py --f results/runs/asqa_cands_C.json --split critique/splits/asqa_C.json \
  --ref results/runs/cand_ground_C.jsonl --out results/runs/time_ground_C.json
