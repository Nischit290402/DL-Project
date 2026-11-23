#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=00:40:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=seg2_wh
#SBATCH --output=logs/seg2_whole_%j.log
#SBATCH --error=logs/seg2_whole_%j.err
# Whole-answer check (first segment vs each candidate second segment): grounding/seg2_whole_answer.py
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
set -eo pipefail
for T in AB BC; do
  [ -f out/seg2_$T.json ] && python grounding/seg2_whole_answer.py --seg2 out/seg2_$T.json --picks results/runs/combined/final_948.json --out out/seg2_$T.whole.jsonl
done
