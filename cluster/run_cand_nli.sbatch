#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=cand_nli
#SBATCH --output=logs/cand_nli_%j.log
#SBATCH --error=logs/cand_nli_%j.err
# Self-contradiction across candidates, GPU step: DeBERTa NLI over every pair of depth-1 candidates.
#   sbatch cluster/run_cand_nli.sbatch                                   # Set A (default)
#   sbatch --export=ALL,SET=B cluster/run_cand_nli.sbatch                # Set B, only after Set A is analysed
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
SET=${SET:-A}
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) on $(hostname)"
python contradiction/cand_nli.py --f results/runs/asqa_cands_0_end.json \
  --split "critique/splits/asqa_${SET}.json" --out "results/runs/cand_nli_${SET}.jsonl"
