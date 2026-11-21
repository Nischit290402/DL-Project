#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=qa_rel
#SBATCH --output=logs/qa_rel_%j.log
#SBATCH --error=logs/qa_rel_%j.err
# Question-passage relevance of every A+B candidate (feasibility check for a D2 relevance gate).
# Models must be in $HF_HOME first (download on the login node: see grounding/qa_relevance.py MODELS).
#   sbatch cluster/run_qa_relevance.sbatch
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
python grounding/qa_relevance.py --f results/runs/asqa_cands_0_end.json \
  --split critique/splits/asqa_A.json,critique/splits/asqa_B.json --out results/runs/cand_qarel_AB.jsonl
