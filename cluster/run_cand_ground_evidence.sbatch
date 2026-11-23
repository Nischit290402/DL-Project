#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=ground_ev
#SBATCH --output=logs/ground_ev_%j.log
#SBATCH --error=logs/ground_ev_%j.err
# Evidence-sentence grounding of every candidate on the reranked passages (grounding/cand_ground_evidence.py).
#   sbatch cluster/run_cand_ground_evidence.sbatch
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
set -eo pipefail
mkdir -p logs
R=results/runs/retrieval
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) on $(hostname)"
python grounding/cand_ground_evidence.py --f $R/asqa_rrR2K20_0_end.json --split critique/splits/asqa_A.json,critique/splits/asqa_B.json --out $R/cand_ground_ev_rr_AB.jsonl
python grounding/cand_ground_evidence.py --f $R/asqa_rrR2K20C_0_end.json --split critique/splits/asqa_C.json --out $R/cand_ground_ev_rr_C.jsonl
