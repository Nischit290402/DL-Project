#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=contra_eval
#SBATCH --output=logs/contra_eval_%j.log
#SBATCH --error=logs/contra_eval_%j.err
# Self-contradiction, held-out test on Set B in one job: NLI on Set B candidates -> re-pick with the settings fixed
# on Set A (noctx, lambda 0.3) -> per-question ALCE scores (str-em, ROUGE, citation rec/prec, length).
#   sbatch cluster/run_contra_eval.sbatch
# Writes results/runs/cand_nli_B.jsonl, results/runs/rr_contra_B.json and rr_contra_B.json.perq.jsonl.
set -e
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) on $(hostname)"
python contradiction/cand_nli.py --f results/runs/asqa_cands_0_end.json \
  --split critique/splits/asqa_B.json --out results/runs/cand_nli_B.jsonl
python contradiction/rerank_contra.py --f results/runs/asqa_cands_0_end.json \
  --nli results/runs/cand_nli_B.jsonl --split critique/splits/asqa_B.json --out results/runs/rr_contra_B.json
python critique/score_per_question.py --f results/runs/rr_contra_B.json --alce_dir "$ALCE_DIR"
echo "CONTRA_EVAL_OK"
