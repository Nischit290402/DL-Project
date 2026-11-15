#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=crit_score
#SBATCH --output=logs/crit_score_%j.log
#SBATCH --error=logs/crit_score_%j.err
# Per-question ASQA scores (str-em, ROUGE-Lsum, citation recall/precision) for bootstrap intervals.
#   sbatch --export=ALL,F=out/asqa_keeptok_0_end.json,SPLIT=critique/splits/asqa_A.json cluster/run_critique_score.sbatch
# SPLIT empty = all questions. FLAGS="--mauve" adds MAUVE (one number), FLAGS="--no_citations" skips the T5-11B model.
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
: "${F:?set F=path/to/asqa_output.json}"
SPLIT=${SPLIT:-}
FLAGS=${FLAGS:-}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME on $(hostname)"
python critique/score_per_question.py --f "$F" ${SPLIT:+--split "$SPLIT"} --alce_dir "$ALCE_DIR" $FLAGS
