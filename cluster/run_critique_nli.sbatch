#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=crit_nli
#SBATCH --output=logs/crit_nli_%j.log
#SBATCH --error=logs/crit_nli_%j.err
# Honesty check, GPU step: T5-11B verdict for every support grade Self-RAG wrote.
#   With candidates (full check): sbatch --export=ALL,F=out/asqa_cands_0_end.json cluster/run_critique_nli.sbatch
#   Label-only, from C1:          sbatch --export=ALL,F=out/asqa_keeptok_0_end.json cluster/run_critique_nli.sbatch
# Optional: MAXQ=5 (quick test). A SPLIT list has commas, so set it in front of sbatch, not inside --export:
#   SPLIT=critique/splits/asqa_A.json,critique/splits/asqa_B.json sbatch --export=ALL,F=... cluster/run_critique_nli.sbatch
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
: "${F:?set F=path/to/output.json}"
SPLIT=${SPLIT:-}
MAXQ=${MAXQ:-}
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) on $(hostname)"
python critique/nli_verdicts.py --f "$F" ${SPLIT:+--split "$SPLIT"} ${MAXQ:+--max_questions "$MAXQ"} --alce_dir "$ALCE_DIR"
