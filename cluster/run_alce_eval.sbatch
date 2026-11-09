#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --job-name=alce_eval
#SBATCH --output=logs/alce_%j.log
#SBATCH --error=logs/alce_%j.err
# Score an ASQA output file with ALCE. Writes <file>.score next to the input.
#   sbatch --export=ALL,F=out/asqa_baseline.json cluster/run_alce_eval.sbatch
# Paper metrics (Table 2): str-em, rougeLsum, mauve, citation_prec, citation_rec.
# Set FLAGS="--citations" to skip QA-F1/MAUVE, or FLAGS="--qa --mauve" to skip citations (no T5-11B).
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
: "${F:?set F=path/to/asqa_output.json}"
FLAGS=${FLAGS:---citations --qa --mauve}
F_ABS=$(readlink -f "$F")
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$ALCE_DIR"
python eval.py --f "$F_ABS" $FLAGS
echo "SCORE_FILE ${F_ABS}.score"
