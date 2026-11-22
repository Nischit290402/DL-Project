#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --exclude=TC2N01,TC2N02
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
# GPU pin: TC2N01/N02 are A40 (8 each), TC2N03-N08 are L40S (4 each; probed 2026-11-11). A40 vs L40S changes ~3% of answers,
# so experiments run on L40S to match the reference run. REQUIRE_GPU="" disables the check.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
REQUIRE_GPU=${REQUIRE_GPU-L40S}
echo "GPU: $GPU_NAME on $(hostname)"
if [ -n "$REQUIRE_GPU" ] && [[ "$GPU_NAME" != *"$REQUIRE_GPU"* ]]; then
  echo "WRONG_GPU: got $GPU_NAME on $(hostname), need $REQUIRE_GPU. Add that node to --exclude and resubmit."
  exit 3
fi
cd "$ALCE_DIR"
python eval.py --f "$F_ABS" $FLAGS
echo "SCORE_FILE ${F_ABS}.score"
