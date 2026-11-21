#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=cand_exact_cit
#SBATCH --output=logs/cand_exact_cit_%j.log
#SBATCH --error=logs/cand_exact_cit_%j.err
# D2: exact ALCE citation recall/precision for EVERY candidate (5 per question), A+B and C, one job.
# Needs setup_env.sh + setup_alce.sh WITH the T5-11B model (do not use SKIP_T5=1).
#   sbatch cluster/run_cand_exact_cit.sbatch
#   SETS="AB" sbatch --export=ALL cluster/run_cand_exact_cit.sbatch    # only Set A+B (5 files instead of 10)
# Writes results/runs/cand_metrics_{AB,C}_exact.jsonl (cand_metrics rows + citation_rec / citation_prec).
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
set -eo pipefail
mkdir -p out logs
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME on $(hostname)"
SETS=${SETS:-"AB C"}
for S in $SETS; do
  if [ "$S" = "AB" ]; then CANDS=results/runs/asqa_cands_0_end.json; REL=results/runs/rr_released.json.perq.jsonl
  else CANDS=results/runs/asqa_cands_C.json; REL=results/runs/rr_released_C.json.perq.jsonl; fi
  python grounding/make_pick_files.py --f "$CANDS" --out_prefix "out/pick_$S"
  for K in 0 1 2 3 4; do
    echo "== $S slot $K  $(date +%T)"
    python critique/score_per_question.py --f "out/pick_${S}_p${K}.json" --alce_dir "$ALCE_DIR"
  done
  python grounding/merge_exact_cit.py --metrics "results/runs/cand_metrics_$S.jsonl" --perq_prefix "out/pick_$S" \
    --released "$REL" --out "results/runs/cand_metrics_${S}_exact.jsonl"
done
echo "EXACT_CIT_OK"
