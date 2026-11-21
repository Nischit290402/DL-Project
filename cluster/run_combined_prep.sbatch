#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=05:00:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=combined_prep
#SBATCH --output=logs/combined_prep_%j.log
#SBATCH --error=logs/combined_prep_%j.err
# Combined system, GPU preparation (combined/PREREG.md). For the retrieval division's candidates on the reranked
# passages (Set A+B and Set C), compute everything the combined ranking needs, with the divisions' own scripts:
#   1. ALCE scores of the retrieval runs as released (ladder row R1), also used for the one multi-segment question
#   2. exact ALCE metrics of every candidate (5 per question), for scoring any ranking by lookup
#   3. DeBERTa grounding of every candidate (grounding division, D2)
#   4. bge-reranker-base question-passage relevance of every candidate (grounding division, D2-v2 gate)
# Needs setup_env.sh + setup_alce.sh with the T5-11B model, and BAAI/bge-reranker-base in $HF_HOME (download it on
# the login node first, see combined/README.md). About 1 hour on an L40S.
#   sbatch cluster/run_combined_prep.sbatch
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
set -eo pipefail
mkdir -p out/combined logs results/runs/combined
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
REQUIRE_GPU=${REQUIRE_GPU-L40S}
echo "GPU: $GPU_NAME on $(hostname)"
if [ -n "$REQUIRE_GPU" ] && [[ "$GPU_NAME" != *"$REQUIRE_GPU"* ]]; then
  echo "WRONG_GPU: got $GPU_NAME on $(hostname), need $REQUIRE_GPU. Add that node to --exclude and resubmit."
  exit 3
fi
D=results/runs/combined
for S in AB C; do
  if [ "$S" = "AB" ]; then
    CANDS=results/runs/retrieval/asqa_rrR2K20_0_end.json
    SPLIT=critique/splits/asqa_A.json,critique/splits/asqa_B.json
  else
    CANDS=results/runs/retrieval/asqa_rrR2K20C_0_end.json
    SPLIT=critique/splits/asqa_C.json
  fi
  echo "== $S 1/4 ALCE scores of the retrieval run as released  $(date +%T)"
  [ -s "$CANDS.perq.jsonl" ] && echo "exists, skipped" || \
    python critique/score_per_question.py --f "$CANDS" --alce_dir "$ALCE_DIR"
  echo "== $S 2/4 exact metrics of every candidate  $(date +%T)"
  python grounding/cand_metrics.py --f "$CANDS" --out "$D/cand_metrics_rr_$S.jsonl"
  python grounding/make_pick_files.py --f "$CANDS" --out_prefix "out/combined/pick_rr_$S"
  for K in 0 1 2 3 4; do
    echo "   slot $K  $(date +%T)"
    [ -s "out/combined/pick_rr_${S}_p${K}.json.perq.jsonl" ] && echo "   exists, skipped" || \
      python critique/score_per_question.py --f "out/combined/pick_rr_${S}_p${K}.json" --alce_dir "$ALCE_DIR"
  done
  python grounding/merge_exact_cit.py --metrics "$D/cand_metrics_rr_$S.jsonl" --perq_prefix "out/combined/pick_rr_$S" \
    --released "$CANDS.perq.jsonl" --out "$D/cand_metrics_rr_${S}_exact.jsonl"
  echo "== $S 3/4 DeBERTa grounding  $(date +%T)"
  python grounding/cand_ground.py --f "$CANDS" --split "$SPLIT" --out "$D/cand_ground_rr_$S.jsonl"
  echo "== $S 4/4 question-passage relevance  $(date +%T)"
  python grounding/qa_relevance.py --f "$CANDS" --split "$SPLIT" --out "$D/cand_qarel_rr_$S.jsonl" \
    --models BAAI/bge-reranker-base
done
echo "COMBINED_PREP_OK $(date +%T)"
