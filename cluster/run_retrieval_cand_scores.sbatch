#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=d1_candsc
#SBATCH --output=logs/retrieval_cand_scores_%j.log
#SBATCH --error=logs/retrieval_cand_scores_%j.err
# Scores every candidate of the reranked runs so that any picking rule can be evaluated offline:
#   exact ALCE metrics per candidate (T5-11B), DeBERTa grounding, question-passage relevance.
# Uses the grounding and critique divisions' scripts unchanged; only the input files differ.
# Needs setup_alce.sh WITH the T5 model, and the two small models in $HF_HOME (download on the login node).
#   sbatch cluster/run_retrieval_cand_scores.sbatch
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
set -eo pipefail
mkdir -p out logs
R=results/runs/retrieval
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) on $(hostname)"
for S in AB C; do
  if [ "$S" = "AB" ]; then CANDS=$R/asqa_rrR2K20_0_end.json; SPLIT=critique/splits/asqa_A.json,critique/splits/asqa_B.json
  else CANDS=$R/asqa_rrR2K20C_0_end.json; SPLIT=critique/splits/asqa_C.json; fi
  echo "== $S grounding $(date +%T)"
  python grounding/cand_ground.py --f "$CANDS" --split "$SPLIT" --out "$R/cand_ground_rr_$S.jsonl"
  echo "== $S relevance $(date +%T)"
  python grounding/qa_relevance.py --f "$CANDS" --split "$SPLIT" --out "$R/cand_qarel_rr_$S.jsonl" --models BAAI/bge-reranker-base
  echo "== $S candidate metrics $(date +%T)"
  python grounding/cand_metrics.py --f "$CANDS" --out "$R/cand_metrics_rr_$S.jsonl"
  python grounding/make_pick_files.py --f "$CANDS" --out_prefix "out/pick_rr_$S"
  for K in 0 1 2 3 4; do
    echo "== $S slot $K  $(date +%T)"
    python critique/score_per_question.py --f "out/pick_rr_${S}_p${K}.json" --alce_dir "$ALCE_DIR"
  done
  python grounding/merge_exact_cit.py --metrics "$R/cand_metrics_rr_$S.jsonl" --perq_prefix "out/pick_rr_$S" \
    --out "$R/cand_metrics_rr_${S}_exact.jsonl"
done
echo "RETRIEVAL_CAND_SCORES_OK $(date +%T)"
