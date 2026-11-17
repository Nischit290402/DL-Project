#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=05:45:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=contra_C
#SBATCH --output=logs/contra_C_%j.log
#SBATCH --error=logs/contra_C_%j.err
# Pre-registered tests on Set C (contradiction/PREREG_C.md), all steps in one job, about 2.5-3 h:
#   candidates (~45 min) -> NLI (~20 min) -> re-pick (seconds) -> scoring x2 (~60-90 min) -> H1, H2, sanity check.
#   sbatch cluster/run_contra_C.sbatch
# Each step is skipped if its output already exists, so a resubmit continues where a failed run stopped.
set -e
cd "$SLURM_SUBMIT_DIR"
source cluster/env_alce.sh
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME on $(hostname)"
if [[ "$GPU_NAME" != *L40S* ]]; then echo "WRONG_GPU: $GPU_NAME"; exit 3; fi
R=results/runs
SPLIT=critique/splits/asqa_C.json
step() { echo; echo "=== $1  $(date +%H:%M:%S)"; }

step "1/5 candidates for Set C"
if [ ! -s $R/asqa_cands_C.json ]; then
  (cd self-rag/retrieval_lm && python run_long_form_static.py \
    --model_name "$SELFRAG_MODEL" --download_dir "$HF_HOME" \
    --input_file ../../data/eval_data/asqa_eval_gtr_top100.json \
    --task asqa --mode always_retrieve --ndocs 5 --max_new_tokens 300 --threshold 0.2 \
    --use_grounding --use_utility --use_seqscore --max_depth 7 --beam_width 2 --dtype half \
    --keep_special_tokens --save_candidates --indices_file ../../$SPLIT \
    --output_file ../../$R/asqa_cands_C.json)
else echo "exists, skipped"; fi

step "2/5 NLI over candidate pairs"
[ -s $R/cand_nli_C.jsonl ] && echo "exists, skipped" || \
  python contradiction/cand_nli.py --f $R/asqa_cands_C.json --split $SPLIT --out $R/cand_nli_C.jsonl

step "3/5 re-pick: released (lambda 0) and penalty (noctx, lambda 0.3)"
python contradiction/rerank_contra.py --f $R/asqa_cands_C.json --nli $R/cand_nli_C.jsonl --split $SPLIT \
  --out $R/rr_released_C.json --lam 0
python contradiction/rerank_contra.py --f $R/asqa_cands_C.json --nli $R/cand_nli_C.jsonl --split $SPLIT \
  --out $R/rr_contra_C.json

step "4/5 per-question scores"
for f in rr_released_C rr_contra_C; do
  [ -s $R/$f.json.perq.jsonl ] && echo "$f exists, skipped" || \
    python critique/score_per_question.py --f $R/$f.json --alce_dir "$ALCE_DIR"
done

step "5/5 pre-registered tests"
echo "--- H1: disagreement vs correctness"
python contradiction/uncertainty_test.py --cands $R/asqa_cands_C.json --nli $R/cand_nli_C.jsonl \
  --perq $R/rr_released_C.json.perq.jsonl --split $SPLIT --out results/uncertainty_C.json
echo "--- H2: penalty vs released"
python critique/bootstrap.py --base $R/rr_released_C.json.perq.jsonl --new $R/rr_contra_C.json.perq.jsonl \
  --split $SPLIT --out results/contra_C_bootstrap.json
echo "--- sanity: released (regenerated) vs C1 on Set C"
python critique/bootstrap.py --base $R/asqa_keeptok_0_end.json.perq.jsonl --new $R/rr_released_C.json.perq.jsonl \
  --split $SPLIT
echo "CONTRA_C_OK"
