#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --exclude=TC2N01,TC2N02
#SBATCH --job-name=selfrag_asqa
#SBATCH --output=logs/asqa_%j.log
#SBATCH --error=logs/asqa_%j.err
# ASQA long-form, settings from the upstream README (paper Table 2, Self-RAG 7B).
# Shard with START/END (example indices, END exclusive). TAG names the run.
#   Pilot:   sbatch --export=ALL,START=0,END=30,TAG=pilot --time=01:00:00 cluster/run_asqa.sbatch
#   Shard A: JOB_A=$(sbatch --parsable --export=ALL,START=0,END=474,TAG=baseline cluster/run_asqa.sbatch)
#   Shard B: sbatch --dependency=afterany:$JOB_A --export=ALL,START=474,TAG=baseline cluster/run_asqa.sbatch
# (QoS limits are per user: one GPU at a time, so shard B waits for shard A.)
# Extra flags for experiments go in EXTRA, e.g. EXTRA="--beam_width 4".
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
mkdir -p out
START=${START:-0}
END=${END:-}            # empty = run to the end of the file
TAG=${TAG:-baseline}
EXTRA=${EXTRA:-}
# GPU pin: TC2N01/N02 are A40 (8 each), TC2N03-N08 are L40S (4 each; probed 2026-11-11). A40 vs L40S changes ~3% of answers,
# so experiments run on L40S to match the reference run. REQUIRE_GPU="" disables the check.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
REQUIRE_GPU=${REQUIRE_GPU-L40S}
echo "GPU: $GPU_NAME on $(hostname)"
if [ -n "$REQUIRE_GPU" ] && [[ "$GPU_NAME" != *"$REQUIRE_GPU"* ]]; then
  echo "WRONG_GPU: got $GPU_NAME on $(hostname), need $REQUIRE_GPU. Add that node to --exclude and resubmit."
  exit 3
fi
cd self-rag/retrieval_lm
python run_long_form_static.py \
  --model_name "$SELFRAG_MODEL" \
  --download_dir "$HF_HOME" \
  --input_file ../../data/eval_data/asqa_eval_gtr_top100.json \
  --task asqa --mode always_retrieve \
  --ndocs 5 --max_new_tokens 300 --threshold 0.2 \
  --use_grounding --use_utility --use_seqscore \
  --max_depth 7 --beam_width 2 \
  --dtype half \
  --start "$START" ${END:+--end "$END"} \
  --output_file "../../out/asqa_${TAG}_${START}_${END:-end}.json" \
  $EXTRA
