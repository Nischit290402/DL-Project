#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --job-name=selfrag_asqa
#SBATCH --output=logs/asqa_%j.log
#SBATCH --error=logs/asqa_%j.err
# ASQA long-form, settings from the upstream README (paper Table 2, Self-RAG 7B).
# Shard with START/END (example indices, END exclusive). TAG names the run.
#   Pilot:   sbatch --export=ALL,START=0,END=30,TAG=pilot --time=01:00:00 cluster/run_asqa.sbatch
#   Shard A: sbatch --export=ALL,START=0,END=474,TAG=baseline cluster/run_asqa.sbatch
#   Shard B: sbatch --export=ALL,START=474,END=948,TAG=baseline cluster/run_asqa.sbatch
# Extra flags for experiments go in EXTRA, e.g. EXTRA="--beam_width 4".
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
mkdir -p out
START=${START:-0}
END=${END:-}            # empty = run to the end of the file
TAG=${TAG:-baseline}
EXTRA=${EXTRA:-}
nvidia-smi --query-gpu=name,memory.total --format=csv
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
