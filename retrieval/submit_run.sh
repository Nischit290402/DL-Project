#!/bin/bash
# Run Self-RAG with C1 settings through the team's cluster/run_asqa.sbatch, changing only what the retrieval
# division owns: the passages file and/or the number of passages. Candidates are saved for offline analysis.
# usage (from the repo root): bash retrieval/submit_run.sh TAG NDOCS INDICES [INPUT_FILE]
#   INDICES: comma-separated split files, e.g. critique/splits/asqa_A.json,critique/splits/asqa_B.json
#   output:  out/asqa_${TAG}_0_end.json
set -e
TAG=$1; ND=$2; IDX=$3; INP=${4:-}
R=$(pwd)
IDX=$(echo "$IDX" | sed "s#\([^,]*\)#$R/\1#g")
export TAG START=0
export EXTRA="--keep_special_tokens --save_candidates --indices_file $IDX --ndocs $ND ${INP:+--input_file $R/$INP}"
sbatch --parsable --time=02:00:00 --job-name=d1_$TAG --export=ALL cluster/run_asqa.sbatch
