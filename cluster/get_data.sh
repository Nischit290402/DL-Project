#!/bin/bash
# Downloads the released 7B checkpoint and the paper's evaluation data (pre-retrieved contexts).
# Run on the login node (has internet):  nohup bash cluster/get_data.sh > data/get_data.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/env.sh
mkdir -p data
echo "== model"
huggingface-cli download selfrag/selfrag_llama2_7b --cache-dir "$HF_HOME" >/dev/null
echo "MODEL_OK"
echo "== eval data (HF mirror jesse-tong/selfrag-eval-data; original Google Drive link 1TLKhWjez63H4uBtgCxyoyJsZi-IMgnDb now 404s)"
huggingface-cli download jesse-tong/selfrag-eval-data --repo-type dataset --local-dir data/ >/dev/null
wc -l data/eval_data/popqa_longtail_w_gs.jsonl
echo "DATA_OK"
