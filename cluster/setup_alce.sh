#!/bin/bash
# One-time setup for ASQA scoring with ALCE (str-EM, ROUGE-L, QA-F1, MAUVE, citation precision/recall)
# and for the contradiction diagnostic. Run on the login node (has internet), AFTER setup_env.sh:
#   bash cluster/setup_alce.sh                 # everything (~52 GB of scorer models)
#   SKIP_T5=1 bash cluster/setup_alce.sh       # no citation model (~6 GB); score with FLAGS="--qa --mauve"
#
# Disk: home quota is 100 GB and the cluster misbehaves above 98% (login/job failures, data loss).
#   selfrag env ~8 GB + 7B model ~13.5 GB + eval data + citation NLI model (T5-11B) ~45 GB + others ~6 GB.
#   Check first with `du -sh ~ ~/.cache ~/hf_cache ~/envs` and clear caches at the end (done below).
#
# Installs the few extra scoring packages INTO the existing selfrag env, constrained by
# requirements-lock.txt so no pinned package (vllm, torch, transformers, ...) can change.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/env.sh

ALCE_DIR=${ALCE_DIR:-$HOME/ALCE}
ALCE_COMMIT=246c476a4edfc564266b7346b6e29ef4861ae937

pip install --no-cache-dir rouge_score mauve-text faiss-cpu -c cluster/requirements-lock.txt
python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt', 'punkt_tab')]"

if [ ! -d "$ALCE_DIR" ]; then
  git clone https://github.com/princeton-nlp/ALCE.git "$ALCE_DIR"
fi
git -C "$ALCE_DIR" checkout -q "$ALCE_COMMIT"

echo "== downloading scorer models into $HF_HOME/hub (one weight format per model)"
MODELS="gaotianyu1350/roberta-large-squad gpt2-large cross-encoder/nli-deberta-v3-large"
if [ "${SKIP_T5:-0}" != "1" ]; then MODELS="$MODELS google/t5_xxl_true_nli_mixture"; fi
python - $MODELS <<'PY'
import sys
from huggingface_hub import list_repo_files, snapshot_download
for repo in sys.argv[1:]:
    files = list_repo_files(repo)
    weights = ["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["*.bin"]
    allow = weights + ["*.json", "*.txt", "*.model", "*.spm", "vocab*", "merges*"]
    print("downloading", repo, weights, flush=True)
    snapshot_download(repo, allow_patterns=allow)
print("MODELS_OK")
PY

pip cache purge >/dev/null 2>&1 || true
conda clean -a -y >/dev/null 2>&1 || true
python -c "import mauve, rouge_score, nltk; nltk.sent_tokenize('A b. C d.'); print('imports ok')"
du -sh "$HF_HOME" "$HOME/envs" 2>/dev/null || true
echo "ALCE_OK $ALCE_DIR"
