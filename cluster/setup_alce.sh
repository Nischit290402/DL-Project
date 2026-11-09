#!/bin/bash
# One-time setup for ASQA scoring with ALCE (str-EM, ROUGE-L, QA-F1, MAUVE, citation precision/recall).
# Separate conda env so the pinned Self-RAG env (vllm 0.2.6) is untouched.
# Run on the login node (has internet):  bash cluster/setup_alce.sh
# Disk: the citation NLI model (google/t5_xxl_true_nli_mixture, 11B) is about 45 GB on disk in fp32.
#       Check your quota first (e.g. `quota -s` or `df -h ~`). If home is too small, point HF_HOME at a
#       larger scratch space before running this AND in cluster/env_alce.sh.
set -euo pipefail

ENV_DIR=$HOME/envs/alce
ALCE_DIR=${ALCE_DIR:-$HOME/ALCE}
ALCE_COMMIT=246c476a4edfc564266b7346b6e29ef4861ae937
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
mkdir -p "$HOME/envs" "$HF_HOME"

module load miniconda3/25.5.1 >/dev/null 2>&1 || true
eval "$(conda shell.bash hook)"
if [ ! -d "$ENV_DIR" ]; then
  conda create -y -p "$ENV_DIR" python=3.10
fi
conda activate "$ENV_DIR"

pip install --upgrade pip
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.36.2 accelerate==0.25.0 sentencepiece protobuf "numpy<2" \
  rouge_score nltk==3.8.1 mauve-text faiss-cpu scikit-learn tqdm "huggingface_hub[cli]"
python -c "import nltk; nltk.download('punkt')"

if [ ! -d "$ALCE_DIR" ]; then
  git clone https://github.com/princeton-nlp/ALCE.git "$ALCE_DIR"
fi
git -C "$ALCE_DIR" checkout -q "$ALCE_COMMIT"

echo "== downloading scorer models into $HF_HOME (the T5 one is large, be patient)"
huggingface-cli download gaotianyu1350/roberta-large-squad >/dev/null
huggingface-cli download gpt2-large --exclude "*.h5" "*.msgpack" "*.ot" "*.onnx" "*.tflite" >/dev/null
huggingface-cli download google/t5_xxl_true_nli_mixture --exclude "*.h5" "*.msgpack" >/dev/null
# NLI model for the contradiction diagnostic (about 1.7 GB)
huggingface-cli download cross-encoder/nli-deberta-v3-large --exclude "*.h5" "*.msgpack" "*.onnx" >/dev/null

python - <<'PY'
import torch, transformers, mauve, rouge_score
print("torch", torch.__version__, "transformers", transformers.__version__)
PY
echo "ALCE_OK $ENV_DIR $ALCE_DIR"
