#!/bin/bash
# One-shot environment build for Self-RAG inference on the NTU CCDS TC2 cluster.
# Run on the login node:  bash cluster/setup_env.sh
# Pins follow upstream requirements.txt (vllm 0.2.6 era) so retrieval_lm/run_short_form.py
# runs unmodified: it relies on full-vocab logprobs returned as plain floats, which
# vllm >= 0.4 no longer does.
set -euo pipefail

ENV_NAME=${ENV_NAME:-selfrag}
ENV_DIR=$HOME/envs/$ENV_NAME
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
mkdir -p "$HOME/envs" "$HF_HOME"

module load miniconda3/25.5.1 >/dev/null 2>&1 || true
eval "$(conda shell.bash hook)"

if [ ! -d "$ENV_DIR" ]; then
  conda create -y -p "$ENV_DIR" python=3.10
fi
conda activate "$ENV_DIR"

pip install --upgrade pip
# torch 2.1.2 (cu121 wheels from PyPI) is what vllm 0.2.6 pins; driver on the A40 nodes supports CUDA <= 12.8.
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.2.6 transformers==4.36.2 tokenizers==0.15.0 accelerate==0.25.0 datasets==2.15.0 \
  sentencepiece jsonlines spacy nltk scikit-learn tqdm "numpy<2" openai==0.28.1 \
  "huggingface_hub[cli]" gdown

python - <<'PY'
import torch, vllm, transformers
print("torch", torch.__version__, "vllm", vllm.__version__, "transformers", transformers.__version__)
PY
echo "ENV_OK $ENV_DIR"
