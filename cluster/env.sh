# source this before running anything: source cluster/env.sh
module load miniconda3/25.5.1 >/dev/null 2>&1 || true
eval "$(conda shell.bash hook)"
conda activate $HOME/envs/selfrag
export HF_HOME=$HOME/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export SELFRAG_MODEL=${SELFRAG_MODEL:-selfrag/selfrag_llama2_7b}
