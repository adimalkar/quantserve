#!/usr/bin/env bash
# Source this file before running experiments to route all cache, downloads, and models to the 1TB drive
export HF_HOME="/mnt/1TB_Drive/Data/MyFiles/models/huggingface"
export TRANSFORMERS_CACHE="/mnt/1TB_Drive/Data/MyFiles/models/huggingface"
export TORCH_HOME="/mnt/1TB_Drive/Data/MyFiles/models/torch"
export PIP_CACHE_DIR="/mnt/1TB_Drive/Data/MyFiles/.cache/pip"
export PATH="/mnt/1TB_Drive/Data/MyFiles/env_quantserve/bin:$PATH"

echo "QuantServe-Bench Environment Activated (Storage -> /mnt/1TB_Drive/Data/MyFiles)"
echo "Python: $(which python)"
