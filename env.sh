#!/usr/bin/env bash
# Optional cache locations for local experiments. Source after activating a venv.
quantserve_cache_root="${QUANTSERVE_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/quantserve}"
export HF_HOME="${HF_HOME:-$quantserve_cache_root/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$quantserve_cache_root/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$quantserve_cache_root/pip}"
unset quantserve_cache_root
