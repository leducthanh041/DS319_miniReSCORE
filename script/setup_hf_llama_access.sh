#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HF_HOME_DEFAULT="/mmlab_students/storageStudents/nguyenvd/Thanhld/.cache/huggingface"
HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
HF_BIN="${HF_BIN:-/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/ReSCORE/bin/hf}"
PYTHON_BIN="${PYTHON_BIN:-/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/ReSCORE/bin/python}"
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"

export HF_HOME

echo "HF_HOME=$HF_HOME"
echo "HF_BIN=$HF_BIN"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "MODEL_ID=$MODEL_ID"
echo
echo "This script will:"
echo "1. log out any stale local Hugging Face session under HF_HOME"
echo "2. prompt you to log in again"
echo "3. verify access to the gated model config"
echo
echo "Important: if you use a fine-grained token, enable:"
echo "'Read access to contents of all public gated repositories you can access'"
echo

"$HF_BIN" auth logout || true
"$HF_BIN" auth login
"$HF_BIN" auth whoami

echo
echo "Verifying gated model access for $MODEL_ID ..."
"$PYTHON_BIN" - <<'PY'
import os
from transformers import AutoConfig

model_id = os.environ.get("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

cfg = AutoConfig.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
print(f"Loaded config successfully for: {model_id}")
print(f"Model type: {cfg.model_type}")
PY

echo
echo "Hugging Face setup looks good."
echo "You can now rerun training."
