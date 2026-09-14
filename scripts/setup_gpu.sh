#!/usr/bin/env bash
set -euo pipefail
command -v nvidia-smi >/dev/null || { echo 'NVIDIA driver/nvidia-smi not found'; exit 1; }
nvidia-smi
if ! command -v nvidia-container-runtime >/dev/null 2>&1; then
  echo 'nvidia-container-runtime is not installed. Install NVIDIA Container Toolkit using your distribution package manager.'
  exit 1
fi
python3 - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))
PY
