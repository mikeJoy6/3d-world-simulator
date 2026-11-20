#!/usr/bin/env bash
# Create the inference environment (.venv) on top of an existing Python 3.10 + PyTorch (CUDA 13.0) install, without
# modifying it or ROS. Needs uv (https://docs.astral.sh/uv/).
#   WORLDSIM_TORCH_PYTHON=/path/to/python-with-torch tools/setup_runtime.sh
set -e
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
torch_python="${WORLDSIM_TORCH_PYTHON:?set WORLDSIM_TORCH_PYTHON to a Python 3.10 interpreter with PyTorch for CUDA 13.0}"
torch_packages="$("$torch_python" -c 'import sys, pathlib, torch; assert sys.version_info[:2] == (3,10) and torch.version.cuda == "13.0"; print(pathlib.Path(torch.__file__).resolve().parents[1])')"
/usr/bin/python3 -m venv --without-pip --system-site-packages .venv
touch .venv/COLCON_IGNORE
uv pip install --python .venv/bin/python --no-deps 'gsplat==1.5.3' 'spconv-cu126==2.3.8'
uv pip install --python .venv/bin/python 'numpy==1.26.4' 'h5py==3.16.0' 'cumm-cu126==0.7.11' 'pccm==0.4.16' 'ccimport==0.4.4' pyquaternion jaxtyping rich ninja 'nvidia-cuda-nvcc==13.0.88' 'nvidia-cuda-crt==13.0.88' 'nvidia-cuda-cccl==13.0.85' 'nvidia-nvvm==13.0.88' 'nvidia-cuda-runtime==13.0.96' 'scikit-image==0.24.0' 'trimesh==4.4.9'
printf '%s\n' "$torch_packages" > .venv/lib/python3.10/site-packages/worldsim_torch.pth
.venv/bin/python tools/link_cuda.py
