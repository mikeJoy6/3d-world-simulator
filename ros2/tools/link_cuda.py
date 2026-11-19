"""Expose the headers, compiler and runtime wheels as a CUDA toolkit for gsplat."""
import os
from pathlib import Path
import torch
root = Path(__file__).resolve().parents[1]
sdk = root / '.venv/lib/python3.10/site-packages/nvidia/cu13'
runtime = Path(torch.__file__).resolve().parents[1] / 'nvidia/cu13'
dest = root / '.runtime/cuda'
def link(src, target):
    if target.is_symlink() and not target.exists() and src.exists():
        target.unlink()
    if not target.is_symlink() and not target.exists():
        target.symlink_to(os.path.relpath(src, target.parent) if src.is_relative_to(root) else src)
for name in ('bin','nvvm'):
    dest.mkdir(parents=True, exist_ok=True)
    link(sdk / name, dest / name)
for name in ('include','lib'):
    target = dest / name
    target.mkdir(exist_ok=True)
    for source in (sdk, runtime):
        for child in (source / name).iterdir():
            link(child, target / child.name)
link(dest / 'lib', dest / 'lib64')
link(dest / 'lib/libcudart.so.13', dest / 'lib/libcudart.so')
