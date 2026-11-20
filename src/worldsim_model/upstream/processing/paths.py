"""Single source of truth for where data and external repos live.

Every processing script takes `--dataset <name>` and resolves its inputs and
outputs through these helpers.  All machine-specific locations are env-driven
with the defaults of the machine the pipeline was built on:

  PW_DATA_ROOT  processed tree   <root>/<ds>/{raw,fs_disp,sam3_masks,packed,train,val,*.json}
  PW_RAW_ROOT   raw tree         <root>/<ds>/episode_N.hdf5 (+ fs_disp/, sam3_masks/ next to it)
  FS_REPO       FoundationStereo checkout (with pretrained_models/23-51-11)
  SAM3_REPO     SAM3 checkout (not pip-installed; added to sys.path)
  PW_URDF       vx300s URDF for the tactile-FK loader (asset not shipped here)

`./pw` exports PW_DATA_ROOT / PW_RAW_ROOT from its own ADATA / APROC knobs;
inside <PW_DATA_ROOT>/<ds>/ the raw/, fs_disp/ and sam3_masks/ entries are
symlinks into PW_RAW_ROOT (created by `./pw setup`).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(os.environ.get("PW_DATA_ROOT", str(REPO_ROOT / "data")))
RAW_ROOT = Path(os.environ.get("PW_RAW_ROOT", str(REPO_ROOT / "raw")))

FS_REPO = Path(os.environ.get("FS_REPO", str(REPO_ROOT / "external" / "FoundationStereo")))
SAM3_REPO = Path(os.environ.get("SAM3_REPO", str(REPO_ROOT / "external" / "sam3")))

# URDF used by the training loader's tactile FK (datasets/aloha_packed.py,
# scripts/train_vae.py --urdf).  The vx300s asset tree is NOT part of this
# repo (see docs/LEFT_BEHIND.md); point PW_URDF at a checkout that has it.
DEFAULT_URDF = Path(os.environ.get(
    "PW_URDF", str(REPO_ROOT / "assets" / "robot" / "trossen_description" / "vx300s_tactile_thin.urdf")))


def root(name: str) -> Path:
    return DATA_ROOT / name


def raw_dir(name: str) -> Path:
    """Raw episode_N.hdf5 dir (the raw/ symlink inside the processed root)."""
    return root(name) / "raw"


def disp_dir(name: str) -> Path:
    return root(name) / "fs_disp"


def sam3_masks_dir(name: str) -> Path:
    return root(name) / "sam3_masks"


def packed_dir(name: str) -> Path:
    return root(name) / "packed"


def train_dir(name: str) -> Path:
    return root(name) / "train"


def val_dir(name: str) -> Path:
    return root(name) / "val"


def config_path(name: str) -> Path:
    return root(name) / "config.json"


def table_bbox_path(name: str) -> Path:
    return root(name) / "table_bbox.json"

