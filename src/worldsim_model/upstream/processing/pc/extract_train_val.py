"""Split packed episodes into train/ and val/ via relative symlinks.

Creates <root>/train/episode_N.hdf5 -> ../packed/episode_N.hdf5 (and val/).
No copy, no transform; re-run any time with different --val-episodes (the
dirs are cleared and rebuilt).  `./pw pack` uses every episode with
id % 10 == 9 as val unless VAL_EPS is given.

Usage:
    python -m processing.pc.extract_train_val --dataset bimanual_cloth_v3 \\
        --val-episodes 9 19 29
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))
from processing import paths  # noqa: E402


def _episode_id(p: Path) -> int:
    m = re.search(r"episode_(\d+)\.hdf5$", p.name)
    if not m:
        raise ValueError(f"unrecognized episode filename: {p}")
    return int(m.group(1))


def _clear_split_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        if f.is_symlink() or f.is_file():
            f.unlink()


def split(dataset: str, val_episodes: set[int], copy: bool = False,
          pc_dir: Path = None, train_dir: Path = None, val_dir: Path = None) -> None:
    src_dir = pc_dir    if pc_dir    is not None else paths.packed_dir(dataset)
    train_d = train_dir if train_dir is not None else paths.train_dir(dataset)
    val_d   = val_dir   if val_dir   is not None else paths.val_dir(dataset)

    src_files = sorted(src_dir.glob("episode_*.hdf5"), key=_episode_id)
    if not src_files:
        raise SystemExit(f"no episode_*.hdf5 found in {src_dir} — run pack_dataset first")

    _clear_split_dir(train_d)
    _clear_split_dir(val_d)

    n_t = n_v = 0
    for f in src_files:
        ep = _episode_id(f)
        dest = (val_d if ep in val_episodes else train_d) / f.name
        if copy:
            import shutil
            shutil.copy2(f, dest)
        else:
            # Symlink. Use relative target so the dataset tree stays portable.
            rel = os.path.relpath(f, dest.parent)
            os.symlink(rel, dest)
        if ep in val_episodes:
            n_v += 1
        else:
            n_t += 1

    print(f"dataset: {dataset}")
    print(f"  source: {src_dir} ({len(src_files)} episodes)")
    print(f"  train:  {train_d} ({n_t} {'copies' if copy else 'symlinks'})")
    print(f"  val:    {val_d} ({n_v} {'copies' if copy else 'symlinks'})")
    missing = sorted(set(val_episodes) - {_episode_id(f) for f in src_files})
    if missing:
        print(f"  WARN: --val-episodes referenced {missing} which are not packed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--val-episodes", type=int, nargs="*", default=[],
                    help="episode ids to put in val/; everything else → train/")
    ap.add_argument("--copy", action="store_true",
                    help="copy files instead of symlinking")
    ap.add_argument("--pc-dir",    type=Path, default=None,
                    help="override packed/ source dir (default: paths.packed_dir(dataset))")
    ap.add_argument("--train-dir", type=Path, default=None,
                    help="override train/ output dir (default: paths.train_dir(dataset))")
    ap.add_argument("--val-dir",   type=Path, default=None,
                    help="override val/ output dir (default: paths.val_dir(dataset))")
    args = ap.parse_args()
    split(args.dataset, set(args.val_episodes), copy=args.copy,
          pc_dir=args.pc_dir, train_dir=args.train_dir, val_dir=args.val_dir)


if __name__ == "__main__":
    main()
