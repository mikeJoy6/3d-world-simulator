"""Recreate train/ and val/ of a downloaded dataset from its splits.json (links into packed/).

    python scripts/link_splits.py data/bimanual_cloth_v3
"""
import json, os, sys
from pathlib import Path

root = Path(sys.argv[1])
for name, eps in json.load(open(root / "splits.json")).items():
    (root / name).mkdir(exist_ok=True)
    for e in eps:
        dst = root / name / f"episode_{e}.hdf5"
        if not dst.is_symlink() and not dst.exists():
            os.symlink(f"../packed/episode_{e}.hdf5", dst)
    print(f"{root / name}: {len(eps)} episodes")
