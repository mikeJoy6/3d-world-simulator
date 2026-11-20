"""Fail closed until every episode has complete packed labels, robot label cache,
gripper halo masks (matching the current cam_crops.json) and SAM3 masks.
Run before launching a VAE training on a dataset root.

    python scripts/preflight_labels.py --root <PW_DATA_ROOT>/bimanual_box
"""
import os
import argparse
import json
from pathlib import Path


import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import h5video  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="dataset root (<PW_DATA_ROOT>/<ds>)")
    a = p.parse_args()
    errors = []
    crops = json.loads((a.root / "cam_crops.json").read_text())
    packs = sorted((a.root / "packed").glob("episode_*.hdf5"),
                   key=lambda q: int(q.stem.split("_")[-1]))
    if not packs:
        raise SystemExit(f"no packed episodes under {a.root / 'packed'}")
    for pack in packs:
        ep = int(pack.stem.split("_")[-1])
        with h5video.File(pack) as f:
            n = int(f.attrs["num_frames"])
            cams = [int(c) for c in f.attrs["cam_ids"]]
            if not f.attrs.get("has_obj_mask", False):
                errors.append(f"{ep}: object labels not packed (pack ran with --no-obj?)")
            for c in cams:
                key = f"img/camera_{c}/obj_mask"
                if key not in f or len(f[key]) != n:
                    errors.append(f"{ep}: incomplete packed labels cam{c}")
                if f"render/camera_{c}/jpg" not in f:
                    errors.append(f"{ep}: no render bake cam{c}")
        for sub in ("robot_label_cache", "gripper_supervision/masks_halo6_surface"):
            path = a.root / sub / f"episode_{ep}.h5"
            if not path.exists():
                errors.append(f"{ep}: missing {sub}")
                continue
            with h5video.File(path) as f:
                if int(f.attrs["frames"]) != n:
                    errors.append(f"{ep}: incomplete {sub}")
                if sub.startswith("gripper"):
                    if f.attrs.get("halo_fine_voxels", 0) != 6:
                        errors.append(f"{ep}: wrong gripper halo")
                    for c in cams:
                        # compare with the crop the render bake used (cam_crops.json clamped to this
                        # episode's image size: policy_eval mixes 1280x720 and 640x360 wrist clips)
                        with h5video.File(pack) as pf:
                            want = [int(v) for v in pf[f"render/camera_{c}"].attrs["crop"]] if f"render/camera_{c}" in pf else crops[f"camera_{c}"]
                        if [int(v) for v in f[f"camera_{c}"].attrs["crop"]] != want:
                            errors.append(f"{ep}: stale gripper crop {c}")
        _skip_sam = {int(x) for x in os.environ.get("PREFLIGHT_SKIP_SAM_CAMS", "").split(",") if x.strip()}   # e.g. plug_v3: wrist cam 3 never segmented (render obj weighting off there)
        for c in cams:
            if c in _skip_sam: continue
            path = a.root / "sam3_masks" / f"episode_{ep}_camera_{c}.h5"
            if not path.exists():
                errors.append(f"{ep}: missing SAM cam{c}")
            else:
                with h5video.File(path) as f:
                    fi = f["frame_indices"]
                    if len(fi) != n or int(fi[-1]) != n - 1:
                        errors.append(f"{ep}: incomplete SAM cam{c}")
    if errors:
        raise SystemExit("Preparation incomplete; no training started:\n"
                         + "\n".join(errors[:15]) + f"\n({len(errors)} issues total)")
    print(f"{a.root.name}: {len(packs)} episodes ready")


if __name__ == "__main__":
    main()
