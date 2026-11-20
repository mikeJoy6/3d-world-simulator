"""Per-dataset config.json: camera calibration + world bbox + robot base.

One JSON per dataset under <PW_DATA_ROOT>/<ds>/config.json.  Keys this
schema does not model (moving_cams, camera_depth_limits, color_calib,
episode_overrides, require_moving_camera_sidecars, ...) are kept verbatim in
`extra` so tool saves never drop them; the packer reads them from there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import shutil
import datetime as _dt

import numpy as np


def _euler_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    import transforms3d
    return transforms3d.euler.euler2mat(
        np.deg2rad(roll_deg), np.deg2rad(pitch_deg), np.deg2rad(yaw_deg),
        axes="sxyz",
    ).astype(np.float32)


@dataclass
class CamCalibration:
    translate: list           # [dx, dy, dz] meters
    rot_local_deg: list       # [droll, dpitch, dyaw]  applied as post-multiply on E.R
    rot_world_deg: list       # [wdroll, wdpitch, wdyaw] applied as pre-multiply on E.R

    @classmethod
    def zero(cls) -> "CamCalibration":
        return cls(translate=[0.0, 0.0, 0.0],
                   rot_local_deg=[0.0, 0.0, 0.0],
                   rot_world_deg=[0.0, 0.0, 0.0])

    def to_dict(self) -> dict:
        return {
            "translate":     list(map(float, self.translate)),
            "rot_local_deg": list(map(float, self.rot_local_deg)),
            "rot_world_deg": list(map(float, self.rot_world_deg)),
        }


@dataclass
class RobotBase:
    translate: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rot_deg:   list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    finger_scale: float = 1.0
    finger_offset_m: float = 0.0

    def to_dict(self) -> dict:
        return {
            "translate":       list(map(float, self.translate)),
            "rot_deg":         list(map(float, self.rot_deg)),
            "finger_scale":    float(self.finger_scale),
            "finger_offset_m": float(self.finger_offset_m),
        }


def _rb_from_dict(d: dict) -> RobotBase:
    return RobotBase(
        translate=d.get("translate", [0.0, 0.0, 0.0]),
        rot_deg=d.get("rot_deg", [0.0, 0.0, 0.0]),
        finger_scale=float(d.get("finger_scale", 1.0)),
        finger_offset_m=float(d.get("finger_offset_m", 0.0)),
    )


# Source-HDF5 conventions for bimanual aloha:
#   robot_base_pose_in_world[t, ARM_INDEX[name]]    is that arm's base pose.
#   tactile keys are named "<arm>_robot_<finger>_finger".
ARM_INDEX = {"left": 0, "right": 1}


@dataclass
class DatasetConfig:
    dataset: str
    cameras: dict             # {"camera_0": CamCalibration, ...}
    bbox: list                # [xmin,xmax,ymin,ymax,zmin,zmax]
    # robot_base: dict[arm_name -> RobotBase]. Single-arm datasets use {"left": ...}.
    # Bimanual datasets use {"left": ..., "right": ...}. Arm names match the
    # tactile-key prefix (left_robot_*, right_robot_*) in the source HDF5 and
    # the `robot_base_pose_in_world[t, arm_idx]` index (left=0, right=1).
    robot_base: dict = field(default_factory=lambda: {"left": RobotBase()})
    # If true, tactile signals from `left_robot_*` topics belong to the arm
    # named "right" (and vice versa): some bimanual ALOHA rigs have the
    # tactile topic names swapped vs the joint convention. Written into the
    # packed attrs (`tactile_stream_swap`) for the training loader.
    tactile_stream_swap: bool = False
    notes: str = ""
    # Keys this schema does not model (moving_cams, color_calib, ...) — captured
    # at load() and merged back at to_dict() so tool saves never drop them.
    extra: dict = field(default_factory=dict)

    # ----- IO --------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "DatasetConfig":
        path = Path(path)
        with open(path) as f:
            d = json.load(f)
        cams = {}
        for k, v in d.get("cameras", {}).items():
            cal = v["calibration"] if "calibration" in v else v
            cams[k] = CamCalibration(
                translate=cal["translate"],
                rot_local_deg=cal["rot_local_deg"],
                rot_world_deg=cal["rot_world_deg"],
            )
        # robot_base is either the new dict-of-arms form or the legacy flat
        # form (single-arm). Detect by presence of `translate` at the top level.
        rb_d = d.get("robot_base") or {}
        if "translate" in rb_d or "rot_deg" in rb_d:
            # Legacy flat (single-arm). Lift into {"left": ...}.
            rb = {"left": _rb_from_dict(rb_d)}
        else:
            rb = {arm: _rb_from_dict(v) for arm, v in rb_d.items()} or {"left": RobotBase()}
        _known = {"dataset", "cameras", "bbox", "robot_base",
                  "tactile_stream_swap", "notes"}
        return cls(
            dataset=d["dataset"],
            cameras=cams,
            bbox=list(d["bbox"]),
            robot_base=rb,
            tactile_stream_swap=bool(d.get("tactile_stream_swap", False)),
            notes=d.get("notes", ""),
            extra={k: v for k, v in d.items() if k not in _known},
        )

    def save(self, path: str | Path, archive_existing: bool = True) -> None:
        """Write to `path`. If a file already exists there and `archive_existing`,
        copy it to a sibling `.config_history/<timestamp>__<basename>` first."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if archive_existing and path.exists():
            hist = path.parent / ".config_history"
            hist.mkdir(exist_ok=True)
            ts = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            shutil.copy2(path, hist / f"{ts}__{path.name}")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "cameras": {
                k: {"calibration": cal.to_dict()}
                for k, cal in self.cameras.items()
            },
            "bbox": list(map(float, self.bbox)),
            "robot_base": {arm: rb.to_dict() for arm, rb in self.robot_base.items()},
            "tactile_stream_swap": bool(self.tactile_stream_swap),
            "notes": self.notes,
            **self.extra,
        }

    # ----- Calibration application ----------------------------------------

    def apply_calibration(self, cam_name: str, E_base: np.ndarray) -> np.ndarray:
        """Bake this camera's calibration into a per-frame extrinsic
        (same math as processing.pc.geometry.apply_tweak):
            R_out = R_world @ E_base.R @ R_local
            t_out = E_base.t + translate
        """
        cal = self.cameras.get(cam_name)
        if cal is None:
            cal = CamCalibration.zero()
        dx, dy, dz = cal.translate
        droll, dpitch, dyaw = cal.rot_local_deg
        wdroll, wdpitch, wdyaw = cal.rot_world_deg
        if all(abs(v) < 1e-9 for v in (droll, dpitch, dyaw,
                                       wdroll, wdpitch, wdyaw,
                                       dx, dy, dz)):
            return E_base.astype(np.float32)
        R_local = _euler_to_R(droll, dpitch, dyaw)
        R_world = _euler_to_R(wdroll, wdpitch, wdyaw)
        out = E_base.astype(np.float32).copy()
        out[:3, :3] = R_world @ E_base[:3, :3] @ R_local
        out[:3, 3]  = E_base[:3, 3] + np.array([dx, dy, dz], np.float32)
        return out


# --------------------------------------------------------------------------
# Flat "tweak" dict form used by the per-frame packing helpers and the
# viser sliders (dx,dy,dz,droll,dpitch,dyaw,wdroll,wdpitch,wdyaw).
# --------------------------------------------------------------------------

def cam_calibration_to_legacy_tweak(cal: CamCalibration) -> dict:
    """CamCalibration -> flat tweak dict (consumed by geometry.apply_tweak)."""
    dx, dy, dz = cal.translate
    droll, dpitch, dyaw = cal.rot_local_deg
    wdroll, wdpitch, wdyaw = cal.rot_world_deg
    return {
        "dx": dx, "dy": dy, "dz": dz,
        "droll": droll, "dpitch": dpitch, "dyaw": dyaw,
        "wdroll": wdroll, "wdpitch": wdpitch, "wdyaw": wdyaw,
    }


# --------------------------------------------------------------------------
# Per-episode-range overrides (config key "episode_overrides", kept in
# cfg.extra). Use case: a camera got physically bumped partway through a
# collection, so episodes [lo, hi] need a different extrinsic tweak and/or
# color gains for that camera while every other episode keeps the base
# config. Schema (list, later entries win on overlap):
#   "episode_overrides": [
#     {"range": [41, 131],                       # inclusive; or "episodes": [..]
#      "cameras":     {"camera_1": {"calibration": {...CamCalibration...}}},
#      "color_calib": {"camera_1": {"gain": [...], "offset": [...]}}}
#   ]
# Both sections are PARTIAL: only the listed cameras are overridden.
# --------------------------------------------------------------------------

def episode_override_for(cfg: DatasetConfig, ep: int):
    """Merged override for episode `ep`:
    ({cam_name -> CamCalibration}, {cam_name -> color_calib dict})."""
    cams, ccal = {}, {}
    for ov in (cfg.extra or {}).get("episode_overrides") or []:
        rng = ov.get("range")
        hit = bool(rng) and len(rng) == 2 and int(rng[0]) <= ep <= int(rng[1])
        hit = hit or ep in (ov.get("episodes") or [])
        if not hit:
            continue
        for k, v in (ov.get("cameras") or {}).items():
            cal = v["calibration"] if "calibration" in v else v
            cams[k] = CamCalibration(translate=cal["translate"],
                                     rot_local_deg=cal["rot_local_deg"],
                                     rot_world_deg=cal["rot_world_deg"])
        ccal.update(ov.get("color_calib") or {})
    return cams, ccal


def update_episode_override(config_path: str | Path, lo: int, hi: int, *,
                            cameras: dict | None = None,
                            color_calib: dict | None = None) -> None:
    """Create/update the episode_overrides entry with range [lo, hi]
    (matched exactly). `cameras` is {cam_name -> CamCalibration}; only the
    passed sections are replaced inside the entry, and only the passed cams
    within them (other cams in the entry are preserved)."""
    cfg = DatasetConfig.load(config_path)
    ovs = list((cfg.extra or {}).get("episode_overrides") or [])
    entry = None
    for ov in ovs:
        if list(map(int, ov.get("range") or [])) == [int(lo), int(hi)]:
            entry = ov
            break
    if entry is None:
        entry = {"range": [int(lo), int(hi)]}
        ovs.append(entry)
    if cameras:
        sec = dict(entry.get("cameras") or {})
        for k, cal in cameras.items():
            sec[k] = {"calibration": cal.to_dict()}
        entry["cameras"] = sec
    if color_calib:
        sec = dict(entry.get("color_calib") or {})
        sec.update(color_calib)
        entry["color_calib"] = sec
    cfg.extra["episode_overrides"] = ovs
    cfg.save(config_path)


# --------------------------------------------------------------------------
# Partial in-place edits for the viser tweaker (only ONE section changes,
# everything else is preserved; cfg.save() archives the previous file).
# --------------------------------------------------------------------------

def update_bbox(config_path: str | Path, bbox: list[float]) -> None:
    cfg = DatasetConfig.load(config_path)
    cfg.bbox = list(map(float, bbox))
    cfg.save(config_path)


def update_all_camera_calibrations(config_path: str | Path,
                                   cameras: dict) -> None:
    """cameras: {"camera_0": CamCalibration, ...}. Replaces the whole
    cameras section atomically."""
    cfg = DatasetConfig.load(config_path)
    cfg.cameras = dict(cameras)
    cfg.save(config_path)


def legacy_tweaks_dict(cfg: DatasetConfig) -> dict:
    """{cam_idx (int) -> flat tweak dict} for every camera in the config."""
    out = {}
    for cam_name, cal in cfg.cameras.items():
        idx = int(cam_name.split("_")[1]) if cam_name.startswith("camera_") else int(cam_name)
        out[idx] = cam_calibration_to_legacy_tweak(cal)
    return out
