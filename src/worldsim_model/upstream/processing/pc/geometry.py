"""Per-pixel geometry helpers shared by the packer, SOR bake and crop tool.

    apply_tweak          bake a config.json camera calibration into E_cam2world
    lrc_mask             left-right disparity consistency (needs disp_right)
    _bilinear_sample_rgb sample the colour image at IR-projected (u, v)
    sor_mask             statistical outlier removal on the fused cloud
    _tweaks_from_config / _base_offset_from_config  picklable config views
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from processing.config import DatasetConfig, cam_calibration_to_legacy_tweak

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None


def _euler_to_R(roll_deg, pitch_deg, yaw_deg):
    import transforms3d
    return transforms3d.euler.euler2mat(
        np.deg2rad(roll_deg), np.deg2rad(pitch_deg), np.deg2rad(yaw_deg),
        axes="sxyz").astype(np.float32)


def apply_tweak(E_base, tweak):
    """R_out = R_world @ E.R @ R_local ; t_out = E.t + [dx,dy,dz]
    (identical to DatasetConfig.apply_calibration, on the flat dict form)."""
    g = lambda k: float(tweak.get(k, 0.0))
    droll, dpitch, dyaw = g("droll"), g("dpitch"), g("dyaw")
    wdroll, wdpitch, wdyaw = g("wdroll"), g("wdpitch"), g("wdyaw")
    dx, dy, dz = g("dx"), g("dy"), g("dz")
    if all(abs(v) < 1e-9 for v in (droll, dpitch, dyaw,
                                     wdroll, wdpitch, wdyaw,
                                     dx, dy, dz)):
        return E_base.astype(np.float32)
    R_local = _euler_to_R(droll, dpitch, dyaw)
    R_world = _euler_to_R(wdroll, wdpitch, wdyaw)
    out = E_base.astype(np.float32).copy()
    out[:3, :3] = R_world @ E_base[:3, :3] @ R_local
    out[:3, 3]  = E_base[:3, 3]  + np.array([dx, dy, dz], np.float32)
    return out


def lrc_mask(disp_l, disp_r, thresh=1.0):
    """Left-right consistency mask on the LEFT-IR disparity.

    For each left pixel (y, x) with disparity d_L, the matching right-image
    column is x_R = round(x - d_L). A pixel survives only if the right-IR-
    referenced disparity at that column agrees: |d_L - d_R[y, x_R]| <= thresh.
    Pixels whose match falls outside the right image are dropped (occluded).

    `disp_r` is the right-referenced map produced by `run_fs --pair both`
    (run_fs.fs_disp_right), in the right-IR pixel frame.
    """
    H, W = disp_l.shape
    xs = np.arange(W, dtype=np.float32)[None, :]
    x_R = np.round(xs - disp_l).astype(np.int32)
    in_R = (x_R >= 0) & (x_R < W)
    x_R_c = np.clip(x_R, 0, W - 1)
    rows = np.arange(H, dtype=np.int32)[:, None]
    d_r_match = disp_r[rows, x_R_c]
    return in_R & (np.abs(disp_l - d_r_match) <= thresh)


def _bilinear_sample_rgb(img, u, v):
    """Bilinear-sample img at (u, v). Returns (N, 3) uint8.

    Uses cv2.remap (~70x faster than a numpy 3-channel loop) when cv2 is
    available. cv2's fixed-point internal interpolation can differ by +-2/255
    from a true float bilinear, invisible for training.
    """
    n = u.shape[0]
    if _cv2 is not None and n > 0:
        # cv2.remap requires map cols < 32767, so reshape into a rectangular
        # 2D map. Pad to a multiple of `cols`, drop the pad at the end.
        cols = 4096
        rows = (n + cols - 1) // cols
        pad = rows * cols - n
        if pad:
            u = np.concatenate([u.astype(np.float32, copy=False),
                                np.zeros(pad, dtype=np.float32)])
            v = np.concatenate([v.astype(np.float32, copy=False),
                                np.zeros(pad, dtype=np.float32)])
        out = _cv2.remap(img,
                         u.reshape(rows, cols),
                         v.reshape(rows, cols),
                         interpolation=_cv2.INTER_LINEAR,
                         borderMode=_cv2.BORDER_REPLICATE)
        return out.reshape(-1, 3)[:n]
    # Fallback: pure-numpy bilinear (slow but no cv2 dependency).
    H, W = img.shape[:2]
    u0 = np.floor(u).astype(np.int32); v0 = np.floor(v).astype(np.int32)
    a = (u - u0).astype(np.float32); b = (v - v0).astype(np.float32)
    u0c = np.clip(u0, 0, W - 2); v0c = np.clip(v0, 0, H - 2)
    out = np.zeros((u.shape[0], 3), dtype=np.float32)
    for ch in range(3):
        top = (1 - a) * img[v0c, u0c, ch] + a * img[v0c, u0c + 1, ch]
        bot = (1 - a) * img[v0c + 1, u0c, ch] + a * img[v0c + 1, u0c + 1, ch]
        out[:, ch] = (1 - b) * top + b * bot
    return out.astype(np.uint8)


def sor_mask(pts, k, std_ratio, n_threads=-1):
    """Statistical outlier removal. `n_threads` is the cKDTree.query worker
    count: -1 = all cores (over-subscribes if the caller already runs several
    processes), positive = exact count."""
    if len(pts) < k + 1:
        return np.ones(len(pts), dtype=bool)
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k + 1, workers=n_threads)
    mean_d = dists[:, 1:].mean(axis=1)
    mu, sd = float(mean_d.mean()), float(mean_d.std())
    return mean_d <= mu + std_ratio * sd


def _tweaks_from_config(cfg: DatasetConfig) -> dict:
    """{cam_idx -> flat tweak dict} so pool workers (which call apply_tweak)
    don't need to know about CamCalibration."""
    out = {}
    for cam_name, cal in cfg.cameras.items():
        idx = int(cam_name.split("_")[1]) if cam_name.startswith("camera_") else int(cam_name)
        out[idx] = cam_calibration_to_legacy_tweak(cal)
    return out


def _base_offset_from_config(cfg: DatasetConfig) -> dict:
    """{arm_name -> {x,y,z,roll_deg,pitch_deg,yaw_deg,finger_scale,finger_offset_m}}:
    the picklable form written into the packed `robot_base` attr, which the
    training loader feeds to datasets.aloha_pc.base_offset_matrix()."""
    out = {}
    for arm, rb in cfg.robot_base.items():
        out[arm] = {
            "x": rb.translate[0], "y": rb.translate[1], "z": rb.translate[2],
            "roll_deg": rb.rot_deg[0], "pitch_deg": rb.rot_deg[1], "yaw_deg": rb.rot_deg[2],
            "finger_scale": float(rb.finger_scale),
            "finger_offset_m": float(rb.finger_offset_m),
        }
    return out
