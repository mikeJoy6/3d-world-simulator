"""All-camera frame browser (fixed cams + wrist cams) for a raw episode (`./pw cams`).

Shows every `camera_*_color` stream of an episode side by side with a frame
slider. Optional overlay projects the SAM3-masked object (lifted to world
from the calibrated FIXED cams) into EVERY view: fixed cams through their
config.json calibration (+ episode_overrides), moving cams through the
per-frame pose sidecar cam_perframe/ep_<E>_cam<C>.npz, so wrist-cam
extrinsics can be verified frame by frame.

usage:
  python -m processing.viz.view_all_cams --dataset bimanual_cloth_v3 --episode 0 --port 8095
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import viser

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))
import h5video  # noqa: E402
from processing import paths                                                   # noqa: E402
from processing.config import DatasetConfig, episode_override_for, _euler_to_R  # noqa: E402

VIEW_W = 480


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--port", type=int, default=8095)
    args = ap.parse_args()

    cfg = DatasetConfig.load(paths.config_path(args.dataset))
    fixed_cams = sorted(int(k.split("_")[1]) for k in cfg.cameras)
    moving_cams = sorted(int(str(n).split("_")[-1])
                         for n in (cfg.extra.get("moving_cams") or []))
    _ov_cache = {}

    def apply_cal_ep(ep, cam_name, E):
        """config calibration, with this episode's episode_overrides entry if any."""
        if ep not in _ov_cache:
            _ov_cache[ep] = episode_override_for(cfg, ep)[0]
        cal = _ov_cache[ep].get(cam_name)
        if cal is None:
            return cfg.apply_calibration(cam_name, E)
        out = E.astype(np.float32).copy()
        out[:3, :3] = (_euler_to_R(*cal.rot_world_deg) @ E[:3, :3]
                       @ _euler_to_R(*cal.rot_local_deg))
        out[:3, 3] = E[:3, 3] + np.asarray(cal.translate, np.float32)
        return out

    server = viser.ViserServer(port=args.port)
    state = {"ep": args.episode, "raw": None, "dsp": None, "sam": {}, "T": 1,
             "cams": [], "perframe": {}}

    def open_ep(ep):
        for h in [state["raw"], state["dsp"], *state["sam"].values()]:
            try:
                h and h.close()
            except Exception:
                pass
        state["raw"] = h5video.File(paths.raw_dir(args.dataset) / f"episode_{ep}.hdf5", "r")
        state["sam"] = {}
        try:
            state["dsp"] = h5video.File(paths.disp_dir(args.dataset) / f"disp_episode_{ep}.hdf5", "r")
            for c in fixed_cams:
                p = paths.sam3_masks_dir(args.dataset) / f"episode_{ep}_camera_{c}.h5"
                if p.exists():
                    state["sam"][c] = h5video.File(p, "r")
        except Exception as e:
            print(f"[overlay unavailable for ep{ep}: {e}]")
            state["dsp"] = None
        # per-frame IR-cam poses of the moving cams (what the packer uses)
        state["perframe"] = {}
        for c in moving_cams:
            sc = paths.root(args.dataset) / "cam_perframe" / f"ep_{ep}_cam{c}.npz"
            if sc.exists():
                state["perframe"][c] = np.load(sc)["E_cam2world"].astype(np.float32)
        imgs = state["raw"]["observations/images"]
        state["cams"] = sorted({k.split("_color")[0] for k in imgs if k.endswith("_color")})
        state["T"] = imgs[f"{state['cams'][0]}_color"].shape[0]
        state["ep"] = ep

    open_ep(args.episode)

    with server.gui.add_folder("controls"):
        ep_box = server.gui.add_number("episode", initial_value=args.episode, min=0, step=1)
        frame = server.gui.add_slider("frame", min=0, max=state["T"] - 1, step=1,
                                      initial_value=0)
        overlay = server.gui.add_checkbox("project object into views", initial_value=False)
        status = server.gui.add_text("status", initial_value="", disabled=True)
    handles = {}
    with server.gui.add_folder("cameras"):
        for cam in state["cams"]:
            handles[cam] = server.gui.add_image(
                np.zeros((VIEW_W * 9 // 16, VIEW_W, 3), np.uint8), label=cam)

    def lift_obj(t):
        """World points of the SAM3 object from the fixed cams (subsampled)."""
        dsp, sam = state["dsp"], state["sam"]
        pts = []
        for c, sh in sam.items():
            disp = dsp[f"disp/camera_{c}"][t].astype(np.float32)
            K = dsp[f"meta/camera_{c}/K_ir"][t]
            E = apply_cal_ep(state["ep"], f"camera_{c}",
                             dsp[f"meta/camera_{c}/E_cam2world"][t])
            b = float(dsp[f"meta/camera_{c}/baseline"][t])
            m = (sh["mask_union"][t] > 0) & (disp > 1.0)
            vs, us = np.nonzero(m)
            if len(vs) == 0:
                continue
            sel = np.random.default_rng(0).choice(len(vs), min(800, len(vs)), replace=False)
            vs, us = vs[sel], us[sel]
            z = K[0, 0] * b / disp[vs, us]
            ok = (z > 0.15) & (z < 2.0)
            vs, us, z = vs[ok], us[ok], z[ok]
            x = (us - K[0, 2]) / K[0, 0] * z
            y = (vs - K[1, 2]) / K[1, 1] * z
            pts.append(np.stack([x, y, z], -1) @ E[:3, :3].T + E[:3, 3])
        return np.concatenate(pts) if pts else np.zeros((0, 3), np.float32)

    def cam_pose_world(cam, t):
        """world-from-RGB-camera 4x4 for projection."""
        raw = state["raw"]
        idx = int(cam.split("_")[1])
        T_r2i = raw[f"observations/images/{cam}_rgb_to_ir_extrinsic"][t].astype(np.float32)
        if idx in state["perframe"]:                       # moving cam: packed IR pose
            E = state["perframe"][idx][min(t, len(state["perframe"][idx]) - 1)]
            return E @ T_r2i
        if state["dsp"] is not None and f"meta/camera_{idx}" in state["dsp"]:
            E = state["dsp"][f"meta/camera_{idx}/E_cam2world"][t].astype(np.float32)
        else:
            E = np.linalg.inv(raw[f"observations/images/{cam}_extrinsics"][t].astype(np.float32))
        if idx in fixed_cams:
            E = apply_cal_ep(state["ep"], cam, E)
        return E @ T_r2i

    def render(t):
        t = int(np.clip(t, 0, state["T"] - 1))
        raw = state["raw"]
        wpts = lift_obj(t) if (overlay.value and state["dsp"] is not None) else None
        for cam in state["cams"]:
            img = np.ascontiguousarray(raw[f"observations/images/{cam}_color"][t])
            if wpts is not None and len(wpts):
                K = raw[f"observations/images/{cam}_rgb_intrinsics"][t]
                try:
                    w2c = np.linalg.inv(cam_pose_world(cam, t))
                    pc = wpts @ w2c[:3, :3].T + w2c[:3, 3]
                    z = pc[:, 2]
                    vis = z > 0.05
                    u = (pc[vis, 0] / z[vis] * K[0, 0] + K[0, 2]).astype(int)
                    v = (pc[vis, 1] / z[vis] * K[1, 1] + K[1, 2]).astype(int)
                    inb = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
                    img[v[inb], u[inb]] = (0, 255, 0)
                except Exception as e:
                    print(f"[overlay {cam}: {e}]")
            h = VIEW_W * img.shape[0] // img.shape[1]
            handles[cam].image = cv2.resize(img, (VIEW_W, h))
        status.value = f"ep {state['ep']}  frame {t}/{state['T']-1}  cams {len(state['cams'])}"

    @frame.on_update
    def _(_):
        render(frame.value)

    @ep_box.on_update
    def _(_):
        try:
            open_ep(int(ep_box.value))
            frame.max = state["T"] - 1
            frame.value = 0
            render(0)
        except Exception as e:
            status.value = f"episode {int(ep_box.value)}: {e}"

    @overlay.on_update
    def _(_):
        render(frame.value)

    render(0)
    print(f"view_all_cams on http://localhost:{args.port} — "
          f"{len(state['cams'])} cams, {state['T']} frames, "
          f"perframe poses for cams {sorted(state['perframe'])}")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
