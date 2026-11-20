"""Occupancy-at-resolution explorer — PRE-PACK, straight from fs_disp + raw.

For a dataset that only has raw episodes + fs_disp (nothing packed yet), shows
per frame the fused multi-cam scene point cloud built EXACTLY like the packer
builds it — disp -> depth (LRC-masked when disp_right exists), backprojected,
placed with config.json's current camera calibration (cfg.apply_calibration,
same math as pack), cropped to config.json's bbox — PLUS the occupied voxels
of that cloud at a selectable grid resolution (16 / 32 / 64 / 128 / ...).

Grid resolution is set per axis (nx, ny, nz number fields — voxels may be
non-cubic), with a cubic-preset dropdown (16/32/64/128/...) that fills the
three fields proportionally to the bbox (N cells along the longest axis).
Editing any field flips the preset to "custom". The status line prints the
exact (nx,ny,nz) grid and per-axis voxel edge in mm.

The "reload config.json" button re-reads calibration + bbox, so you can tweak
in tweak_extrinsics (./pw camera), save there, and refresh here — no restart.

Usage (`./pw voxels <ds> [ep]`):
    python -m processing.viz.view_voxel_res --dataset bimanual_cloth_v3 --episode 0 --port 8097
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import viser

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))
import h5video  # noqa: E402
from processing import paths                                    # noqa: E402
from processing.config import DatasetConfig                     # noqa: E402
from processing.viz.tweak_extrinsics import (                   # noqa: E402
    back_project_to_cam_frame, lrc_mask)

_BOX_EDGES = np.array([
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)], dtype=np.int32)

RES_CHOICES = [8, 16, 32, 64, 128, 256]
WIRE_MAX = 15000   # occupied-cell budget for wireframe mode in "auto"


def cubic_dims(bbox, n_long):
    """Cubic voxels: n_long cells along the longest bbox axis -> (nx, ny, nz)."""
    ext = np.array(bbox[1::2], np.float32) - np.array(bbox[0::2], np.float32)
    vs = float(ext.max()) / n_long
    return np.maximum(1, np.round(ext / vs).astype(int))


def grid_for(bbox, dims):
    """Per-axis grid: dims (nx, ny, nz) -> (bmin, voxel size (3,), dims)."""
    bmin = np.array(bbox[0::2], np.float32)
    bmax = np.array(bbox[1::2], np.float32)
    dims = np.asarray(dims, int)
    vs = (bmax - bmin) / dims.astype(np.float32)
    return bmin, vs, dims


def voxelize(pts, rgb, bmin, vs, dims):
    """-> (centers (K,3), mean rgb (K,3) uint8, K). vs is a (3,) per-axis size."""
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    idx = np.floor((pts - bmin) / vs).astype(np.int64)
    ok = ((idx >= 0).all(1) & (idx[:, 0] < nx) & (idx[:, 1] < ny) & (idx[:, 2] < nz))
    idx, rgb = idx[ok], rgb[ok]
    if idx.shape[0] == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), 0
    flat = (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]
    uniq, inv = np.unique(flat, return_inverse=True)
    csum = np.zeros((uniq.size, 3), np.float64)
    np.add.at(csum, inv, rgb.astype(np.float64))
    cnt = np.bincount(inv, minlength=uniq.size).astype(np.float64)
    mean_rgb = (csum / cnt[:, None]).clip(0, 255).astype(np.uint8)
    ux = uniq // (ny * nz); uy = (uniq // nz) % ny; uz = uniq % nz
    centers = bmin + (np.stack([ux, uy, uz], -1).astype(np.float32) + 0.5) * vs
    return centers, mean_rgb, int(uniq.size)


def wireframes(centers, vs):
    """(K,3) centers + per-axis size (3,) -> (12K, 2, 3) line segments."""
    if centers.shape[0] == 0:
        return np.zeros((0, 2, 3), np.float32)
    vs = np.asarray(vs, np.float32)
    cmin = centers - vs / 2
    cmax = centers + vs / 2
    K = centers.shape[0]
    corners = np.empty((K, 8, 3), np.float32)
    for c in range(8):
        bx, by, bz = (c >> 2) & 1, (c >> 1) & 1, c & 1
        corners[:, c, 0] = cmax[:, 0] if bx else cmin[:, 0]
        corners[:, c, 1] = cmax[:, 1] if by else cmin[:, 1]
        corners[:, c, 2] = cmax[:, 2] if bz else cmin[:, 2]
    a = corners[:, _BOX_EDGES[:, 0]]
    b = corners[:, _BOX_EDGES[:, 1]]
    return np.stack([a, b], axis=2).reshape(K * 12, 2, 3)


def bbox_segments(bbox):
    x0, x1, y0, y1, z0, z1 = bbox
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float32)
    e = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7)], np.int32)
    return np.stack([c[e[:, 0]], c[e[:, 1]]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--lrc-thresh", type=float, default=2.0, help="pw pack default (LRC=2); 0 disables")
    ap.add_argument("--min-depth", type=float, default=0.10)
    ap.add_argument("--max-depth", type=float, default=2.0)
    args = ap.parse_args()

    cfg_path = paths.config_path(args.dataset)
    if not cfg_path.exists():
        raise SystemExit(f"no config.json at {cfg_path} — run './pw camera {args.dataset}' first")

    import re as _re
    episodes = []
    for p in sorted(paths.disp_dir(args.dataset).glob("disp_episode_*.hdf5"),
                    key=lambda p: int(_re.search(r"(\d+)", p.name).group(1))):
        e = int(_re.search(r"(\d+)", p.name).group(1))
        if (paths.raw_dir(args.dataset) / f"episode_{e}.hdf5").exists():
            episodes.append(e)
    if not episodes:
        raise SystemExit(f"no disp+raw episode pairs under {paths.disp_dir(args.dataset)}")
    if args.episode not in episodes:
        raise SystemExit(f"episode {args.episode} not available (have {episodes[0]}..{episodes[-1]})")

    state = {"df": None, "sf": None, "T": 0, "has_right": False, "ep": None,
             "cfg": None, "bbox": None, "cams": []}

    def load_config():
        state["cfg"] = DatasetConfig.load(cfg_path)
        state["bbox"] = list(state["cfg"].bbox)

    def open_episode(ep):
        for h in ("df", "sf"):
            if state[h] is not None:
                try: state[h].close()
                except Exception: pass
        state["df"] = h5video.File(paths.disp_dir(args.dataset) / f"disp_episode_{ep}.hdf5", "r")
        state["sf"] = h5video.File(paths.raw_dir(args.dataset) / f"episode_{ep}.hdf5", "r")
        state["T"] = int(state["df"].attrs["num_frames"])
        state["has_right"] = "disp_right" in state["df"]
        state["cams"] = sorted(int(k.split("_")[1]) for k in state["df"]["disp"].keys()
                               if k.startswith("camera_"))
        state["ep"] = ep
        print(f"episode {ep}  T={state['T']}  cams={state['cams']}  "
              f"disp_right={state['has_right']}", flush=True)

    load_config()
    open_episode(args.episode)
    print(f"{args.dataset}: {len(episodes)} episodes, bbox={state['bbox']}"
          + ("" if state["has_right"] else "  (no disp_right — LRC OFF)"), flush=True)

    # T_max for the shared frame slider (viser sliders can't be resized live).
    T_max = 0
    for e in episodes:
        with h5video.File(paths.disp_dir(args.dataset) / f"disp_episode_{e}.hdf5", "r") as f:
            T_max = max(T_max, int(f.attrs["num_frames"]))

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    ep_dd = server.gui.add_dropdown("episode", options=[str(e) for e in episodes],
                                    initial_value=str(args.episode))
    frame_sl = server.gui.add_slider("frame", min=0, max=T_max - 1, step=1,
                                     initial_value=min(100, state["T"] - 1))
    d0 = cubic_dims(state["bbox"], 32)
    res_dd = server.gui.add_dropdown("cubic preset (longest axis)",
                                     options=["custom"] + [str(r) for r in RES_CHOICES],
                                     initial_value="32")
    nx_w = server.gui.add_number("nx", initial_value=int(d0[0]), min=1, max=1024, step=1)
    ny_w = server.gui.add_number("ny", initial_value=int(d0[1]), min=1, max=1024, step=1)
    nz_w = server.gui.add_number("nz", initial_value=int(d0[2]), min=1, max=1024, step=1)
    style_dd = server.gui.add_dropdown("voxel style", options=["auto", "wireframe", "points"],
                                       initial_value="auto")
    show_pc = server.gui.add_checkbox("show scene points", initial_value=True)
    show_vox = server.gui.add_checkbox("show voxels", initial_value=True)
    bbox_crop = server.gui.add_checkbox("crop scene to bbox", initial_value=True)
    stride_w = server.gui.add_slider("stride", min=1, max=8, step=1, initial_value=args.stride)
    lrc_w = server.gui.add_slider("LRC thresh (px, 0=off)", min=0.0, max=10.0, step=0.5,
                                  initial_value=args.lrc_thresh if state["has_right"] else 0.0)
    pt_size = server.gui.add_slider("scene point size (m)", min=0.0005, max=0.01,
                                    step=0.0005, initial_value=0.002)
    reload_btn = server.gui.add_button("reload config.json (calib + bbox)")
    status = server.gui.add_text("status", initial_value="-", disabled=True)

    bbox_handle = server.scene.add_line_segments(
        "/bbox", points=bbox_segments(state["bbox"]),
        colors=np.array((50, 200, 230), np.uint8), line_width=2.0)
    scene = server.scene.add_point_cloud("/scene", points=np.zeros((1, 3), np.float32),
                                         colors=np.zeros((1, 3), np.uint8), point_size=0.002)
    vox_pts = server.scene.add_point_cloud("/vox_pts", points=np.zeros((1, 3), np.float32),
                                           colors=np.zeros((1, 3), np.uint8),
                                           point_size=0.01, point_shape="square")
    vox_wire = server.scene.add_line_segments("/vox_wire", points=np.zeros((0, 2, 3), np.float32),
                                              colors=np.array((255, 220, 40), np.uint8),
                                              line_width=1.5)

    cache = {"key": None, "xyz": None, "rgb": None}

    def load_frame(t, stride, lrc):
        """Fused world-frame cloud with the CURRENT config calibration applied."""
        df, sf, cfg = state["df"], state["sf"], state["cfg"]
        pts_all, rgb_all = [], []
        for c in state["cams"]:
            disp = df[f"disp/camera_{c}"][t].astype(np.float32)
            K_ir = df[f"meta/camera_{c}/K_ir"][t]
            K_rgb = df[f"meta/camera_{c}/K_rgb"][t]
            T_r2i = df[f"meta/camera_{c}/T_rgb_to_ir"][t]
            E_orig = df[f"meta/camera_{c}/E_cam2world"][t]
            B = float(df[f"meta/camera_{c}/baseline"][t])
            color = sf[f"/observations/images/camera_{c}_color"][t]
            depth = (float(K_ir[0, 0]) * B) / np.maximum(disp, 1e-3)
            if lrc > 0 and state["has_right"]:
                m = lrc_mask(disp, df[f"disp_right/camera_{c}"][t].astype(np.float32),
                             thresh=lrc)
                depth = np.where(m, depth, 0.0)
            pts_cam, rgb = back_project_to_cam_frame(
                depth, color, K_ir, K_rgb, T_r2i,
                stride=stride, min_z=args.min_depth, max_z=args.max_depth)
            E = cfg.apply_calibration(f"camera_{c}", E_orig)
            pts_all.append((pts_cam @ E[:3, :3].T + E[:3, 3]).astype(np.float32))
            rgb_all.append(rgb)
        return np.concatenate(pts_all, 0), np.concatenate(rgb_all, 0)

    def render():
        T = state["T"]
        t = min(int(frame_sl.value), T - 1)
        stride = int(stride_w.value)
        lrc = float(lrc_w.value)
        key = (state["ep"], t, stride, lrc)
        if cache["key"] != key:
            cache["xyz"], cache["rgb"] = load_frame(t, stride, lrc)
            cache["key"] = key
        xyz, rgb = cache["xyz"], cache["rgb"]

        b = state["bbox"]
        inb = ((xyz[:, 0] >= b[0]) & (xyz[:, 0] <= b[1]) &
               (xyz[:, 1] >= b[2]) & (xyz[:, 1] <= b[3]) &
               (xyz[:, 2] >= b[4]) & (xyz[:, 2] <= b[5]))
        sxyz, srgb = (xyz[inb], rgb[inb]) if bbox_crop.value else (xyz, rgb)

        scene.visible = bool(show_pc.value)
        if show_pc.value:
            scene.points = sxyz if len(sxyz) else np.zeros((1, 3), np.float32)
            scene.colors = srgb if len(sxyz) else np.zeros((1, 3), np.uint8)
            scene.point_size = float(pt_size.value)

        dims_req = (int(nx_w.value), int(ny_w.value), int(nz_w.value))
        bmin, vs, dims = grid_for(b, dims_req)
        centers, vrgb, K = voxelize(xyz[inb], rgb[inb], bmin, vs, dims)  # voxels always bbox-only
        style = style_dd.value
        if style == "auto":
            style = "wireframe" if K <= WIRE_MAX else "points"
        use_wire = show_vox.value and style == "wireframe"
        use_pts = show_vox.value and style == "points"
        vox_wire.visible = use_wire
        vox_pts.visible = use_pts
        if use_wire:
            segs = wireframes(centers, vs)
            vox_wire.points = segs
            # colors must be re-sent at the new segment count, else the stale
            # (old-N,2,3) array mismatches and nothing is drawn
            vox_wire.colors = np.tile(np.array([255, 220, 40], np.uint8),
                                      (segs.shape[0], 2, 1))
        if use_pts:
            vox_pts.points = centers if K else np.zeros((1, 3), np.float32)
            vox_pts.colors = vrgb if K else np.zeros((1, 3), np.uint8)
            vox_pts.point_size = float(vs.min()) * 0.95

        total = int(np.prod(dims))
        vox_mm = "x".join(f"{v*1000:.1f}" for v in vs)
        status.value = (f"ep {state['ep']}  frame {t}/{T-1}  grid {dims[0]}x{dims[1]}x{dims[2]} "
                        f"(vox {vox_mm}mm)  occupied {K:,}/{total:,} "
                        f"({100.0*K/max(1,total):.2f}%)  in-bbox pts {int(inb.sum()):,}  [{style}]")

    @ep_dd.on_update
    def _switch_ep(_):
        ep = int(ep_dd.value)
        if ep == state["ep"]:
            return
        open_episode(ep)
        if int(frame_sl.value) > state["T"] - 1:
            frame_sl.value = state["T"] - 1   # triggers render via its own cb
        else:
            render()

    @reload_btn.on_click
    def _reload(_):
        load_config()
        bbox_handle.points = bbox_segments(state["bbox"])
        cache["key"] = None   # calibration changed -> recompute cloud
        render()
        status.value = "config.json reloaded: " + status.value

    # preset <-> per-axis sync (suppress flag breaks the update loop)
    sup = {"v": False}

    @res_dd.on_update
    def _preset(_):
        if sup["v"] or res_dd.value == "custom":
            return
        d = cubic_dims(state["bbox"], int(res_dd.value))
        sup["v"] = True
        nx_w.value, ny_w.value, nz_w.value = int(d[0]), int(d[1]), int(d[2])
        sup["v"] = False
        render()

    def _axis_edit(_):
        if sup["v"]:
            return
        sup["v"] = True
        res_dd.value = "custom"
        sup["v"] = False
        render()

    for w in (nx_w, ny_w, nz_w):
        w.on_update(_axis_edit)
    for w in (frame_sl, style_dd, show_pc, show_vox, bbox_crop,
              stride_w, lrc_w, pt_size):
        w.on_update(lambda _: render())
    render()
    print(f"server: http://localhost:{args.port}", flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
