"""Viser viewer for processed (packed) episodes: the data as training sees it, straight from a download.

Reads only `data/<dataset>/packed/episode_<E>.hdf5`. Scroll through episodes and frames and look at

  the fused point cloud of all cameras (the training cloud, or the full depth of every camera),
  the cameras, with their poses and images, moving wrist cameras included,
  the workspace box and the occupied voxels at a grid resolution of your choice,
  the manipulated object's points, highlighted.

    ./pw camera <dataset>      # point cloud, cameras and workspace box
    ./pw voxels <dataset>      # the same with the occupied voxels switched on
    ./pw cams   <dataset>      # the same with the camera images switched on
    python -m processing.viz.view_packed --dataset assembly --port 8099
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import viser

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5video  # noqa: E402
from processing import paths  # noqa: E402
from processing.viz.view_voxel_res import bbox_segments, grid_for, voxelize, wireframes  # noqa: E402

WIRE_MAX = 15000  # occupied cells drawn as wire boxes; above this they are drawn as points
VIEW_W = 420


def bits(row, h, w):
    return np.unpackbits(np.asarray(row))[: h * w].reshape(h, w).astype(bool)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--episode", type=int, default=None, help="first episode shown (default: the lowest id)")
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--show", default="camera", choices=("camera", "voxels", "cams"), help="what is switched on at start")
    a = p.parse_args()

    files = {int(f.stem.split("_")[1]): f for f in paths.packed_dir(a.dataset).glob("episode_*.hdf5")}
    if not files:
        raise SystemExit(f"no packed episodes in {paths.packed_dir(a.dataset)}")
    episodes = sorted(files)
    S = dict(f=None, ep=None)

    def open_episode(ep):
        if S["f"] is not None:
            S["f"].close()
        f = h5video.File(files[ep], "r")
        S.update(f=f, ep=ep, T=int(f.attrs["num_frames"]), H=int(f.attrs["H"]), W=int(f.attrs["W"]), cams=[int(c) for c in f.attrs["cam_ids"]], bbox=[float(v) for v in f.attrs["bbox"]])

    open_episode(a.episode if a.episode in files else episodes[0])
    try:  # the task's latent grid, when the dataset is a task of this repository
        from tasks import TASKS

        grid = next(t["grid"] for t in TASKS.values() if a.dataset in t["datasets"])
    except Exception:
        grid = (64, 32, 64)

    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    g = server.gui
    with g.add_folder("episode"):
        ep_sl = g.add_slider("episode", min=0, max=max(len(episodes) - 1, 1), step=1, initial_value=episodes.index(S["ep"]), disabled=len(episodes) < 2)
        frame_sl = g.add_slider("frame", min=0, max=S["T"] - 1, step=1, initial_value=0)
        play = g.add_checkbox("play", initial_value=False)
        status = g.add_text("status", initial_value="", disabled=True)
    with g.add_folder("point cloud"):
        cloud_dd = g.add_dropdown("points", ("training cloud", "full depth"), initial_value="training cloud")
        show_obj = g.add_checkbox("highlight object", initial_value=False)
        stride_w = g.add_slider("pixel stride", min=1, max=8, step=1, initial_value=2)
        size_w = g.add_slider("point size (mm)", min=0.5, max=8.0, step=0.5, initial_value=2.0)
        show_box = g.add_checkbox("workspace box", initial_value=True)
    with g.add_folder("voxels"):
        show_vox = g.add_checkbox("occupied voxels", initial_value=a.show == "voxels")
        nx_w = g.add_number("nx", initial_value=int(grid[0]), min=1, max=512, step=1)
        ny_w = g.add_number("ny", initial_value=int(grid[1]), min=1, max=512, step=1)
        nz_w = g.add_number("nz", initial_value=int(grid[2]), min=1, max=512, step=1)
    with g.add_folder("cameras"):
        show_frusta = g.add_checkbox("camera poses", initial_value=True)
        show_imgs = g.add_checkbox("camera images", initial_value=a.show == "cams")
        img_h = {c: g.add_image(np.zeros((VIEW_W * 9 // 16, VIEW_W, 3), np.uint8), label=f"camera {c}", visible=a.show == "cams") for c in S["cams"]}

    cloud = server.scene.add_point_cloud("/cloud", np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8), point_size=0.002, point_shape="circle")
    box = server.scene.add_line_segments("/bbox", bbox_segments(S["bbox"]), colors=np.array([0, 200, 255], np.uint8), line_width=2.0)
    vox_wire = server.scene.add_line_segments("/vox_wire", np.zeros((1, 2, 3), np.float32), colors=np.array([255, 220, 40], np.uint8), line_width=1.0, visible=False)
    vox_pts = server.scene.add_point_cloud("/vox_pts", np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8), point_size=0.004, point_shape="square", visible=False)
    frusta = {}

    @server.on_client_connect
    def _(client):  # start close to the workspace, looking at its centre
        b = S["bbox"]
        centre = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        client.camera.position = centre + np.array([0.0, -1.0, 0.75]) * max(b[1] - b[0], b[3] - b[2]) * 1.6
        client.camera.look_at = centre

    def camera(c, t):
        f = S["f"]
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(f[f"img/camera_{c}/rgb_in_ir"][t], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return rgb, f[f"cam/camera_{c}/K_ir"][t].astype(np.float32), f[f"cam/camera_{c}/E_cam2world"][t].astype(np.float32)

    def points(c, t, rgb, K, E, stride, full):
        f, H, W = S["f"], S["H"], S["W"]
        depth = f[f"img/camera_{c}/depth"][t].astype(np.float32)
        if full:
            m = depth > 0
        else:
            name = "filter_mask" if f"img/camera_{c}/filter_mask" in f else "valid_mask"
            m = bits(f[f"img/camera_{c}/{name}"][t], H, W) & (depth > 0)
        m[np.arange(H) % stride != 0] = False
        m[:, np.arange(W) % stride != 0] = False
        v, u = np.nonzero(m)
        z = depth[v, u]
        xyz = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1) @ E[:3, :3].T + E[:3, 3]
        obj = f[f"img/camera_{c}/obj_mask"][t][v, u] > 0 if f"img/camera_{c}/obj_mask" in f else np.zeros(len(v), bool)
        return xyz.astype(np.float32), rgb[v, u], obj

    lock = threading.Lock()

    def render():
        with lock:
            draw()

    def draw():
        t = min(int(frame_sl.value), S["T"] - 1)
        xyz, col, obj = [], [], []
        for c in S["cams"]:
            rgb, K, E = camera(c, t)
            x, k, o = points(c, t, rgb, K, E, int(stride_w.value), cloud_dd.value == "full depth")
            xyz.append(x), col.append(k), obj.append(o)
            if show_frusta.value:
                small = cv2.resize(rgb, (160, 160 * rgb.shape[0] // rgb.shape[1]))
                q = viser.transforms.SO3.from_matrix(E[:3, :3]).wxyz
                fov = 2 * np.arctan2(S["H"] / 2, K[1, 1])
                if c not in frusta:
                    frusta[c] = server.scene.add_camera_frustum(f"/cams/camera_{c}", fov=float(fov), aspect=S["W"] / S["H"], scale=0.06, line_width=1.0, color=(110, 110, 110), image=small, wxyz=q, position=E[:3, 3])
                else:
                    frusta[c].wxyz, frusta[c].position, frusta[c].image = q, E[:3, 3], small
            if show_imgs.value:
                img_h[c].image = cv2.resize(rgb, (VIEW_W, VIEW_W * rgb.shape[0] // rgb.shape[1]))
        for c, h in frusta.items():
            h.visible = bool(show_frusta.value)
        for h in img_h.values():
            h.visible = bool(show_imgs.value)
        xyz, col, obj = np.concatenate(xyz), np.concatenate(col), np.concatenate(obj)
        if show_obj.value:
            col = np.where(obj[:, None], np.array([255, 40, 160], np.uint8), (col * 0.5).astype(np.uint8))
        cloud.points, cloud.colors = (xyz, col) if len(xyz) else (np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8))
        cloud.point_size = float(size_w.value) / 1000
        box.visible = bool(show_box.value)
        b = S["bbox"]
        note = ""
        vox_wire.visible = vox_pts.visible = False
        if show_vox.value:
            inb = (xyz[:, 0] >= b[0]) & (xyz[:, 0] <= b[1]) & (xyz[:, 1] >= b[2]) & (xyz[:, 1] <= b[3]) & (xyz[:, 2] >= b[4]) & (xyz[:, 2] <= b[5])
            bmin, vs, dims = grid_for(b, (int(nx_w.value), int(ny_w.value), int(nz_w.value)))
            centers, vrgb, n = voxelize(xyz[inb], col[inb], bmin, vs, dims)
            if n and n <= WIRE_MAX:
                seg = wireframes(centers, vs)
                vox_wire.points, vox_wire.colors, vox_wire.visible = seg, np.tile(np.array([255, 220, 40], np.uint8), (seg.shape[0], 2, 1)), True
            elif n:
                vox_pts.points, vox_pts.colors, vox_pts.point_size, vox_pts.visible = centers, vrgb, float(vs.min()) * 0.95, True
            note = f", {n:,} of {int(np.prod(dims)):,} cells occupied ({'x'.join(f'{v * 1000:.1f}' for v in vs)} mm)"
        status.value = f"episode {S['ep']} ({episodes.index(S['ep']) + 1}/{len(episodes)}), frame {t}/{S['T'] - 1}, {len(xyz):,} points{note}"

    @ep_sl.on_update
    def _(_):
        ep = episodes[min(int(ep_sl.value), len(episodes) - 1)]
        if ep != S["ep"]:
            with lock:
                open_episode(ep)
                box.points = bbox_segments(S["bbox"])
                frame_sl.max = S["T"] - 1
            render()

    for w in (frame_sl, cloud_dd, show_obj, stride_w, size_w, show_box, show_vox, nx_w, ny_w, nz_w, show_frusta, show_imgs):
        w.on_update(lambda _: render())
    render()
    print(f"[view] {a.dataset}: {len(episodes)} episodes, open http://localhost:{a.port}", flush=True)
    while True:
        if play.value:
            frame_sl.value = (int(frame_sl.value) + 1) % S["T"]
            time.sleep(0.1)
        else:
            time.sleep(0.2)


if __name__ == "__main__":
    main()
