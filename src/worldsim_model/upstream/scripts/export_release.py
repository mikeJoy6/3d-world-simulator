"""Build the released checkpoint folders from the workstation's training outputs (the exact files the public demo runs).

    python scripts/export_release.py --out release [--tasks cloth box pusht assembly]

A task folder for your own run (no scanned room: the scene renders on white):

    python scripts/export_release.py --out runs/assembly/task --run runs/assembly/vae/checkpoints/latest.pth \
        runs/assembly/dynamics/checkpoints/latest.pth runs/assembly/latents/val/episode_9_latents.h5 data/assembly/val/episode_9.hdf5

release/
  background/scan_room.pt          the scanned room, already in the world frame (shared by all tasks)
  <task>/vae.pth                   autoencoder decoder + gaussian head + the learned scan colour map (training state stripped)
  <task>/dynamics.pth              latent dynamics model + action normalisation
  <task>/seed_episode_<E>.h5       one latent episode: compact latents, recorded actions, the dataset cameras and some of their images
  <task>/task.json                 start frame, workspace, control step, camera and display settings, arm base frames, latent colour map
  <task>/thumbs/*.jpg              the tile images of the interactive demo
  arms/rig.pt, <task>/arms.pt      the kinematic robot model of the interactive demo (scripts/export_arms.py)
"""
import argparse, glob, json, os, shutil, subprocess
from pathlib import Path
import cv2
import h5py

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import h5video  # noqa: E402
from tasks import TASKS as RELEASED  # noqa: E402
import numpy as np
import torch
from scipy.spatial.transform import Rotation

J = os.environ.get("WS_ROOT", "")   # root of the authors' workstation tree (only for the table below; --run needs none of it)
V = f"{J}/rollout_view_1019/3dws/data/view"
CK = f"{J}/3dws/checkpoints"
C0 = 0.28209479177387814
SCAN = (f"{J}/rollout_view_1019/3dws/data/background/scan_difix_r3_fedex.pt", f"{J}/rollout_view_1019/3dws/data/background/T_scan2world_cloth_scanfinal.npz")
WB = [0.97, 1.0, 1.06]
POINT_GAIN = dict(cloth=0.999, box=1.88, pusht=1.85, assembly=1.544)   # exposure of the particle view, matched to the website clips
POINT_MM = dict(cloth=0.5, box=1.0, pusht=1.0, assembly=0.5)   # dot diameter of the particle view (mm)
XYZ_DUAL = {"a": [1, 0, -1], "d": [1, 0, 1], "w": [1, 1, 1], "s": [1, 1, -1], "q": [1, 2, 1], "e": [1, 2, -1], "j": [0, 0, 1], "l": [0, 0, -1], "i": [0, 1, -1], "k": [0, 1, 1], "u": [0, 2, 1], "o": [0, 2, -1]}
XY_PUSH = {"d": [1, 0, 1], "a": [1, 0, -1], "w": [1, 1, 1], "s": [1, 1, -1], "l": [0, 0, -1], "j": [0, 0, 1], "i": [0, 1, -1], "k": [0, 1, 1]}
XZ_ARM0 = {"j": [0, 0, 1], "l": [0, 0, -1], "i": [0, 2, 1], "k": [0, 2, -1]}
TASKS = dict(
    cloth=dict(
        title="Cloth Folding", vae=f"{V}/cloth/arms_final_1102/decoder_colgeo.pth",
        dyn=f"{CK}/bimanual_cloth_all/dynamics_particle_vae_bighead_aniso_tactile_v2_v3only_4val_h8_ar1to10_500_l3_film_full_tactile20_e499_dropft_1101c",
        dyn_ckpt="checkpoints/model_11.pth", episode=38, latents=f"{J}/3dws/data/bimanual_cloth_all/latents_particle_vae_bighead_aniso_tactile_v2_compact_v3only_4val/train",
        packed=f"{J}/aloha_processed/bimanual_cloth_v3/packed", dataset="bimanual_cloth_v3", multi_view="latent_cams",   # the public demo has no packed episode for cloth: its multi-view shows the cameras stored with the latents
        control=dict(step=0.004, keys=XYZ_DUAL, orient="aim25", gripper=dict(keys={"h": 0}, open=1.2, close=[0.2, 0.1], rate=0.2, fixed={"1": "close"})),
        view=dict(az=-129, el=17, radius=0.74), exposure=1.0, scan_room_gain=1.0, class_filters=True,
    ),
    box=dict(
        title="Box Packing", vae=f"{V}/box/vae_aniso.pth", dyn=f"{V}/box/dynamics_cont/ar10", dyn_ckpt="checkpoints/model_689.pth", episode=24,
        latents=f"{V}/box/latents/all", packed=f"{J}/rollout_view_1019/aloha_processed/_view_packed/box", dataset="bimanual_box_free",
        control=dict(step=0.004, keys=XYZ_DUAL, orient="aim25", run_on=25, soft_limit=True), view=dict(az=-157, el=31.8, radius=0.74),
        exposure=1.45, scan_room_gain=1.16, class_filters=True, table_edge_faces=[0.0, 0.0, 0.05, 0.0],   # a camera mount sits on the table's front edge in the box data
    ),
    pusht=dict(
        title="T Pushing", vae=f"{V}/pusht/vae_aniso.pth", dyn=f"{V}/pusht/dynamics_e299", dyn_ckpt="checkpoints/latest.pth", episode=0,
        latents=f"{V}/pusht/latents/*", packed=f"{J}/rollout_view_1019/aloha_processed/_view_packed/pusht", dataset="bimanual_pusht_v2",
        control=dict(step=0.0035, keys=XY_PUSH, orient="seed"), view=dict(az=-97, el=49, radius=0.74), exposure=1.67, scan_room_gain=1.10,
        class_filters=False,
    ),
    assembly=dict(   # called "plug" in the workstation tree (``bundle`` below is the name used in its paths)
        title="Part Assembly", bundle="plug", vae=f"{V}/plug/vae_aniso_opaque.pth",
        dyn=f"{CK}/bimanual_plug_all/dynamics_particle_vae_bighead_aniso_tactile_v2_noteleop_h8_ar1to10_500_l3_film_full_tactile20_e499_plugft_1102e",
        dyn_ckpt="checkpoints/model_9.pth", episode=10, latents=f"{V}/plug/latents/*", packed=f"{J}/rollout_view_1019/aloha_processed/_view_packed/plug",
        dataset="bimanual_plug_insertion_v3", control=dict(step=0.0035, keys=XZ_ARM0, orient="seed", locked_arms=[1]),
        view=dict(az=-93.4, el=34.2, radius=0.74), exposure=1.55, scan_room_gain=1.0, class_filters=False,
    ),
)


def export_scan(out):
    pt, T_path = SCAN
    g = torch.load(pt, map_location="cpu", weights_only=False)
    g = g.get("splats", g)
    T = np.load(T_path)["T"].astype(np.float64)
    R = T[:3, :3]
    x = g["means"].numpy().astype(np.float64) @ R.T + T[:3, 3]
    q = g["quats"].numpy().astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q = (Rotation.from_matrix(R) * Rotation.from_quat(q[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    d = dict(
        xyz=torch.from_numpy(x.astype(np.float32)), scale=torch.exp(g["scales"]).float(), opacity=torch.sigmoid(g["opacities"]).reshape(-1).float(),
        rgb01=(g["sh0"].reshape(-1, 3) * C0 + 0.5).clamp(0, 1).float(), quat=torch.from_numpy(q.astype(np.float32)),
    )
    out.mkdir(parents=True, exist_ok=True)
    torch.save(d, out / "scan_room.pt")
    print(f"[release] background: {len(x)} gaussians -> {out / 'scan_room.pt'}")


def strip_vae(src, dst):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sb = ck.get("scan_bg") or {}
    tactile = ck["args"].get("tactile_particle_arch") in ("v2", "v3")  # the tactile particle head of the demo's particle view
    spec = dict(ck["new_head"])
    head = spec.pop("state_dict", None)  # the Gaussian head the demo runs (a later decoder fine-tune updates only this copy)
    keep = dict(
        model_state_dict={k: v for k, v in ck["model_state_dict"].items() if tactile or not k.startswith("tactile_particle_head.")},
        seed_head_state_dict=ck["seed_head_state_dict"] if head is None else {f"head.{k}": v for k, v in head.items()}, new_head=spec,
        pos_mean=ck["pos_mean"], pos_std=ck["pos_std"],
        epoch=ck.get("epoch"), args={k: ck["args"][k] for k in ("x_range", "y_range", "z_range", "grid", "nx", "ny", "nz", "dataset", "tactile_particle_reach") if k in ck["args"]},
        scan_color=None if "gain" not in sb else dict(gain=np.asarray(sb["gain"], np.float32).tolist(), bias=np.asarray(sb["bias"], np.float32).tolist()),
    )
    torch.save(keep, dst)
    return keep


def strip_dyn(src, dst):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    a = ck["args"]
    norm = Path(src).parents[1] / "action_norm_params.npz"
    n = np.load(norm) if norm.exists() else dict(action_mean=ck["action_mean"], action_std=ck["action_std"])
    keep = dict(
        model_state_dict={k.replace("_orig_mod.", ""): v for k, v in ck["model_state_dict"].items()}, epoch=ck.get("epoch"),
        action_mean=np.asarray(n["action_mean"], np.float32), action_std=np.asarray(n["action_std"], np.float32),
        args={k: a[k] for k in ("n_history", "n_levels", "film", "dim", "infer_hist_noise", "ar_schedule", "object_alpha", "tactile_alpha") if a.get(k) is not None},
    )
    torch.save(keep, dst)
    return keep


def export_seed(latents, packed, dst, gt_every=5, latent_cams=False):
    """latent_cams: also store the cameras recorded with the latents (``latent_cams``); the demo's multi-view of that task shows those."""
    with h5video.File(latents, "r") as f, h5py.File(dst, "w") as g:
        for k, v in f.attrs.items():
            g.attrs[k] = v
        for k in ("z_bg", "z_mask", "z_off", "z_vals", "action", "obj_occ", "tac_occ"):
            if k in f:
                g.create_dataset(k, data=f[k][:], compression="gzip" if k in ("obj_occ", "tac_occ") else None)
        T = int(f["action"].shape[0])
        if latent_cams:
            lc = g.create_group("latent_cams")
            K, E = f["K"][:].astype(np.float32), f["E"][:].astype(np.float64)
            lc.attrs["cam_ids"] = list(range(K.shape[1]))
            for c in range(K.shape[1]):
                gc = lc.create_group(f"camera_{c}")
                gc.attrs["W"], gc.attrs["H"] = (int(round(2 * K[0, c, 0, 2])), int(round(2 * K[0, c, 1, 2])))
                gc.create_dataset("viewmat", data=np.linalg.inv(E[:, c]).astype(np.float32))
                gc.create_dataset("K", data=K[:, c])
        with h5video.File(packed, "r") as ph:
            rc = g.create_group("render_cams")
            cids = [int(x) for x in ph.attrs.get("render_cam_ids", ph.attrs["cam_ids"])]
            rc.attrs["cam_ids"] = cids
            frames = list(range(0, T, gt_every))
            for c in cids:
                gp = ph[f"render/camera_{c}"]
                h, w = cv2.imdecode(np.asarray(gp["jpg"][0], np.uint8), cv2.IMREAD_COLOR).shape[:2]
                gc = rc.create_group(f"camera_{c}")
                gc.attrs["W"], gc.attrs["H"] = (w, h)
                gc.create_dataset("viewmat", data=gp["viewmat"][:T])
                gc.create_dataset("K", data=gp["K"][:T])
                gc.create_dataset("gt_frames", data=np.asarray(frames, np.int64))
                gc.create_dataset("gt_jpg", data=[np.asarray(gp["jpg"][t], np.uint8) for t in frames], dtype=h5py.vlen_dtype(np.uint8))
    print(f"[release] seed {Path(latents).name}: {T} frames, cameras {cids} -> {dst} ({os.path.getsize(dst) >> 20} MB)")


def demo_assets(t, s, d):
    """What only the interactive demo needs: where each arm's action frame sits in the world (the gizmo is drawn there), the
    fitted principal component that colours the latent view, and the tile images of the page."""
    cal = f"{J}/rollout_view_1019/aloha_processed/{s['dataset']}/packing/calibration"
    base = [np.load(f"{cal}/arm{a}_base_and_joint_offsets.npz")["T_base"].astype(np.float64) for a in (0, 1)]
    if os.path.exists(f"{cal}/base_offset_z.json"):
        for a, dz in json.load(open(f"{cal}/base_offset_z.json")).items():
            base[int(a)][2, 3] += float(dz)
    b = s.get("bundle", t)
    P = json.load(open(f"{V}/{b}/latent_site.json"))
    (d / "thumbs").mkdir(exist_ok=True)
    for name, src in (("splats", "splats"), ("particles", "points"), ("multi", "multiview"), ("voxels", "voxels"), ("latents", "latents"), ("raw", "raw")):
        shutil.copy(f"{J}/rollout_view_1019/relight/web/site/pan/{b}/th_{src}.jpg", d / "thumbs" / f"{name}.jpg")
    return dict(arm_base=[b.round(6).tolist() for b in base], latent_pca={k: P[k] for k in ("ch", "mean", "pc", "q")}, multi_cams=[0, 1, 2], multi_view=s.get("multi_view", "render_cams"), point_gain=POINT_GAIN[t], point_size_mm=POINT_MM[t])


@torch.no_grad()
def table_match_gain(task_dir, W=960, H=640):
    """Per-channel gain on the scan's table top so it meets the world model's own table layer without a step (the public demo
    measures it on its first frame): the model's table layer of the seed frame is rendered alone from the opening view; over
    the pixels it covers where the scan is white, gain = the layer's unshadowed (75th percentile) colour / the scan's median."""
    from models.gsplat_render_loss import render_gaussians
    from models.world_model import TABLE_Z, WorldModel

    wm = WorldModel(task_dir, arms=False)
    z, _ = wm.load_episode()
    g, _ = wm.decode_raw(z[wm.cfg["seed"]["frame"] + wm.n_hist - 1])
    m = (g["cls"] == 2) & (g["xyz"][:, 2] < TABLE_Z + 0.03) & (g["opacity"] > 0.3)
    vm, K, _, _ = wm.orbit_camera(W=W, H=H)
    col, al = render_gaussians(g["xyz"][m], g["scale"][m], g["opacity"][m], g["rgb01"][m], vm[None], K[None], W, H, quats=g["quat"][m])
    cov = al[0, ..., 0] > 0.5
    layer, scan = col[0][cov] / al[0][cov], wm.scan.render(vm, K, W, H)[0][cov]
    white = (scan.min(1).values > 0.45) & ((scan.max(1).values - scan.min(1).values) < 0.12)
    if cov.sum() < 500 or white.sum() < 200:
        return None
    gain = (torch.quantile(layer[white], 0.75, dim=0) / scan[white].median(0).values.clamp(min=0.05)).clamp(0.5, 2.0)
    return [round(float(v), 4) for v in gain]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="release")
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--run", nargs=4, metavar=("VAE", "DYNAMICS", "LATENTS", "PACKED"), help="build one task folder from your own checkpoints and one episode")
    p.add_argument("--frame", type=int, default=0, help="with --run: start frame of the seed episode")
    p.add_argument("--no-arms", action="store_true", help="skip the kinematic robot model (scripts/export_arms.py)")
    a = p.parse_args()
    out = Path(a.out)
    if a.run:
        vae_p, dyn_p, lat, packed = a.run
        out.mkdir(parents=True, exist_ok=True)
        vae, dyn = (strip_vae(vae_p, out / "vae.pth"), strip_dyn(dyn_p, out / "dynamics.pth"))
        ep = int(Path(lat).name.split("_")[1])
        export_seed(lat, packed, out / f"seed_episode_{ep}.h5")
        cfg = dict(
            task=out.name, seed=dict(file=f"seed_episode_{ep}.h5", episode=ep, frame=a.frame), n_history=dyn["args"]["n_history"],
            view=dict(az=-129, el=17, radius=0.74), exposure=1.0, class_filters=False, background=None,
            sources=dict(vae=vae_p, dynamics=dyn_p, latents=lat, vae_epoch=vae["epoch"], dynamics_epoch=dyn["epoch"]),
        )
        json.dump(cfg, open(out / "task.json", "w"), indent=1)
        print(f"[release] task folder -> {out}")
        return
    export_scan(out / "background")
    for t in a.tasks:
        s = TASKS[t]
        d = out / t
        d.mkdir(parents=True, exist_ok=True)
        vae = strip_vae(s["vae"], d / "vae.pth")
        dyn = strip_dyn(f"{s['dyn']}/{s['dyn_ckpt']}", d / "dynamics.pth")
        lat = sorted(glob.glob(f"{s['latents']}/episode_{s['episode']}_latents.h5"))[0]
        seed = f"seed_episode_{s['episode']}.h5"
        export_seed(lat, f"{s['packed']}/episode_{s['episode']}.hdf5", d / seed, latent_cams=s.get("multi_view") == "latent_cams")
        su = json.load(open(f"{V}/{s.get('bundle', t)}/studio_setup.json"))
        cfg = dict(
            task=t, title=s["title"], dataset=RELEASED[t]["datasets"][0], seed=dict(file=seed, episode=s["episode"], frame=int(su["frame"])),
            n_history=dyn["args"]["n_history"], workspace=dict(min=su["ws_min"], max=su["ws_max"], keepout=su.get("keepout", [])),
            control=s["control"], view=s["view"], exposure=s["exposure"], scan_white_balance=WB, scan_room_gain=s["scan_room_gain"],
            scan_wall_saturation=0.5, class_filters=s["class_filters"], table_feather=0.15, table_edge_faces=s.get("table_edge_faces"),
            background="../background/scan_room.pt", arms="../arms/rig.pt",
            sources=dict(vae=s["vae"], dynamics=f"{s['dyn']}/{s['dyn_ckpt']}", latents=lat, vae_epoch=vae["epoch"], dynamics_epoch=dyn["epoch"]),
            **demo_assets(t, s, d),
        )
        json.dump(cfg, open(d / "task.json", "w"), indent=1)
        cfg["scan_table_gain"] = table_match_gain(d)
        json.dump(cfg, open(d / "task.json", "w"), indent=1)
        print(f"[release] {t}: scan table-top gain {cfg['scan_table_gain']}, vae {os.path.getsize(d / 'vae.pth') >> 20} MB, dynamics {os.path.getsize(d / 'dynamics.pth') >> 20} MB, n_history {cfg['n_history']}")
    if not a.no_arms:
        subprocess.run([sys.executable, str(_P(__file__).with_name("export_arms.py")), "--out", str(out.resolve()), "--tasks", *[f"{t}={TASKS[t].get('bundle', t)}" for t in a.tasks]], check=True)


if __name__ == "__main__":
    main()
