#!/usr/bin/env python3
"""Render the README's still figures from the model itself (same camera for every panel).

    tools/wspython tools/render_readme_figures.py          # writes docs/media/urdf_extraction.png and scene_only.png

urdf_extraction.png  the upstream splat arm rig vs the extracted URDF meshes, at the same joints (the joints the model
                     predicts for the seed frame), drawn over white. The meshes are drawn as densely sampled, shaded
                     surface points through the same Gaussian renderer, so both panels share one camera exactly.
scene_only.png       per task: the upstream output (model Gaussians + arm rig) vs this repository's scene-only output.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/worldsim_core"))
from worldsim_core.engine import WorldSimEngine  # noqa: E402
from worldsim_core.robot import ARM_JOINTS, RobotCalibration  # noqa: E402
from worldsim_core.transforms import axis_rotation, pose_to_transform  # noqa: E402

OUT = ROOT / "docs/media"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def label(img, text, size=22):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, size)
    x0, y0, x1, y1 = d.textbbox((14, 10), text, font=f)
    d.rectangle((x0 - 8, y0 - 6, x1 + 8, y1 + 6), fill=(255, 255, 255))
    d.text((14, 10), text, font=f, fill=(30, 40, 36))
    return np.asarray(im)


def link_transforms(calib, q6, opening):
    """base_link -> every link of one arm."""
    T = {"base_link": np.eye(4)}
    for j in calib.chain:
        M = pose_to_transform(j["origin"])
        if j["type"] == "revolute":
            M[:3, :3] = M[:3, :3] @ axis_rotation(j["axis"], q6[ARM_JOINTS.index(j["name"])])
        elif j["type"] == "prismatic":
            sign = j.get("mimic", {}).get("multiplier", 1.0)
            M[:3, 3] = M[:3, 3] + M[:3, :3] @ (np.asarray(j["axis"]) * opening * sign)
        T[j["child"]] = T[j["parent"]] @ M
    return T


def mesh_points(calib, mounts, pose, view_dir, n=900000):
    import trimesh
    pts, cols = [], []
    light = -view_dir / np.linalg.norm(view_dir) + np.array([0, 0, 0.6])
    light /= np.linalg.norm(light)
    for arm in (0, 1):
        T = link_transforms(calib, pose[arm * 7: arm * 7 + 6], pose[arm * 7 + 6])
        for f in sorted((ROOT / "src/worldsim_description/meshes").glob(f"arm{arm}_*.stl")):
            if f.stem.endswith("_collision"):
                continue
            link = f.stem[len(f"arm{arm}_"):]
            m = trimesh.load(f)
            k = max(2000, int(n * m.area / 0.6))
            p, fi = trimesh.sample.sample_surface_even(m, k) if hasattr(trimesh.sample, "sample_surface_even") else trimesh.sample.sample_surface(m, k)
            W = mounts[arm] @ T[link]
            pw = p @ W[:3, :3].T + W[:3, 3]
            nw = m.face_normals[fi] @ W[:3, :3].T
            shade = 0.28 + 0.62 * np.clip(nw @ light, 0, 1) + 0.10 * np.clip(-nw @ view_dir, 0, 1)
            base = np.array([0.42, 0.43, 0.45]) if "finger" not in link else np.array([0.55, 0.56, 0.58])
            pts.append(pw)
            cols.append(np.clip(shade[:, None] * base, 0, 1))
    return np.concatenate(pts), np.concatenate(cols)


@torch.no_grad()
def urdf_figure(calib):
    eng = WorldSimEngine(image=True)  # puts the upstream code on the path
    from models.gsplat_render_loss import render_gaussians
    eng.load("pusht")
    eng.reset()
    for _ in range(30):
        eng.step(eng.recorded_action())
    wm = eng.wm
    g, coarse = wm.decode_raw(eng.latent)
    st = wm.arms.new_state()
    rig = wm.arms(st, coarse, g)
    vm, K, W, H = wm.orbit_camera(az=-60, el=32, radius=1.05, W=960, H=600)
    # the rig as the demo draws it: its fingers are hidden there, the model's own gripper Gaussians take over
    left = wm.render(rig, vm, K, W, H, background=False)
    V = vm.cpu().numpy()
    view_dir = V[2, :3]  # camera forward in the world
    mounts = [np.asarray(b, np.float64) for b in wm.arms.base]  # this task's mounts, as the rig uses
    xyz, rgb = mesh_points(calib, mounts, st.pose, view_dir)
    x = torch.tensor(xyz, dtype=torch.float32, device="cuda")
    c = torch.tensor(rgb, dtype=torch.float32, device="cuda")
    fg, alpha = render_gaussians(x, torch.full((len(x), 3), 0.0011, device="cuda"), torch.ones(len(x), device="cuda"), c, vm[None], K[None], W, H)
    right = (fg[0] + (1 - alpha[0])).clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
    img = np.concatenate([label(left, "Upstream: fitted Gaussian arm rig (fingers come from the model)", 19), np.full((H, 8, 3), 255, np.uint8),
                          label(right, "This repo: URDF meshes extracted from that rig", 19)], 1)
    Image.fromarray(img).save(OUT / "urdf_extraction.png", optimize=True)
    eng.unload()


@torch.no_grad()
def scene_figure():
    rows = []
    for task, frames in (("cloth", 60), ("assembly", 45)):
        eng = WorldSimEngine(image=True)
        eng.load(task)
        eng.reset()
        for _ in range(frames):
            eng.step(eng.recorded_action())
        vm, K, W, H = eng.wm.orbit_camera(W=800, H=520)
        up = eng.wm.render(eng.wm.decode(eng.latent, arms=eng.wm.arms.new_state()), vm, K, W, H)
        ours = eng.render((vm, K, W, H))
        title = eng.cfg.title
        rows.append(np.concatenate([label(up, f"{title}: upstream output (model + arm rig)", 20), np.full((H, 8, 3), 255, np.uint8),
                                    label(ours, f"{title}: scene only (this repo)", 20)], 1))
        rows.append(np.full((8, rows[-1].shape[1], 3), 255, np.uint8))
        eng.unload()
    Image.fromarray(np.concatenate(rows[:-1], 0)).save(OUT / "scene_only.png", optimize=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    urdf_figure(RobotCalibration())
    scene_figure()
    print("wrote", OUT / "urdf_extraction.png", OUT / "scene_only.png")
