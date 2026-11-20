#!/usr/bin/env python3
"""Extract the robot that the world model was trained with (two ViperX 300 S arms) from the released assets.

    .venv/bin/python tools/extract_robot.py            # writes src/worldsim_description/{meshes,config/calibration.yaml}

What comes out, and where it comes from:

  kinematic chain     ``arms/rig.pt`` -> ``fk``: the joint table the authors exported from their URDF (origins, axes,
                      joint types, finger mimic). Joint limits are the ViperX 300 S limits widened to cover every joint
                      angle the model predicts on the four seed episodes.
  arm bases           ``<task>/arms.pt`` -> ``T_base``: where each arm's base_link sits in the world, per task.
  meshes              ``arms/rig.pt``: the fitted Gaussian splats of each link, moved into that link's frame
                      (``rest_inv``), voxelised and meshed (marching cubes). One visual mesh and one convex collision
                      hull per link and arm.
  action calibration  The model's 14-d action is the *commanded* end-effector pose of each arm in the dataset's
                      teleoperation frame, not a URDF link pose. For every frame of the four seed episodes the arm
                      head of the release (``ArmRig.joint_net``) gives the joint angles the model believes in; a
                      least-squares fit then finds, per arm, the fixed ``action_frame`` (relative to base_link) and
                      ``action_tcp`` (relative to gripper_link) such that
                          action(t) = pose of action_tcp in action_frame at joints(t + lag).
                      With these two fixed links in the URDF the action is a plain TF lookup, for any joint source.

Needs the GPU environment of this workspace (torch + the upstream model code) and about two minutes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = next(p for p in (ROOT.parent, ROOT / "src/worldsim_model/upstream") if (p / "models/world_model.py").is_file())
sys.path.insert(0, str(UPSTREAM))

TASKS = ("pusht", "cloth", "box", "assembly")
FRAMES = ["base_link", "shoulder_link", "upper_arm_link", "upper_forearm_link", "lower_forearm_link", "wrist_link",
          "gripper_link", "ee_arm_link", "gripper_bar_link", "fingers_link", "left_finger_link", "right_finger_link"]
JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "ee_arm", "gripper_bar", "ee_bar",
          "left_finger", "right_finger"]
# ViperX 300 S (Interbotix) limits in rad; widened below where the model's predictions go further.
VX300S_LIMITS = {"waist": (-3.1416, 3.1416), "shoulder": (-1.85, 1.2566), "elbow": (-1.7628, 1.6057),
                 "forearm_roll": (-3.1416, 3.1416), "wrist_angle": (-1.8675, 2.2340), "wrist_rotate": (-3.1416, 3.1416)}
VX300S_VELOCITY = {"waist": 3.14, "shoulder": 3.14, "elbow": 3.14, "forearm_roll": 3.14, "wrist_angle": 3.14, "wrist_rotate": 3.14}


def axis_rotation(axis, q):
    x, y, z = axis
    c, s = np.cos(q), np.sin(q)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s], [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def fk_frames(fk, q6, opening=0.02):
    val = np.concatenate([np.asarray(q6, np.float64), [opening, 0.0]])
    T = [None] * fk["n_frames"]
    T[fk["base"]] = np.eye(4)
    for par, chi, O, typ, ax, src, mul, off in zip(fk["parent"], fk["child"], fk["origin"], fk["type"], fk["axis"], fk["source"], fk["mul"], fk["off"]):
        x = mul * val[src] + off
        M = np.array(O, np.float64)
        if typ == 1:
            M[:3, :3] = M[:3, :3] @ axis_rotation(ax, x)
        elif typ == 2:
            M[:3, 3] = M[:3, 3] + M[:3, :3] @ (np.asarray(ax) * x)
        T[chi] = T[par] @ M
    return T


def pose_dict(T):
    return dict(xyz=[round(float(v), 6) for v in T[:3, 3]], rpy=[round(float(v), 6) for v in Rotation.from_matrix(T[:3, :3]).as_euler("xyz")])


# -- 1. what the model believes the joints are, frame by frame ----------------------------------------------------------------
@torch.no_grad()
def predicted_joints(assets, cache):
    from models.world_model import WorldModel
    out = {}
    for task in TASKS:
        f = cache / f"{task}_joints.npz"
        if f.is_file():
            d = np.load(f)
            out[task] = (d["q"], d["act"])
            continue
        wm = WorldModel(assets / task, bf16=True, background=False, arms=True)
        z, act = wm.load_episode()
        Q = []
        for t in range(len(z)):
            cells = wm.decode_cells(torch.from_numpy(z[t]).to(wm.device))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = wm.arms.joint_net(cells["coarse"].float())[0].float()
            Q.append((o * wm.arms.y_std + wm.arms.y_mean).cpu().numpy())
        out[task] = (np.array(Q, np.float64), act.astype(np.float64))
        cache.mkdir(parents=True, exist_ok=True)
        np.savez(f, q=out[task][0], act=out[task][1])
        print(f"[extract] {task}: joints of {len(z)} frames", flush=True)
        del wm
        torch.cuda.empty_cache()
    return out


# -- 2. the fixed action frame and tool point per arm --------------------------------------------------------------------------
def fit_action_frames(fk, data, lag):
    result = []
    for arm in (0, 1):
        P, Rf, A = [], [], []
        for q, act in data.values():
            for i in range(len(q) - lag):
                a = act[i, arm * 7: arm * 7 + 6]
                if not np.any(a):  # unrecorded frames hold all-zero actions
                    continue
                T = fk_frames(fk, q[i + lag, arm * 7: arm * 7 + 6])[6]
                P.append(T[:3, 3])
                Rf.append(T[:3, :3])
                A.append(a)
        P, Rf, A = np.array(P), np.array(Rf), np.array(A)
        Ra = Rotation.from_euler("xyz", A[:, 3:])

        def residual(x):
            Rb, tb, Rt, d = Rotation.from_rotvec(x[:3]), x[3:6], Rotation.from_rotvec(x[6:9]), x[9:12]
            pos = Rb.apply(P + np.einsum("nij,j->ni", Rf, d)) + tb
            rot = (Ra.inv() * Rb * Rotation.from_matrix(Rf) * Rt).as_rotvec()
            return np.concatenate([(pos - A[:, :3]).ravel(), 0.05 * rot.ravel()])

        sol = least_squares(residual, np.zeros(12), loss="soft_l1", f_scale=0.01)
        r = residual(sol.x)
        n = len(A)
        pe = np.linalg.norm(r[: 3 * n].reshape(-1, 3), axis=1)
        re = np.degrees(np.linalg.norm(r[3 * n:].reshape(-1, 3), axis=1) / 0.05)
        C = np.eye(4)  # action frame <- base_link
        C[:3, :3] = Rotation.from_rotvec(sol.x[:3]).as_matrix()
        C[:3, 3] = sol.x[3:6]
        tool = np.eye(4)  # gripper_link -> tcp
        tool[:3, :3] = Rotation.from_rotvec(sol.x[6:9]).as_matrix()
        tool[:3, 3] = sol.x[9:12]
        print(f"[extract] arm{arm}: {n} frames, position error median {np.median(pe) * 1e3:.1f} mm (p90 {np.percentile(pe, 90) * 1e3:.1f}),"
              f" rotation median {np.median(re):.2f} deg", flush=True)
        result.append(dict(action_frame=pose_dict(np.linalg.inv(C)), action_tcp=pose_dict(tool),
                           fit=dict(frames=int(n), lag_steps=int(lag), position_median_mm=round(float(np.median(pe) * 1e3), 2),
                                    position_p90_mm=round(float(np.percentile(pe, 90) * 1e3), 2), rotation_median_deg=round(float(np.median(re)), 2))))
    return result


# -- 3. meshes from the link splats ---------------------------------------------------------------------------------------------
def link_mesh(points, voxel=0.003):
    import trimesh
    from skimage import measure
    lo = points.min(0) - 4 * voxel
    dims = np.ceil((points.max(0) + 4 * voxel - lo) / voxel).astype(int) + 1
    ijk = np.floor((points - lo) / voxel).astype(int)
    grid = np.zeros(dims, np.float32)
    np.add.at(grid, tuple(ijk.T), 1.0)
    occ = grid >= 2  # at least two splat centres: drops lone floaters
    occ = ndimage.binary_closing(occ, iterations=2)
    occ = ndimage.binary_fill_holes(occ)
    lab, n = ndimage.label(occ)
    if n > 1:  # keep pieces with at least 2 % of the largest one
        sizes = ndimage.sum(occ, lab, range(1, n + 1))
        occ = np.isin(lab, 1 + np.nonzero(sizes >= 0.02 * sizes.max())[0])
    field = ndimage.gaussian_filter(occ.astype(np.float32), 1.0)
    v, f, _, _ = measure.marching_cubes(field, 0.5)
    mesh = trimesh.Trimesh(v * voxel + lo, f[:, ::-1], process=True)
    trimesh.smoothing.filter_taubin(mesh, iterations=10)
    if len(mesh.faces) > 12000:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=12000)
        except Exception:  # fast_simplification not installed: keep the full mesh
            pass
    mesh.fix_normals()
    return mesh


def extract_meshes(rig, out_dir):
    fk = rig["fk"]
    L = len(fk["links"]) - 1
    gauss_frames = fk["links"][:L]  # frame index of each splat link
    means = rig["means"].numpy().astype(np.float64)
    labels = rig["labels"].numpy()
    opacity = rig["opacity"].numpy()
    scale = rig["scale"].numpy().max(1)
    rgb = rig["rgb01"].numpy()
    rest_inv = rig["rest_inv"].numpy().astype(np.float64)
    colors = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm in (0, 1):
        for l in range(L):
            m = (labels == arm * L + l) & (opacity > 0.2) & (scale < 0.012)
            if m.sum() < 50:
                continue
            P = means[m] @ rest_inv[arm][l][:3, :3].T + rest_inv[arm][l][:3, 3]
            name = FRAMES[gauss_frames[l]]
            mesh = link_mesh(P)
            mesh.export(out_dir / f"arm{arm}_{name}.stl")
            mesh.convex_hull.export(out_dir / f"arm{arm}_{name}_collision.stl")
            w = opacity[m][:, None]
            colors[f"arm{arm}_{name}"] = [round(float(c), 3) for c in np.clip((rgb[m] * w).sum(0) / w.sum(), 0, 1)]
            print(f"[extract] mesh arm{arm} {name}: {int(m.sum())} splats -> {len(mesh.faces)} faces", flush=True)
    return colors


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", default=str(ROOT / "assets/checkpoints"))
    p.add_argument("--out", default=str(ROOT / "src/worldsim_description"))
    p.add_argument("--cache", default=str(ROOT / "log/extract_cache"))
    p.add_argument("--lag", type=int, default=2, help="model steps between a commanded action and the arm reaching it")
    p.add_argument("--skip-meshes", action="store_true")
    a = p.parse_args()
    assets, out = Path(a.assets), Path(a.out)
    rig = torch.load(assets / "arms/rig.pt", map_location="cpu", weights_only=False)
    fk = rig["fk"]
    fk = dict(fk, origin=np.asarray(fk["origin"], np.float64), axis=np.asarray(fk["axis"], np.float64))

    data = predicted_joints(assets, Path(a.cache))
    arms = fit_action_frames(fk, data, a.lag)
    allq = np.concatenate([q for q, _ in data.values()])
    for arm, entry in enumerate(arms):  # a typical working pose: the median joints the model predicts on the seeds
        entry["rest"] = [round(float(v), 4) for v in np.median(allq[:, arm * 7: arm * 7 + 6], axis=0)]

    joints = []
    type_name = {0: "fixed", 1: "revolute", 2: "prismatic"}
    for i, (par, chi, O, typ, ax, src, mul) in enumerate(zip(fk["parent"], fk["child"], fk["origin"], fk["type"], fk["axis"], fk["source"], fk["mul"])):
        j = dict(name=JOINTS[i], parent=FRAMES[par], child=FRAMES[chi], type=type_name[typ], origin=pose_dict(O),
                 axis=[float(v) for v in ax])
        if typ == 1:
            lo, hi = VX300S_LIMITS[JOINTS[i]]
            seen = allq[:, [src, src + 7]]
            j.update(lower=round(min(lo, float(seen.min()) - 0.05), 4), upper=round(max(hi, float(seen.max()) + 0.05), 4),
                     velocity=VX300S_VELOCITY[JOINTS[i]], effort=10.0)
        elif typ == 2:
            lo, hi = sorted((0.005 * mul, 0.06 * mul))  # the mimic finger moves the other way: its limits are mirrored
            j.update(lower=round(lo, 4), upper=round(hi, 4), velocity=0.2, effort=10.0)
            if mul < 0:
                j["mimic"] = dict(joint="left_finger", multiplier=float(mul), offset=0.0)
        joints.append(j)

    tasks = {}
    for task in TASKS:
        A = torch.load(assets / task / "arms.pt", map_location="cpu", weights_only=False)
        cfg = json.load(open(assets / task / "task.json"))
        tasks[task] = dict(bases=[pose_dict(np.asarray(T, np.float64)) for T in A["T_base"].numpy()],
                           title=cfg.get("title", task))

    colors = {} if a.skip_meshes else extract_meshes(rig, out / "meshes")
    calib_path = out / "config/calibration.yaml"
    if a.skip_meshes and calib_path.is_file():
        colors = yaml.safe_load(open(calib_path)).get("colors", {})
    gripper_frame = FRAMES[fk["links"][-1]]
    calib = dict(
        robot="ViperX 300 S x2 (ALOHA), extracted from arms/rig.pt by tools/extract_robot.py",
        root_link="worldsim_world",
        arm_prefixes=["arm0_", "arm1_"],
        gripper_link=gripper_frame,
        chain=joints,
        arms=arms,
        tasks=tasks,
        # display only: the training data holds no usable map from the gripper command to the finger opening
        gripper_display=dict(scale=0.025, offset=0.01, lower=0.008, upper=0.045),
        colors=colors,
    )
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calib_path, "w") as f:
        f.write("# Generated by tools/extract_robot.py. Do not edit by hand; re-run the tool instead.\n")
        yaml.safe_dump(calib, f, sort_keys=False, default_flow_style=None, width=140)
    print(f"[extract] wrote {calib_path}", flush=True)

    sys.path.insert(0, str(ROOT / "src/worldsim_core"))
    from worldsim_core.robot import RobotCalibration, build_srdf, build_urdf
    c = RobotCalibration(calib_path)
    (out / "urdf").mkdir(exist_ok=True)
    (out / "urdf/worldsim_aloha.urdf").write_text(build_urdf(c))
    (out / "config/worldsim_aloha.srdf").write_text(build_srdf(c))
    print(f"[extract] wrote {out / 'urdf/worldsim_aloha.urdf'} and {out / 'config/worldsim_aloha.srdf'}", flush=True)


if __name__ == "__main__":
    main()
