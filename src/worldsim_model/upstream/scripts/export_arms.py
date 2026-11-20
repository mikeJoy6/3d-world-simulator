"""Bake the kinematic robot model of the public demo into the release (run by scripts/export_release.py; authors' workstation).

    WS_ROOT=<workstation tree> python scripts/export_arms.py --out release [--tasks cloth box pusht assembly=plug]

This is the only place that imports the demo's original code ("the bundle", with urdfpy and the arm-splat repository): the rig
is loaded, edited and reordered by that code exactly as the public demo does at start-up, and written out as plain tensors, so
the runtime (models/arm_rig.py) needs none of it. Every per-task switch of the public demo (deploy/tasks.json: flags and
environment variables) is resolved to a plain value here.

release/arms/rig.pt       both arms' Gaussians (canonical frame, opacity-pruned, ordered [fixed links | wrist + gripper]),
                          link labels, the lock template and finger subsets, and the forward-kinematics tables of the URDF
release/<task>/arms.pt    everything per task in one file: arm base frames in the world, the two networks of ArmRig
                          (joints from the latent, snap onto the predicted gripper) and the settings
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

J = os.environ.get("WS_ROOT", "")
BUNDLE = f"{J}/rollout_view_1019/3dws"
JOINT_SOURCE = {"waist": 0, "shoulder": 1, "elbow": 2, "forearm_roll": 3, "wrist_angle": 4, "wrist_rotate": 5, "left_finger": 6, "right_finger": 6}
JOINT_TYPE = {"revolute": 1, "continuous": 1, "prismatic": 2}


def fk_tables(npfk):
    """interactive.np_fk.NpFK (built from the URDF) as arrays: one row per joint on the chains to the rig's links, parents first."""
    names = [npfk.base] + [s[1] for s in npfk.steps]
    return dict(
        n_frames=len(names), base=0, parent=[names.index(s[0]) for s in npfk.steps], child=[names.index(s[1]) for s in npfk.steps],
        origin=np.stack([s[2] for s in npfk.steps]), type=[JOINT_TYPE.get(s[3], 0) for s in npfk.steps], axis=np.stack([s[4] for s in npfk.steps]),
        source=[JOINT_SOURCE.get(s[5], -1) for s in npfk.steps], mul=[float(s[6]) for s in npfk.steps], off=[float(s[7]) for s in npfk.steps],
        links=[names.index(l) for l in npfk.links], link_names=list(npfk.links),
    )


def settings(E, cfg, A):
    """The public demo's environment variables and task table (interactive/teleop.py) as plain values."""
    assert not E.get("GRIP_SEG") and not E.get("ARM_STILL_HOLD") and A.mode == "gripper" and A.lock and A.attach and not A.weld and A.filt_on and not A.rebase
    class_filters = bool(cfg.get("class_filters", True))
    return dict(
        lock_cls=[int(c) for c in E.get("ARM_LOCK_CLS", "1").split(",")], lock_zmin=float(E["ARM_LOCK_ZMIN"]) if E.get("ARM_LOCK_ZMIN") else None,
        lock_max_t=float(E.get("ARM_LOCK_MAX_T", 0.03)), lock_max_ang=float(E.get("ARM_LOCK_MAX_ANG", 15.0)), lock_split=E.get("ARM_LOCK_SPLIT", "1") == "1",
        hold_tol_deg=float(E.get("ARM_HOLD_TOL") or 0), filter_min_cutoff=float(A.filt_min), filter_beta=float(A.filt_beta), filter_deadband_deg=float(A.filt_dead),
        grip_pad=float(A.grip_pad), grip_back=[float(b) for b in A.grip_back], rig_overlap=[float(v) for v in A.rig_overlap],
        grip_open_box=float(E.get("GRIP_OPEN_BOX") or 0),
        grip_box_grow={int(t.split(":")[0]): [float(v) for v in t.split(":")[1].split(",")] for t in filter(None, E.get("GRIP_BOX_GROW", "").split(";"))},
        table_float=(E.get("TABLE_FLOAT", "1") == "1" and not class_filters and bool(cfg.get("table_float", True))) or E.get("TABLE_FLOAT") == "force",
        table_float_z=float(E.get("TABLE_FLOAT_Z") or (cfg["table_float"] if isinstance(cfg.get("table_float"), float) else 0.03)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="release")
    p.add_argument("--tasks", nargs="+", default=["cloth", "box", "pusht", "assembly=plug"], help="release task name, or name=bundle name where the two differ")
    a = p.parse_args()
    out = Path(a.out).resolve()
    deploy = json.load(open(f"{BUNDLE}/deploy/tasks.json"))
    names = [(t.split("=")[0], t.split("=")[-1]) for t in a.tasks]  # (release name, bundle name)
    os.environ.update({k: deploy[names[0][1]]["env"][k] for k in ("JIMMY", "ARM_SPLATS", "ARM_DIFIX")})
    os.chdir(BUNDLE)
    sys.path.insert(0, BUNDLE)
    from interactive.arm_layer import ArmLayer
    from interactive.teleop import TASKS, make_parser
    from interactive.teleop_fast import FAST_DEFAULTS

    rig = None
    for t, bundle in names:
        E = deploy[bundle]["env"]
        assert all(E[k] == os.environ[k] for k in ("JIMMY", "ARM_SPLATS", "ARM_DIFIX")), "the rig is shared: every task must use the same arm splats"
        os.environ.pop("ARM_BASE_DZ", None)
        if E.get("ARM_BASE_DZ"):
            os.environ["ARM_BASE_DZ"] = E["ARM_BASE_DZ"]
        args = make_parser(FAST_DEFAULTS).parse_args(deploy[bundle]["flags"])
        assert args.arms_fast and args.lock_every == 1 and not args.arm_full and args.lock_net == E.get("LOCK_OVERRIDE", args.lock_net)
        cfg = TASKS[bundle]
        A = ArmLayer(args.arm_head, f"{E['JIMMY']}/aloha_processed/{cfg['ds']}", "cuda", np.zeros(3), np.zeros(3), margin=args.arm_margin, vae=object(), lock_net=args.lock_net, prune_opac=args.rig_prune)
        A.set_mode("gripper")
        A._prepare_fast()
        assert A.kind == "joint"
        R = dict(
            means=A.f_means, quats=A.f_quats, labels=A.f_lab, **A._st, template=A.f_tmpl, n_template=[int(n) for n in A.f_ntm], finger_sub=list(A.f_fsub),
            region=[(int(s0), int(s1)) for s0, s1 in A.f_cand], region_finger=list(A.f_cand_finger), rest_inv=torch.from_numpy(A.f_inv), attach_w=torch.from_numpy(np.asarray(A.f_w)),
        )
        R = {k: (v.cpu() if torch.is_tensor(v) else [x.cpu() if torch.is_tensor(x) else x for x in v]) for k, v in R.items()}
        R["fk"] = fk_tables(A.npfk)
        if rig is None:
            rig = R
            (out / "arms").mkdir(parents=True, exist_ok=True)
            torch.save(rig, out / "arms" / "rig.pt")
            print(f"[arms] rig: {len(rig['labels'])} gaussians, {len(rig['fk']['links']) - 1} links -> {out / 'arms/rig.pt'} ({os.path.getsize(out / 'arms/rig.pt') >> 20} MB)", flush=True)
        else:
            assert all(torch.equal(R[k], rig[k]) for k in ("means", "quats", "labels", "opacity", "rgb01", "template")), "rig differs between tasks"
        ck = torch.load(args.arm_head, map_location="cpu", weights_only=False)
        lk = torch.load(args.lock_net, map_location="cpu", weights_only=False)
        S = settings(E, cfg, A)
        torch.save(
            dict(
                T_base=torch.from_numpy(np.asarray(A.f_Tb, np.float64)),
                joints=dict(state=ck["joint"], channels=int(ck["CJ"]), width=int(ck["width"]), kind=ck["head"], feat=ck["feat"], y_mean=torch.as_tensor(ck["y_mean"]).float(), y_std=torch.as_tensor(ck["y_std"]).float(), step=ck.get("step"), val=ck.get("val")),
                snap=dict(state=lk["model"], width=int(lk["width"]), pool=lk.get("pool", "meanmax"), tau=float(lk.get("tau", 0.2)), step=lk.get("step"), val=lk.get("val")),
                settings=S, sources=dict(joint_head=args.arm_head, lock_net=args.lock_net),
            ),
            out / t / "arms.pt",
        )
        print(f"[arms] {t}: {out / t / 'arms.pt'} ({os.path.getsize(out / t / 'arms.pt') >> 20} MB) settings {json.dumps(S)}", flush=True)
        del A
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
