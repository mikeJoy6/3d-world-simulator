"""Golden comparison of the kinematic robot arms: this repository against the authors' demo code (workstation only).

The public demo's code ("the bundle", not part of this repository) is run on a fixed action sequence and everything the arm path
produces is saved; this repository is then run on the SAME latents and compared: the 14-d arm pose, the positions of the arm
Gaussians, the gripper boxes, the rendered free-view frame over the scanned room on every frame, and the other five "Show"
tiles of the public page (3D particles, multi-view, upsampled voxels, latent representation, raw model output) on every sixth.

    # 1. the bundle side (its own process: the two code bases share module names); TensorRT off
    python tests/golden_arms.py bundle --task cloth --mode ref  --out /tmp/golden
    python tests/golden_arms.py bundle --task cloth --mode fast --out /tmp/golden     # bf16 + sparse decoder, same latents
    # 2. this repository on the same latents
    python tests/golden_arms.py clean --task cloth --out /tmp/golden [--fast | --head-bf16] [--release release]

The bundle's reference decode still runs its Gaussian head in bfloat16, this repository's reference mode runs it in fp32:
``--head-bf16`` switches that one thing and the comparison becomes exact.

BUNDLE_ROOT points at the bundle (default: the authors' path).
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ["BUNDLE_ROOT"])
BUNDLE_NAME = dict(assembly="plug")  # release task name -> the bundle's name for it
LATENTS = dict(cloth="data/view/cloth/latents_demo/all", box="data/view/box/latents/all", pusht="data/view/pusht/latents/train", assembly="data/view/plug/latents/train")
N_STEPS = 36
# the six "Show" tiles of the public page (interactive/demo_clean.html) as the controls they set: view, display, arms and room
VIEWS = dict(
    splats=("orbit", "splats", True), particles=("orbit", "particles", True), multi=("cams", "splats", True),
    voxels=("orbit", "voxels", True), latents=("orbit", "latents", True), raw=("orbit_white", "splats", False),
)
VIEW_EVERY = 6  # the rendered scene is compared on every frame, the other five tiles on every sixth


def key_plan(cfg):
    """Held keys per tick: arm 0 along one axis, then another arm / axis, a gripper tap where the task has one, then back."""
    free = [(k, v) for k, v in sorted(cfg["keymap"].items()) if v[0] not in cfg.get("locked_arms", ())]
    k0 = next(k for k, v in free if v[0] == 0)
    k1 = next((k for k, v in free if v[0] == 1), None) or next(k for k, v in free if v[:2] != cfg["keymap"][k0][:2])
    back = next(k for k, v in free if v[:2] == cfg["keymap"][k0][:2] and v[2] == -cfg["keymap"][k0][2])
    grip = sorted(cfg.get("grip_keys") or {})
    plan = []
    for i in range(N_STEPS):
        held = {k0} if i < 12 else {k1} if i < 20 else set() if i < 26 else {back}
        plan.append((held, grip[:1] if i == 20 else []))
    return plan


def run_bundle(a):
    import torch

    name = BUNDLE_NAME.get(a.task, a.task)
    T = json.load(open(BUNDLE / "deploy/tasks.json"))[name]
    os.environ.update(T["env"])
    os.environ["DEMO_UI"] = ""
    os.chdir(BUNDLE)
    sys.path.insert(0, str(BUNDLE))
    flags = list(T["flags"])
    flags[flags.index("--latent-dir") + 1] = LATENTS[a.task]
    flags.insert(flags.index("--packed-dirs") + 1, f"{os.environ['JIMMY']}/aloha_processed/_view_packed/{name}")
    from interactive.teleop import Teleop, build_world, make_parser

    torch.set_grad_enabled(False)
    # the public demo's options (interactive/teleop_fast.py) without TensorRT and torch.compile; ref: also fp32 and the dense decoder
    defaults = dict(lowsync=True, arms_fast=True) if a.mode == "ref" else dict(bf16=True, sparse_fine=[1, 2, 2], lowsync=True, arms_fast=True)
    args = make_parser(defaults).parse_args(flags)
    cfg, wm, eps = build_world(args)
    app = Teleop(args, cfg, wm, eps)
    app.load_dyn_options()
    app.ctl["view"] = "orbit"
    A = wm.arms
    rec = dict(z=[], pose=[], head=[], xyz=[], opacity=[], box_Wi=[], box_lo=[], box_hi=[], Tloc=[], img=[], img_noarms=[], hold=[], action=[])
    raw = {}
    stab = A.stabilise
    A.stabilise = lambda p: (raw.__setitem__("p", p.copy()), stab(p))[1]
    arm_g = wm.arm_gaussians
    wm.arm_gaussians = lambda z, g=None: (lambda o: (raw.__setitem__("g", o), o)[1])(arm_g(z, g))

    def publish(z):
        app.R["z_last"] = z
        app.ctl["arms"] = True
        im = app.frame(z, as_tensor=True).cpu().numpy()
        g = raw["g"]
        rec["z"].append(z.float().cpu().numpy())
        rec["pose"].append(np.array(A._pose))
        rec["head"].append(raw["p"])
        rec["xyz"].append(g["xyz"].cpu().numpy())
        rec["opacity"].append(g["opacity"].cpu().numpy())
        for k, i in (("box_Wi", 0), ("box_lo", 1), ("box_hi", 2)):
            rec[k].append(np.stack([b[i].cpu().numpy() for b in A.last_boxes]))
        rec["Tloc"].append(np.stack(A.Tloc))
        rec["img"].append(im)
        rec["hold"].append(np.array(getattr(A, "still_hold", (False, False))))
        if (len(rec["z"]) - 1) % VIEW_EVERY == 0:  # the other tiles, in this order (each tile with arms advances the arm filter once)
            for name, (view, display, on) in list(VIEWS.items())[1:]:
                app.ctl.update(view=view, display=display, arms=on, bg=on)
                rec.setdefault(f"view_{name}", []).append(app.frame(z, as_tensor=True).cpu().numpy())
            app.ctl.update(view="orbit", display="splats", arms=True, bg=True)
        app.ctl["arms"] = False
        rec["img_noarms"].append(app.frame(z, as_tensor=True).cpu().numpy())
        app.ctl["arms"] = True

    app.publish = lambda z: app.frame(z)
    app.reseed()  # warm-up frame: the bundle measures its scan table gain on the first frame it renders, which is drawn without it
    app.publish = publish
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    given = np.load(out / f"{a.task}_ref.npz") if a.mode != "ref" else None
    app.reseed()
    rec["action"].append(np.zeros(14))
    prev = None
    plan = key_plan(cfg)
    plan_back = plan[-1][0]
    for i, (held, taps) in enumerate(plan):
        if given is None:
            a14, mode = app.tick(held, taps, False)
            if a14 is None:  # nothing commanded and nothing settling: the demo would not step, so keep driving
                a14, mode = app.tick(plan_back, [], False)
            assert a14 is not None, (i, mode)
        else:
            a14 = given["action"][i + 1]
        A.still_hold = tuple(bool(prev is not None and np.allclose(a14[j * 7 : j * 7 + 6], prev[j * 7 : j * 7 + 6], atol=1e-7)) for j in range(2))
        prev = np.array(a14, copy=True)
        z = app.advance(a14) if given is None else torch.from_numpy(given["z"][i + 1]).to(wm.dev)
        if given is None:
            app.after_step(z, a14, mode)
        else:
            app.R["step"] = i + 1  # the dataset cameras of the multi-view follow the frame
        rec["action"].append(np.array(a14))
        publish(z)
    view = dict(az=float(app.ctl["az"]), el=float(app.ctl["el"]), rad=float(app.ctl["rad"]))
    np.savez(out / f"{a.task}_{a.mode}.npz", view=json.dumps(view), **{k: np.stack(v) for k, v in rec.items()})
    print(f"[golden] {a.task} {a.mode}: {len(rec['z'])} frames -> {out / f'{a.task}_{a.mode}.npz'}; view {view}", flush=True)


def psnr(x, y):
    return float(10 * np.log10(255.0**2 / max(np.mean((x.astype(np.float32) - y.astype(np.float32)) ** 2), 1e-12)))


def run_clean(a):
    """This repository's demo session (demo/session.py) on the bundle's latents: the same tiles, rendered in the same order."""
    import cv2
    import torch

    sys.path.insert(0, str(REPO))
    from demo.session import Session
    from models.world_model import WorldModel

    torch.set_grad_enabled(False)
    out = Path(a.out)
    G = np.load(out / f"{a.task}_{'fast' if a.fast else 'ref'}.npz")
    view = json.loads(str(G["view"]))
    wm = WorldModel(Path(a.release) / a.task, fast=a.fast)
    if a.head_bf16:  # the bundle always runs the Gaussian head in bfloat16; this repository's reference mode runs it in fp32
        wm.head.head.bf16 = True
    s = Session(wm)
    s.reseed()
    s.ctl.update(az=view["az"], el=view["el"], rad=view["rad"])
    arms = s.arms = wm.arms.new_state()
    t0 = s.st.t
    d = dict(pose=[], head=[], xyz=[], box=[], tloc=[], psnr=[], psnr_noarms=[], hidden=[], **{f"view_{k}": [] for k in list(VIEWS)[1:]})
    tag = "fast" if a.fast else "ref_headbf16" if a.head_bf16 else "ref"

    def tile(name):
        v, disp, on = VIEWS[name]
        s.ctl.update(view=v, display=disp, arms=on, bg=on)
        return s.frame(z, s.cameras())

    def side_by_side(ref, im, name):
        diff = np.abs(im.astype(np.int16) - ref.astype(np.int16)).clip(0, 255).astype(np.uint8) * 4
        cv2.imwrite(str(out / name), np.concatenate([ref, im, diff], 0 if ref.shape[1] > 1200 else 1)[..., ::-1])

    for i in range(len(G["z"])):
        z = torch.from_numpy(G["z"][i]).to(wm.device)
        arms.still = tuple(bool(x) for x in G["hold"][i])
        s.st.t = t0 + i  # the dataset cameras of the multi-view follow the frame
        im = tile("splats")
        d["pose"].append(np.abs(arms.pose - G["pose"][i]).max())
        d["head"].append(np.abs(arms.head_pose - G["head"][i]).max())
        d["xyz"].append(np.abs(arms.rig["xyz"].cpu().numpy() - G["xyz"][i]).max())
        d["hidden"].append(int(((arms.rig["opacity"].cpu().numpy() > 0) != (G["opacity"][i] > 0)).sum()))
        Wi, lo, hi = arms.boxes
        d["box"].append(max(np.abs(Wi.cpu().numpy() - G["box_Wi"][i]).max(), np.abs(lo.cpu().numpy() - G["box_lo"][i]).max(), np.abs(hi.cpu().numpy() - G["box_hi"][i]).max()))
        d["tloc"].append(np.abs(np.stack(arms.Tloc) - G["Tloc"][i]).max())
        d["psnr"].append(psnr(im, G["img"][i]))
        if i in (0, len(G["z"]) // 2, len(G["z"]) - 1) or d["psnr"][-1] < 38:
            side_by_side(G["img"][i], im, f"{a.task}_{tag}_{i:02d}.png")
        if i % VIEW_EVERY == 0:
            for name in list(VIEWS)[1:]:
                ref, im = G[f"view_{name}"][i // VIEW_EVERY], tile(name)
                assert ref.shape == im.shape, (name, ref.shape, im.shape)
                d[f"view_{name}"].append(psnr(im, ref))
                if i == 18 or d[f"view_{name}"][-1] < 38:
                    side_by_side(ref, im, f"{a.task}_{tag}_{name}_{i:02d}.png")
        s.ctl.update(view="orbit", display="splats", arms=False, bg=True)  # the room without the rig (no tile of the page)
        d["psnr_noarms"].append(psnr(s.frame(z, s.cameras()), G["img_noarms"][i]))
    P = np.array(d["psnr"])
    print(
        f"[golden] {a.task} {tag}: frames {len(P)}  pose max {max(d['pose']):.2e} (rad | m)  unfiltered head max {max(d['head']):.2e}  "
        f"rig xyz max {max(d['xyz']) * 1e3:.3f} mm  boxes max {max(d['box']) * 1e3:.3f} mm  Tloc max {max(d['tloc']):.2e}  hidden-splat mismatches max {max(d['hidden'])}",
        flush=True,
    )
    views = [("splats", d["psnr"])] + [(k, d[f"view_{k}"]) for k in list(VIEWS)[1:]] + [("(room, no rig)", d["psnr_noarms"])]
    print(f"[golden] {a.task} {tag} PSNR min / mean dB: " + "   ".join(f"{k} {min(v):.1f} / {np.mean(v):.1f}" for k, v in views), flush=True)
    json.dump({k: [float(x) for x in v] for k, v in d.items()}, open(out / f"{a.task}_{tag}_result.json", "w"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("side", choices=["bundle", "clean"])
    p.add_argument("--task", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="ref", choices=["ref", "fast"], help="bundle: reference decode, or bf16 + sparse finest level (same latents as ref)")
    p.add_argument("--fast", action="store_true", help="clean: WorldModel(fast=True), compared with the bundle's fast dump")
    p.add_argument("--head-bf16", action="store_true", help="clean, reference mode: run the Gaussian head in bfloat16 as the bundle does (isolates that one difference)")
    p.add_argument("--release", default=str(REPO / "release"))
    a = p.parse_args()
    run_bundle(a) if a.side == "bundle" else run_clean(a)
