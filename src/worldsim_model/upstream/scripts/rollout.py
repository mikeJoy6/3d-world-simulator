"""Roll the world model forward from a seed episode and render a video.

    python scripts/rollout.py --task release/cloth --steps 80 --out out/cloth.mp4
    python scripts/rollout.py --task release/box --actions hold --view cams --out out/box_cams.mp4
    python scripts/rollout.py --task release/assembly --actions my_actions.npy --mode particles --out out/assembly_particles.mp4

Actions: ``replay`` feeds the episode's recorded actions (open loop, the check that the model follows real commands), ``hold``
repeats the action at the seed frame, or a ``.npy`` file of shape (steps, 14) with raw actions (per arm x y z, roll pitch yaw,
gripper). ``--compare`` renders the ground-truth latent next to the prediction and writes the latent error per step.
"""
import argparse, json, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.world_model import WorldModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="released task folder, e.g. release/cloth")
    p.add_argument("--episode", default=None, help="latent episode file (default: the task's seed episode)")
    p.add_argument("--frame", type=int, default=None, help="start frame (default: the task's seed frame)")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--actions", default="replay", help="replay | hold | <file.npy>")
    p.add_argument("--view", default="orbit", help="orbit | cams")
    p.add_argument("--az", type=float, default=None)
    p.add_argument("--el", type=float, default=None)
    p.add_argument("--radius", type=float, default=None)
    p.add_argument("--size", type=int, nargs=2, default=(960, 640), metavar=("W", "H"))
    p.add_argument("--cams", default=None, help="dataset cameras for --view cams, e.g. 0,1,2")
    p.add_argument("--mode", default="splats", help="splats | particles")
    p.add_argument("--no-background", action="store_true")
    p.add_argument("--bf16", action="store_true", help="bf16 decoder, head and dynamics (what the demo runs)")
    p.add_argument("--compare", action="store_true", help="ground-truth latent next to the prediction + metrics.json")
    p.add_argument("--rng", type=int, default=0)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.manual_seed(a.rng)
    wm = WorldModel(a.task, bf16=a.bf16, background=not a.no_background)
    z, act = wm.load_episode(a.episode)
    st = wm.seed(z, act, a.frame)
    t0 = st.t
    if a.actions == "replay":
        acts = [act[min(t, len(act) - 1)] for t in range(t0, t0 + a.steps)]
    elif a.actions == "hold":
        acts = [act[t0 - 1]] * a.steps
    else:
        acts = list(np.load(a.actions).astype(np.float32))[: a.steps]
    cam_ids = None if a.cams is None else [int(c) for c in a.cams.split(",")]
    if a.view == "cams":
        cams = wm.episode_cameras(a.episode, frame=t0, cams=cam_ids)
    else:
        cams = [wm.orbit_camera(a.az, a.el, a.radius, W=a.size[0], H=a.size[1])]

    def frame(z_t, t):
        if a.mode == "particles":
            xyz, rgb = wm.particles(z_t)
            tiles = [wm.render_points(xyz, rgb, *c) for c in cams]
        else:
            g = wm.decode(z_t)
            tiles = [wm.render(g, *c, background=not a.no_background) for c in cams]
        h = min(x.shape[0] for x in tiles)
        return np.concatenate([cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h)) if x.shape[0] != h else x for x in tiles], 1)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer, metrics, tick = (None, [], time.time())
    for i, ac in enumerate(acts):
        t = st.t
        if a.view == "cams":
            cams = wm.episode_cameras(a.episode, frame=t, cams=cam_ids)
        zp = wm.step(st, ac)
        im = frame(zp, t)
        if a.compare and t < len(z):
            gt = torch.from_numpy(z[t]).to(wm.device)
            metrics.append(dict(step=i, frame=int(t), latent_mse=float(torch.mean((zp - gt) ** 2))))
            im = np.concatenate([im, frame(gt, t)], 0 if len(cams) > 1 else 1)   # prediction, then ground truth
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (im.shape[1], im.shape[0]))
            cv2.imwrite(str(out.with_suffix(".png")), im[..., ::-1])
        writer.write(np.ascontiguousarray(im[..., ::-1]))
    writer.release()
    dt = (time.time() - tick) / max(1, len(acts))
    print(f"[rollout] {len(acts)} steps -> {out} ({dt * 1000:.0f} ms per step incl. rendering)")
    if metrics:
        json.dump(metrics, open(out.with_suffix(".metrics.json"), "w"), indent=1)
        print(f"[rollout] latent mse step 1 {metrics[0]['latent_mse']:.5f}, step {len(metrics)} {metrics[-1]['latent_mse']:.5f}")


if __name__ == "__main__":
    main()
