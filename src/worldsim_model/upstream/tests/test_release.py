"""Checks on every released task folder (GPU): the decoder renders the seed frame close to the real camera images, and the
dynamics model follows the recorded actions.

    RELEASE_DIR=release pytest -s tests/test_release.py
"""
import glob, os, sys
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
RELEASE = Path(os.environ.get("RELEASE_DIR", "release"))
TASKS = sorted(Path(p).parent.name for p in glob.glob(str(RELEASE / "*" / "task.json")))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not TASKS, reason="needs a GPU and a release folder")
_cache = {}


def world(task, fast=False):
    from models.world_model import WorldModel

    if (task, fast) not in _cache:
        _cache[task, fast] = WorldModel(RELEASE / task, fast=fast)
    return _cache[task, fast]


def psnr(a, b):
    return float(10 * np.log10(255.0**2 / np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)))


@pytest.mark.parametrize("task", TASKS)
def test_decode_matches_cameras(task):
    wm = world(task)
    z, act = wm.load_episode()
    t = wm.cfg["seed"]["frame"] + wm.n_hist - 1
    g = wm.decode(z[t])
    assert g is not None and g["xyz"].shape[0] > 1000
    import h5py

    scores, exposure, wm.exposure = [], wm.exposure, 1.0  # the camera images are compared without the display exposure
    with h5py.File(wm.dir / wm.cfg["seed"]["file"], "r") as f:
        for vm, K, W, H, c in wm.episode_cameras(frame=t):
            gc = f["render_cams"][f"camera_{c}"]
            i = int(np.argmin(np.abs(gc["gt_frames"][:] - t)))
            tg = int(gc["gt_frames"][i])
            vm_t = torch.from_numpy(gc["viewmat"][tg].astype(np.float32)).to(wm.device)
            gt = cv2.imdecode(np.asarray(gc["gt_jpg"][i], np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
            im = wm.render(wm.decode(z[tg]), vm_t, K, W, H)
            scores.append((c, psnr(im, gt)))
    wm.exposure = exposure
    print(f"\n[{task}] render vs camera image at frame {t}: " + ", ".join(f"cam {c} {s:.1f} dB" for c, s in scores))
    assert all(s > 15.0 for _, s in scores)


@pytest.mark.parametrize("task", TASKS)
def test_rollout_follows_recorded_actions(task):
    wm = world(task)
    torch.manual_seed(0)
    z, act = wm.load_episode()
    st = wm.seed(z, act)
    errs = []
    for _ in range(20):
        t = st.t
        zp = wm.step(st, act[t])
        errs.append(float(torch.mean((zp - torch.from_numpy(z[t]).to(wm.device)) ** 2)))
    base = float(np.mean((z[st.t - 1] - z[st.t - 21]) ** 2))
    print(f"\n[{task}] open-loop latent mse: step 1 {errs[0]:.5f}, step 20 {errs[-1]:.5f} (frozen seed would be {base:.5f})")
    assert np.all(np.isfinite(errs)) and errs[-1] < 0.05


@pytest.mark.parametrize("task", TASKS)
def test_orbit_and_particles(task):
    wm = world(task)
    z, act = wm.load_episode()
    t = wm.cfg["seed"]["frame"] + wm.n_hist - 1
    cam = wm.orbit_camera(W=480, H=320)
    im = wm.render(wm.decode(z[t]), *cam)
    assert im.shape == (320, 480, 3) and im.std() > 10
    xyz, rgb = wm.particles(z[t])
    assert xyz.shape[0] > 1000
    pts = wm.render_points(xyz, rgb, *cam)
    assert pts.shape == (320, 480, 3) and pts.mean() < 250


@pytest.mark.parametrize("task", TASKS)
def test_demo_session(task):
    """The interactive demo's control and views, without a browser: a held key moves the commanded position by one keyboard
    step per tick inside the workspace, the gizmo geometry is on screen, and every view renders."""
    wm = world(task)
    if "arm_base" not in wm.cfg:
        pytest.skip("no demo assets in this task folder")
    from demo.session import Session

    s = Session(wm)
    s.reseed()
    for _ in range(60):  # let a gripper ramp, a start outside the limits and the settle finish
        if s.tick(set(), []) is None:
            break
    best = None
    for key, (arm, axis, sign) in sorted(s.keys.items()):
        if arm in s.locked:
            continue
        p0 = np.array(s.S["pos"][arm], copy=True)
        for _ in range(5):
            a = s.tick({key}, [])
            if a is not None:
                s.z_last = wm.step(s.st, a)
        d = s.S["pos"][arm] - p0
        assert abs(d[axis]) <= 5 * s.ctl["step"] + 1e-9 and d[axis] * sign >= -1e-9, (key, d)
        assert np.all(s.S["pos"][arm] >= s.ws_min[arm] - 1e-9) and np.all(s.S["pos"][arm] <= s.ws_max[arm] + 1e-9)
        if best is None or abs(d[axis]) > abs(best[1]):
            best = (key, d[axis], arm, axis)
    assert abs(best[1]) > 0.5 * s.ctl["step"]  # some key moves its arm (keys at a workspace limit do not)
    key, moved, arm, axis = best
    for view, display in (("orbit", "splats"), ("orbit", "particles"), ("cams", "splats"), ("orbit", "voxels"), ("orbit", "latents"), ("orbit_white", "splats")):
        s.ctl.update(view=view, display=display)
        s.publish()
        im = cv2.imdecode(np.frombuffer(s.jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert im is not None and im.std() > 5, (view, display)
        assert s.gizmo["anchors"] and s.gizmo["w"] == im.shape[1]
    print(f"\n[{task}] demo session: key '{key}' moved arm {arm} axis {axis} by {moved * 1000:+.1f} mm in 5 ticks; six views rendered")


@pytest.mark.parametrize("task", TASKS)
def test_fast_matches_reference(task):
    """The optimised inference (bf16, sparse finest decoder level, compiled dynamics) renders the same picture."""
    wm, fast = world(task), world(task, fast=True)
    z, act = wm.load_episode()
    t = wm.cfg["seed"]["frame"] + wm.n_hist - 1
    cam = wm.orbit_camera(W=480, H=320)
    s = psnr(wm.render(wm.decode(z[t]), *cam), fast.render(fast.decode(z[t]), *cam))
    print(f"\n[{task}] optimised vs reference render: {s:.1f} dB")
    assert s > 35.0


@pytest.mark.parametrize("task", TASKS)
def test_kinematic_arms(task):
    """The kinematic robot model: joints read from the latent, the rig locked onto the predicted grippers, the world model's
    robot Gaussians kept only at the grippers, the joints of an arm that is commanded still held, and the same picture from
    the optimised inference."""
    wm = world(task)
    if wm.arms is None:
        pytest.skip("no kinematic arms in this release (release/arms/rig.pt, <task>/arms.pt)")
    from models.arm_rig import in_box

    z, act = wm.load_episode()
    t = wm.cfg["seed"]["frame"] + wm.n_hist - 1
    cam = wm.orbit_camera(W=480, H=320)
    arms = wm.arms.new_state()
    g = wm.decode(z[t], arms=arms)
    assert arms.pose.shape == (14,) and np.all(np.isfinite(arms.pose)) and np.abs(arms.pose[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]).max() < np.pi
    assert all(0.0 < arms.pose[i] < 0.12 for i in (6, 13)), arms.pose  # finger openings in metres
    rig, robot = g["cls"] == 3, g["cls"] == 1
    assert int(rig.sum()) == wm.arms.means.shape[0] and "sh1" in g
    Wi, lo, hi = arms.boxes
    kept = in_box(g["xyz"][robot], Wi[0], lo[0], hi[0]) | in_box(g["xyz"][robot], Wi[1], lo[1], hi[1])
    assert bool(kept.all()) and int(robot.sum()) > 500  # the world model keeps its grippers and nothing else of the robot
    for arm in (0, 1):  # the lock moved the rig gripper by a small accepted correction, and it sits inside the model's box
        T = arms.Tloc[arm]
        assert np.linalg.norm(T[:3, 3]) < wm.arms.cfg["lock_max_t"]
        p = wm.arms.gripper_anchor(arms, arm)
        assert np.all(p > wm.bmin.cpu().numpy() - 0.15) and np.all(p < wm.bmax.cpu().numpy() + 0.15), (arm, p)
    with_arms, without = wm.render(g, *cam), wm.render(wm.decode(z[t]), *cam)
    assert psnr(with_arms, without) < 35.0  # the arms are drawn
    again = wm.arms.new_state()
    wm.decode(z[t], arms=again)
    assert np.array_equal(again.pose, arms.pose)  # a fresh state gives the same pose: nothing but the latent goes in
    shown = arms.pose.copy()
    arms.still = (True, True)
    wm.decode(z[t + 3], arms=arms)
    assert np.array_equal(arms.pose[:6], shown[:6]) and np.array_equal(arms.pose[7:13], shown[7:13])  # commanded still: joints held
    fast = world(task, fast=True)
    s = psnr(with_arms, fast.render(fast.decode(z[t], arms=fast.arms.new_state()), *cam))
    print(f"\n[{task}] kinematic arms: {int(rig.sum())} rig gaussians, {int(robot.sum())} world-model gripper gaussians, lock {[round(float(np.linalg.norm(T[:3, 3])) * 1e3, 1) for T in arms.Tloc]} mm; optimised vs reference {s:.1f} dB")
    assert s > 35.0
