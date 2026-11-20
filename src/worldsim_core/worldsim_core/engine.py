"""The world model as one object with no ROS in it: load a task, seed it, step it with 14-d actions, and read the
predicted scene back as particles or as a rendered image.

Only the scene is particles. The world model predicts the whole workspace, arms included, as Gaussians with a class
label (0 object, 1 robot, 2 table). ``scene()`` drops the robot class; the arms are drawn from the URDF instead, at the
joints that produced the action. The upstream demo's splat arm rig (``models/arm_rig.py``) is not drawn; its joint head
can still be read (``model_joints``) to show where the model itself believes the arms are.

    eng = WorldSimEngine()
    eng.load("pusht")
    seed = eng.reset()                       # 14-d action of the seed frame
    eng.step(action)                         # one model step = 0.1 s of the recorded data
    xyz, rgb = eng.points(max_points=100000) # scene particles of the current prediction
    img = eng.render()                       # same prediction over the scanned room, from the task's camera
"""
import gc
import sys
from pathlib import Path

import numpy as np

from .tasks import TaskConfig, assets_root

ROBOT_CLASS = 1


def upstream_root():
    """The upstream model code (the folder with models/world_model.py): installed with worldsim_model, or found from
    the source tree, either as the repository root (this repository is a fork of the upstream one; the ROS packages are
    in ros2/) or as src/worldsim_model/upstream."""
    try:
        from ament_index_python.packages import get_package_share_directory
        p = Path(get_package_share_directory("worldsim_model")) / "upstream"
        if (p / "models/world_model.py").is_file():
            return p
    except Exception:
        pass
    for parent in Path(__file__).resolve().parents:
        for cand in (parent, parent / "worldsim_model/upstream"):
            if (cand / "models/world_model.py").is_file():
                return cand
    raise FileNotFoundError("upstream model code (models/world_model.py) not found")


class WorldSimEngine:
    def __init__(self, assets=None, device="cuda", image=True, robot_mask=True, bf16=True, fast=False, camera_size=(640, 400)):
        """image: load the scanned room and render camera images. robot_mask: load the release's arm rig to find the
        model's own arms (better robot removal, and ``model_joints``). fast: the upstream interactive settings (sparse
        decoder, compiled dynamics; the first load of a task compiles for about a minute)."""
        up = str(upstream_root())
        if up not in sys.path:
            sys.path.insert(0, up)
        self.assets = assets_root(assets)
        self.device, self.image, self.use_rig, self.bf16, self.fast = device, bool(image), bool(robot_mask), bf16, bool(fast)
        self.camera_size = camera_size
        self.wm = self.task = self.cfg = None
        self.state = self.latent = self._scene = self.arm_state = None
        self.z = self.actions = None

    # -- tasks -----------------------------------------------------------------------------------------------------------
    def load(self, task):
        import torch
        from models.world_model import WorldModel
        self.unload()
        self.cfg = TaskConfig(task, self.assets)
        self.wm = WorldModel(self.assets / task, device=self.device, bf16=self.bf16, background=self.image, arms=self.use_rig, fast=self.fast)
        self.task = task
        self.z, self.actions = self.wm.load_episode()
        W, H = self.camera_size
        self.camera = self.wm.orbit_camera(W=W, H=H)
        torch.cuda.empty_cache()

    def unload(self):
        if self.wm is None:
            return
        import torch
        self.wm = self.state = self.latent = self._scene = self.z = self.actions = None
        gc.collect()
        torch.cuda.empty_cache()

    # -- rollout ----------------------------------------------------------------------------------------------------------
    def reset(self):
        """Back to the task's seed frame. Returns the 14-d action the seed continues with (the arms' starting command)."""
        self.state = self.wm.seed(self.z, self.actions)
        self.latent = self.state.z_hist[0, -1]
        self._scene = None
        self.arm_state = self.wm.arms.new_state() if self.wm.arms is not None else None
        return self.seed_action()

    def seed_action(self):
        return np.asarray(self.actions[min(self.state.t, len(self.actions) - 1)], np.float32).copy()

    @property
    def t(self):
        return self.state.t

    def recorded_action(self, t=None):
        """The recorded action at step t (default: the next one), or None past the end of the seed episode."""
        t = self.state.t if t is None else t
        return None if t >= len(self.actions) else np.asarray(self.actions[t], np.float32).copy()

    def step(self, action):
        a = np.asarray(action, np.float32)
        if a.shape != (14,) or not np.isfinite(a).all():
            raise ValueError("an action is 14 finite numbers")
        self.latent = self.wm.step(self.state, a)
        self._scene = None
        return self.latent

    # -- reading the prediction -------------------------------------------------------------------------------------------
    def scene(self):
        """Gaussians of the current prediction without the robot: dict(xyz, scale, opacity, rgb01, quat, cls) or None."""
        if self._scene is None:
            import torch
            with torch.no_grad():
                self._scene = self._decode_scene()
        return self._scene

    def _robot_mask(self, g, coarse, voxel=0.008):
        """Where the model's own arms are: the release's arm rig posed from this latent (joint head + snap onto the predicted
        grippers), as a dilated voxel occupancy. Returns a function xyz -> bool, or None without the rig."""
        if self.wm.arms is None:
            return None
        import torch
        rig = self.wm.arms(self.arm_state, coarse, g)
        lo = self.wm.bmin - 0.1
        dims = (((self.wm.bmax + 0.1) - lo) / voxel).ceil().long() + 1
        occ = torch.zeros(int(dims[0]), int(dims[1]), int(dims[2]), device=self.device)
        ijk = ((rig["xyz"] - lo) / voxel).long()
        ok = ((ijk >= 0) & (ijk < dims)).all(1)
        ijk = ijk[ok]
        occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1.0
        occ = torch.nn.functional.max_pool3d(occ[None, None], 3, 1, 1)[0, 0] > 0

        def inside(xyz):
            q = ((xyz - lo) / voxel).long()
            valid = ((q >= 0) & (q < dims)).all(1)
            q = q.clamp(min=0)
            q = torch.minimum(q, dims - 1)
            return valid & occ[q[:, 0], q[:, 1], q[:, 2]]
        return inside

    def _decode_scene(self):
        """``WorldModel.decode`` without the arm rig, minus everything that belongs to the robot:
          1. the robot class;
          2. table-class Gaussians floating above the table (cables, arm parts; the upstream "table float" filter, which
             upstream applies only together with the rig);
          3. near the model's own arms (see ``_robot_mask``): table-class Gaussians off the table surface, and the red /
             orange gripper pads, which the autoencoder labels as object. Other object Gaussians there are kept: they are
             what the gripper holds."""
        import torch
        from models.world_model import TABLE_Z, _rgb_to_hsv
        wm = self.wm
        g, coarse = wm.decode_raw(self.latent)
        if g is None:
            return None
        robot = self._robot_mask(g, coarse)
        x, cls, rgb = g["xyz"], g["cls"], g["rgb01_lock"]
        keep = (g["scale"].max(1).values <= 0.02) & (g["opacity"] > 0.003) & (cls != ROBOT_CLASS)
        keep &= ~((cls == 2) & (x[:, 2] > TABLE_Z + 0.03))
        if wm.class_filters:
            keep &= ~wm._isolated(x, cls == 0)
        if robot is not None:
            near = robot(x)
            h, s, v = _rgb_to_hsv(rgb.clamp(0, 1))
            h = torch.where(h > 180, h - 360, h)
            pad = (h > -12) & (h < 45) & (s > 0.4) & (v > 0.15)
            keep &= ~(near & (cls == 2) & (x[:, 2] > TABLE_Z + 0.005))
            keep &= ~(near & pad)  # gripper pads, whichever class they were given
        if wm.table_feather > 0 or wm.table_edge_faces:
            dface = torch.stack([x[:, 0] - wm.bmin[0], wm.bmax[0] - x[:, 0], x[:, 1] - wm.bmin[1], wm.bmax[1] - x[:, 1]], 1)
            if wm.table_edge_faces:
                band = (dface < torch.tensor(wm.table_edge_faces, device=x.device, dtype=x.dtype)).any(1) & (x[:, 2] < TABLE_Z + 0.08)
                keep &= ~band
            if wm.table_feather > 0:
                w = (dface.min(1).values / wm.table_feather).clamp(0, 1)
                w = w * w * (3 - 2 * w)
                low = (cls == 2) & (x[:, 2] < TABLE_Z + 0.03)
                g["opacity"] = g["opacity"] * torch.where(low, w, torch.ones_like(w))
        g.pop("rgb01_lock")
        idx = keep.nonzero().squeeze(1)
        if idx.numel() == 0:
            return None
        return {k: v.index_select(0, idx) for k, v in g.items()}

    def points(self, max_points=100000):
        """Scene particle centres (N, 3) float32 and colours (N, 3) in [0, 1]; evenly thinned to at most max_points (0: all)."""
        g = self.scene()
        if g is None:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
        n = g["xyz"].shape[0]
        stride = max(1, -(-n // max_points)) if max_points else 1
        xyz = g["xyz"][::stride].detach().float().cpu().numpy()
        rgb = g["rgb01"][::stride].detach().float().clamp(0, 1).cpu().numpy()
        return xyz, rgb

    def render(self, camera=None):
        """The scene over the scanned room: uint8 (H, W, 3)."""
        return self.wm.render(self.scene(), *(camera or self.camera))

    def camera_model(self):
        """(world -> optical 4x4, K 3x3, W, H) of the task's default view."""
        vm, K, W, H = self.camera
        return vm.detach().cpu().numpy().astype(np.float64), K.detach().cpu().numpy().astype(np.float64), int(W), int(H)

    def model_joints(self):
        """Where the world model believes the arms are: (14,) = per arm 6 joints (rad) + finger opening (m), from the
        release's arm head on the current latent; None without it."""
        if self.wm.arms is None:
            return None
        self.scene()
        if self.arm_state is not None and self.arm_state.pose is not None:
            return np.array(self.arm_state.pose, np.float64)
        import torch
        with torch.no_grad():
            cells = self.wm.decode_cells(self.latent)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = self.wm.arms.joint_net(cells["coarse"].float())[0].float()
            return (o * self.wm.arms.y_std + self.wm.arms.y_mean).cpu().numpy().astype(np.float64)
