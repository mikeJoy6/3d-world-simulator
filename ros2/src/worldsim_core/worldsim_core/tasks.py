"""Released tasks and the 14-d action layout (no torch: every node can read a task).

Action layout (the training data's ``cartesian_action``): per arm, in that arm's action frame,
    x, y, z (m), roll, pitch, yaw (rad, fixed axes), gripper command (task's own units)
arm 0 occupies [0:7], arm 1 [7:14].
"""
import json
import os
from pathlib import Path

import numpy as np

TASKS = ("pusht", "cloth", "box", "assembly")
ARM_SLICE = (slice(0, 7), slice(7, 14))
SETTLE_TICKS = 20  # model steps kept running after a gripper change, so the object can drop or settle (upstream demo)


def assets_root(path=None):
    """The released task folders: ``path``, $WORLDSIM_ASSETS, or ``assets/checkpoints`` of the workspace this module
    lives in (found by walking up from the source tree or from the colcon install tree)."""
    p = path or os.environ.get("WORLDSIM_ASSETS")
    if p:
        return Path(p)
    here = Path(__file__)
    for start in (here.resolve(), here.absolute()):
        for parent in start.parents:
            if (parent / "assets/checkpoints").is_dir():
                return parent / "assets/checkpoints"
    raise FileNotFoundError("assets/checkpoints not found; set WORLDSIM_ASSETS or run tools/fetch_weights.py")


def split_action(action):
    a = np.asarray(action, np.float64)
    return [a[s] for s in ARM_SLICE]


def join_action(arms):
    return np.concatenate([np.asarray(arms[0], np.float64), np.asarray(arms[1], np.float64)]).astype(np.float32)


def aim_rotation(pos, pitch_deg):
    """Orientation rule of the cloth and box data: yaw towards the commanded xy, fixed pitch, zero roll."""
    return np.array([0.0, np.radians(pitch_deg), np.arctan2(pos[1], pos[0])])


class TaskConfig:
    """task.json of a released task: control keys, workspace, orientation rule, gripper, view."""

    def __init__(self, task, root=None):
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; released tasks: {', '.join(TASKS)}")
        self.name = task
        self.dir = assets_root(root) / task
        with open(self.dir / "task.json") as f:
            self.cfg = json.load(f)
        c = self.cfg["control"]
        self.title = self.cfg.get("title", task)
        self.step = float(c["step"])
        self.keys = {k: tuple(v) for k, v in c["keys"].items()}  # key -> (arm, axis, sign)
        self.orient = c["orient"]  # "seed" or "aim<deg>"
        self.locked = set(int(a) for a in c.get("locked_arms", []))
        self.run_on = int(c.get("run_on", 0))
        self.soft_limit = bool(c.get("soft_limit"))
        self.gripper = c.get("gripper")  # None or {keys {key: arm}, open, close [a0, a1], rate, fixed {arm: open|close}}
        ws = self.cfg["workspace"]
        self.ws_min = np.array(ws["min"], np.float64)
        self.ws_max = np.array(ws["max"], np.float64)
        self.keepouts = ws.get("keepout") or []
        self.view = self.cfg["view"]

    @property
    def aim_pitch(self):
        return float(self.orient[3:]) if self.orient.startswith("aim") else None

    def orientation_for(self, pos, seed_rpy):
        """The orientation the task's data uses at commanded position ``pos``."""
        pitch = self.aim_pitch
        return aim_rotation(pos, pitch) if pitch is not None else np.array(seed_rpy, np.float64)

    def clamp(self, arm, pos):
        return np.clip(pos, self.ws_min[arm], self.ws_max[arm])

    def gripper_keys(self):
        return dict(self.gripper["keys"]) if self.gripper else {}

    def gripper_target_value(self, arm, state):
        g = self.gripper
        return g["open"] if state == "open" else g["close"][arm]
