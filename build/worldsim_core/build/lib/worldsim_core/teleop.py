"""Keyboard / target teleoperation in the model's action space (the upstream demo's control rule, without the model).

Every tick a held key moves its arm's target by one recorded keyboard step along one axis of that arm's action frame;
orientation follows the task's rule; the target is clamped to the recorded workspace (optional); a gripper key toggles
the gripper, which then ramps at the recorded rate. The result is a per-arm target pose for MoveIt and a gripper command.
"""
import numpy as np

from .tasks import TaskConfig


class CartesianTeleop:

    def __init__(self, task: TaskConfig, clamp_to_workspace=True):
        self.task = task
        self.clamp = clamp_to_workspace
        self.pos = [np.zeros(3), np.zeros(3)]
        self.rpy = [np.zeros(3), np.zeros(3)]
        self.seed_rpy = [np.zeros(3), np.zeros(3)]
        self.grip = [0.0, 0.0]
        self.grip_target = [None, None]
        self.goal = [None, None]  # absolute target (pos, rpy) approached one keyboard step per tick
        self.rot_step = 0.05  # rad per tick towards a goal orientation
        self.ready = False

    def reseed(self, action):
        """Start from a 14-d action (the seed frame of the model)."""
        a = np.asarray(action, np.float64)
        for i in range(2):
            self.pos[i] = a[i * 7: i * 7 + 3].copy()
            self.rpy[i] = a[i * 7 + 3: i * 7 + 6].copy()
            self.seed_rpy[i] = self.rpy[i].copy()
            self.grip[i] = float(a[i * 7 + 6])
        self.goal = [None, None]
        g = self.task.gripper
        self.grip_target = [None, None]
        if g:
            for i in range(2):
                self.grip_target[i] = "open" if abs(self.grip[i] - g["open"]) <= abs(self.grip[i] - g["close"][i]) else "close"
            for i, state in g.get("fixed", {}).items():
                self.grip_target[int(i)] = state
        self.ready = True

    def sync_pose(self, arm, pos, rpy):
        """Adopt the arm's measured pose as the target (after it was moved by something else, e.g. the MoveIt panel)."""
        self.goal[arm] = None
        self.pos[arm] = np.asarray(pos, np.float64).copy()
        self.rpy[arm] = np.asarray(rpy, np.float64).copy()

    def set_target(self, arm, pos, rpy=None):
        """An absolute target (external command or a pose topic). Like the upstream gizmo it is approached at the
        recorded keyboard speed (one step per axis and tick), so the model sees motions of the size it was trained on.
        Orientation follows the task rule when rpy is None."""
        if arm in self.task.locked:
            return
        pos = np.asarray(pos, np.float64)
        pos = self.task.clamp(arm, pos) if self.clamp else pos.copy()
        rpy = np.asarray(rpy, np.float64) if rpy is not None else self.task.orientation_for(pos, self.seed_rpy[arm])
        self.goal[arm] = (pos, rpy)

    def toggle_gripper(self, key):
        arm = self.task.gripper_keys().get(key)
        if arm is None or self.grip_target[arm] is None:
            return False
        self.grip_target[arm] = "close" if self.grip_target[arm] == "open" else "open"
        return True

    def tick(self, held_keys):
        """Apply one tick of held keys. Returns (moved arms, gripper ramping)."""
        moved = set()
        for key in held_keys:
            if key not in self.task.keys:
                continue
            arm, axis, sign = self.task.keys[key]
            if arm in self.task.locked:
                continue
            self.pos[arm][axis] += sign * self.task.step
            self.goal[arm] = None  # a key on that arm cancels its goal
            moved.add(arm)
        goal_driven = set()
        for arm in range(2):
            if self.goal[arm] is None or arm in moved:
                continue
            gpos, grpy = self.goal[arm]
            dp = np.clip(gpos - self.pos[arm], -self.task.step, self.task.step)
            dr = (grpy - self.rpy[arm] + np.pi) % (2 * np.pi) - np.pi
            dr = np.clip(dr, -self.rot_step, self.rot_step)
            self.pos[arm] = self.pos[arm] + dp
            self.rpy[arm] = self.rpy[arm] + dr
            if np.abs(gpos - self.pos[arm]).max() < 1e-9 and np.abs((grpy - self.rpy[arm] + np.pi) % (2 * np.pi) - np.pi).max() < 1e-9:
                self.goal[arm] = None
            goal_driven.add(arm)
        for arm in moved:
            if self.clamp:
                self.pos[arm] = self.task.clamp(arm, self.pos[arm])
            self.rpy[arm] = self.task.orientation_for(self.pos[arm], self.rpy[arm] if self.task.aim_pitch is None else self.seed_rpy[arm])
        ramping = False
        g = self.task.gripper
        if g:
            for i in range(2):
                if self.grip_target[i] is None:
                    continue
                want = self.task.gripper_target_value(i, self.grip_target[i])
                v = self.grip[i]
                if abs(v - want) > 1e-9:
                    self.grip[i] = min(want, v + g["rate"]) if v < want else max(want, v - g["rate"])
                    ramping = True
        return moved | goal_driven, ramping

    def target(self, arm):
        return self.pos[arm].copy(), self.rpy[arm].copy()
