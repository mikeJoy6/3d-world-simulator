"""From a robot's end-effector poses to the model's 14-d action, and when the model should step.

The same code serves the simulated arms and a real robot; only the configuration differs:

  absolute   action pose = offset @ measured pose. The simulated ViperX arms use this with the identity, because their URDF
             carries the fitted action frame and tcp: the TF pose of ``arm<i>_action_tcp`` in ``arm<i>_action_frame`` IS the
             action.
  relative   for another robot (G2): when following starts, the measured pose P0 is tied to the seed action A0; after that
             the action is A0 moved by the robot's motion since P0, with the robot's axes turned into the action frame by
             ``axes`` (3x3) and translations scaled by ``scale``. The model then sees the motion it knows, from the
             configuration it was seeded in, whatever the other robot's absolute geometry.
"""
import numpy as np

from .tasks import SETTLE_TICKS
from .transforms import make_transform, matrix_to_rpy, nearest_rpy, rotation_angle, rpy_to_matrix


class PoseMapping:
    def __init__(self, mode="absolute", axes=None, scale=1.0, offset=None):
        if mode not in ("absolute", "relative"):
            raise ValueError("mapping mode must be absolute or relative")
        self.mode = mode
        self.axes = np.eye(3) if axes is None else np.asarray(axes, np.float64).reshape(3, 3)
        self.scale = float(scale)
        self.offset = np.eye(4) if offset is None else np.asarray(offset, np.float64)
        self.anchor = None  # (P0 4x4 robot, A0 4x4 action)

    def reset(self):
        self.anchor = None

    def engage(self, measured, seed_pose6):
        self.anchor = (np.asarray(measured, np.float64), make_transform(seed_pose6[:3], seed_pose6[3:6]))

    @property
    def engaged(self):
        return self.mode == "absolute" or self.anchor is not None

    def to_action(self, measured):
        """4x4 robot pose -> 4x4 pose in the arm's action frame."""
        T = np.asarray(measured, np.float64)
        if self.mode == "absolute":
            return self.offset @ T
        P0, A0 = self.anchor
        out = np.eye(4)
        out[:3, 3] = A0[:3, 3] + self.scale * (self.axes @ (T[:3, 3] - P0[:3, 3]))
        dR = T[:3, :3] @ P0[:3, :3].T  # rotation since engagement, robot frame
        out[:3, :3] = (self.axes @ dR @ self.axes.T) @ A0[:3, :3]
        return out

    def to_robot(self, action_T):
        """Inverse of ``to_action``: 4x4 action-frame pose -> 4x4 robot pose (for commanding the robot from action space)."""
        A = np.asarray(action_T, np.float64)
        if self.mode == "absolute":
            return np.linalg.inv(self.offset) @ A
        P0, A0 = self.anchor
        out = np.eye(4)
        out[:3, 3] = P0[:3, 3] + self.axes.T @ (A[:3, 3] - A0[:3, 3]) / self.scale
        out[:3, :3] = self.axes.T @ (A[:3, :3] @ A0[:3, :3].T) @ self.axes @ P0[:3, :3]
        return out


def pose6_from_transform(T, reference_rpy=None):
    rpy = matrix_to_rpy(T[:3, :3])
    if reference_rpy is not None:
        rpy = nearest_rpy(rpy, reference_rpy)
    return np.concatenate([T[:3, 3], rpy])


def pose_distance(a6, b6):
    """(position distance m, rotation angle rad) between two 6-d poses."""
    Ra, Rb = rpy_to_matrix(a6[3:6]), rpy_to_matrix(b6[3:6])
    return float(np.linalg.norm(np.asarray(a6[:3]) - np.asarray(b6[:3]))), rotation_angle(Ra @ Rb.T)


class StepGate:
    """The upstream demo steps the model only while something is commanded; a still scene holds. Here the command is the
    measured action: step when it moved since the last step, for ``run_on`` steps after it stopped (the real arm lags its
    command), and for SETTLE_TICKS steps after a gripper change."""

    def __init__(self, run_on=2, pos_eps=5e-4, rot_eps=4e-3, grip_eps=1e-3):
        self.run_on_ticks, self.pos_eps, self.rot_eps, self.grip_eps = int(run_on), pos_eps, rot_eps, grip_eps
        self.reset()

    def reset(self):
        self.last = None
        self.run_on = 0
        self.settle = None

    def decide(self, action):
        a = np.asarray(action, np.float64)
        if self.last is None:
            self.last = a.copy()
            return False
        moved = grip = False
        for i in range(2):
            d, r = pose_distance(a[i * 7: i * 7 + 6], self.last[i * 7: i * 7 + 6])
            moved |= d > self.pos_eps or r > self.rot_eps
            grip |= abs(a[i * 7 + 6] - self.last[i * 7 + 6]) > self.grip_eps
        if grip:
            self.settle = 0
        if moved or grip:
            self.last = a.copy()
            self.run_on = self.run_on_ticks
            return True
        if self.settle is not None:
            self.settle += 1
            if self.settle >= SETTLE_TICKS:
                self.settle = None
            return True
        if self.run_on > 0:
            self.run_on -= 1
            return True
        return False
