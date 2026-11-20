"""Small rigid-transform helpers (numpy only, so every node can use them without scipy or torch).

Euler angles follow the action convention of the training data and of ROS: roll, pitch, yaw about the fixed x, y, z
axes, R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Quaternions are (x, y, z, w) as in geometry_msgs.
"""
import numpy as np


def rpy_to_matrix(rpy):
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_to_rpy(R):
    R = np.asarray(R, np.float64)
    sp = -R[2, 0]
    if abs(sp) < 1 - 1e-9:
        return np.array([np.arctan2(R[2, 1], R[2, 2]), np.arcsin(sp), np.arctan2(R[1, 0], R[0, 0])])
    # gimbal lock: roll and yaw share one axis, put it all in yaw
    return np.array([0.0, np.copysign(np.pi / 2, sp), np.arctan2(-R[0, 1], R[1, 1])])


def matrix_to_quat(R):
    R = np.asarray(R, np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_to_matrix(q):
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError("zero quaternion")
    s = 2.0 / n
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def make_transform(xyz=(0, 0, 0), rpy=None, R=None):
    T = np.eye(4)
    T[:3, :3] = R if R is not None else rpy_to_matrix(rpy if rpy is not None else (0, 0, 0))
    T[:3, 3] = xyz
    return T


def pose_to_transform(pose):
    """``{xyz, rpy}`` dict -> 4x4."""
    return make_transform(pose["xyz"], pose["rpy"])


def axis_rotation(axis, q):
    x, y, z = axis
    c, s = np.cos(q), np.sin(q)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s], [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def rotation_angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def rotation_log(R):
    """Rotation vector of R (axis * angle)."""
    th = rotation_angle(R)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - th < 1e-6:  # 180 degrees: take the axis from the symmetric part
        M = (R + np.eye(3)) / 2
        axis = np.sqrt(np.clip(np.diag(M), 0, None))
        k = int(np.argmax(axis))
        axis = M[:, k] / max(np.sqrt(M[k, k]), 1e-12)
        return axis / np.linalg.norm(axis) * th
    return v / (2 * np.sin(th)) * th


def nearest_rpy(rpy, reference):
    """The Euler triple of the same orientation closest to ``reference`` (stays on the reference's branch across +-pi)."""
    r, p, y = rpy
    options = np.array([[r, p, y], [r + np.pi, np.pi - p, y + np.pi]])
    deltas = (options - np.asarray(reference) + np.pi) % (2 * np.pi) - np.pi
    return np.asarray(reference) + deltas[np.argmin(np.linalg.norm(deltas, axis=1))]
