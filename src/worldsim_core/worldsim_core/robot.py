"""The simulated robot: two ViperX 300 S arms, extracted from the world model's release by tools/extract_robot.py.

``RobotCalibration`` reads ``worldsim_description/config/calibration.yaml``. From it this module builds

  * the URDF (``build_urdf``): root ``worldsim_world``; per arm ``arm<i>_`` links, the meshes extracted from the rig splats,
    and two extra fixed links per arm, ``arm<i>_action_frame`` and ``arm<i>_action_tcp``. The model's 6-d arm action is
    exactly the pose of ``action_tcp`` in ``action_frame`` (xyz, then roll pitch yaw about fixed axes).
  * the SRDF (``build_srdf``): MoveIt planning groups ``arm0``/``arm1`` (chain base_link -> action_tcp) and
    ``arm0_gripper``/``arm1_gripper``, end effectors on the tcp, and the disabled collision pairs inside each arm.
  * ``ArmKinematics``: forward kinematics of that chain and a damped least-squares IK, used to put the simulated arms
    at a commanded action (reset to the seed frame, replay) without a round trip through MoveIt.
"""
from pathlib import Path

import numpy as np
import yaml

from .transforms import axis_rotation, make_transform, matrix_to_rpy, pose_to_transform, rotation_log, rpy_to_matrix

ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
FINGER_JOINT = "left_finger"
MIMIC_FINGER_JOINT = "right_finger"


def default_calibration_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("worldsim_description")) / "config/calibration.yaml"
    except Exception:  # running from the source tree
        return Path(__file__).resolve().parents[2] / "worldsim_description/config/calibration.yaml"


class RobotCalibration:
    def __init__(self, path=None):
        self.path = Path(path) if path else default_calibration_path()
        with open(self.path) as f:
            self.data = yaml.safe_load(f)
        self.chain = self.data["chain"]
        self.arms = self.data["arms"]
        self.prefixes = self.data["arm_prefixes"]
        self.root = self.data["root_link"]
        self.gripper_link = self.data["gripper_link"]
        self.tasks = self.data["tasks"]
        self.gripper_display = self.data["gripper_display"]

    # -- names ------------------------------------------------------------------------------------------------------------
    def joint_names(self, arm, with_fingers=True):
        p = self.prefixes[arm]
        names = [p + j for j in ARM_JOINTS]
        return names + [p + FINGER_JOINT, p + MIMIC_FINGER_JOINT] if with_fingers else names

    def action_frame(self, arm):
        return self.prefixes[arm] + "action_frame"

    def action_tcp(self, arm):
        return self.prefixes[arm] + "action_tcp"

    def group(self, arm):
        return self.prefixes[arm].rstrip("_")

    # -- arm mounting -----------------------------------------------------------------------------------------------------
    def mounts(self, task=None):
        """World pose of each arm's base_link. One task: that task's calibration. None: the per-arm median over all tasks,
        which is what the URDF uses so that a runtime task switch does not change the robot model (the tasks differ by at
        most 3 mm, except the locked right arm of part assembly by about 2 cm)."""
        if task is not None:
            return [pose_to_transform(b) for b in self.tasks[task]["bases"]]
        out = []
        for arm in range(2):
            xyz = np.median([t["bases"][arm]["xyz"] for t in self.tasks.values()], axis=0)
            rpy = np.median([t["bases"][arm]["rpy"] for t in self.tasks.values()], axis=0)
            out.append(make_transform(xyz, rpy))
        return out

    # -- gripper (display only) ---------------------------------------------------------------------------------------------
    def finger_command(self, opening):
        """Inverse of ``finger_opening``: the gripper command a finger opening (m) shows."""
        g = self.gripper_display
        return (float(opening) - g["offset"]) / g["scale"]

    def finger_opening(self, command):
        g = self.gripper_display
        return float(np.clip(g["offset"] + g["scale"] * float(command), g["lower"], g["upper"]))


class ArmKinematics:
    """FK / IK of one arm from base_link to the action tcp, and the tcp pose in the action frame."""

    def __init__(self, calib, arm):
        self.calib, self.arm = calib, arm
        self.steps = []  # (origin 4x4, type, axis, index of the driven arm joint or -1), base_link -> gripper_link
        chain = {j["child"]: j for j in calib.chain}
        link, path = calib.gripper_link, []
        while link in chain:
            path.append(chain[link])
            link = chain[link]["parent"]
        for j in reversed(path):
            idx = ARM_JOINTS.index(j["name"]) if j["name"] in ARM_JOINTS else -1
            self.steps.append((pose_to_transform(j["origin"]), j["type"], np.asarray(j["axis"], np.float64), idx))
        lim = {j["name"]: (j.get("lower", -np.pi), j.get("upper", np.pi)) for j in calib.chain}
        self.lower = np.array([lim[j][0] for j in ARM_JOINTS])
        self.upper = np.array([lim[j][1] for j in ARM_JOINTS])
        a = calib.arms[arm]
        self.base_to_action = pose_to_transform(a["action_frame"])  # action frame in base_link
        self.action_from_base = np.linalg.inv(self.base_to_action)
        self.tool = pose_to_transform(a["action_tcp"])  # tcp in gripper_link

    def fk_gripper(self, q):
        T = np.eye(4)
        for O, typ, ax, idx in self.steps:
            M = O.copy()
            if typ == "revolute":
                M[:3, :3] = M[:3, :3] @ axis_rotation(ax, q[idx])
            T = T @ M
        return T

    def tcp_in_action(self, q):
        """4x4 pose of the action tcp in the action frame at arm joints q (6,)."""
        return self.action_from_base @ self.fk_gripper(q) @ self.tool

    def action_pose(self, q):
        """The arm's 6-d action (x y z roll pitch yaw) at joints q."""
        T = self.tcp_in_action(q)
        return np.concatenate([T[:3, 3], matrix_to_rpy(T[:3, :3])])

    def jacobian(self, q, eps=1e-6):
        """Numerical 6x6 Jacobian of (position, rotation vector) of the tcp in the action frame."""
        T0 = self.tcp_in_action(q)
        J = np.zeros((6, 6))
        for i in range(6):
            dq = np.array(q, np.float64)
            dq[i] += eps
            T = self.tcp_in_action(dq)
            J[:3, i] = (T[:3, 3] - T0[:3, 3]) / eps
            J[3:, i] = rotation_log(T[:3, :3] @ T0[:3, :3].T) / eps
        return J

    def ik(self, pose6, q0, iters=200, tol_pos=1e-4, tol_rot=1e-3, damping=0.02, rot_weight=0.3, singular_eps=0.05,
           singular_damping=0.12):
        """Joints reaching the 6-d action pose, started from q0. Returns (q, position error m, rotation error rad).
        Damped least squares with adaptive damping: near a singularity (smallest singular value of the weighted Jacobian
        below ``singular_eps``) the damping grows towards ``singular_damping``, which stops the joints from spinning
        in the near-null space (at the ViperX wrist singularity: forearm_roll against wrist_rotate)."""
        target = make_transform(pose6[:3], pose6[3:6])
        q = np.clip(np.array(q0, np.float64), self.lower, self.upper)
        W = np.diag([1, 1, 1, rot_weight, rot_weight, rot_weight])
        for _ in range(iters):
            T = self.tcp_in_action(q)
            e = np.concatenate([target[:3, 3] - T[:3, 3], rotation_log(target[:3, :3] @ T[:3, :3].T)])
            if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot:
                break
            J = W @ self.jacobian(q)
            ew = W @ e
            smin = np.linalg.svd(J, compute_uv=False)[-1]
            lam2 = damping ** 2 + (singular_damping ** 2 * (1 - (smin / singular_eps) ** 2) if smin < singular_eps else 0.0)
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), ew)
            n = np.abs(dq).max()
            if n > 0.2:
                dq *= 0.2 / n
            q = np.clip(q + dq, self.lower, self.upper)
        T = self.tcp_in_action(q)
        return q, float(np.linalg.norm(target[:3, 3] - T[:3, 3])), float(np.linalg.norm(rotation_log(target[:3, :3] @ T[:3, :3].T)))


def cartesian_joint_path(kin, q0, target, max_step=0.003, max_rot=0.03, max_joint_step=0.25, tol_pos=1e-3, tol_rot=0.02,
                         rot_weight=0.1):
    """Joint waypoints that move the tcp on a straight line (and a constant-axis rotation) from where joints q0 put it
    to ``target`` (4x4 in the action frame), at most ``max_step`` m / ``max_rot`` rad apart. Each waypoint is solved by
    the damped least-squares IK seeded with the previous one, so the arm stays on its branch: through the ViperX wrist
    singularity (wrist_angle = 0) the wrist simply crosses zero, where KDL, LMA and pick_ik let forearm_roll and
    wrist_rotate counter-rotate by pi or stall. Position is weighted above orientation (``rot_weight``) and orientation
    may deviate by ``tol_rot`` (1.1 deg; the fitted action frame itself is accurate to about 3 deg): through the
    singularity, holding the exact orientation would need a fast roll swing that the recorded arms never made.
    Stops at the first waypoint that is out of reach or needs a joint jump.
    Returns (list of joint vectors, fraction of the line covered)."""
    T0 = kin.tcp_in_action(q0)
    dp = target[:3, 3] - T0[:3, 3]
    w = rotation_log(target[:3, :3] @ T0[:3, :3].T)
    ang = float(np.linalg.norm(w))
    n = max(1, int(np.ceil(max(np.linalg.norm(dp) / max_step, ang / max_rot))))
    q, out = np.array(q0, np.float64), []
    for k in range(1, n + 1):
        s = k / n
        R = (axis_rotation(w / ang, s * ang) if ang > 1e-9 else np.eye(3)) @ T0[:3, :3]
        pose6 = np.concatenate([T0[:3, 3] + s * dp, matrix_to_rpy(R)])
        qn, pe, re = kin.ik(pose6, q, iters=60, rot_weight=rot_weight)
        if pe > tol_pos or re > tol_rot or np.abs(qn - q).max() > max_joint_step:
            break
        out.append(qn)
        q = qn
    return out, len(out) / n


# -- URDF / SRDF ---------------------------------------------------------------------------------------------------------------
def _origin(T):
    xyz = " ".join(f"{v:.6f}" for v in T[:3, 3])
    rpy = " ".join(f"{v:.6f}" for v in matrix_to_rpy(T[:3, :3]))
    return f'<origin xyz="{xyz}" rpy="{rpy}"/>'


def build_urdf(calib, mesh_uri="package://worldsim_description/meshes", task=None, name="worldsim_aloha"):
    mesh_dir = Path(calib.path).parent.parent / "meshes"
    colors = calib.data.get("colors") or {}
    out = [f'<?xml version="1.0"?>', f'<!-- Generated by worldsim_core.robot.build_urdf from {Path(calib.path).name} -->',
           f'<robot name="{name}">', f'  <link name="{calib.root}"/>']
    links = sorted({j["parent"] for j in calib.chain} | {j["child"] for j in calib.chain})
    for arm, (p, mount) in enumerate(zip(calib.prefixes, calib.mounts(task))):
        for link in links:
            out.append(f'  <link name="{p}{link}">')
            key = f"arm{arm}_{link}"
            if (mesh_dir / f"{key}.stl").is_file():
                # neutral grey from the splats' brightness (their hue is the room light's, the real arms are black)
                v = min(1.0, 0.12 + 1.2 * float(np.mean(colors.get(key, [0.1, 0.1, 0.1]))))
                rgb = [v, v, v * 1.04]
                out.append(f'    <visual><geometry><mesh filename="{mesh_uri}/{key}.stl"/></geometry>'
                           f'<material name="{key}"><color rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1"/></material></visual>')
                out.append(f'    <collision><geometry><mesh filename="{mesh_uri}/{key}_collision.stl"/></geometry></collision>')
            out.append('    <inertial><mass value="0.1"/><inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/></inertial>')
            out.append("  </link>")
        out.append(f'  <joint name="{p}mount" type="fixed"><parent link="{calib.root}"/><child link="{p}base_link"/>{_origin(mount)}</joint>')
        for j in calib.chain:
            O = pose_to_transform(j["origin"])
            body = f'<parent link="{p}{j["parent"]}"/><child link="{p}{j["child"]}"/>{_origin(O)}'
            if j["type"] != "fixed":
                ax = " ".join(f"{v:g}" for v in j["axis"])
                lo, hi = j["lower"], j["upper"]
                if "mimic" in j:  # a mimic joint's range is its driver's range times the multiplier
                    drv = next(d for d in calib.chain if d["name"] == j["mimic"]["joint"])
                    lo, hi = sorted(j["mimic"]["multiplier"] * v + j["mimic"]["offset"] for v in (drv["lower"], drv["upper"]))
                body += f'<axis xyz="{ax}"/><limit lower="{lo:g}" upper="{hi:g}" effort="{j["effort"]}" velocity="{j["velocity"]}"/>'
                if "mimic" in j:
                    m = j["mimic"]
                    body += f'<mimic joint="{p}{m["joint"]}" multiplier="{m["multiplier"]}" offset="{m["offset"]}"/>'
            out.append(f'  <joint name="{p}{j["name"]}" type="{j["type"]}">{body}</joint>')
        a = calib.arms[arm]
        out.append(f'  <link name="{p}action_frame"/>')
        out.append(f'  <joint name="{p}action_frame_joint" type="fixed"><parent link="{p}base_link"/><child link="{p}action_frame"/>'
                   f'{_origin(pose_to_transform(a["action_frame"]))}</joint>')
        out.append(f'  <link name="{p}action_tcp"/>')
        out.append(f'  <joint name="{p}action_tcp_joint" type="fixed"><parent link="{p}{calib.gripper_link}"/><child link="{p}action_tcp"/>'
                   f'{_origin(pose_to_transform(a["action_tcp"]))}</joint>')
    out.append("</robot>")
    return "\n".join(out) + "\n"


def build_srdf(calib, name="worldsim_aloha"):
    out = ['<?xml version="1.0"?>', f'<robot name="{name}">']
    parent = {j["child"]: j["parent"] for j in calib.chain}
    links = sorted(set(parent) | set(parent.values()))

    def depth_between(a, b):  # links apart along the chain, or None if neither is an ancestor of the other
        for x, y in ((a, b), (b, a)):
            d, cur = 0, x
            while cur in parent:
                cur, d = parent[cur], d + 1
                if cur == y:
                    return d
        return None

    for arm, p in enumerate(calib.prefixes):
        g = calib.group(arm)
        out.append(f'  <group name="{g}"><chain base_link="{p}base_link" tip_link="{p}action_tcp"/></group>')
        out.append(f'  <group name="{g}_gripper"><joint name="{p}{FINGER_JOINT}"/><joint name="{p}{MIMIC_FINGER_JOINT}"/></group>')
        out.append(f'  <end_effector name="{g}_hand" parent_link="{p}action_tcp" group="{g}_gripper" parent_group="{g}"/>')
        rest = calib.arms[arm].get("rest", [0.0] * 6)
        out.append(f'  <group_state name="rest" group="{g}">' + "".join(
            f'<joint name="{p}{j}" value="{v:.4f}"/>' for j, v in zip(ARM_JOINTS, rest)) + "</group_state>")
        # the cloth task's gripper commands: open 1.2, closed 0.2
        out.append(f'  <group_state name="open" group="{g}_gripper"><joint name="{p}{FINGER_JOINT}" value="{calib.finger_opening(1.2):.4f}"/></group_state>')
        out.append(f'  <group_state name="closed" group="{g}_gripper"><joint name="{p}{FINGER_JOINT}" value="{calib.finger_opening(0.2):.4f}"/></group_state>')
        mesh_links = [l for l in links if (Path(calib.path).parent.parent / "meshes" / f"arm{arm}_{l}.stl").is_file()]
        for i, a in enumerate(mesh_links):
            for b in mesh_links[i + 1:]:
                d = depth_between(a, b)
                fingers = {a, b} <= {"gripper_link", "left_finger_link", "right_finger_link"}
                if (d is not None and d <= 3) or fingers or {a, b} == {"left_finger_link", "right_finger_link"}:
                    out.append(f'  <disable_collisions link1="{p}{a}" link2="{p}{b}" reason="Adjacent"/>')
        out.append(f'  <disable_collisions link1="{p}base_link" link2="{p}upper_arm_link" reason="Never"/>')
    out.append(f'  <disable_collisions link1="arm0_base_link" link2="arm1_base_link" reason="Never"/>')
    out.append("</robot>")
    return "\n".join(out) + "\n"
