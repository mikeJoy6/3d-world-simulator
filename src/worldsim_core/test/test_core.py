"""Offline tests of the core (no ROS, no GPU): python3 -m pytest src/worldsim_core/test -q"""
import numpy as np
import pytest

from worldsim_core.mapping import PoseMapping, StepGate, pose6_from_transform
from worldsim_core.robot import ArmKinematics, RobotCalibration, build_srdf, build_urdf, cartesian_joint_path
from worldsim_core.tasks import SETTLE_TICKS, TaskConfig, assets_root
from worldsim_core.teleop import CartesianTeleop
from worldsim_core.transforms import make_transform, matrix_to_quat, matrix_to_rpy, nearest_rpy, quat_to_matrix, rpy_to_matrix

HAVE_ASSETS = (assets_root() / "pusht/task.json").is_file()


def test_rpy_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(500):
        rpy = rng.uniform(-3, 3, 3)
        rpy[1] = rng.uniform(-1.5, 1.5)
        R = rpy_to_matrix(rpy)
        assert np.allclose(matrix_to_rpy(R), rpy, atol=1e-9)
        assert np.allclose(quat_to_matrix(matrix_to_quat(R)), R, atol=1e-9)
    assert np.allclose(nearest_rpy([0, 0, -3.1], [0, 0, 3.1]), [0, 0, 2 * np.pi - 3.1])


@pytest.fixture(scope="module")
def calib():
    return RobotCalibration()


def test_ik_reaches_fk(calib):
    rng = np.random.default_rng(1)
    for arm in (0, 1):
        k = ArmKinematics(calib, arm)
        rest = np.array(calib.arms[arm]["rest"])
        for _ in range(20):
            q = rest + rng.uniform(-0.3, 0.3, 6)
            pose = k.action_pose(q)
            q2, pe, re = k.ik(pose, rest)
            assert pe < 1e-3 and re < 5e-3
            assert np.allclose(k.action_pose(q2)[:3], pose[:3], atol=1e-3)


def test_cartesian_path_through_wrist_singularity(calib):
    """Lowering arm 0 from the cloth seed pose crosses wrist_angle = 0. The path must stay on its branch
    (forearm_roll and wrist_rotate do not spin by pi) and keep the tcp on the line."""
    k = ArmKinematics(calib, 0)
    q0 = np.array([0.119, -0.362, 0.841, -0.09, 0.155, 0.066])
    T0 = k.tcp_in_action(q0)
    target = T0.copy()
    target[2, 3] -= 0.10
    qs, frac = cartesian_joint_path(k, q0, target)
    assert frac == 1.0
    qs = np.array(qs)
    assert np.abs(qs[:, 3]).max() < 0.3 and np.abs(qs[:, 5]).max() < 0.3  # no 180 degree wrist flip
    assert qs[0, 4] > 0 > qs[-1, 4]  # the wrist crossed zero instead
    for q in qs:
        p = k.tcp_in_action(q)[:3, 3]
        assert np.linalg.norm(p[:2] - T0[:2, 3]) < 1.5e-3  # on the vertical line
    assert np.allclose(k.tcp_in_action(qs[-1])[:3, 3], target[:3, 3], atol=1e-3)


def test_urdf_and_srdf(calib):
    urdf, srdf = build_urdf(calib), build_srdf(calib)
    for i in (0, 1):
        assert f'<link name="arm{i}_action_tcp"/>' in urdf and f'<link name="arm{i}_action_frame"/>' in urdf
        assert f'<group name="arm{i}"><chain base_link="arm{i}_base_link" tip_link="arm{i}_action_tcp"/>' in srdf


def test_relative_mapping_roundtrip():
    rng = np.random.default_rng(2)
    axes = rpy_to_matrix([0, 0, np.pi / 2])
    m = PoseMapping("relative", axes, 0.5)
    P0 = make_transform([0.4, -0.2, 0.9], [0.1, 0.2, 0.3])
    seed = np.array([0.3, 0.0, 0.1, 0.0, 0.26, 0.0])
    m.engage(P0, seed)
    assert np.allclose(pose6_from_transform(m.to_action(P0)), seed, atol=1e-9)
    for _ in range(20):
        T = make_transform(rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3))
        assert np.allclose(m.to_robot(m.to_action(T)), T, atol=1e-9)
    moved = P0.copy()
    moved[:3, 3] += [0.02, 0, 0]  # robot +x 2 cm -> action +y 1 cm (axes: z rotation, scale 0.5)
    assert np.allclose(m.to_action(moved)[:3, 3] - m.to_action(P0)[:3, 3], [0, 0.01, 0], atol=1e-12)


def test_step_gate():
    g = StepGate(run_on=2)
    a = np.zeros(14)
    assert not g.decide(a)  # first sample is the reference
    assert not g.decide(a)
    b = a.copy()
    b[0] = 0.01
    assert g.decide(b)
    assert g.decide(b) and g.decide(b) and not g.decide(b)  # run-on, then still
    c = b.copy()
    c[6] = 1.0  # gripper change: settle
    assert g.decide(c)
    assert sum(g.decide(c) for _ in range(SETTLE_TICKS + 5)) == SETTLE_TICKS + 2  # settle, then the run-on


@pytest.mark.skipif(not HAVE_ASSETS, reason="task assets not installed")
def test_teleop_keys_and_goal():
    t = TaskConfig("pusht")
    tel = CartesianTeleop(t)
    seed = np.array([0.3, 0, 0.07, 0, 0.26, 0, 0.15] * 2)
    tel.reseed(seed)
    key = next(k for k, (arm, axis, sign) in t.keys.items() if arm == 0 and axis == 0 and sign > 0)
    moved, _ = tel.tick({key})
    assert moved == {0} and np.isclose(tel.pos[0][0], 0.3 + t.step)
    tel.set_target(1, [0.3 + 10 * t.step, 0.0, 0.07])
    for _ in range(9):
        tel.tick(set())
    assert tel.goal[1] is not None
    tel.tick(set())
    assert tel.goal[1] is None and np.isclose(tel.pos[1][0], 0.3 + 10 * t.step)
    tel.set_target(0, [5.0, 0, 0.07])  # clamped to the workspace
    assert tel.goal[0][0][0] == t.ws_max[0][0]


@pytest.mark.skipif(not HAVE_ASSETS, reason="task assets not installed")
def test_cloth_gripper_toggle():
    t = TaskConfig("cloth")
    tel = CartesianTeleop(t)
    tel.reseed(np.array([0.3, 0, 0.2, 0, 0.43, 0, t.gripper["open"], 0.3, 0, 0.2, 0, 0.43, 0, t.gripper["close"][1]]))
    assert tel.toggle_gripper("h")
    _, ramping = tel.tick(set())
    assert ramping and tel.grip[0] < t.gripper["open"]
