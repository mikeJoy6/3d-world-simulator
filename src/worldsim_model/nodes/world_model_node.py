"""ROS front end of the world model (worldsim_core.engine), the same node for the simulated arms and for a real robot.

follow (default)  Every tick, the pose of each arm's end effector is read from TF (``armN.frame`` -> ``armN.tip``) and
                  mapped to the model's 14-d action (worldsim_core.mapping). The model steps when that action moved
                  (StepGate): the robot drives the world. With the extracted ViperX URDF the TF pose IS the action
                  (frames arm0_action_frame -> arm0_action_tcp, absolute mapping); a real robot uses its own frames with
                  the relative mapping.
replay            The recorded actions of the seed episode, looped. They are also sent to /worldsim/robot_command so the
                  simulated arms show them.

Publishes the scene only as particles (/worldsim/points: the prediction without the robot) and the scene rendered from
the task's camera (/worldsim/image + camera_info). The robot itself is the URDF, drawn by robot_state_publisher.
"""
import array
import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2, PointField
from std_msgs.msg import Empty, Float32MultiArray, Header, MultiArrayDimension, MultiArrayLayout, String
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from worldsim_core.mapping import PoseMapping, StepGate, pose6_from_transform, pose_distance
from worldsim_core.robot import ARM_JOINTS, FINGER_JOINT, MIMIC_FINGER_JOINT
from worldsim_core.tasks import TASKS, join_action
from worldsim_core.transforms import make_transform, matrix_to_quat, quat_to_matrix

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def cloud_message(points, colors, header):
    packed = np.empty(len(points), dtype=[("xyz", "<f4", 3), ("rgb", "<u4")])
    packed["xyz"] = points
    c = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint32)
    packed["rgb"] = c[:, 0] << 16 | c[:, 1] << 8 | c[:, 2]
    msg = PointCloud2(header=header, height=1, width=len(points), is_dense=True, point_step=16, row_step=16 * len(points),
                      is_bigendian=False)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
    msg.fields.append(PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1))
    msg.data = array.array("B", packed.tobytes())
    return msg


def transform_to_matrix(t):
    q, p = t.transform.rotation, t.transform.translation
    return make_transform([p.x, p.y, p.z], R=quat_to_matrix([q.x, q.y, q.z, q.w]))


class WorldModelNode(Node):
    def __init__(self):
        super().__init__("world_model")
        P = self.declare_parameter
        self.mode = P("mode", "follow").value
        task = P("task", "pusht").value
        rate = float(P("rate", 5.0).value)
        self.max_points = int(P("max_points", 100000).value)
        self.image_on = bool(P("image", True).value)
        self.world = P("world_frame", "worldsim_world").value
        self.camera_frame = P("camera_frame", "worldsim_camera_optical").value
        # topic: /worldsim/gripper_command (real robot); joint: the finger joints in /joint_states (simulated arms, so the
        # MoveIt gripper groups work too); seed: fixed at the seed frame's values
        self.gripper_source = P("gripper_source", "topic").value
        self.gripper_joints = list(P("gripper_joints", ["arm0_left_finger", "arm1_left_finger"]).value)
        if self.gripper_source not in ("topic", "joint", "seed"):
            raise ValueError("gripper_source must be topic, joint or seed")
        self.wait_for_seed = bool(P("wait_for_seed", True).value)
        self.seed_timeout = float(P("seed_timeout", 6.0).value)
        self.seed_tol = (float(P("seed_tolerance_m", 0.01).value), float(P("seed_tolerance_rad", 0.05).value))
        self.publish_model_joints = bool(P("publish_model_joints", False).value)
        self.republish_period = float(P("republish_period", 1.0).value)
        # A robot that moved further in one tick than the data's keyboard step is fed to the model in up to
        # max_substeps interpolated steps (the MoveIt panel and real robots move at their own speed).
        self.substep_scale = float(P("substep_scale", 1.5).value)  # x the task's keyboard step
        self.substep_rad = float(P("substep_rad", 0.06).value)
        self.max_substeps = int(P("max_substeps", 4).value)
        if self.mode not in ("follow", "replay") or task not in TASKS or not 0 < rate <= 60 or self.max_points < 0:
            raise ValueError("mode follow|replay, a released task, rate in (0, 60] and max_points >= 0 expected")
        self.arms = []
        for i in range(2):
            frame = P(f"arm{i}.frame", f"arm{i}_action_frame").value
            tip = P(f"arm{i}.tip", f"arm{i}_action_tcp").value
            mode = P(f"arm{i}.mapping", "absolute").value
            axes = list(P(f"arm{i}.axes", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).value)
            scale = float(P(f"arm{i}.scale", 1.0).value)
            off = make_transform(list(P(f"arm{i}.offset_xyz", [0.0, 0.0, 0.0]).value), list(P(f"arm{i}.offset_rpy", [0.0, 0.0, 0.0]).value))
            self.arms.append(dict(frame=frame, tip=tip, map=PoseMapping(mode, np.reshape(axes, (3, 3)), scale, off)))

        from worldsim_core.engine import WorldSimEngine
        self.engine = WorldSimEngine(image=self.image_on, robot_mask=bool(P("robot_mask", True).value), fast=bool(P("fast", False).value))

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self, spin_thread=True)
        self.static_tf = StaticTransformBroadcaster(self)
        self.pub_points = self.create_publisher(PointCloud2, "/worldsim/points", 1)
        self.pub_image = self.create_publisher(Image, "/worldsim/image", 1)
        self.pub_info = self.create_publisher(CameraInfo, "/worldsim/camera_info", 1)
        self.pub_applied = self.create_publisher(Float32MultiArray, "/worldsim/applied_action", 1)
        self.pub_seed = self.create_publisher(Float32MultiArray, "/worldsim/seed_action", LATCHED)
        self.pub_task = self.create_publisher(String, "/worldsim/active_task", LATCHED)
        self.pub_state = self.create_publisher(String, "/worldsim/state", LATCHED)
        self.pub_robot = self.create_publisher(Float32MultiArray, "/worldsim/robot_command", 1)
        self.pub_model_joints = self.create_publisher(JointState, "/worldsim/model/joint_states", 1) if self.publish_model_joints else None
        self.create_subscription(String, "/worldsim/select_task", self.on_select, 1)
        self.create_subscription(Empty, "/worldsim/reset", lambda _: setattr(self, "reset_requested", True), 1)
        if self.gripper_source == "topic":
            self.create_subscription(Float32MultiArray, "/worldsim/gripper_command", self.on_gripper, 1)
        elif self.gripper_source == "joint":
            from worldsim_core.robot import RobotCalibration
            self.calib = RobotCalibration()
            self.create_subscription(JointState, "/joint_states", self.on_joints, 10)

        self.pending_task, self.reset_requested = task, False
        self.task = None
        self.gate = StepGate()
        self.last_msgs = None
        self.last_publish = 0.0
        self.status = ""
        self.create_timer(1.0 / rate, self.tick)

    # -- inputs ---------------------------------------------------------------------------------------------------------
    def on_select(self, msg):
        if msg.data in TASKS and msg.data != self.task:
            self.pending_task = msg.data

    def on_gripper(self, msg):
        if len(msg.data) == 2 and np.isfinite(msg.data).all():
            self.grip = [self.gripper_limit(i, float(v)) for i, v in enumerate(msg.data)]

    def on_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if self.task is None or not all(j in pos for j in self.gripper_joints):
            return
        self.grip = [self.gripper_limit(i, self.calib.finger_command(pos[j])) for i, j in enumerate(self.gripper_joints)]

    def gripper_limit(self, arm, value):
        """Only a gripper the task's data operates follows its input, kept in the recorded open/close range; the others
        stay at the seed value (in the data they never change)."""
        g = self.engine.cfg.gripper
        fixed = g is None or arm not in g["keys"].values() or str(arm) in g.get("fixed", {})
        if fixed:
            return float(self.seed[arm * 7 + 6])
        return float(np.clip(value, min(g["close"][arm], g["open"]), max(g["close"][arm], g["open"])))

    def set_state(self, text, **extra):
        if text != self.status:
            self.status = text
            self.pub_state.publish(String(data=json.dumps(dict(task=self.task, mode=self.mode, state=text, **extra), ensure_ascii=False)))
            self.get_logger().info(f"[{self.task}] {text}")

    # -- task and reset -------------------------------------------------------------------------------------------------
    def load(self, task):
        self.set_state(f"loading {task}")
        t0 = time.time()
        self.engine.load(task)
        self.task = task
        self.gate = StepGate(run_on=max(2, self.engine.cfg.run_on))
        vm, K, W, H = self.engine.camera_model()
        cam = np.linalg.inv(vm)  # renderer: world -> optical; TF: optical in world
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = self.world, self.camera_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in cam[:3, 3])
        q = matrix_to_quat(cam[:3, :3])
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = (float(v) for v in q)
        self.static_tf.sendTransform(t)
        self.camera_info = CameraInfo(width=W, height=H, distortion_model="plumb_bob", d=[0.0] * 5,
                                      k=K.reshape(-1).tolist(), r=np.eye(3).reshape(-1).tolist(),
                                      p=np.hstack([K, np.zeros((3, 1))]).reshape(-1).tolist())
        self.get_logger().info(f"Loaded {task} in {time.time() - t0:.1f} s ({self.mode} mode).")
        self.pub_task.publish(String(data=task))
        self.reset()

    def reset(self):
        self.seed = self.engine.reset()
        self.grip = [float(self.seed[6]), float(self.seed[13])]
        self.prev_action = self.seed.copy()
        self.applied = self.seed.copy()
        self.gate.reset()
        for arm in self.arms:
            arm["map"].reset()
        self.following = self.mode == "replay"
        # the seed names its task (layout label), so a listener never pairs it with another task's settings
        layout = MultiArrayLayout(dim=[MultiArrayDimension(label=self.task, size=14, stride=14)])
        self.pub_seed.publish(Float32MultiArray(layout=layout, data=self.seed.tolist()))
        if self.mode == "replay":
            self.pub_robot.publish(Float32MultiArray(data=self.seed.tolist()))
        self.publish_scene()
        self.reset_time = time.time()  # after the first frame: its render (or a renderer compile) is not the robot's time
        self.set_state("following the robot" if self.following else "waiting for the robot to reach the seed pose")

    # -- reading the robot ----------------------------------------------------------------------------------------------
    def measured_poses(self):
        out = []
        for arm in self.arms:
            try:
                out.append(transform_to_matrix(self.tf.lookup_transform(arm["frame"], arm["tip"], Time())))
            except TransformException as e:
                self.set_state(f"no TF {arm['frame']} -> {arm['tip']}: {e.__class__.__name__}")
                return None
        return out

    def robot_action(self, poses):
        arms = []
        for i, (arm, T) in enumerate(zip(self.arms, poses)):
            if not arm["map"].engaged:
                arm["map"].engage(T, self.seed[i * 7: i * 7 + 6])
            pose6 = pose6_from_transform(arm["map"].to_action(T), reference_rpy=self.prev_action[i * 7 + 3: i * 7 + 6])
            arms.append(np.concatenate([pose6, [self.grip[i] if self.gripper_source != "seed" else self.seed[i * 7 + 6]]]))
        return join_action(arms)

    # -- the loop ---------------------------------------------------------------------------------------------------------
    def tick(self):
        if self.pending_task:
            task, self.pending_task = self.pending_task, None
            self.load(task)
            return
        if self.reset_requested:
            self.reset_requested = False
            self.reset()
            return
        if self.mode == "replay":
            action = self.engine.recorded_action()
            if action is None:
                self.reset()
                return
            self.pub_robot.publish(Float32MultiArray(data=action.tolist()))
            self.advance(action)
            return
        poses = self.measured_poses()
        if poses is None:
            return self.republish()
        action = self.robot_action(poses)
        if not self.following:
            d = [pose_distance(action[i * 7: i * 7 + 6], self.seed[i * 7: i * 7 + 6]) for i in range(2)]
            at_seed = all(p <= self.seed_tol[0] and r <= self.seed_tol[1] for p, r in d)
            waited = time.time() - self.reset_time
            if not (at_seed or not self.wait_for_seed or waited > self.seed_timeout):
                return self.republish()
            if not at_seed and self.wait_for_seed:
                self.get_logger().warn(f"The robot did not reach the seed pose in {self.seed_timeout:.0f} s "
                                       f"(off by {max(p for p, _ in d) * 100:.1f} cm); following from where it is.")
            self.following = True
            self.gate.reset()
            self.gate.decide(action)  # the first measured action is the reference, not a motion
            self.prev_action = self.applied = action
            self.set_state("following the robot")
            return self.republish()
        self.prev_action = action
        if self.gate.decide(action):
            self.advance(action, interpolate=True)
        else:
            self.republish()

    def substeps(self, action):
        """Actions from the last applied one to ``action``, each at most about one keyboard step apart."""
        a, b = np.asarray(self.applied, np.float64), np.asarray(action, np.float64)
        n = 1
        for i in range(2):
            d, r = pose_distance(a[i * 7: i * 7 + 6], b[i * 7: i * 7 + 6])
            n = max(n, int(np.ceil(d / (self.substep_scale * self.engine.cfg.step))), int(np.ceil(r / self.substep_rad)))
        n = min(n, self.max_substeps)
        delta = b - a
        for i in range(2):
            sl = slice(i * 7 + 3, i * 7 + 6)
            delta[sl] = (delta[sl] + np.pi) % (2 * np.pi) - np.pi
        return [(a + delta * k / n).astype(np.float32) for k in range(1, n + 1)]

    def advance(self, action, interpolate=False):
        for a in (self.substeps(action) if interpolate else [action]):
            self.engine.step(a)
        self.applied = np.asarray(action, np.float32)
        self.prev_action = np.asarray(action, np.float32)
        self.pub_applied.publish(Float32MultiArray(data=self.prev_action.tolist()))
        self.publish_scene()

    def publish_scene(self):
        stamp = self.get_clock().now().to_msg()
        xyz, rgb = self.engine.points(self.max_points)
        msgs = [(self.pub_points, cloud_message(xyz, rgb, Header(stamp=stamp, frame_id=self.world)))]
        if self.image_on:
            px = self.engine.render()
            img = Image(header=Header(stamp=stamp, frame_id=self.camera_frame), height=px.shape[0], width=px.shape[1],
                        encoding="rgb8", step=px.shape[1] * 3)
            img.data = array.array("B", np.ascontiguousarray(px).tobytes())
            self.camera_info.header = Header(stamp=stamp, frame_id=self.camera_frame)
            msgs += [(self.pub_image, img), (self.pub_info, self.camera_info)]
        if self.pub_model_joints is not None:
            q = self.engine.model_joints()
            if q is not None:
                names, pos = [], []
                for i in range(2):
                    names += [f"arm{i}_{j}" for j in ARM_JOINTS] + [f"arm{i}_{FINGER_JOINT}", f"arm{i}_{MIMIC_FINGER_JOINT}"]
                    pos += list(q[i * 7: i * 7 + 6]) + [q[i * 7 + 6], -q[i * 7 + 6]]
                msgs.append((self.pub_model_joints, JointState(header=Header(stamp=stamp), name=names, position=[float(v) for v in pos])))
        for pub, m in msgs:
            pub.publish(m)
        self.last_msgs, self.last_publish = msgs, time.time()

    def republish(self):
        """A still scene is re-sent now and then, for displays that join late."""
        if self.last_msgs and time.time() - self.last_publish > self.republish_period:
            stamp = self.get_clock().now().to_msg()
            for pub, m in self.last_msgs:
                m.header.stamp = stamp
                pub.publish(m)
            self.last_publish = time.time()


def main():
    rclpy.init()
    node = WorldModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        if rclpy.ok():  # a failure while running; errors from a shutdown during a task load are expected
            raise
    finally:
        node.engine.unload()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
