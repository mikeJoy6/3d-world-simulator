"""The world model driven by the real G2: its end-effector poses (TF) -> model actions -> predicted scene.

Run inside the G2 ROS environment, with the G2 stack up (robot_state_publisher; for teleoperation also move_group and
the arm controllers, e.g. `ros2 launch genie_robot_agibot_g2_hardware hardware.launch.py`):

    source ~/agi@g2_robot/env_g2.sh && source install/local_setup.bash
    ros2 launch worldsim_g2 real.launch.py task:=pusht [teleop:=true enable_motion:=false] [observe:=true]

The robot is moved by its own controller, teleoperation, or (teleop:=true enable_motion:=true) this workspace's
keyboard through the G2 MoveIt planning groups. Parameters: config/g2.yaml.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def nodes(context):
    arg = lambda name: context.launch_configurations[name]
    flag = lambda name: arg(name).lower() in ("true", "1", "yes")
    share = Path(get_package_share_directory("worldsim_g2"))
    cfg = str(share / "config/g2.yaml")
    xyz = [float(v) for v in arg("world_xyz").split()]
    rpy = [float(v) for v in arg("world_rpy").split()]
    out = [
        Node(package="tf2_ros", executable="static_transform_publisher", name="worldsim_world_in_g2", output="log",
             arguments=["--x", str(xyz[0]), "--y", str(xyz[1]), "--z", str(xyz[2]), "--roll", str(rpy[0]), "--pitch", str(rpy[1]),
                        "--yaw", str(rpy[2]), "--frame-id", arg("robot_frame"), "--child-frame-id", "worldsim_world"]),
        Node(package="worldsim_model", executable="world_model", output="screen", parameters=[cfg, {
            "task": arg("task"), "rate": float(arg("rate")), "image": flag("image"), "max_points": int(arg("max_points"))}]),
    ]
    if flag("teleop"):
        out.append(Node(package="worldsim_model", executable="ee_teleop", output="screen", parameters=[cfg, {
            "task": arg("task"), "rate": float(arg("rate")), "enable_motion": flag("enable_motion")}]))
        if flag("panel"):
            out.append(Node(package="worldsim_sim", executable="control_panel", output="screen", parameters=[{"image": flag("image")}]))
    if flag("observe"):
        out.append(Node(package="worldsim_g2", executable="observe", output="screen"))
    if flag("rviz"):
        out.append(Node(package="rviz2", executable="rviz2", output="log", arguments=["-d", str(share / "rviz/real.rviz")]))
    return out


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("task", default_value="pusht", choices=["pusht", "cloth", "box", "assembly"]),
        DeclareLaunchArgument("rate", default_value="5.0"),
        DeclareLaunchArgument("image", default_value="true"),
        DeclareLaunchArgument("max_points", default_value="100000"),
        DeclareLaunchArgument("teleop", default_value="true", description="keyboard end-effector control through the G2 MoveIt groups"),
        DeclareLaunchArgument("enable_motion", default_value="false", description="false: plan and report only, the G2 does not move"),
        DeclareLaunchArgument("panel", default_value="true"),
        DeclareLaunchArgument("observe", default_value="false", description="also republish the G2 camera and arm feedback"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("robot_frame", default_value="base_link"),
        DeclareLaunchArgument("world_xyz", default_value="0.75 0.0 0.80", description="where the model's scene is shown in robot_frame"),
        DeclareLaunchArgument("world_rpy", default_value="0 0 0"),
        OpaqueFunction(function=nodes),
    ])
