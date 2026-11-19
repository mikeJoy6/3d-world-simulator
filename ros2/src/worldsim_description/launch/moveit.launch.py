"""The extracted ViperX pair in ROS: robot_state_publisher and MoveIt's move_group (planning groups arm0 / arm1).

    ros2 launch worldsim_description moveit.launch.py           # needs /joint_states (worldsim_sim's sim_robot)

Included by worldsim_sim/launch/sim.launch.py. ``moveit_parameters()`` is what an RViz MotionPlanning display needs too.
"""
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def moveit_parameters():
    share = Path(get_package_share_directory("worldsim_description"))
    cfg = share / "config"
    ompl = _yaml(cfg / "ompl_planning.yaml")
    return {
        "robot_description": (share / "urdf/worldsim_aloha.urdf").read_text(),
        "robot_description_semantic": (cfg / "worldsim_aloha.srdf").read_text(),
        "robot_description_kinematics": _yaml(cfg / "kinematics.yaml"),
        "robot_description_planning": _yaml(cfg / "joint_limits.yaml"),
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": ompl,
        **_yaml(cfg / "moveit_controllers.yaml"),
        "moveit_manage_controllers": False,
        "planning_scene_monitor": {"publish_planning_scene": True, "publish_geometry_updates": True, "publish_state_updates": True,
                                   "publish_transforms_updates": True, "publish_robot_description": True,
                                   "publish_robot_description_semantic": True},
        "publish_robot_description_semantic": True,
    }


def generate_launch_description():
    params = moveit_parameters()
    return LaunchDescription([
        DeclareLaunchArgument("rsp", default_value="true"),
        DeclareLaunchArgument("move_group", default_value="true"),
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="log",
             parameters=[{"robot_description": params["robot_description"], "publish_frequency": 50.0}],
             condition=IfCondition(LaunchConfiguration("rsp"))),
        Node(package="moveit_ros_move_group", executable="move_group", output="log", parameters=[params],
             arguments=["--ros-args", "--log-level", "warn"], condition=IfCondition(LaunchConfiguration("move_group"))),
    ])
