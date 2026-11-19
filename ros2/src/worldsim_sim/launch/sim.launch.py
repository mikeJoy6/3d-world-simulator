"""The whole simulation: the extracted ViperX pair (URDF + MoveIt planning groups), its kinematic controllers, the world
model following it, end-effector control through MoveIt, the keyboard panel and RViz.

    ros2 launch worldsim_sim sim.launch.py [task:=pusht|cloth|box|assembly] [mode:=follow|replay] [rviz:=true] ...
"""
import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def nodes(context):
    arg = lambda name: context.launch_configurations[name]
    flag = lambda name: arg(name).lower() in ("true", "1", "yes")
    desc = get_package_share_directory("worldsim_description")
    spec = importlib.util.spec_from_file_location("worldsim_moveit_launch", str(Path(desc) / "launch/moveit.launch.py"))
    moveit_launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(moveit_launch)
    params = moveit_launch.moveit_parameters()
    mode, task = arg("mode"), arg("task")
    rate = float(arg("rate"))
    model = Node(package="worldsim_model", executable="world_model", output="screen", parameters=[{
        "task": task, "mode": mode, "rate": rate, "max_points": int(arg("max_points")), "image": flag("image"),
        "fast": flag("fast"), "publish_model_joints": flag("model_arms"),
        "gripper_source": "joint", "gripper_joints": ["arm0_left_finger", "arm1_left_finger"]}])
    out = [
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(Path(desc) / "launch/moveit.launch.py")),
                                 launch_arguments={"move_group": "true" if mode == "follow" else "false"}.items()),
        Node(package="worldsim_sim", executable="sim_robot", output="screen"),
        model,
        RegisterEventHandler(OnProcessExit(target_action=model, on_exit=[EmitEvent(event=Shutdown(reason="world model exited"))])),
    ]
    if mode == "follow":
        out.append(Node(package="worldsim_model", executable="ee_teleop", output="screen", parameters=[{
            "task": task, "rate": rate, "clamp_workspace": flag("clamp"),
            "gripper_joints": ["arm0_left_finger", "arm1_left_finger"], "arm0.ik": "core", "arm1.ik": "core"}]))
        if flag("panel"):
            out.append(Node(package="worldsim_sim", executable="control_panel", output="screen", parameters=[{"image": flag("image")}]))
    if flag("model_arms"):
        out.append(Node(package="robot_state_publisher", executable="robot_state_publisher", name="model_arms_state_publisher",
                        output="log", parameters=[{"robot_description": params["robot_description"], "frame_prefix": "model/"}],
                        remappings=[("joint_states", "/worldsim/model/joint_states"), ("robot_description", "/worldsim/model/robot_description")]))
        out.append(Node(package="tf2_ros", executable="static_transform_publisher", output="log",
                        arguments=["--frame-id", "worldsim_world", "--child-frame-id", "model/worldsim_world"]))
    if flag("rviz"):
        rviz_params = {k: params[k] for k in ("robot_description", "robot_description_semantic", "robot_description_kinematics",
                                              "robot_description_planning")}
        config = arg("rviz_config") or str(Path(get_package_share_directory("worldsim_sim")) / "rviz/worldsim.rviz")
        out.append(Node(package="rviz2", executable="rviz2", output="log", parameters=[rviz_params], arguments=["-d", config]))
    return out


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("task", default_value="pusht", choices=["pusht", "cloth", "box", "assembly"]),
        DeclareLaunchArgument("mode", default_value="follow", choices=["follow", "replay"],
                              description="follow: the world model follows the arms (keyboard, MoveIt panel, /worldsim/action); replay: the recorded actions"),
        DeclareLaunchArgument("rate", default_value="5.0", description="model steps per second (one step = 0.1 s of the data)"),
        DeclareLaunchArgument("max_points", default_value="100000", description="scene particles published (0: all)"),
        DeclareLaunchArgument("image", default_value="true", description="render the scene from the task camera"),
        DeclareLaunchArgument("fast", default_value="false", description="upstream fast inference (compiles about a minute per task)"),
        DeclareLaunchArgument("clamp", default_value="true", description="keep teleoperation targets in the recorded workspace"),
        DeclareLaunchArgument("model_arms", default_value="false", description="also show where the model believes the arms are"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value="", description="RViz config file (default: worldsim_sim/rviz/worldsim.rviz)"),
        DeclareLaunchArgument("panel", default_value="true"),
        OpaqueFunction(function=nodes),
    ])
