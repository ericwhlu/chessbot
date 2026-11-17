"""Launch chess_brain.  Example:

    ros2 launch chess_brain brain.launch.py robot_color:=black think_time:=0.5 stable_time:=1.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# name -> (default, type)
ARGS = {
    "stockfish_path": ("stockfish", str),
    "robot_color": ("black", str),
    "think_time": ("0.5", float),
    "skill_level": ("5", int),
    "stable_time": ("1.0", float),
    "arm_timeout": ("120.0", float),
    "vision_timeout": ("0.0", float),
    "allow_promotion": ("false", bool),
}


def generate_launch_description():
    declared = [DeclareLaunchArgument(name, default_value=default) for name, (default, _) in ARGS.items()]
    params = {name: ParameterValue(LaunchConfiguration(name), value_type=kind) for name, (_, kind) in ARGS.items()}
    node = Node(
        package="chess_brain",
        executable="brain_node",
        name="chess_brain",
        output="screen",
        emulate_tty=True,
        parameters=[params],
    )
    return LaunchDescription(declared + [node])
