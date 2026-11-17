"""Play a whole game with no hardware: chess_brain against a simulated human, camera and arm.

    ros2 launch chess_brain sim_game.launch.py
    ros2 launch chess_brain sim_game.launch.py max_plies:=10            # short demo
    ros2 launch chess_brain sim_game.launch.py arm_fail_every:=4        # show the recovery path
    ros2 launch chess_brain sim_game.launch.py robot_color:=white human:=random

The launch ends when the game does (or after max_plies half-moves).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# name -> (default, type, description)
SHARED = {
    "robot_color": ("black", str, "white or black; the simulated human plays the other colour"),
    "stockfish_path": ("stockfish", str, "engine binary (found automatically in /usr/games)"),
}
BRAIN = {
    "think_time": ("0.3", float, "robot seconds per move"),
    "skill_level": ("5", int, "robot Stockfish Skill Level 0-20"),
    "stable_time": ("0.5", float, "seconds a detection must stay unchanged"),
    "allow_promotion": ("true", bool, "the simulated table has spare queens"),
}
WORLD = {
    "human": ("engine", str, "engine | random | script"),
    "human_moves": ("", str, "UCI moves for human:=script, e.g. 'e2e4 g1f3'"),
    "human_skill": ("0", int, "simulated human's Stockfish Skill Level"),
    "human_delay": ("1.0", float, "seconds the human waits before moving"),
    "arm_action_time": ("0.7", float, "seconds per pick-and-place"),
    "arm_fail_every": ("0", int, "N >= 2: every Nth arm command fails once"),
    "max_plies": ("0", int, "stop after this many half-moves; 0 = play to the end"),
}


def params(*groups):
    out = {}
    for group in groups:
        for name, (_, kind, _) in group.items():
            out[name] = ParameterValue(LaunchConfiguration(name), value_type=kind)
    return out


def generate_launch_description():
    declared = [
        DeclareLaunchArgument(name, default_value=default, description=text)
        for group in (SHARED, BRAIN, WORLD)
        for name, (default, _, text) in group.items()
    ]
    brain = Node(
        package="chess_brain",
        executable="brain_node",
        name="chess_brain",
        output="screen",
        emulate_tty=True,
        parameters=[params(SHARED, BRAIN)],
        on_exit=Shutdown(reason="chess_brain exited"),
    )
    world = Node(
        package="chess_brain",
        executable="sim_world_node",
        name="sim_world",
        output="screen",
        emulate_tty=True,
        parameters=[params(SHARED, WORLD)],
        on_exit=Shutdown(reason="simulated game finished"),
    )
    return LaunchDescription(declared + [brain, world])
