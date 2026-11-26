"""Play a game in the MuJoCo simulation over ROS 2: chess_brain decides, mujoco_world_node is
the camera, the arm, the opponent and the operator.

    ros2 launch chessbot_sim_ros mujoco_game.launch.py                       # 12 plies, robot black
    ros2 launch chessbot_sim_ros mujoco_game.launch.py plies:=30 robot_color:=white
    ros2 launch chessbot_sim_ros mujoco_game.launch.py scan_mode:=on_request script:=e2e4,d2d4

The launch ends when the game does. Images and summary.json go to out_dir.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# name -> (default, type, description)
SHARED = {
    "robot_color": ("black", str, "white or black; the simulated opponent plays the other colour"),
}
BRAIN = {
    "stockfish_path": ("stockfish", str, "engine binary (found automatically in /usr/games)"),
    "think_time": ("0.3", float, "robot seconds per move"),
    "skill_level": ("5", int, "robot Stockfish Skill Level 0-20"),
    "stable_time": ("0.5", float, "seconds a detection must stay unchanged; keep it below human_delay"),
    "arm_timeout": ("300.0", float, "seconds chess_brain waits for the arm (a capture takes 20-40 s here)"),
}
WORLD = {
    "plies": ("12", int, "stop after this many half-moves; 0 = play to the end"),
    "seed": ("7", int, "seed of the simulated opponent"),
    "script": ("", str, "opponent moves to play first, e.g. e2e4,d2d4"),
    "scan_mode": ("continuous", str, "continuous (camera reports every 2 s) | on_request (only when asked)"),
    "human_delay": ("1.5", float, "seconds after a scan before the opponent moves; above stable_time"),
    "out_dir": ("game_run_ros", str, "directory for images and summary.json"),
    "repo_root": ("", str, "the repository (where chessbot_sim/ is); empty = find it, e.g. /chessbot"),
    "video": ("false", bool, "also record game.mp4"),
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
        package="chessbot_sim_ros",
        executable="mujoco_world_node",
        name="mujoco_world",
        output="screen",
        emulate_tty=True,
        parameters=[params(SHARED, WORLD)],
        on_exit=Shutdown(reason="the simulated game finished"),
    )
    return LaunchDescription(declared + [brain, world])
