"""The real system: chess_brain + the vision adapter + the arm adapter.

    ros2 launch chess_brain robot.launch.py                                  # arm in dry run, no detector yet
    ros2 launch chess_brain robot.launch.py \
        detector:=chessbot_vision.detector:detect image_topic:=/camera/image_raw \
        executor:=chessbot_arm.motion:pick_and_place board_x:=0.30 board_y:=-0.20

On the course's Kinova Gen3 Lite (bring-up and MoveIt already running; add use_sim_time:=true in Gazebo):
    ros2 launch chess_brain robot.launch.py executor:=chess_brain.kinova_executor:pick_and_place \
        board_x:=0.60 board_y:=-0.20 board_yaw_deg:=90 surface_z:=-0.056 offboard_drop:=0.0066 arm_timeout:=300

Until a detector is plugged in, be the vision yourself from a second shell:
    ros2 run chess_brain operator_node        (paste a placement FEN, or press Enter for the detect button)

Not for use together with sim_game.launch.py (the simulator also answers arm_actions).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# name -> (default, type, description)
BRAIN = {
    "robot_color": ("black", str, "white or black"),
    "stockfish_path": ("stockfish", str, "engine binary (found automatically in /usr/games)"),
    "think_time": ("0.5", float, "robot seconds per move"),
    "skill_level": ("5", int, "robot Stockfish Skill Level 0-20"),
    "stable_time": ("1.0", float, "seconds a detection must stay unchanged"),
    "arm_timeout": ("120.0", float, "seconds to wait for arm_done"),
    "allow_promotion": ("false", bool, "true only if spare queens are on the table"),
}
VISION = {
    "detector": ("", str, "package.module:function returning a placement FEN from a BGR image"),
    "image_topic": ("chessbot/camera/image", str, "sensor_msgs/Image topic of the camera"),
    "period": ("0.0", float, "> 0: also detect every N seconds (continuous mode)"),
    "max_image_age": ("2.0", float, "seconds; older camera frames are not used"),
}
ARM = {
    "executor": ("", str, "package.module:function doing one pick-and-place; empty = dry run"),
    "dry_run_time": ("1.0", float, "seconds per action in dry-run mode"),
    "board_x": ("0.0", float, "world x of the board's a1 corner (m)"),
    "board_y": ("0.0", float, "world y of the board's a1 corner (m)"),
    "board_yaw_deg": ("0.0", float, "rotation of the board about +Z (degrees)"),
    "surface_z": ("0.00656168", float, "world z of the playing surface (m)"),
    "discard_x": ("-0.075", float, "first discard slot, board frame x (m)"),
    "discard_y": ("0.025", float, "first discard slot, board frame y (m)"),
    "offboard_drop": ("0.0", float, "discard slots and spares are this far below the surface (m)"),
}
# Used by executor:=chess_brain.kinova_executor:pick_and_place (the course's Kinova Gen3 Lite).
KINOVA = {
    "use_sim_time": ("false", bool, "true when the arm runs in Gazebo"),
    "speed": ("0.2", float, "velocity scaling of the large arm moves (at most 0.2 on the real arm)"),
    "slow_speed": ("0.1", float, "velocity scaling of the descents onto the board"),
    "grip_height": ("0.02", float, "fingertips above the square when gripping a piece (m)"),
    "travel_clearance": ("0.15", float, "fingertips above the board while carrying (m)"),
    "place_clearance": ("0.003", float, "a piece is released this much higher than it was gripped (m)"),
    "gripper_length": ("0.157", float, "fingertips below end_effector_link (m)"),
    "open_position": ("0.5", float, "finger joint when open over the board (0 = fully open)"),
    "grasp_check": ("true", bool, "check every finger command against the measured position"),
    "missed_reading": ("0.75", float, "finger position at or above this after closing = no piece"),
    "park": ("work", str, "where the arm waits between moves: work or retract"),
    "add_table": ("true", bool, "add the lab table and mount to MoveIt's planning scene"),
}


def params(group):
    return {name: ParameterValue(LaunchConfiguration(name), value_type=kind) for name, (_, kind, _) in group.items()}


def generate_launch_description():
    declared = [
        DeclareLaunchArgument(name, default_value=default, description=text)
        for group in (BRAIN, VISION, ARM, KINOVA)
        for name, (default, _, text) in group.items()
    ]
    common = dict(package="chess_brain", output="screen", emulate_tty=True)
    brain = Node(
        executable="brain_node",
        name="chess_brain",
        parameters=[params(BRAIN)],
        on_exit=Shutdown(reason="chess_brain exited"),
        **common,
    )
    # No name= on the adapters: launch would rename EVERY node in that process, including a node
    # the plug-in creates for its own ROS calls. The executables already name themselves.
    vision = Node(executable="vision_adapter_node", parameters=[params(VISION)], **common)
    # The KINOVA parameters are read by the node that kinova_executor creates inside this process.
    arm = Node(executable="arm_adapter_node", parameters=[params(ARM), params(KINOVA)], **common)
    return LaunchDescription(declared + [brain, vision, arm])
