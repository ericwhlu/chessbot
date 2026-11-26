"""The Kinova Gen3 Lite with ros2_control mock hardware and MoveIt: no Gazebo, no arm.

For checking arm code on a machine that cannot run the course image (Apple Silicon): joints
follow their commands exactly, there is no physics and no gripper contact, so a finger always
ends where it was told to go. Uses the description, SRDF and controller settings of the
ros-jazzy-kortex-description / ros-jazzy-kinova-gen3-lite-moveit-config packages; only the
hardware block of the description is replaced (those packages name the real Kortex driver).

    ros2 launch chess_brain mock_gen3_lite.launch.py
"""
import re
import tempfile
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

ARM_JOINTS = [f"joint_{i}" for i in range(1, 7)]
FINGER = "right_finger_bottom_joint"


def mock_description() -> str:
    """The packaged gen3_lite.urdf with its hardware block swapped for mock components."""
    source = Path(get_package_share_directory("kortex_description")) / "robots" / "gen3_lite.urdf"
    text = source.read_text()
    joints = "".join(
        f'<joint name="{name}"><command_interface name="position"/>'
        f'<state_interface name="position"><param name="initial_value">0.0</param></state_interface>'
        f'<state_interface name="velocity"/></joint>'
        for name in ARM_JOINTS + [FINGER]
    )
    block = (
        '<ros2_control name="MockGen3Lite" type="system"><hardware>'
        "<plugin>mock_components/GenericSystem</plugin></hardware>" + joints + "</ros2_control>"
    )
    text, count = re.subn(r"<ros2_control\b.*?</ros2_control>", block, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"no ros2_control block found in {source}")
    return text


def generate_launch_description():
    config_dir = Path(get_package_share_directory("kinova_gen3_lite_moveit_config")) / "config"
    urdf = Path(tempfile.mkdtemp(prefix="mock_gen3_lite_")) / "gen3_lite_mock.urdf"
    urdf.write_text(mock_description())

    moveit = (
        MoveItConfigsBuilder("gen3_lite", package_name="kinova_gen3_lite_moveit_config")
        .robot_description(file_path=str(urdf))
        .robot_description_semantic(file_path="config/gen3_lite.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(publish_robot_description=True, publish_robot_description_semantic=True)
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    def spawner(name):
        return Node(package="controller_manager", executable="spawner", arguments=[name, "-c", "/controller_manager"])

    return LaunchDescription([
        Node(
            package="controller_manager", executable="ros2_control_node", output="both",
            parameters=[str(config_dir / "ros2_controllers.yaml")],
            remappings=[("/controller_manager/robot_description", "/robot_description")],
        ),
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="both",
             parameters=[moveit.robot_description]),
        spawner("joint_state_broadcaster"),
        spawner("joint_trajectory_controller"),
        spawner("gen3_lite_2f_gripper_controller"),
        Node(package="moveit_ros_move_group", executable="move_group", output="screen",
             parameters=[moveit.to_dict()]),
    ])
