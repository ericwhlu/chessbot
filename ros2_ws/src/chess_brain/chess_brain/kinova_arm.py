"""The course's Kinova Gen3 Lite through MoveIt: the ROS side of kinova_executor.py.

Adapted from the course's Lab 6 helpers (arm.py and table.py), which drive both the Gazebo
station and the real arm at the lab benches: MoveIt through pymoveit2, the gripper through its
own action, measured finger positions from /joint_states. It therefore only runs inside the
course Kinova container, with the arm's bring-up (Gazebo or the real arm) and MoveIt running.

Differences from the lab helper:

* every motion takes a speed factor, and outside simulation the settings refuse more than the
  lab benches' limit of 0.2;
* straight-line motions must be planned completely (the lab helper accepts 90 % of the path) and
  are checked for collisions; a straight-line motion that fails while executing is not retried;
* planning and execution get timeouts where the installed pymoveit2 offers them;
* ``last_move_started`` tells whether a failed motion may have moved the arm;
* ``can_reach`` answers "is this top-down pose reachable" without moving, for reach_check.py.
"""

from __future__ import annotations

import inspect
import time
from typing import List, Optional

from geometry_msgs.msg import Pose
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionFK
from pymoveit2 import GripperCommand, MoveIt2
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .kinova_executor import FINGER_CLOSED, LAB_SPEED_LIMIT, TABLE_TOP_Z, Settings

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
GRIPPER_JOINT = "right_finger_bottom_joint"
GRIPPER_ACTION = "/gen3_lite_2f_gripper_controller/gripper_cmd"
BASE_LINK = "base_link"
END_EFFECTOR_LINK = "end_effector_link"

# Gripper pointing straight down, as a quaternion (x, y, z, w).
TOP_DOWN = (0.0, 1.0, 0.0, 0.0)

# The lab station (from the course's table.py): the arm stands on a 50 mm quick mount bolted to
# a 12 x 12 x 0.5 inch plate, so base_link is MOUNT_OFFSET above the tabletop.
QUICK_MOUNT_SIZE = (0.11, 0.11, 0.050)
PLATE_SIZE = (0.3048, 0.3048, 0.0127)
MOUNT_OFFSET = QUICK_MOUNT_SIZE[2] + PLATE_SIZE[2]  # 0.0627; TABLE_TOP_Z = -MOUNT_OFFSET
TABLE_SIZE = (1.829, 1.016, 0.038)
_GAP = 0.001  # air under the arm's base, so it is not reported as touching the mount

# name -> default: the ROS parameters behind kinova_executor.Settings, plus the station.
SETTING_PARAMETERS = {
    "gripper_length": Settings.gripper_length,
    "grip_height": Settings.grip_height,
    "travel_clearance": Settings.travel_clearance,
    "place_clearance": Settings.place_clearance,
    "open_position": Settings.open_position,
    "speed": Settings.speed,
    "slow_speed": Settings.slow_speed,
    "grasp_check": Settings.grasp_check,
    "missed_reading": Settings.missed_reading,
    "park": Settings.park,
    "floor_z": Settings.floor_z,
}
PLAN_TIMEOUT = 30.0        # s for MoveIt to answer one planning request
EXECUTION_TIMEOUT = 180.0  # s for one motion to finish


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def settings_from_parameters(node: Node) -> Settings:
    """Read kinova_executor.Settings from the node's parameters (declared by KinovaArm).

    Raises ValueError for a value that is not usable. On the real arm (``use_sim_time`` false)
    the speeds may not exceed the lab limit; a simulation may go faster.
    """
    values = {}
    for name, default in SETTING_PARAMETERS.items():
        value = node.get_parameter(name).value
        try:
            if isinstance(default, bool):
                values[name] = _as_bool(value)
            elif isinstance(default, str):
                values[name] = str(value).strip().lower()
            else:
                values[name] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"parameter {name}: {value!r} is not a number") from None
    simulated = _as_bool(node.get_parameter("use_sim_time").value)
    return Settings(max_speed=1.0 if simulated else LAB_SPEED_LIMIT, **values)


class KinovaArm(Node):
    """A ROS 2 node with MoveIt, the gripper and the measured joint positions of the Gen3 Lite."""

    def __init__(self, name: str = "chess_kinova_arm") -> None:
        super().__init__(name)
        for parameter, default in SETTING_PARAMETERS.items():
            self.declare_parameter(parameter, default, ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("add_table", True, ParameterDescriptor(dynamic_typing=True))

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ARM_JOINTS,
            base_link_name=BASE_LINK,
            end_effector_name=END_EFFECTOR_LINK,
            group_name="arm",
        )
        # ignore_new_calls_while_executing=False lets an open replace a close that is still
        # pressing on a piece; with True, that open would be dropped.
        self.gripper = GripperCommand(
            node=self,
            gripper_joint_names=[GRIPPER_JOINT],
            open_gripper_joint_positions=[0.0],
            closed_gripper_joint_positions=[FINGER_CLOSED],
            ignore_new_calls_while_executing=False,
            gripper_command_action_name=GRIPPER_ACTION,
        )
        self._joint_positions = {}
        self._joint_stamp = 0.0
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self._fk_client = self.create_client(GetPositionFK, "compute_fk")
        self.moveit2.max_velocity = 0.1      # never the planner's default, even before the first _run
        self.moveit2.max_acceleration = 0.1
        if hasattr(self.moveit2, "cartesian_avoid_collisions"):
            self.moveit2.cartesian_avoid_collisions = True   # straight lines respect the table too

        # pymoveit2 versions differ: adapt to the installed one instead of guessing.
        plan_arguments = inspect.signature(self.moveit2.plan).parameters
        self._step_argument = "max_step" if "max_step" in plan_arguments else "cartesian_max_step"
        self._plan_extra = {"timeout_sec": PLAN_TIMEOUT} if "timeout_sec" in plan_arguments else {}
        wait_arguments = inspect.signature(self.moveit2.wait_until_executed).parameters
        self._wait_extra = {}
        if "timeout_sec" in wait_arguments:
            self._wait_extra["timeout_sec"] = EXECUTION_TIMEOUT
        if "cancel_on_timeout" in wait_arguments:
            self._wait_extra["cancel_on_timeout"] = True
        self._has_ik = callable(getattr(self.moveit2, "compute_ik", None))
        self._started = False   # did the last move_to_* send a trajectory to the arm?

    # ------------------------------------------------------------ measured state
    def _on_joint_state(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            self._joint_positions[name] = position
        # The robot's own clock: simulated time in Gazebo, real time on the arm.
        self._joint_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def finger_position(self) -> Optional[float]:
        """The latest measured finger position, or None before the first /joint_states message."""
        return self._joint_positions.get(GRIPPER_JOINT)

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Wait until MoveIt answers for the arm's current state. Wall-clock timeouts only, so a
        paused simulation cannot hang it."""
        log = self.get_logger()
        deadline = time.monotonic() + timeout
        if not self._fk_client.wait_for_service(timeout_sec=timeout):
            log.error("MoveIt's compute_fk service is not available. Is MoveIt running?")
            return False

        while True:  # all six arm joints must have been reported before asking for FK
            state = self.moveit2.joint_state
            if (
                state is not None
                and len(state.name) == len(state.position)
                and all(name in state.name for name in ARM_JOINTS)
            ):
                break
            if time.monotonic() >= deadline:
                log.error("Timed out waiting for the arm's joint states. Is the arm's bring-up running?")
                return False
            time.sleep(0.05)

        if _as_bool(self.get_parameter("use_sim_time").value):
            first = self.get_clock().now().nanoseconds
            while self.get_clock().now().nanoseconds == first:
                if time.monotonic() >= deadline:
                    log.error("The simulation clock is not advancing. Is Gazebo paused?")
                    return False
                time.sleep(0.05)

        request = GetPositionFK.Request()
        request.header.frame_id = BASE_LINK
        request.fk_link_names = [END_EFFECTOR_LINK]
        request.robot_state.joint_state = state
        request.robot_state.is_diff = True
        future = self._fk_client.call_async(request)
        while not future.done():
            if time.monotonic() >= deadline:
                future.cancel()
                log.error("Timed out waiting for MoveIt's FK answer.")
                return False
            time.sleep(0.05)
        try:
            response = future.result()
        except Exception as error:
            log.error(f"MoveIt's FK request failed: {error}")
            return False
        if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS or not response.pose_stamped:
            log.error("MoveIt did not return a valid pose for the arm's current state.")
            return False
        log.info("MoveIt is ready.")
        return True

    # ------------------------------------------------------------ planning scene
    def add_table(self) -> None:
        """Add the lab table, plate and quick mount to MoveIt's planning scene (parameter add_table)."""
        if not _as_bool(self.get_parameter("add_table").value):
            return
        boxes = {
            "table": (TABLE_SIZE, TABLE_TOP_Z - TABLE_SIZE[2] / 2.0),
            "mount_plate": (PLATE_SIZE, TABLE_TOP_Z + PLATE_SIZE[2] / 2.0),
            "quick_mount": (
                (QUICK_MOUNT_SIZE[0], QUICK_MOUNT_SIZE[1], QUICK_MOUNT_SIZE[2] - _GAP),
                TABLE_TOP_Z + PLATE_SIZE[2] + (QUICK_MOUNT_SIZE[2] - _GAP) / 2.0,
            ),
        }
        for object_id, (size, center_z) in boxes.items():
            self.moveit2.add_collision_box(
                id=object_id,
                size=size,
                position=(0.0, 0.0, center_z),
                quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id=BASE_LINK,
            )
        time.sleep(0.5)  # let the planning scene take the boxes before the first plan
        self.get_logger().info("Added the table and mount to MoveIt's planning scene.")

    # ------------------------------------------------------------------- motion
    @staticmethod
    def _top_down_pose(x: float, y: float, z: float) -> Pose:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z)
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = TOP_DOWN
        return pose

    def _run(self, describe: str, speed: float, attempts: int, retry_execution: bool, **plan_arguments) -> bool:
        """Plan, execute and wait; a failed plan is retried, the lab helper's sequence."""
        log = self.get_logger()
        speed = min(max(float(speed), 0.01), 1.0)
        self.moveit2.max_velocity = speed
        self.moveit2.max_acceleration = speed
        self._started = False
        for attempt in range(1, attempts + 1):
            trajectory = self.moveit2.plan(**plan_arguments, **self._plan_extra)
            if trajectory is None:
                log.warning(f"Planning attempt {attempt} of {attempts} for {describe} failed.")
                continue
            self._started = True
            self.moveit2.execute(trajectory)
            if self.moveit2.wait_until_executed(**self._wait_extra):
                return True
            log.warning(f"Execution attempt {attempt} of {attempts} for {describe} failed.")
            if not retry_execution:
                break
        log.error(f"Could not plan or execute {describe}.")
        return False

    def last_move_started(self) -> bool:
        """False if the last move_to_* failed before anything was sent to the arm (planning only)."""
        return self._started

    def move_to_joints(self, joint_positions: List[float], speed: float = 0.1, attempts: int = 3) -> bool:
        """Plan and execute a joint-space motion."""
        return self._run("a joint-space motion", speed, attempts, True, joint_positions=list(joint_positions))

    def move_to_pose(
        self, x: float, y: float, z: float, cartesian: bool = False, speed: float = 0.1, attempts: int = 3
    ) -> bool:
        """Move end_effector_link to (x, y, z) in base_link, gripper pointing down.

        cartesian=True moves in a straight line, and only if the whole line can be planned. If a
        straight-line motion fails while executing (e.g. it touched something) it is not retried.
        """
        kind = "a straight-line motion" if cartesian else "a pose-goal motion"
        arguments = self._pose_arguments(x, y, z, cartesian)
        return self._run(f"{kind} to ({x:.3f}, {y:.3f}, {z:.3f})", speed, attempts, not cartesian, **arguments)

    def _pose_arguments(self, x: float, y: float, z: float, cartesian: bool) -> dict:
        return {
            "pose": self._top_down_pose(x, y, z),
            "cartesian": cartesian,
            "cartesian_fraction_threshold": 0.999 if cartesian else 0.0,
            self._step_argument: 0.005,
        }

    def reach_method(self) -> str:
        return "MoveIt inverse kinematics" if self._has_ik else "a full motion plan from the current pose"

    def can_reach(self, x: float, y: float, z: float, tries: int = 3) -> bool:
        """True if MoveIt finds joint angles for the top-down pose at (x, y, z). Nothing moves.

        The IK solver starts from a random guess and can miss a solution that exists, so a "no"
        is asked again a few times.
        """
        for _ in range(tries):
            if self._has_ik:
                found = self.moveit2.compute_ik((float(x), float(y), float(z)), TOP_DOWN) is not None
            else:
                arguments = self._pose_arguments(x, y, z, cartesian=False)
                found = self.moveit2.plan(**arguments, **self._plan_extra) is not None
            if found:
                return True
        return False

    # ------------------------------------------------------------------ gripper
    def _move_fingers(self, position: float, timeout: float) -> Optional[float]:
        """Command the fingers, wait until they stop moving or the timeout passes, and return
        the measured position. Times are read from /joint_states, not the wall clock, so a slow
        simulation gets the same amount of robot time as the real arm."""
        self.gripper.move_to_position(position)
        start = self._joint_stamp
        last = self.finger_position()
        still_since = start
        wall_limit = time.time() + 10.0 * timeout  # in case /joint_states stops
        while time.time() < wall_limit:
            time.sleep(0.05)
            now, stamp = self.finger_position(), self._joint_stamp
            if stamp - start > timeout:
                break
            if last is None or now is None or abs(now - last) > 0.002:
                last, still_since = now, stamp
            elif stamp - still_since > 0.5 and stamp - start > 0.5:
                break
        return self.finger_position()

    def open_gripper(self, position: float = 0.0) -> Optional[float]:
        """Open the fingers to ``position`` (0.0 = fully open) and return the measured position."""
        return self._move_fingers(float(position), timeout=3.0)

    def close_gripper(self) -> Optional[float]:
        """Close the fingers and return the measured position: about 0.80 on nothing, less when
        they stopped on a piece (the gripper action never reports success in that case, so this
        does not wait for it)."""
        return self._move_fingers(FINGER_CLOSED, timeout=4.0)
