"""Pick-and-place for the course's Kinova Gen3 Lite: the arm plug-in of arm_adapter_node.

    ros2 run chess_brain arm_adapter_node --ros-args \\
        -p executor:=chess_brain.kinova_executor:pick_and_place \\
        -p board_x:=0.60 -p board_y:=-0.20 -p board_yaw_deg:=90 -p surface_z:=-0.056

``pick_and_place`` moves one piece with the same top-down sequence as the Lab 6 block stacking:
above the square -> open -> straight down -> close -> straight up -> above the target ->
straight down -> open -> straight up, and parks the arm after the last action of a move.

This file is two layers:

* ``PickPlace`` - the sequence, the heights and what to do when a step fails. Pure Python: it
  talks to any object with the small ``arm`` interface below, so it is unit-tested without ROS.
* ``pick_and_place`` (+ its ``prepare`` and ``operator`` hooks) - create the real arm
  (kinova_arm.py, MoveIt through pymoveit2) on first use, inside the arm_adapter_node process.

The ``arm`` interface (kinova_arm.KinovaArm implements it):

    wait_until_ready() -> bool
    add_table() -> None
    move_to_joints(joints, speed) -> bool
    move_to_pose(x, y, z, cartesian=False, speed=...) -> bool   # end_effector_link, gripper down
        # with a grip direction it is also called with grip=<angle of the line the fingers
        # open along, rad, world frame> and then_z=<height it will be lowered to next, or None>
    last_move_started() -> bool      # did the last move_to_* get as far as moving the arm?
    open_gripper(position) -> float | None                      # measured finger position
    close_gripper() -> float | None
    finger_position() -> float | None

Rules the sequence keeps, whatever fails:

* the arm only moves sideways at the carrying height; after a descent it first lifts straight up;
* the gripper only opens over the board at the gripping height, never while carrying;
* a piece that was gripped is either on its target, back on its square, or still in the closed
  gripper - and in that last case no further move is accepted until the operator has emptied it;
* every finger command is checked against the measured finger position (``grasp_check``).

All heights are derived from the z of the surface that arrives with every action, so nothing
here depends on where the board is. What must be tuned on the real arm are the numbers in
``Settings`` (ROS parameters of the arm node; see kinova_arm.SETTING_PARAMETERS).
"""

from __future__ import annotations

import inspect
import math
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from .arm_adapter import PlacedButFailed

XYZ = Tuple[float, float, float]

# Joint configurations of the Gen3 Lite (radians, joint_1 first), from the course's Lab 6 helper.
# WORK is an elbow-up pose above the table from which top-down poses are reachable; RETRACT is
# the arm's folded parking pose (most straight-line moves are impossible from it).
WORK = (0.40, 0.02, 2.27, -1.57, -0.84, 1.97)
RETRACT = (-0.0527, 0.3669, 2.59, -1.5359, -0.6988, -1.5189)
PARK_POSES = {"work": WORK, "retract": RETRACT}

# Finger joint: 0.0 = fully open (about 111 mm between the fingertips), 0.8 = the close command.
# Lab 6 readings (Gazebo): 0.80 with nothing in the fingers, about 0.44 on a 50 mm block; a
# straight line through them gives 111 mm * (1 - position / 0.8). The manufacturer's finger
# meshes give a wider opening for the same reading (about 60 mm at 0.5). Neither has been
# measured on the real gripper: set open_position and missed_reading from measurements.
FINGER_CLOSED = 0.8
FINGER_TOLERANCE = 0.06   # an "open" within this of open_position counts as done
FINGER_MOVED = 0.01       # a "close" must end at least this far past open_position

# Upper limit of the velocity scaling when the max_speed parameter is not set. 0.2 is what the
# course's midterm required of every motion on the real arm; nothing here knows whether the
# final project has such a rule, so going faster on the real arm takes an explicit max_speed.
LAB_SPEED_LIMIT = 0.2     # outside simulation
SIM_SPEED_LIMIT = 1.0     # in a simulation
TABLE_TOP_Z = -0.0627     # the tabletop in base_link (the arm stands on a 62.7 mm mount)


def speed_limit(*, simulated: bool, max_speed=0.0) -> float:
    """The upper limit for ``Settings.speed`` and ``Settings.slow_speed``.

    ``max_speed`` is the arm node's parameter of that name as it arrives (a number, or a string
    holding one): 0 (not set) gives 0.2 outside simulation and 1.0 in a simulation; a value from
    0.01 to 1.0 is the limit wherever the arm runs. Anything else raises ValueError, true/false
    included. Keyword arguments only, so that the two cannot be swapped.
    """
    if isinstance(max_speed, bool):
        raise ValueError(f"kinova_executor settings: max_speed {max_speed!r} is not a number")
    try:
        limit = float(max_speed)
    except (TypeError, ValueError):
        raise ValueError(f"kinova_executor settings: max_speed {max_speed!r} is not a number") from None
    if limit == 0.0:
        return SIM_SPEED_LIMIT if simulated else LAB_SPEED_LIMIT
    if not 0.01 <= limit <= 1.0:
        raise ValueError(
            f"kinova_executor settings: max_speed {max_speed!r} must be 0 (the default limit) "
            "or between 0.01 and 1.0"
        )
    return limit


@dataclass(frozen=True)
class Settings:
    gripper_length: float = 0.157    # fingertips below end_effector_link (m); Lab 6, fingers open
    grip_height: float = 0.020       # fingertips above the square when closing on a piece (m)
    travel_clearance: float = 0.150  # fingertips above the board while moving between squares (m)
    place_clearance: float = 0.003   # a piece is released this much higher than it was gripped (m)
    open_position: float = 0.50      # finger joint when open over the board (42 to 60 mm, see
    #                                  above; fully open is wider than the neighbours allow)
    speed: float = 0.2               # velocity/acceleration scaling of the moves at carrying height
    slow_speed: float = 0.1          # ... of the straight descents and lifts
    max_speed: float = LAB_SPEED_LIMIT  # upper limit for both; see speed_limit()
    grasp_check: bool = True         # check every finger command against the measured position
    missed_reading: float = 0.75     # finger position at or above this after closing = no piece
    park: str = "work"               # where to go after the last action of a move: work | retract
    floor_z: float = TABLE_TOP_Z     # the fingertips are never sent below this (base_link z, m)

    def __post_init__(self) -> None:
        def need(ok: bool, text: str) -> None:
            if not ok:
                raise ValueError(f"kinova_executor settings: {text}")

        need(0.05 <= self.gripper_length <= 0.30, f"gripper_length {self.gripper_length} is not plausible")
        need(self.grip_height >= 0.003, "grip_height must keep the fingertips at least 3 mm above the board")
        need(self.place_clearance >= 0.0, "place_clearance cannot be negative")
        need(
            self.travel_clearance >= self.grip_height + 0.03,
            "travel_clearance must be at least 3 cm above grip_height",
        )
        need(0.01 <= self.max_speed <= 1.0, "max_speed must be between 0.01 and 1.0")
        need(
            0.01 <= self.slow_speed <= self.speed <= self.max_speed,
            f"need 0.01 <= slow_speed <= speed <= {self.max_speed:g}, the speed limit "
            f"({LAB_SPEED_LIMIT} outside simulation unless the max_speed parameter sets another)",
        )
        need(0.0 <= self.open_position < FINGER_CLOSED, "open_position must be between 0.0 and 0.8")
        need(
            self.open_position + FINGER_TOLERANCE < self.missed_reading <= FINGER_CLOSED,
            f"missed_reading must be more than {FINGER_TOLERANCE} above open_position and at most 0.8",
        )
        need(self.park in PARK_POSES, f"park must be one of {sorted(PARK_POSES)}")

    def travel_z(self, surface_z: float) -> float:
        """Height of end_effector_link while carrying a piece over a surface at ``surface_z``."""
        return surface_z + self.gripper_length + self.travel_clearance

    def grip_z(self, surface_z: float) -> float:
        """Height of end_effector_link when the fingers close on a piece."""
        return surface_z + self.gripper_length + self.grip_height

    def release_z(self, surface_z: float) -> float:
        """Height of end_effector_link when a piece is let go."""
        return self.grip_z(surface_z) + self.place_clearance


class PickPlace:
    """Moves pieces with an ``arm``; one call per action. Returns None on success, else the reason."""

    def __init__(self, arm, settings: Optional[Settings] = None, log: Optional[Callable[[str], None]] = None):
        self._arm = arm
        self._s = settings or Settings()
        self._log = log or (lambda text: None)
        self._ready = False
        self._turns = _takes(getattr(arm, "move_to_pose", None), "grip")   # can this arm turn its gripper?
        self._holding: Optional[str] = None  # square whose piece is (or may be) in the closed gripper
        self._low: Optional[tuple] = None    # (x, y, carrying z, grip): the arm may be below carrying height there
        self._over = False                   # the arm is at carrying height over the board (not parked)
        self._at: Optional[tuple] = None     # ... and exactly here (x, y, z, grip), if known

    # ------------------------------------------------------------------ start-up
    def prepare(self) -> Optional[str]:
        """Wait for MoveIt, add the table to the planning scene and go to WORK. Safe to call again.

        This first move starts from wherever the arm was left, so it never goes faster than the
        default limit, whatever ``speed`` and ``max_speed`` say.
        """
        if self._ready:
            return None
        if not self._arm.wait_until_ready():
            return "MoveIt is not ready (are the arm's bring-up and MoveIt running?)"
        self._arm.add_table()
        if not self._arm.move_to_joints(list(WORK), min(self._s.speed, LAB_SPEED_LIMIT)):
            return "could not move the arm to its WORK pose"
        self._ready = True
        self._log("arm ready at the WORK pose")
        return None

    def operator_cleared(self) -> None:
        """The operator says the gripper is empty again (``retry`` or ``reset`` on chessbot/cmd)."""
        if self._holding is not None:
            self._log(f"operator: the piece from {self._holding} was taken out of the gripper")
        self._holding = None

    # -------------------------------------------------------------------- action
    def __call__(self, pick_xyz: Sequence[float], place_xyz: Sequence[float], info: dict) -> Optional[str]:
        s, arm = self._s, self._arm
        pick_name, place_name = str(info.get("pick", "?")), str(info.get("place", "?"))
        px, py, pz = (float(v) for v in pick_xyz)
        qx, qy, qz = (float(v) for v in place_xyz)

        problem = self._check_heights(pz, qz) or self.prepare() or self._check_gripper_is_free()
        if problem:
            return problem
        if not self._rise():
            return "the arm is still low over the board after an earlier failure and cannot lift straight up"

        travel = s.travel_z(max(pz, qz))  # one carrying height for the whole action
        # Lines the fingers open along at the pick and at the place (rad, world frame), chosen by
        # arm_adapter from the squares that are occupied; None = the arm's own default.
        g_pick, g_place = _angle(info.get("pick_grip")), _angle(info.get("place_grip"))
        if not self._turns:          # an arm written before grip directions existed: its own orientation
            g_pick = g_place = None
        grips = "" if g_pick is None and g_place is None else (
            f"; fingers open along {_degrees(g_pick)} at the pick, {_degrees(g_place)} at the place"
        )
        self._log(
            f"action {int(info.get('index', 0)) + 1}/{info.get('count', 1)}: {pick_name} "
            f"({px:.3f}, {py:.3f}) -> {place_name} ({qx:.3f}, {qy:.3f}){grips}"
        )

        # ---- pick: nothing is held yet, so a failure here leaves the board as it was
        if not self._transit(px, py, travel, g_pick, s.grip_z(pz)):
            return self._fail_empty(f"could not reach above {pick_name}")
        if not self._open():
            return self._fail_empty("the gripper did not open")
        if not self._descend(px, py, s.grip_z(pz), travel, g_pick):
            return self._fail_empty(f"could not descend onto {pick_name}")
        reading = arm.close_gripper()
        if reading is None:
            arm.open_gripper(s.open_position)
            return self._fail_empty("no finger position from the gripper")
        if s.grasp_check and reading >= s.missed_reading:
            arm.open_gripper(s.open_position)
            return self._fail_empty(
                f"grasp missed at {pick_name}: the fingers closed on nothing (reading {reading:.2f})"
            )
        if s.grasp_check and reading < s.open_position + FINGER_MOVED:
            return self._fail_empty(f"the gripper did not close at {pick_name} (reading {reading:.2f})")
        self._holding = pick_name  # from here until it is released, whatever happens (even an exception)
        self._log(f"{pick_name}: fingers closed at {reading:.2f}")

        # ---- carry
        if not self._rise():
            return self._fail_holding(f"could not lift the piece from {pick_name}")
        if not self._transit(qx, qy, travel, g_place, s.release_z(qz)):
            return self._put_back(f"could not reach above {place_name}", pick_name, (px, py, pz), travel, g_pick)
        if s.grasp_check:
            reading = arm.finger_position()
            if reading is None or reading >= s.missed_reading:
                self._holding = None
                self._park()
                return f"the piece from {pick_name} is no longer in the gripper: it was lost on the way to {place_name}"
        if not self._descend(qx, qy, s.release_z(qz), travel, g_place):
            return self._put_back(f"could not descend onto {place_name}", pick_name, (px, py, pz), travel, g_pick)

        # ---- release
        if not self._open():
            return self._fail_holding(f"the gripper did not open over {place_name}")
        self._holding = None
        if not self._rise():
            return PlacedButFailed(f"the piece is on {place_name}, but the arm could not lift away from it")
        if info.get("last", True) and not self._park():
            return PlacedButFailed(
                f"the piece is on {place_name}, but the arm could not park out of the camera's view"
            )
        return None

    # ------------------------------------------------------------------- motions
    def _moved(self) -> bool:
        """After a failed motion: may the arm have moved? (Unknown counts as yes.)"""
        probe = getattr(self._arm, "last_move_started", None)
        return True if probe is None else bool(probe())

    def _pose(self, x: float, y: float, z: float, grip: Optional[float], then_z: Optional[float] = None, **how) -> bool:
        """One pose move. The grip direction is only handed on when there is one and the arm
        takes it, so an arm written before grip directions existed keeps working."""
        if grip is None or not self._turns:
            return bool(self._arm.move_to_pose(x, y, z, **how))
        return bool(self._arm.move_to_pose(x, y, z, grip=grip, then_z=then_z, **how))

    def _transit(
        self, x: float, y: float, travel: float, grip: Optional[float] = None, then_z: Optional[float] = None
    ) -> bool:
        """Go above (x, y) at the carrying height, the fingers turned to open along ``grip``. Over
        the board that is a straight line, so the arm cannot swing through the pieces; only if no
        straight line can be planned, or when coming from the parking pose, MoveIt chooses the
        path. ``then_z`` is the height the gripper will be lowered to next."""
        s = self._s
        target = (x, y, travel, grip)
        if (
            self._at is not None
            and all(abs(a - b) < 1e-6 for a, b in zip(self._at[:3], target[:3]))
            and _same_line(self._at[3], grip)
        ):
            return True  # already there, and turned the right way
        if self._over:
            if self._pose(x, y, travel, grip, then_z, cartesian=True, speed=s.speed):
                self._at = target
                return True
            if self._moved():
                self._at = None  # it stopped somewhere on the line: do not push on along another path
                return False
            self._log(f"no straight line to ({x:.3f}, {y:.3f}); letting MoveIt choose a path")
        if self._pose(x, y, travel, grip, then_z, speed=s.speed):
            self._over, self._at = True, target
            return True
        if self._moved():
            self._over, self._at = False, None
        return False

    def _descend(self, x: float, y: float, z: float, travel: float, grip: Optional[float] = None) -> bool:
        above, self._at = self._at, None
        self._low = (x, y, travel, grip)
        if self._pose(x, y, z, grip, cartesian=True, speed=self._s.slow_speed):
            return True
        if not self._moved():
            self._low, self._at = None, above  # never started: still at the carrying height
        return False

    def _rise(self) -> bool:
        """Straight up to the carrying height, if the arm is (or may be) below it."""
        if self._low is None:
            return True
        x, y, travel, grip = self._low
        if self._pose(x, y, travel, grip, cartesian=True, speed=self._s.slow_speed):
            self._low, self._over, self._at = None, True, (x, y, travel, grip)
            return True
        return False

    def _park(self) -> bool:
        parked = bool(self._arm.move_to_joints(list(PARK_POSES[self._s.park]), self._s.speed))
        if parked or self._moved():
            self._over, self._at = False, None
        return parked

    def _open(self) -> bool:
        """Open to ``open_position`` and confirm it on the measured finger position."""
        reading = self._arm.open_gripper(self._s.open_position)
        if not self._s.grasp_check:
            return True
        return reading is not None and abs(reading - self._s.open_position) <= FINGER_TOLERANCE

    # -------------------------------------------------------------------- checks
    def _check_heights(self, *surfaces: float) -> Optional[str]:
        for z in surfaces:
            if z + self._s.grip_height < self._s.floor_z + 0.002:
                return (
                    f"refused: a surface at z {z:.4f} would send the fingertips below the table "
                    f"(z {self._s.floor_z:.4f}); check surface_z"
                )
        return None

    def _check_gripper_is_free(self) -> Optional[str]:
        """After a failure that left a piece in the gripper, refuse to move until it was taken out."""
        if self._holding is None:
            return None
        if self._s.grasp_check:
            reading = self._arm.finger_position()
            if reading is not None and reading >= self._s.missed_reading:  # the fingers have closed on nothing
                self._holding = None
                return None
        return (
            f"the gripper still holds the piece from {self._holding} (failed move): take it out, put it "
            "back on its square, then send 'retry'"
        )

    # ------------------------------------------------------------------ failures
    def _fail_empty(self, reason: str) -> str:
        """A failure with nothing in the gripper: get clear of the board if that is safe."""
        if self._rise():
            self._park()
            return f"{reason}; no piece was moved"
        return f"{reason}; no piece was moved, but the arm could not lift and is still low over the board"

    def _fail_holding(self, reason: str) -> str:
        """A failure with the piece in the gripper and no safe way to put it down."""
        return (
            f"{reason}; the arm stopped HOLDING the piece from {self._holding}: take it out of the gripper "
            "and put it back on its square, then send 'retry'"
        )

    def _put_back(
        self, reason: str, pick_name: str, pick_xyz: XYZ, travel: float, grip: Optional[float] = None
    ) -> str:
        """The target could not be reached while carrying: return the piece to where it came from,
        the fingers turned as they were when it was picked up."""
        px, py, pz = pick_xyz
        returned = (
            self._rise()
            and self._transit(px, py, travel, grip, self._s.release_z(pz))
            and self._descend(px, py, self._s.release_z(pz), travel, grip)
            and self._open()
        )
        if not returned:
            return self._fail_holding(reason)
        self._holding = None
        if self._rise():
            self._park()
        return f"{reason}; the piece was put back on {pick_name}"


def _takes(function, keyword: str) -> bool:
    """Does ``function`` accept ``keyword``?"""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return keyword in parameters or any(p.kind is p.VAR_KEYWORD for p in parameters.values())


def _angle(value) -> Optional[float]:
    """A grip direction from ``info``: a finite number of radians, or None."""
    try:
        angle = float(value)
    except (TypeError, ValueError):
        return None
    return angle if math.isfinite(angle) else None


def _same_line(a: Optional[float], b: Optional[float]) -> bool:
    """Do two grip directions describe the same line? (A line has no sign: 0 and pi are one.)"""
    if a is None or b is None:
        return a is None and b is None
    return abs(math.sin(a - b)) < 1e-6


def _degrees(angle: Optional[float]) -> str:
    return "the default" if angle is None else f"{math.degrees(angle) % 180.0:.0f} deg"


# --------------------------------------------------------------------------- the plug-in itself
_lock = threading.Lock()
_runner: Optional[PickPlace] = None


def _spin(executor) -> None:
    try:
        executor.spin()
    except Exception:  # the context was shut down: the process is exiting
        pass


def _start() -> PickPlace:
    """Create the arm node (once per process) and spin it in the background.

    The node lives in the process that called us (arm_adapter_node), which has already called
    rclpy.init(). Its callbacks need their own executor thread: the motion calls below block.
    """
    try:
        from rclpy.executors import MultiThreadedExecutor

        from .kinova_arm import KinovaArm, settings_from_parameters
    except ImportError as exc:
        raise RuntimeError(
            f"the Kinova arm needs MoveIt and pymoveit2 (run inside the course Kinova container): {exc}"
        ) from exc

    arm = KinovaArm("chess_kinova_arm")
    try:
        settings = settings_from_parameters(arm)  # ValueError if a parameter is out of range
    except Exception:
        arm.destroy_node()  # do not leave a half-made node behind for the next attempt
        raise
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(arm)
    threading.Thread(target=_spin, args=(executor,), daemon=True, name="kinova-arm-spin").start()
    arm.get_logger().info(f"kinova_executor: {settings}")
    return PickPlace(arm, settings, log=arm.get_logger().info)


def _get_runner() -> PickPlace:
    global _runner
    with _lock:
        if _runner is None:
            _runner = _start()
        return _runner


def prepare() -> Optional[str]:
    """Bring the arm up before the first move (arm_adapter_node calls this once at start-up).

    MOVES THE ARM to its WORK pose. Returns None when ready, else the reason.
    """
    return _get_runner().prepare()


def operator(command: str) -> None:
    """Operator commands from chessbot/cmd (arm_adapter_node forwards them).

    ``retry`` and ``reset`` are the operator saying the table is in order again, which includes
    an emptied gripper after a move that failed while holding a piece.
    """
    if str(command).strip().lower() in ("retry", "reset") and _runner is not None:
        _runner.operator_cleared()


def pick_and_place(pick_xyz, place_xyz, info):
    """The executor of arm_adapter_node: see arm_adapter.py for the contract."""
    return _get_runner()(pick_xyz, place_xyz, info)


pick_and_place.prepare = prepare
pick_and_place.operator = operator
