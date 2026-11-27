"""Tests for the Kinova pick-and-place sequence (pure logic, simulated arm) and the reach map.

``FakeArm`` is a small simulation: it knows where the gripper is, what the fingers hold and where
the pieces stand, and records a *violation* whenever the sequence does something that would be
unsafe on the real arm (moving sideways among the pieces, opening while carrying, ...).
"""

import math

import chess
import pytest

from chess_brain import kinova_executor
from chess_brain.actions import actions_payload, move_to_actions
from chess_brain.arm_adapter import PlacedButFailed, run_command
from chess_brain.board_frame import BoardFrame, Locations
from chess_brain.kinova_executor import RETRACT, WORK, PickPlace, Settings
from chess_brain.reach_check import GAME_DISCARDS, format_reach, kinematic_reach, reach_table

SURFACE = -0.056
S = Settings()
TRAVEL = S.travel_z(SURFACE)
GRIP = S.grip_z(SURFACE)
RELEASE = S.release_z(SURFACE)
PICK = (0.30, -0.10, SURFACE)
PLACE = (0.30, 0.00, SURFACE)
INFO = {"id": 1, "move": "e2e4", "index": 0, "count": 1, "last": True, "pick": "e2", "place": "e4"}


def key(x, y):
    return (round(x, 4), round(y, 4))


class FakeArm:
    """A simulated arm. ``fail_pose`` maps the 0-based index of a move_to_pose call to "plan"
    (nothing moves) or "exec" (the arm stops halfway); ``blocked`` is a predicate on
    (x, y, z, cartesian) for poses that can never be planned."""

    def __init__(self, pieces=(PICK,), fail_pose=None, fail_joints=(), blocked=None, ready=True,
                 dead_opens=(), dead_closes=(), safe_z=TRAVEL, low_z=RELEASE + 0.001):
        self.calls = []
        self.grips = []                 # (x, y, z, cartesian, grip, then_z) of every pose move
        self.grip = None                # the line the fingers open along now (None = the default)
        self.violations = []
        self.pieces = {key(p[0], p[1]) for p in pieces}   # where pieces stand
        self.position = None            # (x, y, z) of the gripper; None = at a joint pose, clear of the board
        self.fingers = 0.0
        self.held = False
        self.ready = ready
        self.fail_pose = dict(fail_pose or {})
        self.fail_joints = set(fail_joints)
        self.blocked = blocked or (lambda x, y, z, cartesian: False)
        self.dead_opens, self.dead_closes = set(dead_opens), set(dead_closes)
        self.safe_z, self.low_z = safe_z, low_z
        self._poses = self._joints = self._opens = self._closes = 0
        self._started = False
        self._no_line_to = None         # target of a straight line that just failed
        self.raise_on_pose = None       # index of a move_to_pose call that raises

    # -- the arm interface
    def wait_until_ready(self):
        self.calls.append(("ready",))
        return self.ready

    def add_table(self):
        self.calls.append(("table",))

    def last_move_started(self):
        return self._started

    def move_to_joints(self, joints, speed=0.1):
        self.calls.append(("joints", tuple(joints), speed))
        index, self._joints = self._joints, self._joints + 1
        self._started = False
        if index in self.fail_joints:
            return False
        if self.position is not None and self.position[2] < self.safe_z - 1e-9:
            self.violations.append(f"left for a joint pose from z {self.position[2]:.3f}, among the pieces")
        self._started = True
        self.position = None
        return True

    def move_to_pose(self, x, y, z, cartesian=False, speed=0.1, grip=None, then_z=None):
        self.calls.append(("pose", round(x, 6), round(y, 6), round(z, 6), cartesian, speed))
        self.grips.append((round(x, 6), round(y, 6), round(z, 6), cartesian, grip, then_z))
        index, self._poses = self._poses, self._poses + 1
        self._started = False
        if self.raise_on_pose == index:
            raise RuntimeError("controller died")
        mode = self.fail_pose.get(index) or ("plan" if self.blocked(x, y, z, cartesian) else None)
        no_line_to, self._no_line_to = self._no_line_to, None
        if self.position is not None and self.position == (x, y, z) and grip == self.grip:
            self.violations.append("zero-length motion requested")
        if mode == "plan":
            if cartesian:
                self._no_line_to = (key(x, y), round(z, 6))
            return False
        if self.position is not None:
            cx, cy, cz = self.position
            sideways = key(cx, cy) != key(x, y)
            if sideways and (cz < self.safe_z - 1e-9 or z < self.safe_z - 1e-9):
                self.violations.append(f"sideways move below the carrying height (z {cz:.3f} -> {z:.3f})")
            if sideways and not cartesian and no_line_to != (key(x, y), round(z, 6)):
                self.violations.append("free-space move over the board without trying a straight line first")
        elif cartesian:
            self.violations.append("straight-line move requested from a joint pose")
        self._started = True
        if mode == "exec":
            start = self.position or (x, y, self.safe_z)
            self.position = tuple((a + b) / 2.0 for a, b in zip(start, (x, y, z)))
            return False
        if z < self.safe_z - 1e-9 and self.position is not None and grip != self.grip:
            self.violations.append("the gripper was turned while moving down among the pieces")
        self.position = (x, y, z)
        self.grip = grip
        return True

    def open_gripper(self, position=0.0):
        self.calls.append(("open", position))
        index, self._opens = self._opens, self._opens + 1
        if index in self.dead_opens:
            return self.fingers
        if self.held:
            if self.position is None or self.position[2] > self.low_z:
                self.violations.append("opened the gripper while carrying a piece")
            self.held = False
            if self.position is not None:
                self.pieces.add(key(self.position[0], self.position[1]))
        self.fingers = position
        return position

    def close_gripper(self):
        self.calls.append(("close",))
        index, self._closes = self._closes, self._closes + 1
        if index in self.dead_closes:
            return self.fingers
        here = self.position is not None and self.position[2] <= self.low_z and key(*self.position[:2])
        if here and here in self.pieces and self.fingers < 0.6:
            self.pieces.discard(here)
            self.held = True
            self.fingers = 0.60
        else:
            self.fingers = 0.80
        return self.fingers

    def finger_position(self):
        return self.fingers

    # -- helpers for the tests
    def kinds(self):
        return [call[0] for call in self.calls]

    def poses(self):
        return [call for call in self.calls if call[0] == "pose"]

    def take_piece_out(self):
        """The operator pulls the piece out of the closed gripper: the fingers close on nothing."""
        self.held = False
        self.fingers = 0.80


def pose(xyz, z, cartesian, speed):
    return ("pose", round(xyz[0], 6), round(xyz[1], 6), round(z, 6), cartesian, speed)


def at(xyz, cartesian=None):
    """Predicate for FakeArm(blocked=...): poses above/at this square (optionally only one kind)."""
    def blocked(x, y, z, is_cartesian):
        return key(x, y) == key(xyz[0], xyz[1]) and (cartesian is None or cartesian == is_cartesian)
    return blocked


PREPARE = [("ready",), ("table",), ("joints", WORK, S.speed)]
PARK = ("joints", WORK, S.speed)
HAPPY = PREPARE + [
    pose(PICK, TRAVEL, False, S.speed),        # from the parking pose: MoveIt chooses the path
    ("open", S.open_position),
    pose(PICK, GRIP, True, S.slow_speed),
    ("close",),
    pose(PICK, TRAVEL, True, S.slow_speed),
    pose(PLACE, TRAVEL, True, S.speed),        # over the board: a straight line
    pose(PLACE, RELEASE, True, S.slow_speed),
    ("open", S.open_position),
    pose(PLACE, TRAVEL, True, S.slow_speed),
    PARK,
]


# ------------------------------------------------------------------- settings
def test_heights_are_measured_from_the_board_surface():
    assert math.isclose(TRAVEL, SURFACE + 0.157 + 0.150)
    assert math.isclose(GRIP, SURFACE + 0.157 + 0.020)
    assert math.isclose(RELEASE, GRIP + 0.003)
    assert GRIP - 0.157 > SURFACE          # the fingertips stay above the board


@pytest.mark.parametrize(
    "bad",
    [
        {"grip_height": 0.0},
        {"grip_height": -0.01},
        {"travel_clearance": 0.03},
        {"place_clearance": -0.001},
        {"speed": 0.0},
        {"speed": 0.5},                      # above the lab limit
        {"speed": 1.5, "max_speed": 1.0},
        {"slow_speed": 0.0},
        {"slow_speed": 0.2, "speed": 0.1},   # the descents may not be the fast part
        {"max_speed": 2.0},
        {"open_position": 0.8},
        {"open_position": -0.1},
        {"missed_reading": 0.4},
        {"missed_reading": 0.52},            # too close to open_position to tell apart
        {"missed_reading": 0.9},
        {"park": "home"},
        {"gripper_length": 0.0},
    ],
)
def test_settings_reject_unsafe_values(bad):
    with pytest.raises(ValueError):
        Settings(**bad)


def test_speed_limit_is_the_lab_limit_unless_raised_for_simulation():
    assert S.speed <= 0.2 and S.slow_speed <= S.speed and S.max_speed == 0.2
    fast = Settings(speed=0.8, slow_speed=0.3, max_speed=1.0)
    assert fast.speed == 0.8


# ----------------------------------------------------------------- happy path
def test_one_action_runs_the_lab6_sequence_and_parks():
    arm = FakeArm()
    assert PickPlace(arm, S)(PICK, PLACE, INFO) is None
    assert arm.calls == HAPPY
    assert arm.violations == []
    assert arm.pieces == {key(*PLACE[:2])} and not arm.held


def test_no_motion_is_faster_than_the_configured_speed():
    arm = FakeArm()
    PickPlace(arm, S)(PICK, PLACE, INFO)
    speeds = [call[-1] for call in arm.calls if call[0] in ("pose", "joints")]
    assert speeds and max(speeds) <= S.speed
    assert all(call[-1] == S.slow_speed for call in arm.poses() if call[3] < TRAVEL)   # every descent


def test_the_gripper_never_opens_fully_over_the_board():
    arm = FakeArm()
    PickPlace(arm, S)(PICK, PLACE, INFO)
    assert [call[1] for call in arm.calls if call[0] == "open"] == [S.open_position, S.open_position]


def test_arm_parks_only_after_the_last_action():
    arm = FakeArm(pieces=(PICK, (0.40, 0.10)))
    runner = PickPlace(arm, S)
    assert runner(PICK, PLACE, dict(INFO, index=0, count=2, last=False)) is None
    assert arm.calls[-1][0] == "pose"                       # still above the board
    second = len(arm.calls)
    assert runner((0.40, 0.10, SURFACE), PICK, dict(INFO, index=1, count=2, last=True)) is None
    assert arm.calls[second] == pose((0.40, 0.10), TRAVEL, True, S.speed)   # straight there, no detour
    assert arm.calls[-1] == PARK
    assert arm.kinds().count("ready") == 1                  # prepared once
    assert arm.violations == []


def test_park_pose_can_be_retract():
    arm = FakeArm()
    assert PickPlace(arm, Settings(park="retract"))(PICK, PLACE, INFO) is None
    assert arm.calls[-1] == ("joints", RETRACT, S.speed)


def test_carrying_height_follows_the_higher_surface():
    high = (0.20, 0.20, SURFACE + 0.03)                     # e.g. a raised tray for captured pieces
    travel = S.travel_z(SURFACE + 0.03)
    arm = FakeArm(safe_z=travel, low_z=S.release_z(SURFACE + 0.03) + 0.001)
    assert PickPlace(arm, S)(PICK, high, INFO) is None
    poses = arm.poses()
    assert poses[0] == pose(PICK, travel, False, S.speed)
    assert poses[1] == pose(PICK, GRIP, True, S.slow_speed)
    assert poses[4] == pose(high, S.release_z(SURFACE + 0.03), True, S.slow_speed)
    assert arm.violations == []


def test_straight_line_falls_back_to_a_planned_path():
    arm = FakeArm(blocked=at(PLACE, cartesian=True))        # no straight line to above the target ...
    runner = PickPlace(arm, S)
    reason = runner(PICK, PLACE, INFO)
    # ... so MoveIt may choose the path there; the straight descent is impossible too, so it is put back
    assert pose(PLACE, TRAVEL, False, S.speed) in arm.calls
    assert "descend onto e4" in reason and "put back on e2" in reason
    assert arm.pieces == {key(*PICK[:2])} and not arm.held


# -------------------------------------------------------------------- prepare
def test_prepare_failure_is_reported_and_retried():
    arm = FakeArm(ready=False)
    runner = PickPlace(arm, S)
    reason = runner(PICK, PLACE, INFO)
    assert "MoveIt is not ready" in reason
    assert arm.kinds() == ["ready"]                         # nothing moved
    arm.ready = True
    assert runner(PICK, PLACE, INFO) is None


def test_prepare_fails_when_work_pose_is_unreachable():
    arm = FakeArm(fail_joints={0})
    reason = PickPlace(arm, S).prepare()
    assert "WORK" in reason
    assert "pose" not in arm.kinds()


def test_prepare_is_idempotent():
    arm = FakeArm()
    runner = PickPlace(arm, S)
    assert runner.prepare() is None and runner.prepare() is None
    assert arm.calls == PREPARE


def test_surface_below_the_table_is_refused_before_anything_moves():
    arm = FakeArm()
    reason = PickPlace(arm, S)((0.3, -0.1, -0.156), PLACE, INFO)   # a typo for -0.056
    assert "below the table" in reason
    assert arm.calls == []
    reason = PickPlace(arm, S)(PICK, (0.3, 0.0, -0.09), INFO)
    assert "below the table" in reason and arm.calls == []


# ------------------------------------------------------- failures while empty
def test_unreachable_pick_square_moves_nothing():
    arm = FakeArm(blocked=at(PICK))
    reason = PickPlace(arm, S)(PICK, PLACE, INFO)
    assert "above e2" in reason and "no piece was moved" in reason
    assert "close" not in arm.kinds() and "open" not in arm.kinds()
    assert arm.calls[-1] == PARK and arm.violations == []


def test_descent_that_cannot_be_planned_parks_without_a_lift():
    arm = FakeArm(fail_pose={1: "plan"})
    reason = PickPlace(arm, S)(PICK, PLACE, INFO)
    assert "descend onto e2" in reason and "no piece was moved" in reason
    assert "close" not in arm.kinds()
    assert arm.calls[-2:] == [pose(PICK, GRIP, True, S.slow_speed), PARK]
    assert arm.violations == []


def test_descent_that_stops_halfway_lifts_straight_up_before_parking():
    arm = FakeArm(fail_pose={1: "exec"})
    reason = PickPlace(arm, S)(PICK, PLACE, INFO)
    assert "descend onto e2" in reason and "no piece was moved" in reason
    assert arm.calls[-2:] == [pose(PICK, TRAVEL, True, S.slow_speed), PARK]
    assert arm.violations == []


def test_arm_stuck_low_stays_put_and_lifts_first_next_time():
    arm = FakeArm(fail_pose={1: "exec", 2: "plan"})         # stops halfway down, then cannot lift
    runner = PickPlace(arm, S)
    reason = runner(PICK, PLACE, INFO)
    assert "still low over the board" in reason
    assert arm.calls[-1][0] == "pose"                       # it did NOT park from among the pieces
    count = len(arm.calls)
    assert runner(PICK, PLACE, INFO) is None
    assert arm.calls[count] == pose(PICK, TRAVEL, True, S.slow_speed)   # first: straight up
    assert arm.violations == []


def test_gripper_that_does_not_open_stops_the_pick_before_the_descent():
    arm = FakeArm(dead_opens={0})
    arm.fingers = 0.80                                      # left closed, and the open is ignored
    reason = PickPlace(arm, S)(PICK, PLACE, INFO)
    assert "did not open" in reason and "no piece was moved" in reason
    assert len(arm.poses()) == 1                            # only the move to above the square
    assert arm.pieces == {key(*PICK[:2])}


def test_missed_grasp_is_a_failure_and_releases():
    arm = FakeArm(pieces=())                                # nothing stands on the pick square
    reason = PickPlace(arm, S)(PICK, PLACE, INFO)
    assert "grasp missed at e2" in reason and "0.80" in reason
    after_close = arm.calls[arm.kinds().index("close") + 1:]
    assert after_close == [("open", S.open_position), pose(PICK, TRAVEL, True, S.slow_speed), PARK]


def test_missed_grasp_is_ignored_when_the_check_is_off():
    arm = FakeArm(pieces=())                                # Gazebo without pieces
    assert PickPlace(arm, Settings(grasp_check=False))(PICK, PLACE, INFO) is None
    assert arm.calls == HAPPY


def test_gripper_that_does_not_close_is_a_failure():
    arm = FakeArm(dead_closes={0})                          # the reading stays at the open position
    reason = PickPlace(arm, S)(PICK, PLACE, INFO)
    assert "did not close at e2" in reason and "no piece was moved" in reason
    assert pose(PLACE, TRAVEL, True, S.speed) not in arm.calls
    assert arm.pieces == {key(*PICK[:2])} and arm.violations == []


def test_no_finger_reading_is_a_failure():
    arm = FakeArm()
    arm.close_gripper = lambda: None
    reason = PickPlace(arm, S)(PICK, PLACE, INFO)
    assert "finger position" in reason
    assert pose(PLACE, TRAVEL, True, S.speed) not in arm.calls


# ----------------------------------------------------- failures while holding
def test_unreachable_target_puts_the_piece_back():
    arm = FakeArm(blocked=at(PLACE))
    reason = PickPlace(arm, S)(PICK, PLACE, INFO)
    assert "above e4" in reason and "put back on e2" in reason
    after = arm.calls[arm.calls.index(pose(PLACE, TRAVEL, False, S.speed)) + 1:]
    assert after == [                                       # it is still above e2: straight back down
        pose(PICK, RELEASE, True, S.slow_speed),
        ("open", S.open_position),
        pose(PICK, TRAVEL, True, S.slow_speed),
        PARK,
    ]
    assert arm.pieces == {key(*PICK[:2])} and not arm.held and arm.violations == []


def test_descent_onto_target_stopping_halfway_lifts_before_going_back():
    arm = FakeArm(fail_pose={4: "exec"})                    # 4 = the descent onto the target
    reason = PickPlace(arm, S)(PICK, PLACE, INFO)
    assert "descend onto e4" in reason and "put back on e2" in reason
    failed = arm.calls.index(pose(PLACE, RELEASE, True, S.slow_speed))
    assert arm.calls[failed + 1] == pose(PLACE, TRAVEL, True, S.slow_speed)   # straight up first
    assert arm.pieces == {key(*PICK[:2])} and arm.violations == []


def test_piece_stays_in_the_gripper_when_it_cannot_be_put_back():
    def blocked(x, y, z, cartesian):                        # the target, and the descent back onto e2
        return key(x, y) == key(*PLACE[:2]) or (key(x, y) == key(*PICK[:2]) and abs(z - RELEASE) < 1e-9)

    arm = FakeArm(blocked=blocked)
    runner = PickPlace(arm, S)
    reason = runner(PICK, PLACE, INFO)
    assert "HOLDING the piece from e2" in reason and "retry" in reason
    assert arm.held and arm.kinds().count("open") == 1      # never opened after the grasp
    count = len(arm.calls)

    # The next command is refused while the fingers are still closed on something ...
    reason = runner(PICK, PLACE, INFO)
    assert "still holds the piece from e2" in reason
    assert len(arm.calls) == count                          # and nothing moved

    # ... and accepted once the operator has taken the piece out and put it back.
    arm.take_piece_out()
    arm.pieces.add(key(*PICK[:2]))
    arm.blocked = lambda x, y, z, cartesian: False          # and whatever was in the way is gone
    assert runner(PICK, PLACE, INFO) is None
    assert arm.pieces == {key(*PLACE[:2])} and arm.violations == []


def test_failed_lift_after_the_grasp_keeps_holding():
    arm = FakeArm(fail_pose={2: "plan"})
    reason = PickPlace(arm, S)(PICK, PLACE, INFO)
    assert "could not lift the piece from e2" in reason and "HOLDING" in reason
    assert arm.held and arm.kinds().count("open") == 1


def test_without_the_finger_check_only_the_operator_clears_a_held_piece():
    # The brain retries a failed move by itself when the board still looks untouched. With the
    # finger check off the arm cannot see that it is still holding, so it must not believe a retry.
    arm = FakeArm(fail_pose={2: "plan"})
    runner = PickPlace(arm, Settings(grasp_check=False))
    assert "HOLDING" in runner(PICK, PLACE, INFO)
    count = len(arm.calls)
    assert "still holds" in runner(PICK, PLACE, INFO)       # the automatic retry
    assert len(arm.calls) == count and arm.held
    arm.take_piece_out()
    arm.pieces.add(key(*PICK[:2]))
    runner.operator_cleared()                               # 'retry' typed by the operator
    assert runner(PICK, PLACE, INFO) is None
    assert arm.violations == []


def test_an_exception_while_carrying_is_remembered_as_holding():
    arm = FakeArm()
    arm.raise_on_pose = 3                                   # the move to above the target raises
    runner = PickPlace(arm, S)
    with pytest.raises(RuntimeError):
        runner(PICK, PLACE, INFO)
    arm.raise_on_pose = None
    count = len(arm.calls)
    assert "still holds the piece from e2" in runner(PICK, PLACE, INFO)
    assert len(arm.calls) == count and arm.held             # the piece was not dropped by the next command


def test_piece_lost_on_the_way_is_reported():
    arm = FakeArm()
    original = arm.move_to_pose

    def slipping(x, y, z, cartesian=False, speed=0.1):
        result = original(x, y, z, cartesian=cartesian, speed=speed)
        if key(x, y) == key(*PLACE[:2]) and arm.held:
            arm.take_piece_out()                            # it slips out during the transit
        return result

    arm.move_to_pose = slipping
    reason = PickPlace(arm, S)(PICK, PLACE, INFO)
    assert "lost on the way to e4" in reason
    assert arm.calls[-1] == PARK
    assert pose(PLACE, RELEASE, True, S.slow_speed) not in arm.calls


def test_release_that_is_not_confirmed_is_not_reported_as_success():
    arm = FakeArm(dead_opens={1})                           # the open over the target is ignored
    runner = PickPlace(arm, S)
    reason = runner(PICK, PLACE, INFO)
    assert "did not open over e4" in reason and "HOLDING" in reason
    assert arm.held and not isinstance(reason, PlacedButFailed)
    assert arm.calls[-1] == ("open", S.open_position)       # it did not lift the piece away again


# ------------------------------------------------------ failures after placing
def test_failed_lift_after_placing_says_the_piece_arrived():
    arm = FakeArm(fail_pose={5: "plan"})
    reason = PickPlace(arm, S)(PICK, PLACE, INFO)
    assert isinstance(reason, PlacedButFailed) and reason.placed
    assert "the piece is on e4" in reason and "lift away" in reason
    assert arm.calls[-1] != PARK                            # no sideways move from between the pieces
    assert arm.pieces == {key(*PLACE[:2])} and arm.violations == []


def test_failed_parking_says_the_piece_arrived():
    arm = FakeArm(fail_joints={1})                          # 0 = WORK at start-up, 1 = parking
    reason = PickPlace(arm, S)(PICK, PLACE, INFO)
    assert isinstance(reason, PlacedButFailed)
    assert "the piece is on e4" in reason and "park" in reason


# ------------------------------------------------- safety, whatever goes wrong
@pytest.mark.parametrize("index,mode", [(i, m) for i in range(12) for m in ("plan", "exec")])
def test_any_single_motion_failure_is_safe(index, mode):
    arm = FakeArm(fail_pose={index: mode})
    runner = PickPlace(arm, S)
    result = runner(PICK, PLACE, INFO)
    assert arm.violations == []
    if result is None or isinstance(result, PlacedButFailed):
        assert arm.pieces == {key(*PLACE[:2])} and not arm.held
    elif arm.held:
        assert "HOLDING" in result                          # never carried silently
    else:
        assert arm.pieces == {key(*PICK[:2])}               # otherwise the board is as it was
        assert "no piece was moved" in result or "put back on e2" in result


@pytest.mark.parametrize("first,second", [(a, b) for a in range(8) for b in range(8)])
def test_any_two_motion_failures_are_safe(first, second):
    arm = FakeArm(fail_pose={first: "exec", first + 1 + second: "plan"})
    result = PickPlace(arm, S)(PICK, PLACE, INFO)
    assert arm.violations == []
    assert len(arm.pieces) + (1 if arm.held else 0) == 1    # the piece is somewhere sensible ...
    assert arm.pieces <= {key(*PICK[:2]), key(*PLACE[:2])}  # ... on its square, the target, or in the gripper
    if arm.held:
        assert result is not None and "HOLDING" in result


@pytest.mark.parametrize("dead_open,dead_close", [(o, c) for o in (None, 0, 1, 2) for c in (None, 0)])
def test_gripper_faults_never_look_like_success(dead_open, dead_close):
    arm = FakeArm(dead_opens=() if dead_open is None else {dead_open},
                  dead_closes=() if dead_close is None else {dead_close})
    result = PickPlace(arm, S)(PICK, PLACE, INFO)
    assert arm.violations == []
    if result is None:
        assert arm.pieces == {key(*PLACE[:2])} and not arm.held


# ------------------------------------------------------- with the arm adapter
def test_a_capture_runs_through_the_arm_adapter():
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    move = chess.Move.from_uci("e4d5")
    payload = actions_payload(7, move, move_to_actions(board, move))
    locations = Locations(frame=BoardFrame(origin_x=0.60, origin_y=-0.20, yaw=math.pi / 2, surface_z=SURFACE))
    d5, e4 = locations.frame.world_xyz("d5"), locations.frame.world_xyz("e4")
    slot = locations.discard_slot_xyz(0)
    arm = FakeArm(pieces=(d5, e4))

    reply = run_command(payload, locations, PickPlace(arm, S))

    assert reply == {"id": 7, "status": "ok"}
    assert locations.discarded == 1
    assert arm.kinds().count("close") == 2 and arm.kinds().count("joints") == 2   # WORK, then one park
    poses = arm.poses()
    assert poses[0] == pose(d5, TRAVEL, False, S.speed)     # the captured pawn goes first ...
    assert poses[3] == pose(slot, TRAVEL, True, S.speed)    # ... to the discard column
    assert poses[-1] == pose(d5, TRAVEL, True, S.slow_speed)
    assert arm.pieces == {key(*slot[:2]), key(*d5[:2])} and arm.violations == []


def test_a_failure_reaches_arm_done_with_its_reason():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    payload = actions_payload(3, move, move_to_actions(board, move))
    arm = FakeArm(pieces=())
    reply = run_command(payload, Locations(frame=BoardFrame(surface_z=SURFACE)), PickPlace(arm, S))
    assert reply["status"] == "failed" and reply["failed_action_index"] == 0
    assert "grasp missed at e2" in reply["reason"]


def test_a_discard_slot_filled_by_a_failed_action_is_not_reused():
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    move = chess.Move.from_uci("e4d5")
    payload = actions_payload(8, move, move_to_actions(board, move))
    locations = Locations(frame=BoardFrame(origin_x=0.60, origin_y=-0.20, yaw=math.pi / 2, surface_z=SURFACE))
    d5, e4 = locations.frame.world_xyz("d5"), locations.frame.world_xyz("e4")
    arm = FakeArm(pieces=(d5, e4), fail_pose={5: "plan"})   # cannot lift away from the discard slot

    reply = run_command(payload, locations, PickPlace(arm, S))

    assert reply["status"] == "failed" and reply["failed_action_index"] == 0
    assert "the piece is on DISCARD" in reply["reason"]
    assert locations.discarded == 1                          # the next capture gets the next slot


def test_offboard_spots_can_lie_below_the_playing_surface():
    locations = Locations(frame=BoardFrame(surface_z=SURFACE), offboard_drop=0.0066)
    assert math.isclose(locations.discard_slot_xyz(0)[2], SURFACE - 0.0066)
    assert math.isclose(locations.place_xyz("DISCARD")[2], SURFACE - 0.0066)
    assert math.isclose(locations.pick_xyz("SPARE_Q_W")[2], SURFACE - 0.0066)
    assert math.isclose(locations.pick_xyz("e2")[2], SURFACE)
    assert math.isclose(Locations().discard_slot_xyz(0)[2], BoardFrame().surface_z)   # default: level


def test_plugin_exposes_its_hooks():
    assert callable(kinova_executor.pick_and_place)
    assert kinova_executor.pick_and_place.prepare is kinova_executor.prepare
    assert kinova_executor.pick_and_place.operator is kinova_executor.operator
    kinova_executor.operator("retry")                       # harmless before the arm exists


# ------------------------------------------------------------------ reach map
def test_reach_map_marks_far_squares():
    locations = Locations(frame=BoardFrame(origin_x=0.20, origin_y=-0.20, surface_z=SURFACE))

    def can_reach(x, y, z):  # a made-up arm: the lower it has to go, the shorter its reach
        return math.hypot(x, y) <= (0.50 if z < 0.15 else 0.58)

    table = reach_table(locations, S.travel_z, S.grip_z, can_reach)
    assert table["a1"] == (True, True)
    assert table["h1"] == (False, False)                    # x = 0.575, y = -0.175
    assert table["g4"] == (True, False)                     # x = 0.525, y = -0.025
    assert set(table) >= {"DISCARD[0]", "DISCARD[7]", "SPARE_Q_W", "SPARE_Q_B"}

    lines = format_reach(table)
    assert lines[0].startswith("  8  ") and lines[7].startswith("  1  ")
    assert lines[7].split()[1:] == ["#", "#", "#", "#", "#", "^", "^", "."]
    assert any("not fully reachable" in line and "h1" in line for line in lines)
    assert any(line.strip().startswith("off-board:") for line in lines)


def test_reach_map_of_a_fully_reachable_board():
    table = reach_table(Locations(), S.travel_z, S.grip_z, lambda x, y, z: True)
    lines = format_reach(table)
    assert "squares fully reachable: 64 of 64" in lines[9]
    assert not any("not fully reachable" in line for line in lines)


def test_the_reach_map_covers_the_captured_pieces_of_a_whole_game():
    # a game can capture every piece but the two kings; the 17th piece starts the third column
    assert GAME_DISCARDS == 30
    table = reach_table(Locations(), S.travel_z, S.grip_z, lambda x, y, z: True)
    assert [name for name in table if name.startswith("DISCARD[")] == [f"DISCARD[{i}]" for i in range(30)]
    lines = format_reach(table)
    assert any("######## ######## ######## ######" in line for line in lines)
    assert any("room for all 30 captured pieces" in line for line in lines)
    assert not any("DISCARD[" in line for line in lines)        # nothing to complain about
    assert any(line.strip().startswith("off-board:") and "SPARE_Q_W #" in line for line in lines)


def test_the_reach_map_says_how_many_captured_pieces_fit():
    locations = Locations()
    out = {locations.discard_slot_xyz(i)[:2] for i in (16, 24, 25)}   # the far corner of the area
    half = locations.discard_slot_xyz(26)[:2]                         # carrying height only

    def can_reach(x, y, z):
        if (x, y) in out:
            return False
        return not ((x, y) == half and z < S.travel_z(SURFACE) - 0.01)

    table = reach_table(locations, S.travel_z, S.grip_z, can_reach)
    lines = format_reach(table, locations.discard_per_column)
    assert any("######## ######## .####### ..^###" in line for line in lines)
    line = next(line for line in lines if "room for" in line)
    # the places are filled in order, so the first one out of reach ends it: 16 pieces fit
    assert "room for 16 captured pieces of the 30 places checked" in line
    assert line.endswith("not fully reachable: DISCARD[16] DISCARD[24] DISCARD[25] DISCARD[26]")


def test_the_reach_map_checks_as_many_places_as_asked():
    table = reach_table(Locations(), S.travel_z, S.grip_z, lambda x, y, z: True, 8)
    assert "DISCARD[7]" in table and "DISCARD[8]" not in table
    assert any("room for all 8 captured pieces" in line for line in format_reach(table))
    none = reach_table(Locations(), S.travel_z, S.grip_z, lambda x, y, z: True, 0)
    assert not any("captured" in line for line in format_reach(none))


def _slot_reachable(locations, index):
    x, y, z = locations.discard_slot_xyz(index)
    return kinematic_reach(x, y, S.travel_z(z)) and kinematic_reach(x, y, S.grip_z(z))


def test_where_the_captured_pieces_of_a_40_mm_board_can_go():
    """The 40 mm board that fits the Gen3 Lite (board_x 0.492, board_y -0.16, turned 90 degrees).
    With the first place beside rank 2 the far corner of the area is beyond the arm's reach from
    the 17th piece on; with it beside rank 4 a whole game fits. By the kinematics alone."""
    frame = BoardFrame(origin_x=0.492, origin_y=-0.16, yaw=math.radians(90.0), surface_z=SURFACE, square=0.04)
    far = Locations(frame=frame, discard_first=(-0.06, 0.06), offboard_drop=0.0066)
    assert [_slot_reachable(far, i) for i in (15, 16, 17, 24, 25, 26)] == [True, False, True, False, False, True]
    near = Locations(frame=frame, discard_first=(-0.05, 0.14), offboard_drop=0.0066)
    assert all(_slot_reachable(near, i) for i in (0, 7, 15, 16, 23, 24, 25, 29))



# ------------------------------------------------------------------ grip directions
GRIP_INFO = dict(INFO, pick_grip=math.radians(45.0), place_grip=math.radians(135.0), grip_clearance=0.011)


def test_the_grip_direction_is_used_at_the_pick_and_at_the_place():
    arm = FakeArm()
    assert PickPlace(arm, S)(PICK, PLACE, GRIP_INFO) is None
    assert arm.violations == []
    at_pick = [g for g in arm.grips if (g[0], g[1]) == PICK[:2]]
    at_place = [g for g in arm.grips if (g[0], g[1]) == PLACE[:2]]
    assert [g[4] for g in at_pick] == [GRIP_INFO["pick_grip"]] * 3      # above, down, up
    assert [g[4] for g in at_place] == [GRIP_INFO["place_grip"]] * 3    # above (turning on the way), down, up
    # the arm is told how far down it will go next, so it can pick a yaw that reaches both heights
    assert at_pick[0][5] == pytest.approx(GRIP) and at_place[0][5] == pytest.approx(RELEASE)
    assert at_pick[1][5] is None and at_pick[2][5] is None


def test_without_a_grip_direction_the_arm_is_called_as_before():
    class OldArm(FakeArm):   # an arm written before grip directions existed
        def move_to_pose(self, x, y, z, cartesian=False, speed=0.1):
            return super().move_to_pose(x, y, z, cartesian, speed)

    arm = OldArm()
    assert PickPlace(arm, S)(PICK, PLACE, INFO) is None
    assert all(g[4] is None for g in arm.grips)


def test_an_arm_that_cannot_turn_its_gripper_makes_the_move_its_own_way():
    class OldArm(FakeArm):   # no ``grip`` keyword, and the command carries grip directions
        def move_to_pose(self, x, y, z, cartesian=False, speed=0.1):
            return super().move_to_pose(x, y, z, cartesian, speed)

    arm = OldArm(pieces=(PICK, PLACE))
    runner = PickPlace(arm, S)
    assert runner(PICK, (0.30, 0.10, SURFACE), dict(GRIP_INFO, last=False)) is None
    # the next action starts where this one ended and names another direction: nothing to turn,
    # so no move on the spot either
    second = dict(INFO, pick="e4", place="e6", pick_grip=math.pi / 2, place_grip=math.pi / 2)
    assert runner((0.30, 0.10, SURFACE), (0.40, 0.10, SURFACE), second) is None
    assert arm.violations == [] and all(g[4] is None for g in arm.grips)


def test_a_grip_direction_that_is_not_a_number_is_ignored():
    arm = FakeArm()
    assert PickPlace(arm, S)(PICK, PLACE, dict(INFO, pick_grip="north", place_grip=float("nan"))) is None
    assert all(g[4] is None for g in arm.grips)


def test_a_failed_carry_puts_the_piece_back_with_the_grip_it_was_picked_with():
    # the place can never be reached: the piece goes back to its square, fingers as at the pick
    arm = FakeArm(blocked=lambda x, y, z, cartesian: key(x, y) == key(*PLACE[:2]))
    reason = PickPlace(arm, S)(PICK, PLACE, GRIP_INFO)
    assert "put back on e2" in reason and arm.violations == []
    back = [g for g in arm.grips if (g[0], g[1]) == PICK[:2]][3:]
    assert back and all(g[4] == GRIP_INFO["pick_grip"] for g in back)


def test_turning_on_the_spot_is_a_move_not_a_shortcut():
    # second action starts where the first ended, but wants the fingers along another line
    arm = FakeArm(pieces=(PICK, PLACE))
    runner = PickPlace(arm, S)
    first = dict(INFO, last=False, pick_grip=0.0, place_grip=0.0)
    assert runner(PICK, (0.30, 0.10, SURFACE), first) is None
    second = dict(INFO, pick="e4", place="e6", pick_grip=math.pi / 2, place_grip=math.pi / 2)
    assert runner((0.30, 0.10, SURFACE), (0.40, 0.10, SURFACE), second) is None
    assert arm.violations == []
    turn = [g for g in arm.grips if g[4] == math.pi / 2][0]
    assert (turn[0], turn[1], turn[2]) == (0.30, 0.10, round(TRAVEL, 6))   # turned at the carrying height


def test_the_same_line_given_half_a_turn_apart_needs_no_move():
    assert kinova_executor._same_line(0.25, 0.25 + math.pi)
    assert kinova_executor._same_line(None, None)
    assert not kinova_executor._same_line(None, 0.0)
    assert not kinova_executor._same_line(0.0, math.pi / 2)
