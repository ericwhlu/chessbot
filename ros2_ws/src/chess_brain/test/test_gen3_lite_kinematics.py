"""gen3_lite_kinematics: the forward kinematics against the Kinova user guide's DH table, and the
top-down inverse kinematics that the reach check hands to MoveIt."""

import math
import random

import pytest

from chess_brain import gen3_lite_kinematics as K
from chess_brain.board_frame import FILES, RANKS
from chess_brain.reach_check import compare_with_kinematics

# Classical DH rows (theta offset, d, a, alpha) from the Kinova Gen3 lite user guide, pages 53-54
# and 135, ending at end_effector_link (the guide's last d also counts 130 mm to the gripper
# centre). An independent description of the same arm: the module uses the URDF's joint origins.
DH = (
    (0.0, 0.2433, 0.0, math.pi / 2),
    (math.pi / 2, 0.030, 0.28, math.pi),
    (math.pi / 2, 0.020, 0.0, math.pi / 2),
    (math.pi / 2, 0.245, 0.0, math.pi / 2),
    (math.pi, 0.057, 0.0, math.pi / 2),
    (math.pi / 2, 0.105, 0.0, 0.0),
)

# end_effector_link heights for a board surface at z = -0.056 (kinova_executor.Settings defaults)
GRIP_Z = -0.056 + 0.157 + 0.020
TRAVEL_Z = -0.056 + 0.157 + 0.150


def dh_pose(joints):
    t = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for (offset, d, a, alpha), angle in zip(DH, joints):
        th = angle + offset
        step = [
            [math.cos(th), -math.sin(th) * math.cos(alpha), math.sin(th) * math.sin(alpha), a * math.cos(th)],
            [math.sin(th), math.cos(th) * math.cos(alpha), -math.cos(th) * math.sin(alpha), a * math.sin(th)],
            [0.0, math.sin(alpha), math.cos(alpha), d],
            [0.0, 0.0, 0.0, 1.0],
        ]
        t = [[sum(t[i][k] * step[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
    return t


def test_forward_matches_the_user_guide_dh_table():
    rng = random.Random(0)
    for _ in range(200):
        joints = [rng.uniform(-limit, limit) for limit in K.JOINT_LIMITS]
        _, _, rotation, position = K.forward(joints)
        expected = dh_pose(joints)
        # the guide rounds the base height to 128.3 mm; the URDF has 128.25 mm
        assert math.dist(position, [expected[i][3] for i in range(3)]) < 1e-4
        for i in range(3):
            for j in range(3):
                assert rotation[i][j] == pytest.approx(expected[i][j], abs=1e-9)


def test_top_down_rotation_is_the_top_down_quaternion():
    x, y, z, w = K.TOP_DOWN
    from_quaternion = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    assert from_quaternion == K.TOP_DOWN_ROTATION


def test_the_work_pose_is_inside_the_joint_limits():
    assert all(abs(angle) <= limit for angle, limit in zip(K.WORK, K.JOINT_LIMITS))


@pytest.mark.parametrize(
    "x, y, z",
    [
        (0.35, 0.10, GRIP_Z),
        (0.35, 0.10, TRAVEL_Z),
        (0.197, -0.175, GRIP_Z),    # the corner nearest the arm
        (0.447, 0.175, TRAVEL_Z),   # third rank from the far edge, at the side
        (0.25, 0.0, 0.20),
    ],
)
def test_solutions_put_the_gripper_there_pointing_down(x, y, z):
    found = K.top_down_solutions(x, y, z)
    assert found
    for joints in found:
        assert all(abs(angle) <= limit + 1e-12 for angle, limit in zip(joints, K.JOINT_LIMITS))
        distance, angle = K.pose_error(joints, (x, y, z), K.TOP_DOWN_ROTATION)
        assert distance < K.POSITION_TOLERANCE and angle < K.ROTATION_TOLERANCE
        _, _, rotation, position = K.forward(joints)
        assert position == pytest.approx((x, y, z), abs=K.POSITION_TOLERANCE)
        assert [rotation[i][2] for i in range(3)] == pytest.approx([0.0, 0.0, -1.0], abs=1e-4)
    for i, a in enumerate(found):   # they are different arm configurations, not one found twice
        for b in found[i + 1:]:
            assert max(abs(p - q) for p, q in zip(a, b)) >= 0.05


@pytest.mark.parametrize(
    "x, y, z",
    [
        (0.547, 0.025, GRIP_Z),     # the far rank of a 400 mm board placed just past the mount plate
        (0.547, 0.175, TRAVEL_Z),
        (0.70, 0.0, GRIP_Z),
        (0.05, 0.0, GRIP_Z),        # on top of the arm's own base
    ],
)
def test_poses_out_of_reach_have_no_solution(x, y, z):
    assert K.top_down_solutions(x, y, z) == []


def test_a_far_corner_can_be_gripped_but_not_carried_to():
    # reach shrinks with height: 0.527 m from the base is within reach low down, not 130 mm higher
    assert K.top_down_solutions(0.497, 0.175, GRIP_Z)
    assert K.top_down_solutions(0.497, 0.175, TRAVEL_Z) == []


def test_the_answer_is_the_same_every_time():
    K._top_down.cache_clear()
    first = K.top_down_solutions(0.40, -0.12, GRIP_Z)
    K._top_down.cache_clear()
    assert K.top_down_solutions(0.40, -0.12, GRIP_Z) == first


def test_a_caller_cannot_spoil_the_cached_answer():
    first = K.top_down_solutions(0.30, 0.05, GRIP_Z)
    first[0][0] = 99.0
    assert K.top_down_solutions(0.30, 0.05, GRIP_Z)[0][0] != 99.0


# ------------------------------------------------------- reach_check.compare_with_kinematics
def _table(default, **others):
    table = {f + r: default for r in RANKS for f in FILES}
    table["DISCARD[0]"] = default
    table.update(others)
    return table


def test_compare_reports_agreement():
    full = (True, True)
    lines = compare_with_kinematics(_table(full, a1=(False, False)), _table(full, a1=(False, False)))
    assert lines[0].endswith("63 of 64 squares")
    assert lines[1].strip() == "MoveIt and the kinematics agree on every location, at both heights"


def test_compare_names_what_moveit_refuses_and_what_only_moveit_finds():
    full = (True, True)
    moveit = _table(full, a1=(False, False), b1=(True, False), h8=full)
    kinematic = _table(full, a1=(False, False), h8=(True, False))
    lines = compare_with_kinematics(moveit, kinematic)
    assert lines[0].endswith("62 of 64 squares")
    assert lines[1].endswith("refuses them: b1")
    assert lines[2].endswith("found nothing: h8")


def test_compare_does_not_call_a_difference_at_one_height_agreement():
    full = (True, True)
    # the kinematics reaches a1 at the carrying height, MoveIt not at all: neither is "full"
    lines = compare_with_kinematics(_table(full, a1=(False, False)), _table(full, a1=(True, False)))
    assert lines[1].endswith("refuses them: a1")
    assert not any("agree" in line for line in lines)
