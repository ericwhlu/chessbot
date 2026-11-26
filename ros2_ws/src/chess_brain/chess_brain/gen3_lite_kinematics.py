"""Kinematics of the Kinova Gen3 Lite, enough to say whether a top-down gripper pose has joint
angles at all. Pure Python (no ROS, no numpy), so it is unit-tested anywhere.

Why this exists: MoveIt's inverse-kinematics service starts from one guess and gives up after a
few milliseconds, so on its own it reports "no solution" for most poses that have one (the first
reach check in Gazebo marked squares unreachable at random). kinova_arm.can_reach computes the
solutions here and hands them to MoveIt as starting points; MoveIt then only has to confirm them
against the joint limits and the collision scene.

The chain is the one in the course image's kortex_description (gen3_lite_macro.xacro): the origin
of each joint in its parent link, every joint turning about its own z axis, the last frame being
end_effector_link. Joint limits are the tighter of the two files in that image.
"""

from __future__ import annotations

import math
import random
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

Vector = Tuple[float, float, float]
Matrix = Tuple[Vector, Vector, Vector]  # rows

# (xyz, rpy) of each joint in its parent link
CHAIN = (
    ((0.0, 0.0, 0.12825), (0.0, 0.0, 0.0)),
    ((0.0, -0.03, 0.115), (math.pi / 2, 0.0, 0.0)),
    ((0.0, 0.28, 0.0), (-math.pi, 0.0, 0.0)),
    ((0.0, -0.14, 0.02), (math.pi / 2, 0.0, 0.0)),
    ((0.0285, 0.0, 0.105), (0.0, math.pi / 2, 0.0)),
    ((-0.105, 0.0, 0.0285), (0.0, -math.pi / 2, 0.0)),
)
JOINT_LIMITS = (2.68, 2.61, 2.61, 2.59, 2.53, 2.59)  # each joint: -limit .. +limit (rad)

# Gripper pointing straight down, as a quaternion (x, y, z, w) and as the same rotation matrix.
# In this orientation ("yaw 0") the two fingers open along the base frame's y axis.
TOP_DOWN = (0.0, 1.0, 0.0, 0.0)
TOP_DOWN_ROTATION: Matrix = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))


def top_down_quaternion(yaw: float = 0.0) -> Tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) of the gripper pointing straight down and turned by ``yaw``
    (rad) about the vertical, counter-clockwise seen from above."""
    return (-math.sin(yaw / 2.0), math.cos(yaw / 2.0), 0.0, 0.0)


def top_down_rotation(yaw: float = 0.0) -> Matrix:
    """The same orientation as a rotation matrix: Rz(yaw) * TOP_DOWN_ROTATION."""
    c, s = math.cos(yaw), math.sin(yaw)
    return ((-c, -s, 0.0), (-s, c, 0.0), (0.0, 0.0, -1.0))


def grip_yaws(grip: float) -> Tuple[float, float]:
    """The two tool yaws (rad, in -pi..pi) for which the fingers open along the line at angle
    ``grip`` (rad, base frame, measured from +x). The fingers are alike, so the two yaws, half a
    turn apart, grip the same way; which one the arm can reach more easily depends on the pose."""
    def wrap(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    return (wrap(grip - math.pi / 2.0), wrap(grip + math.pi / 2.0))

# The course's WORK pose (Lab 6): elbow up above the table, a good first guess for a top-down pose.
WORK = (0.40, 0.02, 2.27, -1.57, -0.84, 1.97)

POSITION_TOLERANCE = 2e-5   # m
ROTATION_TOLERANCE = 1e-4   # rad


# ------------------------------------------------------------------ small 3 x 3 algebra
def _mul(a: Matrix, b: Matrix) -> Matrix:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )  # type: ignore[return-value]


def _apply(a: Matrix, v: Vector) -> Vector:
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))  # type: ignore[return-value]


def _cross(a: Vector, b: Vector) -> Vector:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _column(a: Matrix, j: int) -> Vector:
    return (a[0][j], a[1][j], a[2][j])


def _rot_x(t: float) -> Matrix:
    c, s = math.cos(t), math.sin(t)
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def _rot_y(t: float) -> Matrix:
    c, s = math.cos(t), math.sin(t)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def _rot_z(t: float) -> Matrix:
    c, s = math.cos(t), math.sin(t)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


_FIXED = tuple(
    (xyz, _mul(_mul(_rot_z(yaw), _rot_y(pitch)), _rot_x(roll))) for xyz, (roll, pitch, yaw) in CHAIN
)


# ------------------------------------------------------------------------- kinematics
def forward(joints: Sequence[float]):
    """(origins, axes, rotation, position): where each joint is and which way it turns, in
    base_link, and the pose of end_effector_link."""
    rotation: Matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    position: Vector = (0.0, 0.0, 0.0)
    origins, axes = [], []
    for (xyz, fixed), angle in zip(_FIXED, joints):
        step = _apply(rotation, xyz)
        position = (position[0] + step[0], position[1] + step[1], position[2] + step[2])
        rotation = _mul(rotation, fixed)
        origins.append(position)
        axes.append(_column(rotation, 2))   # the joint axis; turning about z does not move it
        rotation = _mul(rotation, _rot_z(angle))
    return origins, axes, rotation, position


def pose_error(joints: Sequence[float], position: Vector, rotation: Matrix):
    """(distance in m, angle in rad) between end_effector_link at ``joints`` and the wanted pose."""
    _, _, r, p = forward(joints)
    distance = math.sqrt(sum((p[i] - position[i]) ** 2 for i in range(3)))
    trace = sum(sum(r[i][k] * rotation[i][k] for k in range(3)) for i in range(3))
    return distance, math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))


def _solve6(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve a 6 x 6 linear system by Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-15:
            return [0.0] * n
        for r in range(col + 1, n):
            factor = m[r][col] / m[col][col]
            if factor:
                for c in range(col, n + 1):
                    m[r][c] -= factor * m[col][c]
    x = [0.0] * n
    for r in reversed(range(n)):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


def _refine(start: Sequence[float], position: Vector, rotation: Matrix, iterations: int = 120):
    """Damped least squares from one starting guess; the joints stay inside their limits.
    Returns the joint angles, or None if this start did not reach the pose."""
    q = [max(-lim, min(lim, v)) for v, lim in zip(start, JOINT_LIMITS)]
    target_columns = [_column(rotation, j) for j in range(3)]
    for _ in range(iterations):
        origins, axes, r, p = forward(q)
        dp = [position[i] - p[i] for i in range(3)]
        dw = [0.0, 0.0, 0.0]
        for j in range(3):   # half the sum of (current axis) x (wanted axis): the rotation still to make
            c = _cross(_column(r, j), target_columns[j])
            dw = [dw[i] + 0.5 * c[i] for i in range(3)]
        error = dp + dw
        size_p = math.sqrt(sum(v * v for v in dp))
        size_w = math.sqrt(sum(v * v for v in dw))
        if size_p < POSITION_TOLERANCE / 2 and size_w < ROTATION_TOLERANCE / 2:
            distance, angle = pose_error(q, position, rotation)   # rules out the 180-degree twin
            return q if distance < POSITION_TOLERANCE and angle < ROTATION_TOLERANCE else None

        jacobian = []   # 6 rows (vx vy vz wx wy wz) x 6 joints
        columns = []
        for origin, axis in zip(origins, axes):
            arm = (p[0] - origin[0], p[1] - origin[1], p[2] - origin[2])
            columns.append(list(_cross(axis, arm)) + list(axis))
        for i in range(6):
            jacobian.append([columns[k][i] for k in range(6)])

        damping = 1e-6 + 0.01 * min(1.0, size_p + size_w)   # heavy far away, light near the answer
        jjt = [[sum(jacobian[i][k] * jacobian[j][k] for k in range(6)) + (damping if i == j else 0.0)
                for j in range(6)] for i in range(6)]
        y = _solve6(jjt, error)
        step = [sum(jacobian[i][k] * y[i] for i in range(6)) for k in range(6)]
        largest = max(abs(v) for v in step)
        if largest > 0.4:
            step = [v * 0.4 / largest for v in step]
        q = [max(-lim, min(lim, a + d)) for a, d, lim in zip(q, step, JOINT_LIMITS)]
    return None


def solutions(position: Vector, rotation: Matrix, wanted: int = 4, starts: int = 48) -> List[List[float]]:
    """Up to ``wanted`` different sets of joint angles, within the joint limits, that put
    end_effector_link at the pose. An empty list means none was found from ``starts`` starting
    guesses, which for this arm means the pose is out of reach.

    The starting guesses are the same on every call, so the answer does not change from run to
    run.
    """
    rng = random.Random(1234)
    azimuth = math.atan2(position[1], position[0])
    guesses = [(azimuth,) + WORK[1:], (azimuth + WORK[0],) + WORK[1:]]
    guesses += [tuple(rng.uniform(-lim, lim) for lim in JOINT_LIMITS) for _ in range(starts)]
    found: List[List[float]] = []
    for guess in guesses:
        q = _refine(guess, position, rotation)
        if q is None or any(max(abs(a - b) for a, b in zip(q, other)) < 0.05 for other in found):
            continue
        found.append(q)
        if len(found) >= wanted:
            break
    return found


@lru_cache(maxsize=8192)
def _top_down(x: float, y: float, z: float, yaw: float) -> Tuple[Tuple[float, ...], ...]:
    return tuple(tuple(q) for q in solutions((x, y, z), top_down_rotation(yaw)))


def top_down_solutions(x: float, y: float, z: float, yaw: float = 0.0) -> List[List[float]]:
    """Joint angles for end_effector_link at (x, y, z) in base_link with the gripper straight down
    and turned by ``yaw`` about the vertical (0 = the orientation of TOP_DOWN). Empty if the arm
    cannot get there."""
    return [
        list(q)
        for q in _top_down(round(float(x), 5), round(float(y), 5), round(float(z), 5), round(float(yaw), 6))
    ]


def nearest_solution(position: Vector, rotation: Matrix, joints: Sequence[float]):
    """The joint angles for the pose that the arm reaches from ``joints`` without changing its
    configuration (same elbow and wrist branch), and how far that is: ``(largest single-joint
    change in rad, joint angles)``. None if there is no such solution inside the joint limits."""
    found = _refine(joints, position, rotation)
    if found is None:
        return None
    return max(abs(a - b) for a, b in zip(found, joints)), found


def rank_yaws(
    x: float,
    y: float,
    z: float,
    grip: float,
    joints: Optional[Sequence[float]] = None,
    then_z: Optional[float] = None,
    previous_yaw: Optional[float] = None,
) -> List[float]:
    """The two tool yaws that open the fingers along ``grip``, the better one first.

    With the arm's present ``joints`` the better one is the yaw it can turn to without leaving its
    configuration and its joint limits, and of two such the one with the smaller joint motion: the
    last joint turns the gripper, and only one of the two yaws may lie inside its range. ``then_z``
    (the height the tool is lowered to next, same x and y) must be reachable the same way.
    Without ``joints`` the yaw nearer to ``previous_yaw`` comes first, and a yaw with no joint
    solution at all goes last.
    """
    yaws = list(grip_yaws(float(grip)))

    def cost(yaw: float):
        rotation = top_down_rotation(yaw)
        if joints is not None:
            near = nearest_solution((x, y, z), rotation, joints)
            if near is not None and then_z is not None and nearest_solution((x, y, then_z), rotation, near[1]) is None:
                near = None
            if near is not None:
                return (0, near[0])
        reachable = bool(top_down_solutions(x, y, z, yaw)) and (
            then_z is None or bool(top_down_solutions(x, y, then_z, yaw))
        )
        turn = 0.0
        if previous_yaw is not None:
            turn = abs(math.atan2(math.sin(yaw - previous_yaw), math.cos(yaw - previous_yaw)))
        # reachable only by another configuration (1), or not at all (2)
        return (1 if reachable else 2, turn)

    yaws.sort(key=cost)
    return yaws
