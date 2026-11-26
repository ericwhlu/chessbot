"""Which way should the gripper open over a square? Pure Python.

The Gen3 Lite's two fingers open along one line. Seen from above, each finger is a block about
20 mm thick (along that line) and 24 mm wide; over a crowded board the blocks pass a few
millimetres from the pieces on the neighbouring squares. How close depends on the direction of
the line: along the board's grid the fingers reach towards the two nearest neighbours, turned
45 degrees they reach towards the diagonal ones, which are 1.41 squares away.

chess_brain knows which squares hold a piece, so for every pick and every place this module
picks, out of four directions, the one that keeps the fingers farthest from the other pieces.
A direction is the angle of the opening line in the BOARD frame, in degrees, measured from the
board's +X axis: 0 = along a rank (towards the next file, a -> h), 90 = along a file (towards the
next rank), 45 and 135 = the two diagonals. A line has no sign, so 180 is the same as 0.

The model is a plan view: two rectangles for the fingers, a disc for every other piece. It is
used to RANK directions. It is not a collision check and not a promise of clearance; the numbers
it is built on come from the manufacturer's finger meshes and a 3D study of the same question
(see the package README), not from the real gripper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

XY = Tuple[float, float]

DIRECTIONS = (0.0, 90.0, 45.0, 135.0)  # in order of preference: along the grid first


@dataclass(frozen=True)
class Fingers:
    """Plan-view size of the open gripper and of the pieces it must stay clear of (metres)."""

    thickness: float = 0.020     # one finger, along the opening line
    width: float = 0.024         # one finger, across the opening line
    gap: float = 0.040           # free width between the fingers when open over a piece
    piece_radius: float = 0.0125  # another piece at finger height (stem corners, heads)
    enough: float = 0.020        # clearance beyond this is not worth turning the gripper for

    def __post_init__(self) -> None:
        for name in ("thickness", "width", "gap", "piece_radius", "enough"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"grip_direction: {name} must be positive")


def clearance(direction: float, target: XY, obstacles: Iterable[XY], fingers: Fingers = Fingers()) -> float:
    """Smallest distance (m) between the open fingers over ``target`` and the other pieces, with
    the fingers opening along ``direction`` (degrees, board frame). Negative = they overlap.
    With no other piece nearby the answer is ``math.inf``."""
    angle = math.radians(direction)
    ux, uy = math.cos(angle), math.sin(angle)      # along the opening line
    inner, outer = fingers.gap / 2.0, fingers.gap / 2.0 + fingers.thickness
    half_width = fingers.width / 2.0
    best = math.inf
    for ox, oy in obstacles:
        dx, dy = ox - target[0], oy - target[1]
        along = abs(dx * ux + dy * uy)             # the two fingers are mirror images
        across = abs(-dx * uy + dy * ux)
        # distance from the point to the rectangle [inner, outer] x [-half_width, half_width]
        da = max(inner - along, 0.0, along - outer)
        dc = max(across - half_width, 0.0)
        if da == 0.0 and dc == 0.0:                # the centre is inside the finger: depth of overlap
            inside = min(along - inner, outer - along, half_width - across)
            distance = -inside
        else:
            distance = math.hypot(da, dc)
        best = min(best, distance - fingers.piece_radius)
    return best


def _turn(a: float, b: float) -> float:
    """Angle between two opening lines, 0 to 90 degrees."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def choose(
    pick: XY,
    pick_obstacles: Sequence[XY],
    place: XY,
    place_obstacles: Sequence[XY],
    fingers: Fingers = Fingers(),
    directions: Sequence[float] = DIRECTIONS,
) -> Tuple[float, float, float]:
    """The directions to open along at the pick and at the place, and the smaller of the two
    clearances (m, uncapped) that this choice leaves: ``(pick_direction, place_direction, clearance)``.

    The piece is turned by the difference between the two directions while it is carried. Only
    0 and 90 degrees are allowed: the pieces have a square stem, and a quarter turn keeps its
    faces along the grid for the next time it is gripped.

    Rule: the pair with the largest ``min(clearance at pick, clearance at place)``, where a
    clearance above ``fingers.enough`` counts as ``enough``. Among equals: no turn, then more
    room at the roomier end, then the earlier entries of ``directions`` (along the grid before
    the diagonals).
    """
    best: Optional[Tuple[float, int, int, int]] = None
    chosen = (directions[0], directions[0], -math.inf)
    for i, a in enumerate(directions):
        at_pick = clearance(a, pick, pick_obstacles, fingers)
        for j, b in enumerate(directions):
            turn = _turn(a, b)
            if not (turn < 1e-9 or abs(turn - 90.0) < 1e-9):
                continue
            at_place = clearance(b, place, place_obstacles, fingers)
            worst = min(at_pick, at_place)
            # round so that float noise does not decide between equal options
            roomier = round(min(max(at_pick, at_place), fingers.enough), 6)
            key = (round(min(worst, fingers.enough), 6), -int(turn > 1e-9), roomier, -i, -j)
            if best is None or key > best:
                best, chosen = key, (a, b, worst)
    return chosen
