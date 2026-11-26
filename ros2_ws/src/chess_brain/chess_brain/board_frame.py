"""Where things are: chess squares and off-board locations in metres. Pure Python.

chess_brain itself only speaks in square names. This module is the single place that turns
those names into coordinates, following docs/board_setup.md:

* board origin = the a1 corner of the playing area; +X along the files a->h, +Y along the
  ranks 1->8, +Z up; squares are 50 mm;
* square centre x = (file_index + 0.5) * 0.05, y = (rank - 0.5) * 0.05.

``BoardFrame`` places that board in the world (origin position + rotation about Z).
``Locations`` adds the named off-board spots used in ``arm_actions``:

* ``DISCARD``   - captured pieces. Slots are handed out one after another in a column
                  beside the a-file (then a second column, and so on).
* ``SPARE_Q_W`` / ``SPARE_Q_B`` - spare queens for promotion.

All off-board positions are given in the BOARD frame, so they move with the board. They lie
``offboard_drop`` below the playing surface: 0 if they are on the board's own border, the
board's thickness if captured pieces are put on the table next to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

FILES = "abcdefgh"
RANKS = "12345678"
DISCARD = "DISCARD"

XY = Tuple[float, float]
XYZ = Tuple[float, float, float]


@dataclass(frozen=True)
class BoardFrame:
    origin_x: float = 0.0          # world position of the a1 corner (m)
    origin_y: float = 0.0
    yaw: float = 0.0               # rotation of the board about +Z (rad)
    surface_z: float = 0.00656168  # height of the playing surface (m), from docs/board_setup.md
    square: float = 0.05           # square size (m)

    def board_xy(self, square_name: str) -> XY:
        """Centre of a square in the board frame, e.g. ``a1`` -> (0.025, 0.025)."""
        name = str(square_name).strip().lower()
        if len(name) != 2 or name[0] not in FILES or name[1] not in RANKS:
            raise ValueError(f"not a square name: {square_name!r}")
        file_index = FILES.index(name[0])
        rank = int(name[1])
        return ((file_index + 0.5) * self.square, (rank - 0.5) * self.square)

    def to_world(self, bx: float, by: float) -> XYZ:
        """A point of the board frame, on the playing surface, in world coordinates."""
        cos_yaw, sin_yaw = math.cos(self.yaw), math.sin(self.yaw)
        return (
            self.origin_x + cos_yaw * bx - sin_yaw * by,
            self.origin_y + sin_yaw * bx + cos_yaw * by,
            self.surface_z,
        )

    def world_xyz(self, square_name: str) -> XYZ:
        """Centre of a square on the playing surface, in world coordinates."""
        return self.to_world(*self.board_xy(square_name))


def _default_spares() -> Dict[str, XY]:
    # Beside the h-file, next to each side's back rank.
    return {"SPARE_Q_W": (0.475, 0.025), "SPARE_Q_B": (0.475, 0.375)}


@dataclass
class Locations:
    """Resolves the location names used in ``arm_actions`` to world coordinates."""

    frame: BoardFrame = field(default_factory=BoardFrame)
    discard_first: XY = (-0.075, 0.025)   # first discard slot (board frame): beside a1
    discard_step: float = 0.05            # distance between slots (m)
    discard_per_column: int = 8           # slots per column before starting the next one
    spares: Dict[str, XY] = field(default_factory=_default_spares)
    discarded: int = 0                    # slots used so far
    offboard_drop: float = 0.0            # off-board spots are this far below the playing surface (m)

    def _offboard(self, bx: float, by: float) -> XYZ:
        x, y, z = self.frame.to_world(bx, by)
        return (x, y, z - self.offboard_drop)

    def pick_xyz(self, name: str) -> XYZ:
        """Where to pick up: a square or a spare piece. ``DISCARD`` is not a pick location."""
        key = str(name).strip()
        if key.upper() in self.spares:
            return self._offboard(*self.spares[key.upper()])
        if key.upper() == DISCARD:
            raise ValueError("DISCARD is not a pick location")
        return self.frame.world_xyz(key)

    def discard_slot_board_xy(self, index: int) -> XY:
        """Position of discard slot ``index`` (0, 1, 2, ...) in the board frame."""
        column, row = divmod(int(index), self.discard_per_column)
        return (self.discard_first[0] - column * self.discard_step, self.discard_first[1] + row * self.discard_step)

    def discard_slot_xyz(self, index: int) -> XYZ:
        """World position of discard slot ``index`` (0, 1, 2, ...)."""
        return self._offboard(*self.discard_slot_board_xy(index))

    def board_xy(self, name: str, discard_offset: int = 0) -> XY:
        """Board-frame position of any location name: a square, a spare, or ``DISCARD`` (the next
        free slot + ``discard_offset``)."""
        key = str(name).strip()
        if key.upper() == DISCARD:
            return self.discard_slot_board_xy(self.discarded + discard_offset)
        if key.upper() in self.spares:
            return self.spares[key.upper()]
        return self.frame.board_xy(key)

    def place_xyz(self, name: str, discard_offset: int = 0) -> XYZ:
        """Where to put down: a square, or the next free discard slot (+ ``discard_offset``).

        Does not use up a discard slot; call ``commit_discard()`` once the piece is really there.
        """
        key = str(name).strip()
        if key.upper() == DISCARD:
            return self.discard_slot_xyz(self.discarded + discard_offset)
        if key.upper() in self.spares:
            raise ValueError(f"{key} is not a place location")
        return self.frame.world_xyz(key)

    def commit_discard(self) -> None:
        self.discarded += 1

    def reset(self) -> None:
        """New game: all discard slots are free again."""
        self.discarded = 0
