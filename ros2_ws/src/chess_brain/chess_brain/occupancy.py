"""Move inference when vision only knows WHICH squares are occupied. Pure Python, no ROS.

The MuJoCo perception (chessbot_sim) detects one class, "a piece", and reports the square it
stands on; it does not tell a pawn from a queen or white from black. That is still enough to
follow a game, because chess_brain knows the position before the move:

    result = infer_from_occupancy(game.board, {"a1", "b1", ..., "e4", ...})

Every legal move is tried and the squares it would leave occupied are compared with what the
camera saw. The answer is one of (same kinds as GameState.infer_human_move):

* ``no_change``           - the occupied squares are those of the current position;
* ``move``                - exactly one legal move explains them;
* ``ambiguous``           - several do (see below); ``moves`` lists them;
* ``illegal_or_misread``  - none does (an illegal move, a knocked-over piece, a detection error).

What occupancy cannot tell apart:

* which piece was captured, when one piece could capture on two or more squares (the piece
  leaves its square and every candidate target is occupied before and after). The caller has
  to ask (an operator, or a detector that sees colours).
* what a pawn was promoted to: the four promotions of a pawn look the same from above, so a
  promotion is always reported as ``ambiguous`` with the four candidates.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, List, Optional

import chess

from .game import InferResult


def squares_of(board: chess.Board) -> FrozenSet[str]:
    """Names of the occupied squares of ``board``."""
    return frozenset(chess.square_name(square) for square in chess.scan_reversed(board.occupied))


def occupancy_after(board: chess.Board, move: chess.Move) -> FrozenSet[str]:
    """Names of the squares that are occupied once ``move`` has been played on ``board``."""
    after = board.copy(stack=False)
    after.push(move)
    return squares_of(after)


def placement_after(board: chess.Board, move: chess.Move) -> str:
    """Piece-placement FEN after ``move``: what a full detector would have reported."""
    after = board.copy(stack=False)
    after.push(move)
    return after.board_fen()


def normalise(occupied: Iterable[str]) -> Optional[FrozenSet[str]]:
    """Validated, lower-case square names, or None if any entry is not a square."""
    names = set()
    try:
        for entry in occupied:
            name = str(entry).strip().lower()
            chess.parse_square(name)
            names.add(name)
    except (TypeError, ValueError):
        return None
    return frozenset(names)


def infer_from_occupancy(board: chess.Board, occupied: Iterable[str]) -> InferResult:
    """Which move of the side to move leaves exactly ``occupied`` squares occupied?

    ``board`` is not modified. A promotion matches four times (queen, rook, bishop, knight)
    and is therefore ``ambiguous``.
    """
    seen = normalise(occupied)
    if seen is None:
        return InferResult("illegal_or_misread")
    if seen == squares_of(board):
        return InferResult("no_change")

    matches: List[chess.Move] = []
    for move in board.legal_moves:
        if occupancy_after(board, move) == seen:
            matches.append(move)

    if len(matches) == 1:
        return InferResult("move", matches)
    if not matches:
        return InferResult("illegal_or_misread")
    return InferResult("ambiguous", sorted(matches, key=lambda move: move.uci()))


def describe_difference(board: chess.Board, occupied: Iterable[str]) -> str:
    """Human-readable difference between the expected and the seen squares, for logs."""
    seen = normalise(occupied)
    if seen is None:
        return "the detection is not a list of squares"
    expected = squares_of(board)
    emptied = sorted(expected - seen)
    filled = sorted(seen - expected)
    if not emptied and not filled:
        return "no difference"
    parts = []
    if emptied:
        parts.append("now empty: " + " ".join(emptied))
    if filled:
        parts.append("now occupied: " + " ".join(filled))
    return "; ".join(parts)
