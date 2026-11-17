"""Translate a chess move into an ordered list of pick-and-place actions for the arm.

The arm team never needs to know chess: each action is "pick up whatever is on
square X and put it on square Y". Off-board locations are named so the arm team
maps them once:

* ``DISCARD``      — one off-board spot for captured pieces (arm allocates slots)
* ``SPARE_Q_W`` / ``SPARE_Q_B`` — spare queens for promotion (Objective 4)

Invariants the arm team can rely on (checked by the tests):

* actions are executed in order; one move emits 1–3 actions;
* ``place`` is never an occupied square — a capture always emits the
  ``(captured_square, DISCARD)`` action first (a capture-promotion emits two
  ``DISCARD`` actions);
* after all actions, the physical board equals the chess position after the move;
* ``SPARE_Q_W`` / ``SPARE_Q_B`` is emitted on *every* queen promotion of that
  colour; a captured promoted queen goes to ``DISCARD``, not back to the spare
  spot, so the table needs one spare per expected promotion;
* castling is king first, then rook — a choice, not a chess rule.

**Call with the board BEFORE the move.** ``is_castling``, ``is_en_passant``,
``is_capture`` and ``board.turn`` are all computed from the pre-move position;
after ``push()`` they are wrong. The node computes actions, publishes them, and
only applies the move once the arm reports ``ok`` (see CHESS_BRAIN_PLAN.md §3.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import chess

DISCARD = "DISCARD"


class UnsupportedMove(Exception):
    """The move is legal in chess but the arm cannot execute it (yet)."""


@dataclass(frozen=True)
class ArmAction:
    pick: str   # "e2", … or "SPARE_Q_W" / "SPARE_Q_B"
    place: str  # square or "DISCARD"

    def as_dict(self) -> dict:
        return {"pick": self.pick, "place": self.place}


def spare_name(piece_type: chess.PieceType, color: chess.Color) -> str:
    return f"SPARE_{chess.piece_symbol(piece_type).upper()}_{'W' if color == chess.WHITE else 'B'}"


def move_to_actions(board: chess.Board, move: chess.Move, allow_promotion: bool = False) -> List[ArmAction]:
    """Return the arm actions for ``move`` on the PRE-move ``board``.

    Raises ``ValueError`` if the move is not legal and ``UnsupportedMove`` for
    promotions when ``allow_promotion`` is False, or for any non-queen promotion
    (the table only has spare queens).
    """
    if move not in board.legal_moves:
        raise ValueError(f"{move.uci()} is not legal in {board.fen()}")

    frm = chess.square_name(move.from_square)
    to = chess.square_name(move.to_square)

    # En passant FIRST: is_capture() is also True for it, but the captured pawn
    # is not on the destination square.
    if board.is_en_passant(move):
        captured = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        return [ArmAction(chess.square_name(captured), DISCARD), ArmAction(frm, to)]

    if board.is_castling(move):
        if board.chess960:
            raise UnsupportedMove("chess960 castling not supported")
        rank = chess.square_rank(move.from_square)  # mover's back rank
        # Derive the king's destination from the side, NOT from move.to_square:
        # python-chess also accepts the "king takes rook" encoding (e1h1) as legal.
        if board.is_kingside_castling(move):
            king_to, rook_from, rook_to = chess.square(6, rank), chess.square(7, rank), chess.square(5, rank)  # g, h->f
        else:
            king_to, rook_from, rook_to = chess.square(2, rank), chess.square(0, rank), chess.square(3, rank)  # c, a->d
        # King first is a choice, not a chess requirement; swap if the arm prefers rook first.
        return [
            ArmAction(frm, chess.square_name(king_to)),
            ArmAction(chess.square_name(rook_from), chess.square_name(rook_to)),
        ]

    actions: List[ArmAction] = []
    if board.is_capture(move):
        actions.append(ArmAction(to, DISCARD))

    if move.promotion:
        if not allow_promotion:
            raise UnsupportedMove(f"promotion {move.uci()} not supported yet (needs spare pieces)")
        if move.promotion != chess.QUEEN:
            raise UnsupportedMove(f"only queen promotion is supported, got {move.uci()}")
        actions.append(ArmAction(frm, DISCARD))
        actions.append(ArmAction(spare_name(move.promotion, board.turn), to))
        return actions

    actions.append(ArmAction(frm, to))
    return actions


def actions_payload(action_id: int, move: chess.Move, actions: List[ArmAction]) -> dict:
    """The JSON-serialisable message published on ``chessbot/arm_actions``."""
    return {"id": int(action_id), "move": move.uci(), "actions": [a.as_dict() for a in actions]}
