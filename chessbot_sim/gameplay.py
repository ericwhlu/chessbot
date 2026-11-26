"""Bookkeeping for a simulated chess game: no MuJoCo, no camera, only python-chess.

play_game.py uses these to stand in for the things a real game has and a simulation lacks:

* ``PieceBook`` - which simulated piece body stands on which square. Only the simulated human
  hand and the simulated operator use it (to move and tidy pieces); the robot never does: it
  works from the camera.
* ``HumanPlayer`` - the opponent's move choice.
* ``discard_slots`` - where the arm puts captured pieces.
* ``Report`` - what happened, ply by ply, for the summary.
"""
from dataclasses import dataclass, field
import random

import chess


STARTING_SQUARES = tuple(f"{file}{rank}" for rank in (1, 2, 7, 8) for file in "abcdefgh")


@dataclass
class Changes:
    moved: list = field(default_factory=list)      # [(body key, new square name)]
    removed: list = field(default_factory=list)    # [body key]: captured pieces


class PieceBook:
    """Follows every piece body through the game. A body is named by its starting square."""

    def __init__(self):
        self.at = {square: square for square in STARTING_SQUARES}   # square -> body key

    def apply(self, board, move):
        """Update the book for ``move`` played on the PRE-move ``board``; return what changed."""
        if move not in board.legal_moves:
            raise ValueError(f"{move.uci()} is not legal in {board.fen()}")
        changes = Changes()
        source, target = chess.square_name(move.from_square), chess.square_name(move.to_square)

        if board.is_castling(move):
            rank = chess.square_rank(move.from_square)
            if board.is_kingside_castling(move):
                king_to, rook_from, rook_to = (chess.square_name(chess.square(f, rank)) for f in (6, 7, 5))
            else:
                king_to, rook_from, rook_to = (chess.square_name(chess.square(f, rank)) for f in (2, 0, 3))
            for start, end in ((source, king_to), (rook_from, rook_to)):
                key = self.at.pop(start)
                self.at[end] = key
                changes.moved.append((key, end))
            return changes

        if board.is_en_passant(move):
            victim = chess.square_name(
                chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
            )
            changes.removed.append(self.at.pop(victim))
        elif board.is_capture(move):
            changes.removed.append(self.at.pop(target))

        key = self.at.pop(source)
        self.at[target] = key
        changes.moved.append((key, target))
        return changes

    def squares(self):
        return frozenset(self.at)


def discard_slots(count=30):
    """Board-frame (x, y) spots for captured pieces, nearest the arm first.

    Rows on the table beyond rank 1, then beyond rank 8, between the board's corner tags and
    clear of the camera's rectified board image (which extends 65 mm past the squares).
    """
    slots = []
    for y in (-0.12, -0.17, -0.22, 0.52, 0.57, 0.62):
        for x in (0.30, 0.25, 0.20, 0.15, 0.10):
            slots.append((x, y))
    return slots[:count]


class HumanPlayer:
    """Chooses the simulated opponent's moves.

    ``script`` (UCI moves) is played first, as long as each move is legal; after that the
    choice is random but reproducible (``seed``), taking a capture half of the time one is
    available so that games reach captures quickly.
    """

    def __init__(self, seed=7, script=()):
        self.rng = random.Random(seed)
        self.script = [str(move).strip().lower() for move in script if str(move).strip()]

    def choose(self, board):
        while self.script:
            uci = self.script.pop(0)
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if move.promotion not in (None, chess.QUEEN):   # from above, every promotion is a queen
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if move in board.legal_moves:
                return move
        moves = sorted(board.legal_moves, key=lambda move: move.uci())
        if not moves:
            return None
        # Under-promotions look the same as a queen from above; keep the game simple.
        moves = [move for move in moves if move.promotion in (None, chess.QUEEN)] or moves
        captures = [move for move in moves if board.is_capture(move)]
        if captures and self.rng.random() < 0.5:
            return self.rng.choice(captures)
        return self.rng.choice(moves)


@dataclass
class Report:
    plies: list = field(default_factory=list)

    def add(self, **entry):
        self.plies.append(entry)

    def count(self, who, outcome=None):
        return sum(1 for ply in self.plies if ply["who"] == who and (outcome is None or ply["outcome"] == outcome))

    def lines(self):
        robot, human = self.count("robot"), self.count("human")
        out = [
            f"plies played: {len(self.plies)}",
            f"robot moves:  {robot}  (arm did it, camera confirmed: {self.count('robot', 'ok')}; "
            f"finished by the operator: {robot - self.count('robot', 'ok')})",
            f"human moves:  {human}  (camera inferred the move: {self.count('human', 'ok')}; "
            f"ambiguous, operator confirmed: {self.count('human', 'ambiguous')}; "
            f"not recognised: {human - self.count('human', 'ok') - self.count('human', 'ambiguous')})",
        ]
        for ply in self.plies:
            if ply["outcome"] not in ("ok",):
                out.append(f"  ply {ply['ply']:>2} {ply['who']:5} {ply['move']:6} {ply['outcome']}: {ply.get('detail', '')}")
        return out

    def clean(self):
        """True if no move needed the operator's hands (an ambiguous capture only needs a word)."""
        return all(ply["outcome"] in ("ok", "ambiguous") for ply in self.plies)
