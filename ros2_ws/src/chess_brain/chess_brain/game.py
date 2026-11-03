"""Pure-Python chess core for chess_brain. No ROS imports here.

Two jobs:

1. ``GameState`` keeps the *full* game state (turn, castling rights, en-passant
   square, clocks) and infers which move the human made from the detector's
   piece-placement FEN.
2. ``Engine`` wraps a UCI engine (Stockfish) and returns the robot's reply.

Design rules (see docs/CHESS_BRAIN_PLAN.md §3.1):

* ``start_fen`` must be a full 6-field FEN. A placement-only string is accepted
  by python-chess but silently drops castling rights, so every later human
  castling move would be judged illegal.
* Every incoming placement string is canonicalised through ``chess.BaseBoard``
  and never crashes the caller.
* Game-over uses ``claim_draw=True`` so threefold repetition and the 50-move
  rule end the game (python-chess's default ignores both). Insufficient
  material (e.g. K+N vs K) ends the game automatically, no claim needed.
* The engine is only asked on the robot's turn and never on a finished game;
  an engine under-promotion is replaced by a queen promotion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Union

import chess
import chess.engine

InferKind = Literal["no_change", "move", "illegal_or_misread", "ambiguous"]

PLACEMENT_FIELDS = 6  # a full FEN has exactly six space-separated fields


def canonical_placement(text: str) -> Optional[str]:
    """Return the canonical piece-placement FEN for ``text``, or ``None`` if invalid.

    Accepts a bare placement (``rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR``) or a
    full FEN (only the first field is used), with surrounding whitespace.
    """
    if not isinstance(text, str):
        return None
    parts = text.strip().split()
    if not parts:
        return None
    try:
        return chess.BaseBoard(parts[0]).board_fen()
    except ValueError:
        return None


@dataclass
class InferResult:
    """Outcome of matching a detected placement against the legal moves."""

    kind: InferKind
    moves: list = field(default_factory=list)  # list[chess.Move]

    @property
    def move(self) -> Optional[chess.Move]:
        return self.moves[0] if self.kind == "move" else None


class GameState:
    """Full chess game state plus move inference from detected placements."""

    def __init__(self, start_fen: str = chess.STARTING_FEN, robot_color: chess.Color = chess.BLACK):
        fields = start_fen.strip().split()
        if len(fields) != PLACEMENT_FIELDS:
            raise ValueError(
                f"start_fen must be a full 6-field FEN (got {len(fields)} field(s)). "
                "A placement-only FEN would silently drop castling rights."
            )
        try:
            self.board = chess.Board(" ".join(fields))
        except ValueError as exc:
            raise ValueError(f"invalid start_fen: {exc}") from exc
        if not self.board.is_valid():
            raise ValueError(f"start_fen is not a legal position: status={self.board.status()!r}")
        self.robot_color: chess.Color = robot_color

    # ------------------------------------------------------------------ queries
    @property
    def human_color(self) -> chess.Color:
        return not self.robot_color

    @property
    def human_to_move(self) -> bool:
        return self.board.turn == self.human_color

    @property
    def robot_to_move(self) -> bool:
        return self.board.turn == self.robot_color

    def placement(self) -> str:
        """Piece-placement FEN of the internal board (what vision *should* see)."""
        return self.board.board_fen()

    def fen(self) -> str:
        return self.board.fen()

    def outcome(self) -> Optional[chess.Outcome]:
        """Game result including claimable draws (threefold, 50-move)."""
        return self.board.outcome(claim_draw=True)

    def is_game_over(self) -> bool:
        return self.outcome() is not None

    def status(self) -> dict:
        outcome = self.outcome()
        return {
            "fen": self.board.fen(),
            "placement": self.board.board_fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "human_to_move": self.human_to_move,
            "in_check": self.board.is_check(),
            "game_over": outcome is not None,
            "result": outcome.result() if outcome else None,
            "termination": outcome.termination.name.lower() if outcome else None,
            "fullmove": self.board.fullmove_number,
        }

    # ----------------------------------------------------------------- inference
    def infer_human_move(self, placement: str) -> InferResult:
        """Infer the move that the side to move played, from the detected placement.

        Infers for whichever side is to move; the node must only call this when
        ``human_to_move`` is True (it would just as happily "infer" a robot move).

        Vision reports *positions*, not moves. We enumerate every legal move,
        push it, and compare the resulting placement with what was detected.
        Castling, en passant and promotion need no special handling: python-chess
        moves the rook / removes the captured pawn / places the promoted piece,
        and the detector reports piece types, so the placements differ.
        """
        canon = canonical_placement(placement)
        if canon is None:
            return InferResult("illegal_or_misread")
        if canon == self.board.board_fen():
            return InferResult("no_change")

        matches = []
        for move in list(self.board.legal_moves):  # list(): don't push/pop under a live generator
            self.board.push(move)
            try:
                if self.board.board_fen() == canon:
                    matches.append(move)
            finally:
                self.board.pop()  # never leave the board mutated

        if len(matches) == 1:
            return InferResult("move", matches)
        if not matches:
            return InferResult("illegal_or_misread")
        # Provably unreachable for a single move in standard chess; kept as a
        # detector-bug trap rather than silently picking one.
        return InferResult("ambiguous", matches)

    # -------------------------------------------------------------------- update
    def apply(self, move: chess.Move) -> None:
        """Push ``move`` onto the board. Raises ``ValueError`` if it is not legal."""
        if move not in self.board.legal_moves:
            raise ValueError(f"{move.uci()} is not legal in {self.board.fen()}")
        self.board.push(move)


EngineCommand = Union[str, Sequence[str]]


class Engine:
    """Thin wrapper around a UCI engine for the robot's move selection.

    ``command`` is the Stockfish path, or a list such as
    ``[sys.executable, "fake_uci_engine.py"]`` for tests.
    """

    def __init__(
        self,
        command: EngineCommand = "stockfish",
        think_time: float = 0.5,
        skill_level: Optional[int] = None,
    ):
        self.think_time = float(think_time)
        self._engine = chess.engine.SimpleEngine.popen_uci(command)
        self.skill_level_supported = "Skill Level" in self._engine.options
        if skill_level is not None and self.skill_level_supported:
            opt = self._engine.options["Skill Level"]
            lo = opt.min if opt.min is not None else 0
            hi = opt.max if opt.max is not None else 20
            level = max(lo, min(hi, int(skill_level)))  # clamp: out-of-range raises and leaks the process
            try:
                self._engine.configure({"Skill Level": level})
            except Exception:
                self._engine.close()
                raise

    @property
    def name(self) -> str:
        return self._engine.id.get("name", "unknown")

    def reply(self, game: GameState) -> Optional[chess.Move]:
        """Return the robot's move, or ``None`` if the game is over.

        Only valid on the robot's turn. An engine under-promotion is replaced by
        a queen promotion (always legal when the under-promotion is; the table
        only has spare queens).
        """
        board = game.board
        if board.turn != game.robot_color:
            raise RuntimeError("Engine.reply called on the human's turn")
        if game.outcome() is not None:
            return None

        # game=... makes python-chess send `ucinewgame` when a different GameState
        # is played through the same engine (clears hash/history).
        result = self._engine.play(board, chess.engine.Limit(time=self.think_time), game=id(game))
        move = result.move
        if move is None:  # engine answered "bestmove (none)" or resigned
            return None
        if move.promotion not in (None, chess.QUEEN):
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        if move not in board.legal_moves:
            raise RuntimeError(f"engine returned illegal move {move.uci()} for {board.fen()}")
        return move

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:  # EngineError, EngineTerminatedError, asyncio.TimeoutError
            pass
        finally:
            self._engine.close()  # always tear down the event-loop thread

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
