"""A simulated world for testing chess_brain without vision or an arm. Pure Python.

``SimWorld`` plays three parts at once:

* the **physical board** — which piece stands on which square (what a camera would see);
* the **human** — picks and plays the human side's moves;
* the **arm** — executes ``arm_actions`` on the physical board, checking the rules the
  real arm relies on (never pick from an empty square, never place onto an occupied one).

It also acts as an independent referee: after the "arm" executes a move, the physical
board must equal the chess position after that move, otherwise the action list was wrong.

The ROS wrapper is sim_world_node.py. Together with brain_node it plays a whole game
with no hardware, no camera and no Gazebo.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional

import chess

from .actions import DISCARD
from .game import GameState, outcome_of

Chooser = Callable[[chess.Board], Optional[chess.Move]]


# ------------------------------------------------------------------ human players
def random_chooser(seed: int = 0) -> Chooser:
    """A human who plays random legal moves (reproducible)."""
    rng = random.Random(seed)

    def choose(board: chess.Board) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        return rng.choice(moves) if moves else None

    return choose


def scripted_chooser(ucis: List[str], then: Optional[Chooser] = None) -> Chooser:
    """A human who plays the given UCI moves in order, then hands over to ``then``."""
    queue = [u.strip() for u in ucis if u.strip()]

    def choose(board: chess.Board) -> Optional[chess.Move]:
        while queue:
            move = chess.Move.from_uci(queue.pop(0))
            if move in board.legal_moves:
                return move
            queue.clear()  # the game left the script; stop following it
        return then(board) if then is not None else None

    return choose


def engine_chooser(engine) -> Chooser:
    """A human who asks a ``chess_brain.game.Engine`` (e.g. a weak Stockfish) for moves."""

    def choose(board: chess.Board) -> Optional[chess.Move]:
        snapshot = GameState(robot_color=board.turn)  # "robot" here just means the side to move
        snapshot.board = board.copy()
        return engine.reply(snapshot)

    return choose


# ------------------------------------------------------------------------ world
class SimWorld:
    def __init__(
        self,
        human_color: chess.Color = chess.WHITE,
        start_fen: str = chess.STARTING_FEN,
        choose: Optional[Chooser] = None,
    ):
        self.board = chess.Board(start_fen)                         # the true game (referee)
        self.phys = chess.BaseBoard(self.board.board_fen())         # the pieces on the table
        self.human_color = human_color
        self.choose: Chooser = choose or random_chooser()
        self.discarded: List[chess.Piece] = []
        self.history: List[str] = []                                # "human e2e4", "robot e7e5"
        self.arm_commands = 0

    # ---------------------------------------------------------------- observation
    def placement(self) -> str:
        """What the camera sees: the piece-placement FEN of the physical board."""
        return self.phys.board_fen()

    def outcome(self) -> Optional[chess.Outcome]:
        return outcome_of(self.board)

    @property
    def game_over(self) -> bool:
        return self.outcome() is not None

    @property
    def human_to_move(self) -> bool:
        return not self.game_over and self.board.turn == self.human_color

    @property
    def robot_to_move(self) -> bool:
        return not self.game_over and self.board.turn != self.human_color

    def result(self) -> Optional[str]:
        outcome = self.outcome()
        return outcome.result() if outcome else None

    # --------------------------------------------------------------------- human
    def play_human_move(self) -> Optional[chess.Move]:
        """Let the human choose and make a move by hand. Returns it (or None if there is none)."""
        if not self.human_to_move:
            return None
        move = self.choose(self.board)
        if move is None:
            return None
        if move not in self.board.legal_moves:
            raise ValueError(f"human chooser returned illegal move {move.uci()} in {self.board.fen()}")
        self.board.push(move)
        self.phys = chess.BaseBoard(self.board.board_fen())  # humans move pieces perfectly
        self.history.append(f"human {move.uci()}")
        return move

    # ----------------------------------------------------------------------- arm
    def execute_arm_actions(self, payload: dict, fail: bool = False) -> dict:
        """Play the arm: execute an ``arm_actions`` payload and return the ``arm_done`` reply.

        Nothing is changed unless the whole command is valid and reproduces the position
        after ``payload["move"]``. With ``fail=True`` the arm refuses without moving
        anything (used to exercise the brain's recovery path).
        """
        self.arm_commands += 1
        done_id = payload.get("id")

        def failed(reason: str) -> dict:
            return {"id": done_id, "status": "failed", "reason": reason}

        if fail:
            return failed("simulated arm failure (nothing moved)")
        if not self.robot_to_move:
            return failed("it is not the robot's turn")
        try:
            move = chess.Move.from_uci(str(payload["move"]))
            actions = list(payload["actions"])
        except (KeyError, ValueError, TypeError) as exc:
            return failed(f"malformed arm_actions: {exc}")
        if move not in self.board.legal_moves:
            return failed(f"{move.uci()} is not a legal move in {self.board.fen()}")

        phys = self.phys.copy()
        discarded: List[chess.Piece] = []
        for index, action in enumerate(actions):
            try:
                pick, place = str(action["pick"]), str(action["place"])
            except (KeyError, TypeError) as exc:
                return failed(f"action {index} malformed: {exc}")
            if pick.startswith("SPARE_"):
                try:
                    _, symbol, colour = pick.split("_")
                    piece = chess.Piece(chess.PIECE_SYMBOLS.index(symbol.lower()), colour == "W")
                except ValueError:
                    return failed(f"action {index}: unknown spare {pick}")
            else:
                try:
                    square = chess.parse_square(pick)
                except ValueError:
                    return failed(f"action {index}: unknown pick location {pick}")
                piece = phys.remove_piece_at(square)
                if piece is None:
                    return failed(f"action {index}: pick from empty square {pick}")
            if place == DISCARD:
                discarded.append(piece)
                continue
            try:
                square = chess.parse_square(place)
            except ValueError:
                return failed(f"action {index}: unknown place location {place}")
            if phys.piece_at(square) is not None:
                return failed(f"action {index}: place onto occupied square {place}")
            phys.set_piece_at(square, piece)

        after = self.board.copy(stack=False)
        after.push(move)
        if phys.board_fen() != after.board_fen():
            return failed(f"actions for {move.uci()} leave {phys.board_fen()}, expected {after.board_fen()}")

        self.phys = phys
        self.discarded.extend(discarded)
        self.board.push(move)
        self.history.append(f"robot {move.uci()}")
        return {"id": done_id, "status": "ok"}

    # ------------------------------------------------------------------- summary
    def summary(self) -> str:
        outcome = self.outcome()
        tail = f"{outcome.result()} ({outcome.termination.name.lower()})" if outcome else "in progress"
        return f"{len(self.board.move_stack)} plies, {len(self.discarded)} captures, {tail}"
