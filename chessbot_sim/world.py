"""The MuJoCo scene as the *world* of a chess_brain game played over messages.

play_game.py runs a game as one loop that calls chess_brain directly. Over ROS 2 the game is
driven from the other side: chess_brain (brain_node) decides, and whoever owns the simulation
answers its messages. ``MujocoWorld`` is that owner's view of the scene, the interface
``chessbot_sim_ros.session.WorldSession`` drives:

    scan()                          one camera frame through the perception -> occupied squares
    execute(board, move, actions)   the arm plays a robot move, the camera confirms it
    finish_by_hand(board, move)     the simulated operator completes a move the arm failed
    choose_human(board)             the simulated opponent's choice
    play_human(board, move)         ... and hand
    record(entry)                   one finished ply: summary line and camera image

It is a thin adapter over ``Match`` of play_game.py, so both ways of running a game use the
same scan, the same pick-and-place and the same simulated hands. As there, the robot's side
(scan, execute) uses only the camera; MuJoCo's ground truth is used by the opponent's hand, by
the operator, and for the ``pieces_match_game`` column of the summary.
"""
import json
from types import SimpleNamespace

import chess

from .gameplay import Report
from .manipulation import OPEN_GRIP
from .play_game import Match, difference, occupancy_after, quiet, squares_of
from .simulation import Simulation


class MujocoWorld:
    def __init__(self, out="game_run", robot="black", seed=7, script="", video=False):
        self.arguments = {"out": str(out), "robot": robot, "seed": seed, "script": script, "video": video}
        self.sim = Simulation.from_files(movable_pieces=True)
        self.match = Match(SimpleNamespace(**self.arguments), self.sim, engine=None)
        self.report = Report()
        self.frame = None            # the latest camera frame, for the image of a finished ply
        self._removed = []           # bodies captured by the robot move in progress
        self._closed = False
        self.match.arm.hold(0.8, OPEN_GRIP, quiet, "")     # let the scene settle, as play_game does

    # ------------------------------------------------------------------ camera
    def scan(self):
        first = self.frame is None
        self.frame, found = self.match.scan()
        if found is None:
            return None
        if first:
            self.match.recorder.snapshot("00_start", self.frame.bgr, f"start: {len(found)} pieces seen")
        return frozenset(found)

    # --------------------------------------------------------------------- arm
    def execute(self, board, move, actions):
        """Play robot ``move`` (``actions`` from chess_brain) on the PRE-move ``board``.

        Returns None if the arm did it and the camera shows exactly the squares the move should
        leave occupied; otherwise what went wrong. The piece book counts the move as played
        either way: after a failure the caller has the operator finish it (finish_by_hand).
        """
        match = self.match
        match.recorder.caption = f"robot  {move.uci()}"
        expected = occupancy_after(board, move)
        problem = None
        for action in actions:
            problem = match.pick_and_place(SimpleNamespace(pick=action["pick"], place=action["place"]))
            if problem is not None:
                break
        if problem is None:
            self.frame, found = match.look_for(expected)
            if found is None:
                problem = "the camera could not locate the board after the move"
            elif frozenset(found) != expected:
                problem = "the camera does not confirm the move: " + difference(expected, frozenset(found))
        self._removed = match.book.apply(board, move).removed
        if problem is not None:
            really = match.hand.truth() == expected
            problem += " (the pieces really were in place)" if really else " (the pieces really were not in place)"
        return problem

    def finish_by_hand(self, board, move):
        """The operator puts every piece where the game says it is. Simulation only."""
        self.match.hand.tidy(self._removed)
        self.match.arm.hold(0.4, OPEN_GRIP, quiet, "")
        self.frame, _ = self.match.scan()

    # ---------------------------------------------------------------- opponent
    def choose_human(self, board):
        return self.match.human.choose(board)

    def play_human(self, board, move):
        match = self.match
        match.recorder.caption = f"human  {move.uci()}"
        match.hand.play(match.book.apply(board, move))
        match.arm.hold(0.4, OPEN_GRIP, match.recorder.motion, "Opponent moves")

    # ------------------------------------------------------------------ report
    def record(self, entry):
        """A ply chess_brain has arrived at. Adds the ground-truth check to ``entry``."""
        entry["pieces_match_game"] = bool(self.match.hand.truth() == squares_of(chess.Board(entry["fen"])))
        self.report.add(**entry)
        note = "" if entry["outcome"] == "ok" else f"  <- {entry['outcome']}: {entry['detail']}"
        if not entry["pieces_match_game"]:
            note += "  [simulation: the pieces do not match the game]"
        print(f"ply {entry['ply']:>2}  {entry['who']:5}  {entry['move']:6}{note}", flush=True)
        if self.frame is not None:
            self.match.recorder.snapshot(
                f"{entry['ply']:02d}_{entry['who']}_{entry['move']}", self.frame.bgr,
                f"ply {entry['ply']}  {entry['who']}  {entry['move']}  {entry['outcome']}")

    def finish(self, result, failed, fen):
        """Write summary.json and return the summary lines. ``result`` is why the game ended."""
        lines = self.report.lines()
        lines.append(("stopped: " if failed else "finished: ") + result)
        lines.append(f"final position: {fen}")
        directory = self.match.recorder.directory
        summary = {"arguments": self.arguments, "plies": self.report.plies, "result": result,
                   "failed": bool(failed), "final_fen": fen, "summary": lines}
        (directory / "summary.json").write_text(json.dumps(summary, indent=2))
        lines.append(f"images and summary.json are in {directory}")
        return lines

    def close(self):
        if not self._closed:
            self._closed = True
            self.match.recorder.close()
            self.sim.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
