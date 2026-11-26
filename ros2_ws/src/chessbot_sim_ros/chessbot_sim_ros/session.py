"""Who does what, and when, on the simulated side of a chess_brain game. Pure Python: no ROS,
no MuJoCo, so the order of events can be tested with pytest against the real state machine.

chess_brain decides and waits. ``WorldSession`` is everything it waits for: the camera, the
arm, the opponent, and an operator. It talks to a *world* (the MuJoCo simulation, or a fake in
the tests) and exchanges the messages of the ``chessbot/*`` topics with chess_brain:

    in   on_status(text)          chessbot/status           what chess_brain is doing
         on_arm_actions(text)     chessbot/arm_actions      a move for the arm
         on_detect_request(text)  chessbot/detect_request   "look at the board now"
    out  tick() -> [(topic, payload), ...]
         ("detected_occupancy", ["a1", ...])   the occupied squares the camera sees
         ("arm_done", {"id", "status", ...})   the answer to an arm command
         ("cmd", "move e4d5")                  the operator answering status ambiguous_move

``tick`` does one job at a time, in this order:

1. **An arm command.** A repeated id is ignored and a move that is not legal on the session's
   own board is answered ``failed`` without moving anything. Otherwise the arm executes it. If
   that fails, the operator finishes the move by hand BEFORE ``failed`` is reported: every
   later scan then shows the finished move and chess_brain leaves RECOVER through "the board
   shows the post-move position". (A pre-move scan after the report would make chess_brain
   re-send the command, and the arm would play the move a second time.) A command waits while
   the opponent's ply is still open: topics are not ordered among themselves, and the command
   can overtake the status that closes the ply.
2. **A scan that is owed**: asked for on ``detect_request``, or the first.
3. **The opponent's move**: only when chess_brain says it is calmly waiting for the human on
   the same position (state ``wait_human``, status ``waiting_human``, ``verified`` true: it has
   seen the robot's move on the board), and ``human_delay`` after the last ply. The scan that
   follows is the opponent's "I have moved".
4. **The operator**: answers ``ambiguous_move`` with the move that was played if it is one of
   the ``candidates``, and again every ``operator_period`` while the question stays open (an
   answer that arrives before the observation is stable is refused).
5. **The heartbeat**: scan mode ``continuous`` scans every ``rescan_period`` while idle. Scan
   mode ``on_request`` stays quiet while chess_brain waits for a person or for its engine, and
   scans at that period only while chess_brain is waiting for the camera: before the game has
   started, while a ply is open, after a robot move it has not verified, in RECOVER, and while
   it reports anything but ``waiting_human``. That is the opponent pressing the button again
   when nothing happened; without it one lost frame would end the game.

A ply counts once chess_brain's position equals the session's. The session ends after
``plies`` half-moves (chess_brain may then be in the middle of its next turn; whoever runs the
two stops both), at game over, or with ``failed`` set: chess_brain does not start, stops in
ERROR, does not arrive at a ply within ``ply_timeout``, or nothing at all happens for
``idle_timeout`` (a lost arm command, a chess_brain that went its own way).

The world (see chessbot_sim/world.py for the MuJoCo one):

    scan() -> frozenset of square names, or None if the camera cannot read the board
    execute(board, move, actions) -> None if the arm did it and the camera confirms, else why not
    finish_by_hand(board, move)    the operator puts the pieces where ``move`` leaves them
    choose_human(board) -> chess.Move or None
    play_human(board, move)        the opponent's hand
    record(entry)                  one finished ply, for the summary (may add to ``entry``)

``board`` is always the position BEFORE the move.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import chess

Output = Tuple[str, Any]
SCAN_MODES = ("continuous", "on_request")


@dataclass
class SessionConfig:
    robot_color: chess.Color = chess.BLACK
    start_fen: str = chess.STARTING_FEN
    plies: int = 12               # stop after this many half-moves; 0 = play to the end
    scan_mode: str = "continuous"
    rescan_period: float = 2.0    # s between heartbeat scans
    human_delay: float = 1.5      # s the opponent takes to move, counted from the last ply
    operator_period: float = 2.0  # s between repeats of the operator's answer
    start_timeout: float = 120.0  # s for chess_brain to leave SETUP
    ply_timeout: float = 60.0     # s for chess_brain to arrive at a ply after it was played
    idle_timeout: float = 60.0    # s without any ply being played or arrived at (> human_delay)


class WorldSession:
    def __init__(self, world, config: Optional[SessionConfig] = None,
                 clock: Callable[[], float] = time.monotonic, log: Optional[Callable[[str], None]] = None):
        self.cfg = config or SessionConfig()
        if self.cfg.scan_mode not in SCAN_MODES:
            raise ValueError(f"scan_mode must be one of {SCAN_MODES}, got {self.cfg.scan_mode!r}")
        if self.cfg.human_delay >= self.cfg.idle_timeout:
            raise ValueError("human_delay must be shorter than idle_timeout")
        self.world = world
        self.board = chess.Board(self.cfg.start_fen)   # the position on the table, as the opponent knows it
        self.plies: List[dict] = []                    # finished plies
        self.finished: Optional[str] = None            # why the session ended; None while it runs
        self.failed = False
        self._clock = clock
        self._log = log or (lambda message: None)
        self._brain: Optional[dict] = None             # latest status of chess_brain
        self._started = False                          # chess_brain has left SETUP
        self._born: Optional[float] = None             # time of the first tick
        self._command: Optional[dict] = None           # arm command not executed yet
        self._seen_ids: set = set()
        self._scan_owed = True                         # the first scan
        self._scanned_at = float("-inf")
        self._answered_at = float("-inf")
        self._moved_at: Optional[float] = None         # when a ply was last played or arrived at
        self._open: Optional[dict] = None              # the ply chess_brain has not arrived at yet

    # ------------------------------------------------------------------ inputs
    def on_status(self, text: str) -> None:
        try:
            status = json.loads(text)
            state = str(status["state"])
        except (ValueError, TypeError, KeyError, RecursionError):
            self._log(f"ignoring malformed status: {text!r:.80}")
            return
        self._brain = status
        if state != "setup":
            self._started = True

    def on_arm_actions(self, text: str) -> None:
        try:
            command = json.loads(text)
            parsed = {
                "id": int(command["id"]),
                "move": str(command["move"]),
                "actions": [{"pick": str(a["pick"]), "place": str(a["place"])} for a in command["actions"]],
            }
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
            self._log(f"ignoring malformed arm_actions: {text!r:.80}")
            return
        if self._command is not None and self._command["id"] != parsed["id"]:
            self._log(f"arm command {self._command['id']} was replaced by {parsed['id']} before it ran")
        self._command = parsed

    def on_detect_request(self, text: str) -> None:
        self._scan_owed = True

    # -------------------------------------------------------------------- tick
    def tick(self) -> List[Output]:
        """Do the next job, if there is one. Returns the messages to publish, in order."""
        out: List[Output] = []
        if self.finished is not None:
            return out
        now = self._clock()
        if self._born is None:
            self._born = now
        if self._progress(now):
            return out

        opponents_ply_open = self._open is not None and self._open["who"] == "human"
        if self._command is not None and not opponents_ply_open:
            self._run_arm(out)
        elif self._scan_owed:
            self._scan(out)
        elif self._opponent_may_move(now) and self._opponent_moves(out):
            pass
        else:
            self._operator_answers(now, out)
            heartbeat = self.cfg.scan_mode == "continuous" or self._brain_waits_for_the_camera()
            if heartbeat and now - self._scanned_at >= self.cfg.rescan_period:
                self._scan(out)
        return out

    @property
    def clean(self) -> bool:
        """The game ran, and no move needed the operator's hands (an answer is only a word)."""
        return not self.failed and all(ply["outcome"] in ("ok", "ambiguous") for ply in self.plies)

    # --------------------------------------------------------------- internals
    def _brain_fen(self) -> Optional[str]:
        return None if self._brain is None else self._brain.get("fen")

    def _brain_calm(self) -> bool:
        """chess_brain waits for the human, with nothing unsettled about the board."""
        brain = self._brain
        return (brain is not None and brain.get("state") == "wait_human"
                and brain.get("status") == "waiting_human" and brain.get("verified") is True)

    def _brain_waits_for_the_camera(self) -> bool:
        """chess_brain cannot go on without another frame (so not: for a person, or its engine)."""
        if not self._started or self._open is not None:
            return True
        state = self._brain.get("state")
        return state == "recover" or (state == "wait_human" and not self._brain_calm())

    def _brain_says(self) -> str:
        if self._brain is None:
            return "no status received from chess_brain"
        return f"chess_brain: {self._brain.get('state')} / {self._brain.get('status')} {self._brain.get('detail', '')}".strip()

    def _finish(self, why: str, failed: bool = False) -> bool:
        self.finished, self.failed = why, failed
        self._log(("stopped: " if failed else "finished: ") + why)
        return True

    def _progress(self, now: float) -> bool:
        """Close the open ply once chess_brain has arrived at it; decide whether the session ends."""
        state = None if self._brain is None else self._brain.get("state")
        if state == "error":
            return self._finish(self._brain_says(), failed=True)
        if not self._started:
            if now - self._born > self.cfg.start_timeout:
                return self._finish(f"the game did not start within {self.cfg.start_timeout:g} s ({self._brain_says()})",
                                    failed=True)
            return False
        if self._moved_at is None:
            self._moved_at = now                       # the game has started

        if self._open is not None:
            if self._brain_fen() == self.board.fen():
                ply, self._open = self._open, None
                self._moved_at = now
                entry = {"ply": len(self.plies) + 1, "who": ply["who"], "move": ply["move"].uci(),
                         "outcome": ply["outcome"], "detail": ply["detail"], "fen": self.board.fen()}
                self.plies.append(entry)
                self.world.record(entry)
            elif now - self._open["since"] > self.cfg.ply_timeout:
                ply, self._open = self._open, None
                if ply["who"] == "human":
                    outcome, detail = "not recognised", self._brain_says()
                else:
                    outcome = "operator"
                    detail = "; ".join(part for part in (ply["detail"], "not confirmed: " + self._brain_says()) if part)
                entry = {"ply": len(self.plies) + 1, "who": ply["who"], "move": ply["move"].uci(),
                         "outcome": outcome, "detail": detail, "fen": self.board.fen()}
                self.plies.append(entry)
                self.world.record(entry)
                return self._finish(f"chess_brain did not arrive at {ply['who']} move {entry['move']} within "
                                    f"{self.cfg.ply_timeout:g} s ({self._brain_says()})", failed=True)

        if self._open is None:
            if state == "game_over" and self._brain_fen() == self.board.fen():
                return self._finish(f"game over: {self._brain.get('detail', '')}".strip())
            if self.cfg.plies > 0 and len(self.plies) >= self.cfg.plies:
                return self._finish(f"{len(self.plies)} plies played")
            if now - self._moved_at > self.cfg.idle_timeout:
                return self._finish(f"nothing happened for {self.cfg.idle_timeout:g} s ({self._brain_says()})",
                                    failed=True)
        return False

    def _scan(self, out: List[Output]) -> None:
        self._scan_owed = False
        squares = self.world.scan()
        self._scanned_at = self._clock()
        if squares is None:
            self._log("the camera could not read the board; nothing published")
        else:
            out.append(("detected_occupancy", sorted(squares)))

    def _run_arm(self, out: List[Output]) -> None:
        command, self._command = self._command, None
        if command["id"] in self._seen_ids:
            self._log(f"ignoring arm command {command['id']}: already handled")
            return
        self._seen_ids.add(command["id"])
        try:
            move = chess.Move.from_uci(command["move"])
        except ValueError:
            move = None
        if (move is None or self._open is not None or self.board.turn != self.cfg.robot_color
                or move not in self.board.legal_moves):
            reason = f"{command['move']} is not a move for the robot in {self.board.fen()}"
            self._log(f"refusing arm command {command['id']}: {reason}")
            out.append(("arm_done", {"id": command["id"], "status": "failed", "reason": reason}))
            return

        reason = self.world.execute(self.board, move, command["actions"])
        outcome, detail = "ok", ""
        if reason is not None:
            self._log(f"arm command {command['id']} ({command['move']}) failed: {reason}; finished by hand")
            self.world.finish_by_hand(self.board, move)
            outcome, detail = "operator", reason
        self.board.push(move)
        self._moved_at = self._clock()
        self._open = {"who": "robot", "move": move, "since": self._moved_at, "outcome": outcome, "detail": detail}
        reply = {"id": command["id"], "status": "ok" if reason is None else "failed"}
        if reason is not None:
            reply["reason"] = reason
        out.append(("arm_done", reply))

    def _opponent_may_move(self, now: float) -> bool:
        return (
            self._open is None
            and self._brain_calm()
            and self._brain_fen() == self.board.fen()
            and self.board.turn != self.cfg.robot_color
            and now - self._moved_at >= self.cfg.human_delay
        )

    def _opponent_moves(self, out: List[Output]) -> bool:
        move = self.world.choose_human(self.board)
        if move is None:
            return False
        self.world.play_human(self.board, move)
        self.board.push(move)
        self._moved_at = self._clock()
        self._open = {"who": "human", "move": move, "since": self._moved_at, "outcome": "ok", "detail": ""}
        self._scan(out)
        return True

    def _operator_answers(self, now: float, out: List[Output]) -> None:
        ply = self._open
        if (ply is None or ply["who"] != "human" or self._brain is None
                or self._brain.get("status") != "ambiguous_move"
                or ply["move"].uci() not in (self._brain.get("candidates") or ())   # a question about another frame
                or now - self._answered_at < self.cfg.operator_period):
            return
        if ply["outcome"] != "ambiguous":
            ply["outcome"], ply["detail"] = "ambiguous", str(self._brain.get("detail", ""))
        self._answered_at = now
        out.append(("cmd", f"move {ply['move'].uci()}"))
