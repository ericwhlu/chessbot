"""Turn-taking state machine for chess_brain. Pure Python: no ROS, no threads, no clock.

The ROS node (brain_node.py) is a thin shell around this class: its callbacks
only *store* inputs (``on_detection`` / ``on_arm_done`` / ``on_cmd``) and a timer
calls ``tick(now)``, which returns the messages to publish. Keeping all logic
here means it can be tested on any machine with ``pytest``.

    SETUP ──start position seen──▶ WAIT_HUMAN ──human move inferred──▶ THINKING
                                      ▲                                   │ engine done
                                      │ arm ok (move applied)             ▼
                                      └───────────────────────────── EXECUTING
                                      ▲                                   │ arm failed / timeout
                                      └──── board = post-move ──── RECOVER ◀┘
                                            (board = pre-move → re-send)

Rules (docs/CHESS_BRAIN_PLAN.md §3.3):

* A detection only counts once the placement has been unchanged for
  ``stable_time`` seconds. "Unchanged" includes "no new message", so this works
  whether vision publishes continuously or once per button press.
* The robot's move is applied to the internal board only after the arm reports
  ``ok`` — never before — so a failed arm cannot desynchronise the game.
* ``arm_done`` messages carry the id of the ``arm_actions`` they answer; any
  other id is ignored (duplicates, late replies).
* Detections are ignored while THINKING / EXECUTING. After the arm finishes,
  only a detection received *after* its ``arm_done`` is trusted.
* ``illegal_move`` / ``arm_mismatch`` are advisory: the machine keeps waiting
  for the board to become a position it understands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

import chess

from .actions import UnsupportedMove, actions_payload, move_to_actions
from .game import GameState, canonical_placement


class State(str, Enum):
    SETUP = "setup"
    WAIT_HUMAN = "wait_human"
    THINKING = "thinking"
    EXECUTING = "executing"
    RECOVER = "recover"
    GAME_OVER = "game_over"
    ERROR = "error"


Output = Tuple[str, Any]  # (topic, payload): "arm_actions" | "status" -> dict, "board_state" -> str


@dataclass
class Config:
    robot_color: chess.Color = chess.BLACK
    start_fen: str = chess.STARTING_FEN
    stable_time: float = 1.0      # s a placement must stay unchanged before it is believed
    arm_timeout: float = 120.0    # s to wait for arm_done before RECOVER
    vision_timeout: float = 0.0   # s without any detection -> "vision_lost"; 0 disables (single-shot vision)
    allow_promotion: bool = False  # needs spare queens on the table
    max_arm_retries: int = 1      # automatic re-sends when the board is still in the pre-move position
    first_action_id: int = 0      # ids continue from here; the node seeds it per process so a stale
                                  # arm_done from an earlier run cannot match a new command


# think(game_snapshot, root_moves_or_None) -> object with .done() and .result() -> chess.Move | None
ThinkFn = Callable[[GameState, Optional[List[chess.Move]]], Any]


class BrainMachine:
    def __init__(self, think: ThinkFn, config: Optional[Config] = None, log: Optional[Callable[[str], None]] = None):
        self.cfg = config or Config()
        self._think = think
        self._log = log or (lambda msg: None)

        self._action_id = int(self.cfg.first_action_id)  # never reset: stale arm_done can't match a new game
        self._placement: Optional[str] = None  # latest valid detection (canonical)
        self._changed_at = 0.0       # when _placement last changed
        self._seen_at: Optional[float] = None  # when any valid detection last arrived
        self._arm_msgs: List[dict] = []
        self._cmds: List[str] = []
        self._last_status: Optional[dict] = None
        self._last_fen: Optional[str] = None
        self._out: List[Output] = []
        self._new_game(gate=float("-inf"))

    # ------------------------------------------------------------------ inputs
    def on_detection(self, text: str, now: float) -> bool:
        """Store a detected placement. Returns False (and ignores it) if malformed."""
        canon = canonical_placement(text)
        if canon is None:
            self._log(f"ignoring malformed detection: {text!r:.80}")
            return False
        if canon != self._placement:
            self._placement = canon
            self._changed_at = now
        self._seen_at = now
        return True

    def on_arm_done(self, text: str, now: float) -> bool:
        """Store an arm_done message: ``{"id": 7, "status": "ok"}`` or ``{"id": 7, "status": "failed", "reason": "..."}``."""
        try:
            msg = json.loads(text)
            done_id = int(msg["id"])
            status = str(msg["status"]).strip().lower()
        except (ValueError, TypeError, KeyError, OverflowError):
            self._log(f"ignoring malformed arm_done: {text!r:.80}")
            return False
        self._arm_msgs.append({"id": done_id, "ok": status == "ok", "reason": str(msg.get("reason", status)), "at": now})
        return True

    def on_cmd(self, text: str) -> None:
        """Queue an operator command: ``start`` | ``reset`` | ``retry``."""
        self._cmds.append(str(text).strip().lower())

    # -------------------------------------------------------------------- tick
    def tick(self, now: float) -> List[Output]:
        """Advance the machine. Returns the messages to publish, in order."""
        self._out = []
        for cmd in self._cmds:
            self._handle_cmd(cmd, now)
        self._cmds = []

        handler = {
            State.SETUP: self._tick_setup,
            State.WAIT_HUMAN: self._tick_wait_human,
            State.THINKING: self._tick_thinking,
            State.EXECUTING: self._tick_executing,
            State.RECOVER: self._tick_recover,
        }.get(self.state)
        if handler is not None:
            handler(now)

        fen = self.game.fen()
        if fen != self._last_fen:
            self._last_fen = fen
            self._out.append(("board_state", fen))
        status = self.status()
        if status != self._last_status:
            self._last_status = status
            self._out.append(("status", status))
        out, self._out = self._out, []
        return out

    def status(self) -> dict:
        s = self.game.status()
        return {
            "state": self.state.value,
            "status": self._status,
            "detail": self._detail,
            "fen": s["fen"],
            "expected_placement": s["placement"],
            "turn": s["turn"],
            "check": s["in_check"],
            "result": s["result"],
            "move_id": self._action_id,
        }

    # ---------------------------------------------------------------- internals
    def _new_game(self, gate: float) -> None:
        self.game = GameState(self.cfg.start_fen, self.cfg.robot_color)
        self.state = State.SETUP
        self._status = "setup"
        self._detail = "waiting for a stable detection of the start position (or cmd: start)"
        self._gate = gate            # detections received before this time are not trusted
        self._handled: Optional[str] = None  # stable placement already evaluated in this state
        self._verified = True        # board seen to match the internal position since the last robot move
        self._future = None
        self._pending_move: Optional[chess.Move] = None
        self._pending_actions: list = []
        self._pending_post: Optional[str] = None
        self._sent_at = 0.0
        self._retries = 0
        self._recover_kind = ""
        self._force_retry = False
        self._arm_msgs = []

    def _set(self, state: State, status: str, detail: str = "") -> None:
        if state != self.state:
            self._handled = None
        self.state = state
        self._status = status
        self._detail = detail

    def _handle_cmd(self, cmd: str, now: float) -> None:
        if cmd == "reset":
            self._log("cmd reset: new game")
            self._new_game(gate=now)
        elif cmd == "start" and self.state == State.SETUP:
            self._begin(now)
        elif cmd == "retry" and self.state == State.RECOVER:
            # Never re-send blind: the arm may have done half the move (king moved,
            # rook not). Re-send only once a fresh detection shows the pre-move board.
            self._force_retry = True
            self._handled = None
            self._detail = f"retry requested: needs a fresh detection showing {self.game.placement()}"
        elif cmd == "retry" and self.state == State.ERROR and self.game.robot_to_move and not self.game.is_game_over():
            self._start_think(now)
        else:
            self._log(f"ignoring cmd {cmd!r} in state {self.state.value}")

    def _stable(self, now: float) -> Optional[str]:
        """The detected placement, if it is fresh (seen after the gate) and stable."""
        if self._placement is None or self._seen_at is None or self._seen_at < self._gate:
            return None
        if now - max(self._changed_at, self._gate) < self.cfg.stable_time:
            return None
        return self._placement

    def _game_over(self) -> None:
        s = self.game.status()
        self._set(State.GAME_OVER, "game_over", f"{s['result']} ({s['termination']})")

    # SETUP ------------------------------------------------------------------
    def _begin(self, now: float) -> None:
        if self.game.is_game_over():
            self._game_over()
        elif self.game.human_to_move:
            self._verified = True
            self._set(State.WAIT_HUMAN, "waiting_human")
        else:
            self._start_think(now)

    def _tick_setup(self, now: float) -> None:
        placement = self._stable(now)
        if placement is None or placement == self._handled:
            return
        if placement == self.game.placement():
            self._begin(now)
        else:
            self._handled = placement
            self._detail = "detected board is not the start position; set up the pieces (or cmd: start)"

    # WAIT_HUMAN -------------------------------------------------------------
    def _tick_wait_human(self, now: float) -> None:
        timeout = self.cfg.vision_timeout
        last_seen = self._gate if self._seen_at is None else max(self._seen_at, self._gate)
        if timeout > 0 and now - last_seen > timeout:
            self._handled = None  # re-evaluate once vision is back
            self._status, self._detail = "vision_lost", f"no detection for more than {timeout:g} s"
            return

        # Single-shot vision may detect just BEFORE arm_done arrives. Such a frame is
        # still trustworthy if it already shows the finished robot move (optionally
        # plus one legal human move): an arm in view never produces exactly that.
        if (
            not self._verified
            and self._seen_at is not None
            and self._seen_at < self._gate
            and self.game.infer_human_move(self._placement).kind in ("no_change", "move")
        ):
            self._gate = self._seen_at

        placement = self._stable(now)
        if placement is None or placement == self._handled:
            return
        self._handled = placement

        result = self.game.infer_human_move(placement)
        if result.kind == "no_change":
            self._verified = True
            self._status, self._detail = "waiting_human", ""
        elif result.kind == "move":
            move = result.move
            self._log(f"human played {move.uci()}")
            self.game.apply(move)
            self._verified = True
            if self.game.is_game_over():
                self._game_over()
            else:
                self._start_think(now)
        else:
            # Lifted piece, half-finished castling, illegal move or detector misread.
            # Advisory only: keep waiting for a position we understand.
            what = "illegal_move" if self._verified else "arm_mismatch"
            self._status = what
            self._detail = f"detected {placement} matches no legal move from {self.game.placement()}"

    # THINKING ---------------------------------------------------------------
    def _supported(self, move: chess.Move) -> bool:
        try:
            move_to_actions(self.game.board, move, allow_promotion=self.cfg.allow_promotion)
            return True
        except UnsupportedMove:
            return False

    def _start_think(self, now: float) -> None:
        legal = list(self.game.board.legal_moves)
        allowed = [m for m in legal if self._supported(m)]
        if not allowed:
            self._set(State.ERROR, "unsupported_move", "every legal robot move needs a promotion the arm cannot do")
            return
        root_moves = None if len(allowed) == len(legal) else allowed
        try:
            self._future = self._think(self.game.copy(), root_moves)
        except Exception as exc:  # e.g. executor shut down
            self._set(State.ERROR, "engine_error", str(exc))
            return
        self._set(State.THINKING, "thinking")

    def _tick_thinking(self, now: float) -> None:
        future = self._future
        if future is None or not future.done():
            return
        self._future = None
        try:
            move = future.result()
        except Exception as exc:
            self._set(State.ERROR, "engine_error", str(exc))
            return
        if move is None:
            self._set(State.ERROR, "engine_error", "engine returned no move")
            return
        try:
            actions = move_to_actions(self.game.board, move, allow_promotion=self.cfg.allow_promotion)
        except (UnsupportedMove, ValueError) as exc:
            self._set(State.ERROR, "unsupported_move", str(exc))
            return
        after = self.game.board.copy(stack=False)
        after.push(move)
        self._pending_move = move
        self._pending_actions = actions
        self._pending_post = after.board_fen()
        self._retries = 0
        if self._placement is not None and self._placement != self.game.placement():
            # A newer detection differs from the position the move was computed for
            # (takeback, bumped piece). Do not command the arm onto an unknown board.
            self._enter_recover(now, "board_changed", "board changed while the robot was thinking")
        else:
            self._send_actions(now)

    # EXECUTING --------------------------------------------------------------
    def _send_actions(self, now: float) -> None:
        self._arm_msgs = []  # a reply cannot legitimately precede its command
        self._force_retry = False
        self._action_id += 1
        self._sent_at = now
        self._out.append(("arm_actions", actions_payload(self._action_id, self._pending_move, self._pending_actions)))
        self._set(State.EXECUTING, "executing", self._pending_move.uci())

    def _take_arm_done(self) -> Optional[dict]:
        """Pop the reply to the current arm_actions id; drop everything else."""
        msgs, self._arm_msgs = self._arm_msgs, []
        for msg in msgs:
            if msg["id"] == self._action_id:
                return msg
            self._log(f"ignoring arm_done for id {msg['id']} (current id {self._action_id})")
        return None

    def _complete_robot_move(self, gate: float, verified: bool) -> None:
        move = self._pending_move
        self._log(f"robot played {move.uci()}")
        self.game.apply(move)
        self._pending_move, self._pending_actions, self._pending_post = None, [], None
        self._gate = gate
        self._verified = verified
        if self.game.is_game_over():
            self._game_over()
        else:
            hint = "" if verified else "robot move done; waiting for a detection taken after the arm finished"
            self._set(State.WAIT_HUMAN, "waiting_human", hint)

    def _tick_executing(self, now: float) -> None:
        msg = self._take_arm_done()
        if msg is None:
            if now - self._sent_at > self.cfg.arm_timeout:
                self._enter_recover(now, "arm_timeout", f"no arm_done within {self.cfg.arm_timeout:g} s")
            return
        if msg["ok"]:
            # Only trust detections that arrive after the arm reported done.
            self._complete_robot_move(gate=msg["at"], verified=False)
        else:
            self._enter_recover(msg["at"], "arm_failed", msg["reason"])

    # RECOVER ----------------------------------------------------------------
    def _recover_detail(self, kind: str, reason: str) -> str:
        move, pre, post = self._pending_move.uci(), self.game.placement(), self._pending_post
        if kind == "board_changed":
            return f"{reason}; restore the board to {pre} so the robot can play {move}"
        if kind == "arm_timeout":
            return (
                f"{reason}; the arm may still be moving - keep hands clear. Waiting for a late arm_done, "
                f"or a detection showing {post} (move {move} done)"
            )
        return f"{reason}; robot move {move} not confirmed. Board must show {pre} (before) or {post} (after)."

    def _enter_recover(self, gate: float, kind: str, reason: str) -> None:
        self._gate = gate            # only detections from now on can decide what happened
        self._recover_kind = kind    # "arm_failed" | "arm_timeout" | "board_changed"
        self._set(State.RECOVER, kind, self._recover_detail(kind, reason))

    def _tick_recover(self, now: float) -> None:
        msg = self._take_arm_done()
        if msg is not None and self._recover_kind == "arm_timeout":
            if msg["ok"]:
                # The arm was only slow: this is the genuine (late) reply to the current id.
                self._complete_robot_move(gate=msg["at"], verified=False)
                return
            self._recover_kind = self._status = "arm_failed"
            self._detail = self._recover_detail("arm_failed", msg["reason"])
            self._handled = None

        placement = self._stable(now)
        if placement is None or placement == self._handled:
            return
        self._handled = placement
        move = self._pending_move.uci()
        if placement == self._pending_post:
            # The move is physically done (arm finished late, or a human completed it).
            self._complete_robot_move(gate=self._gate, verified=True)
        elif placement == self.game.placement():
            # Nothing moved, so sending the full action list is safe. Automatic only
            # after an explicit failure (after a timeout the arm may still be about
            # to execute the first copy) or when the arm was never commanded.
            auto_retry = self._recover_kind == "arm_failed" and self._retries < self.cfg.max_arm_retries
            if self._force_retry or self._recover_kind == "board_changed":
                self._send_actions(now)
            elif auto_retry:
                self._retries += 1
                self._send_actions(now)
            else:
                self._detail = f"robot move {move} was not executed; make it by hand, or send cmd: retry"
        else:
            wait = " once the arm has stopped" if self._recover_kind == "arm_timeout" else ""
            self._detail = (
                f"board matches neither {self.game.placement()} (before) nor {self._pending_post} (after) "
                f"robot move {move}; fix the board by hand{wait}"
            )
