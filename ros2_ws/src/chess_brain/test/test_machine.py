"""Scenario tests for the turn-taking state machine (no ROS, no real engine, fake clock)."""

import json
import random

import chess
import pytest

from chess_brain.actions import DISCARD
from chess_brain.machine import BrainMachine, Config, State

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
STABLE = 1.0


def referee_outcome(board):
    """Same draw policy as GameState.outcome, written independently for the referee."""
    outcome = board.outcome(claim_draw=False)
    if outcome is None and board.halfmove_clock >= 100 and any(board.legal_moves):
        outcome = chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    if outcome is None and board.is_repetition(3):
        outcome = chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
    return outcome


def after(fen, *ucis):
    """Piece placement after playing ``ucis`` from ``fen``."""
    board = chess.Board(fen)
    for uci in ucis:
        board.push_uci(uci)
    return board.board_fen()


class FakeFuture:
    def __init__(self, value=None, exc=None, done=True):
        self._value, self._exc, self._done = value, exc, done

    def done(self):
        return self._done

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._value

    def finish(self, uci):
        self._value, self._done = chess.Move.from_uci(uci), True


class Harness:
    """Drives a BrainMachine with a fake clock and scripted engine replies."""

    def __init__(self, replies=(), chooser=None, **cfg):
        cfg.setdefault("stable_time", STABLE)
        self.now = 100.0
        self.replies = list(replies)   # UCI strings | Exception | "PENDING"
        self.chooser = chooser         # callable(game, root_moves) -> chess.Move, used when replies run out
        self.think_calls = []          # (fen, root_moves)
        self.pending = None
        self.logs = []
        self.out = []
        self.m = BrainMachine(self._think, Config(**cfg), log=self.logs.append)

    def _think(self, game, root_moves):
        self.think_calls.append((game.fen(), root_moves))
        if not self.replies:
            return FakeFuture(self.chooser(game, root_moves))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            return FakeFuture(exc=reply)
        if reply == "PENDING":
            self.pending = FakeFuture(done=False)
            return self.pending
        return FakeFuture(chess.Move.from_uci(reply))

    # --- time and inputs
    def tick(self, dt=0.0, n=1):
        for _ in range(n):
            self.now += dt
            self.out += self.m.tick(self.now)
            dt = 0.0

    def see(self, placement):
        return self.m.on_detection(placement, self.now)

    def settle(self, placement):
        """Detect ``placement``, let it become stable, and let the machine react."""
        self.see(placement)
        self.tick(STABLE + 0.01)
        self.tick(n=3)

    def arm(self, done_id=None, status="ok", **extra):
        payload = {"id": self.m._action_id if done_id is None else done_id, "status": status, **extra}
        return self.m.on_arm_done(json.dumps(payload), self.now)

    # --- outputs
    def topic(self, name):
        return [payload for topic, payload in self.out if topic == name]

    @property
    def status(self):
        return self.topic("status")[-1]

    @property
    def actions(self):
        return self.topic("arm_actions")


def started(**kw):
    """A machine that has seen the start position and is waiting for the human."""
    h = Harness(**kw)
    h.tick()
    h.settle(START)
    return h


# ------------------------------------------------------------------- SETUP
def test_first_tick_publishes_setup_status_and_board_state():
    h = Harness()
    h.tick()
    assert h.m.state == State.SETUP
    assert h.status["state"] == "setup" and h.status["status"] == "setup"
    assert h.topic("board_state") == [chess.STARTING_FEN]


def test_setup_waits_for_stable_start_position():
    h = Harness()
    h.tick()
    h.see(START)
    h.tick(STABLE / 2)
    assert h.m.state == State.SETUP           # not stable yet
    h.tick(STABLE / 2 + 0.01)
    assert h.m.state == State.WAIT_HUMAN
    assert h.status["status"] == "waiting_human"


def test_setup_reports_wrong_board_then_cmd_start_forces():
    h = Harness(replies=["e7e5"])
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    assert h.m.state == State.SETUP
    assert "not the start position" in h.status["detail"]
    h.m.on_cmd("start")
    h.tick(n=3)
    # Forced start: the board already shows 1.e4, which is read as the human's first move.
    assert h.m.state == State.EXECUTING and h.actions[0]["move"] == "e7e5"


def test_cmd_start_without_any_detection():
    h = Harness()
    h.tick()
    h.m.on_cmd("start")
    h.tick()
    assert h.m.state == State.WAIT_HUMAN and h.status["status"] == "waiting_human"


def test_malformed_inputs_are_ignored():
    h = started()
    assert h.see("hello world") is False
    assert h.m.on_arm_done("not json", h.now) is False
    assert h.m.on_arm_done('{"status": "ok"}', h.now) is False  # no id
    h.m.on_cmd("dance")
    h.tick(STABLE + 0.01)
    assert h.m.state == State.WAIT_HUMAN and h.status["status"] == "waiting_human"


# -------------------------------------------------------------- normal turn
def test_full_turn_human_move_engine_reply_arm_ok_verify():
    h = started(replies=["e7e5"])
    h.see(after(chess.STARTING_FEN, "e2e4"))
    h.tick(STABLE / 2)
    assert h.m.state == State.WAIT_HUMAN and not h.think_calls   # not stable yet

    h.tick(STABLE / 2 + 0.01)                                     # human move inferred
    assert h.m.state == State.THINKING
    assert h.think_calls[0][0].startswith(after(chess.STARTING_FEN, "e2e4") + " b")
    assert h.think_calls[0][1] is None                            # no move restriction needed

    h.tick()                                                      # engine reply ready
    assert h.m.state == State.EXECUTING
    assert h.actions == [{"id": 1, "move": "e7e5", "actions": [{"pick": "e7", "place": "e5"}]}]
    # the robot's move is NOT applied until the arm confirms
    assert h.m.game.board.turn == chess.BLACK
    assert h.m.game.placement() == after(chess.STARTING_FEN, "e2e4")

    h.tick(5.0)
    h.arm()
    h.tick()
    assert h.m.state == State.WAIT_HUMAN
    assert h.m.game.placement() == after(chess.STARTING_FEN, "e2e4", "e7e5")
    assert h.topic("board_state")[-1] == h.m.game.fen()

    h.settle(after(chess.STARTING_FEN, "e2e4", "e7e5"))           # post-move verification
    assert h.status["status"] == "waiting_human" and h.m._verified


def test_capture_sends_discard_first():
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    h = Harness(replies=["d8d5"], start_fen=fen)
    h.settle(chess.Board(fen).board_fen())
    h.settle(after(fen, "e4d5"))
    assert h.actions[-1]["actions"] == [{"pick": "d5", "place": DISCARD}, {"pick": "d8", "place": "d5"}]


def test_detections_are_ignored_while_thinking_and_executing():
    h = started(replies=["PENDING"])
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    assert h.m.state == State.THINKING
    h.settle("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR")     # human fidgets during THINK (also plays d4)
    assert h.m.state == State.THINKING and len(h.think_calls) == 1  # not read as a second move
    h.settle(after(chess.STARTING_FEN, "e2e4"))                   # ...and puts it back
    h.pending.finish("e7e5")
    h.tick()
    assert h.m.state == State.EXECUTING
    h.settle(START)                                               # arm occludes / garbage during EXECUTING
    assert h.m.state == State.EXECUTING
    assert h.m.game.placement() == after(chess.STARTING_FEN, "e2e4")


def test_board_changed_while_thinking_holds_the_arm_until_restored():
    h = started(replies=["PENDING"])
    pre = after(chess.STARTING_FEN, "e2e4")
    h.settle(pre)
    h.settle(START)                                               # takeback during THINK
    h.pending.finish("e7e5")
    h.tick()
    assert h.m.state == State.RECOVER and h.status["status"] == "board_changed"
    assert h.actions == []                                        # arm NOT commanded onto an unknown board
    h.settle(pre)                                                 # board restored
    assert h.m.state == State.EXECUTING and h.actions[0]["move"] == "e7e5"


def test_stale_detection_from_before_arm_done_is_not_trusted():
    h = started(replies=["e7e5"])
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    # While the arm moves, vision keeps reporting an old/odd board, last seen before arm_done:
    h.see(START)
    h.tick(10.0)
    h.arm()
    h.tick()
    h.tick(STABLE + 5.0)
    assert h.m.state == State.WAIT_HUMAN
    assert h.status["status"] == "waiting_human"                  # no arm_mismatch from stale data
    assert not h.m._verified
    h.settle(after(chess.STARTING_FEN, "e2e4", "e7e5"))           # fresh detection after arm_done
    assert h.m._verified


def test_human_moves_before_any_verification_detection():
    h = started(replies=["e7e5", "b8c6"])
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    h.arm()
    h.tick()
    # single detection per turn: next thing vision reports already contains the human's reply
    h.settle(after(chess.STARTING_FEN, "e2e4", "e7e5", "g1f3"))
    assert h.m.state == State.EXECUTING
    assert h.actions[-1]["move"] == "b8c6" and h.actions[-1]["id"] == 2


# ------------------------------------------------------ advisory statuses
def test_transient_then_illegal_then_fixed():
    h = started(replies=["e7e5"])
    lifted = "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNBQKBNR"        # e2 pawn in the hand
    h.see(lifted)
    h.tick(STABLE / 2)
    assert h.status["status"] == "waiting_human"                  # transient: no complaint yet
    h.tick(STABLE)
    assert h.status["status"] == "illegal_move" and h.m.state == State.WAIT_HUMAN
    n = len(h.topic("status"))
    h.tick(STABLE, n=5)
    assert len(h.topic("status")) == n                            # reported once, not spammed
    h.settle(START)                                               # piece put back
    assert h.status["status"] == "waiting_human"
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    assert h.m.state == State.EXECUTING


def test_arm_reports_ok_but_board_is_wrong_gives_arm_mismatch():
    h = started(replies=["e7e5"])
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    h.arm()
    h.tick(1.0)
    h.settle("rnbqkbnr/pppp1ppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")     # e7 pawn vanished
    assert h.status["status"] == "arm_mismatch" and h.m.state == State.WAIT_HUMAN
    h.settle(after(chess.STARTING_FEN, "e2e4", "e7e5"))           # human fixes it
    assert h.status["status"] == "waiting_human"


def test_vision_lost_when_enabled():
    h = started(vision_timeout=3.0)
    h.tick(3.5)
    assert h.status["status"] == "vision_lost"
    h.settle(START)
    assert h.status["status"] == "waiting_human"


# ---------------------------------------------------------- arm_done handling
def test_wrong_id_and_duplicate_arm_done_are_ignored():
    h = started(replies=["e7e5", "g8f6"])
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    h.arm(done_id=99)
    h.tick()
    assert h.m.state == State.EXECUTING
    h.arm()
    h.arm()                                                       # duplicate
    h.tick()
    assert h.m.state == State.WAIT_HUMAN
    h.settle(after(chess.STARTING_FEN, "e2e4", "e7e5", "g1f3"))
    assert h.m.state == State.EXECUTING and h.m._action_id == 2
    h.arm(done_id=1)                                              # late duplicate of the previous id
    h.tick()
    assert h.m.state == State.EXECUTING


def test_arm_failed_board_unchanged_resends_once_then_waits():
    h = started(replies=["e7e5"])
    pre = after(chess.STARTING_FEN, "e2e4")
    h.settle(pre)
    h.arm(status="failed", reason="grasp")
    h.tick()
    assert h.m.state == State.RECOVER and h.status["status"] == "arm_failed"
    assert h.m.game.placement() == pre                            # move not applied
    h.settle(pre)                                                 # nothing moved -> automatic re-send
    assert h.m.state == State.EXECUTING
    assert [a["id"] for a in h.actions] == [1, 2] and h.actions[1]["move"] == "e7e5"
    h.arm(status="failed", reason="grasp")
    h.tick()
    h.settle(pre)                                                 # retries exhausted
    assert h.m.state == State.RECOVER and "by hand" in h.status["detail"]
    h.m.on_cmd("retry")
    h.tick()
    assert h.m.state == State.EXECUTING and h.actions[-1]["id"] == 3
    h.arm()
    h.tick()
    assert h.m.state == State.WAIT_HUMAN
    assert h.m.game.placement() == after(chess.STARTING_FEN, "e2e4", "e7e5")


def test_arm_failed_but_human_completes_the_move():
    h = started(replies=["e7e5"])
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    h.arm(status="failed", reason="collision")
    h.tick()
    h.settle("rnbqkbnr/pppp1ppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")     # piece dropped off the board
    assert h.m.state == State.RECOVER and "neither" in h.status["detail"]
    h.settle(after(chess.STARTING_FEN, "e2e4", "e7e5"))
    assert h.m.state == State.WAIT_HUMAN and h.m._verified
    assert h.m.game.board.turn == chess.WHITE


def test_arm_timeout_does_not_auto_resend_and_accepts_late_ok():
    h = started(replies=["e7e5"], arm_timeout=30.0)
    pre = after(chess.STARTING_FEN, "e2e4")
    h.settle(pre)
    h.tick(31.0)
    assert h.m.state == State.RECOVER and h.status["status"] == "arm_timeout"
    h.settle(pre)                                                 # board unchanged: do NOT re-send
    assert h.m.state == State.RECOVER and len(h.actions) == 1
    h.arm()                                                       # the arm was only slow
    h.tick()
    assert h.m.state == State.WAIT_HUMAN
    assert h.m.game.placement() == after(chess.STARTING_FEN, "e2e4", "e7e5")


# ----------------------------------------------------------------- game over
def test_human_checkmate_ends_game_without_calling_engine():
    fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
    h = Harness(start_fen=fen)
    h.settle(chess.Board(fen).board_fen())
    h.settle("R5k1/5ppp/8/8/8/8/5PPP/6K1")
    assert h.m.state == State.GAME_OVER
    assert h.status["status"] == "game_over" and h.status["result"] == "1-0"
    assert h.think_calls == []


def test_robot_checkmate_ends_game_after_arm_ok():
    h = Harness(replies=["a8a1"], start_fen="r6k/8/8/8/8/8/5PPP/6K1 w - - 0 1")
    h.settle("r6k/8/8/8/8/8/5PPP/6K1")
    h.settle("r6k/8/8/8/8/8/5PPP/7K")              # human plays Kh1; ...Ra1 is a back-rank mate
    assert h.m.state == State.EXECUTING and h.actions[-1]["move"] == "a8a1"
    h.arm()
    h.tick()
    assert h.m.state == State.GAME_OVER and h.status["result"] == "0-1"


def test_reset_starts_a_new_game_and_needs_a_fresh_detection():
    h = started(replies=["PENDING"])
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    assert h.m.state == State.THINKING
    h.m.on_cmd("reset")
    h.tick()
    assert h.m.state == State.SETUP and h.m.game.fen() == chess.STARTING_FEN
    h.pending.finish("e7e5")                       # result of the old game's search is discarded
    h.tick(STABLE + 0.01, n=2)
    assert h.m.state == State.SETUP and h.actions == []
    h.settle(START)
    assert h.m.state == State.WAIT_HUMAN


# --------------------------------------------------------- robot plays white
def test_robot_white_thinks_first():
    h = Harness(replies=["e2e4", "g1f3"], robot_color=chess.WHITE)
    h.settle(START)
    assert h.m.state == State.EXECUTING and h.actions[0]["move"] == "e2e4"
    h.arm()
    h.tick()
    assert h.m.state == State.WAIT_HUMAN
    h.settle(after(chess.STARTING_FEN, "e2e4", "e7e5"))          # human (black) replies
    assert h.m.state == State.EXECUTING and h.actions[-1]["move"] == "g1f3"
    assert h.think_calls[-1][0].split()[1] == "w"


def test_stale_arm_done_queued_before_the_command_is_not_its_reply():
    h = started(replies=["e7e5"])
    h.arm(done_id=1)                                              # e.g. left over from a previous run
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    h.tick(n=3)
    assert h.m.state == State.EXECUTING and h.actions[0]["id"] == 1
    assert h.m.game.board.turn == chess.BLACK                     # move not applied by the stale reply


def test_first_action_id_seed():
    h = started(replies=["e7e5"], first_action_id=4200)
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    assert h.actions[0]["id"] == 4201
    h.arm(done_id=1)
    h.tick()
    assert h.m.state == State.EXECUTING


def test_retry_is_never_sent_blind_onto_a_half_executed_move():
    # Robot (black) castles kingside; the arm moves the king and fails on the rook.
    fen = "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4"
    h = Harness(replies=["e8g8"], start_fen=fen)
    h.settle(chess.Board(fen).board_fen())
    pre = after(fen, "d2d3")
    h.settle(pre)
    assert h.actions[0]["actions"] == [{"pick": "e8", "place": "g8"}, {"pick": "h8", "place": "f8"}]
    h.arm(status="failed", reason="grasp")
    h.tick()
    half = "rnbq2kr/pppp1ppp/5n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R"   # king on g8, rook still on h8
    h.m.on_cmd("retry")
    h.tick(STABLE + 0.01, n=3)
    assert h.m.state == State.RECOVER and len(h.actions) == 1     # no detection yet: nothing sent
    h.settle(half)
    assert h.m.state == State.RECOVER and len(h.actions) == 1     # intermediate board: nothing sent
    h.settle(pre)                                                 # human puts the king back
    assert h.m.state == State.EXECUTING and len(h.actions) == 2
    assert h.actions[1]["id"] == 2 and h.actions[1]["move"] == "e8g8"


def test_late_failed_after_timeout_becomes_arm_failed_and_resends():
    h = started(replies=["e7e5"], arm_timeout=30.0)
    pre = after(chess.STARTING_FEN, "e2e4")
    h.settle(pre)
    h.tick(31.0)
    assert h.status["status"] == "arm_timeout"
    h.arm(status="failed", reason="planning")
    h.tick()
    assert h.m.state == State.RECOVER and h.status["status"] == "arm_failed"
    h.settle(pre)
    assert h.m.state == State.EXECUTING and [a["id"] for a in h.actions] == [1, 2]


def test_timeout_then_human_completes_and_late_ok_is_harmless():
    h = started(replies=["e7e5", "b8c6"], arm_timeout=30.0)
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    h.tick(31.0)
    h.settle(after(chess.STARTING_FEN, "e2e4", "e7e5"))           # someone made the robot's move by hand
    assert h.m.state == State.WAIT_HUMAN and h.m.game.board.turn == chess.WHITE
    h.arm(done_id=1)                                              # the slow arm finally answers
    h.tick()
    assert h.m.game.placement() == after(chess.STARTING_FEN, "e2e4", "e7e5")   # not applied twice
    h.settle(after(chess.STARTING_FEN, "e2e4", "e7e5", "g1f3"))
    assert h.m.state == State.EXECUTING and h.actions[-1]["id"] == 2
    assert h.m.game.board.turn == chess.BLACK                     # stale ok did not complete id 2


def test_detection_taken_just_before_arm_done_is_used_if_it_shows_the_finished_move():
    h = started(replies=["e7e5", "b8c6"])
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    post = after(chess.STARTING_FEN, "e2e4", "e7e5")
    h.see(post)                                                   # single-shot vision fires first...
    h.tick(0.3)
    h.arm()                                                       # ...arm_done arrives a moment later
    h.tick()
    assert "after the arm finished" in h.status["detail"]
    h.tick(STABLE + 0.01)
    assert h.m.state == State.WAIT_HUMAN and h.m._verified and h.status["detail"] == ""


def test_reset_during_executing_ignores_the_old_arm_reply():
    h = Harness(replies=["e2e4", "d2d4"], robot_color=chess.WHITE)
    h.settle(START)
    assert h.m.state == State.EXECUTING and h.actions[0]["id"] == 1
    h.m.on_cmd("reset")
    h.tick()
    assert h.m.state == State.SETUP
    h.arm(done_id=1)
    h.tick(STABLE + 0.01, n=2)
    assert h.m.state == State.SETUP and h.m.game.fen() == chess.STARTING_FEN
    h.settle(START)                                               # fresh detection after the reset
    assert h.m.state == State.EXECUTING and h.actions[-1]["id"] == 2
    assert h.m.game.board.turn == chess.WHITE                     # id 2 not completed by the old reply


def test_draw_only_on_actual_third_repetition():
    replies = ["g8f6", "f6g8", "g8f6", "f6g8"]
    h = started(replies=replies)
    board = chess.Board()
    for i, human in enumerate(["g1f3", "f3g1", "g1f3", "f3g1"]):
        board.push_uci(human)
        h.settle(board.board_fen())
        assert h.m.state == State.EXECUTING, (i, h.status)       # engine still asked, even after 4.Ng1
        board.push_uci(replies[i])
        h.arm()
        h.tick()
    assert h.m.state == State.GAME_OVER                           # start position, third occurrence
    assert h.status["result"] == "1/2-1/2" and "threefold" in h.status["detail"]


def test_illegal_position_is_reported_again_after_being_fixed_and_repeated():
    h = started()
    lifted = "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNBQKBNR"
    h.settle(lifted)
    h.settle(START)
    h.settle(lifted)
    assert [s["status"] for s in h.topic("status")][-3:] == ["illegal_move", "waiting_human", "illegal_move"]


def test_idle_ticks_publish_nothing():
    h = started(replies=["PENDING"])
    n = len(h.out)
    h.tick(0.1, n=50)                                             # WAIT_HUMAN
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    m = len(h.out)
    h.tick(0.1, n=50)                                             # THINKING
    assert len(h.out) == m and m > n
    h.pending.finish("e7e5")
    h.tick()
    k = len(h.out)
    h.tick(0.1, n=50)                                             # EXECUTING
    assert len(h.out) == k


def test_arm_done_parsing():
    h = started(replies=["e7e5", "g8f6"])
    assert h.m.on_arm_done('{"id": 1e999, "status": "ok"}', h.now) is False
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    assert h.m.on_arm_done('{"id": 1, "status": " OK "}', h.now) is True
    h.tick()
    assert h.m.state == State.WAIT_HUMAN
    h.settle(after(chess.STARTING_FEN, "e2e4", "e7e5", "g1f3"))
    assert h.m.on_arm_done('{"id": 2, "status": "success"}', h.now) is True   # anything but "ok" is a failure
    h.tick()
    assert h.m.state == State.RECOVER and "success" in h.status["detail"]


def test_engine_error_then_cmd_retry_thinks_again():
    h = started(replies=[RuntimeError("engine died"), "e7e5"])
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    assert h.m.state == State.ERROR
    h.m.on_cmd("retry")
    h.tick(n=3)
    assert h.m.state == State.EXECUTING and h.actions[0]["move"] == "e7e5"


def test_stable_time_zero_accepts_on_the_next_tick():
    h = Harness(replies=["e7e5"], stable_time=0.0)
    h.see(START)
    h.tick()
    assert h.m.state == State.WAIT_HUMAN
    h.see(after(chess.STARTING_FEN, "e2e4"))
    h.tick(n=2)
    assert h.m.state == State.EXECUTING


# ------------------------------------------------------------ engine problems
def test_engine_error_goes_to_error_state_and_reset_recovers():
    h = started(replies=[RuntimeError("engine died")])
    h.settle(after(chess.STARTING_FEN, "e2e4"))
    assert h.m.state == State.ERROR and h.status["status"] == "engine_error"
    assert "engine died" in h.status["detail"]
    h.m.on_cmd("reset")
    h.tick()
    assert h.m.state == State.SETUP


def test_promotions_are_excluded_from_the_search_when_unsupported():
    fen = "8/8/8/8/8/k7/p6P/7K w - - 0 1"          # black can play a1=Q etc. or king moves
    h = Harness(replies=["a3b3"], start_fen=fen)
    h.settle(chess.Board(fen).board_fen())
    h.settle(after(fen, "h2h3"))
    roots = h.think_calls[0][1]
    assert roots is not None and all(m.promotion is None for m in roots)
    assert chess.Move.from_uci("a3b3") in roots
    assert h.m.state == State.EXECUTING


def test_queen_promotion_allowed_when_enabled():
    fen = "8/8/8/8/8/k7/p6P/7K w - - 0 1"
    h = Harness(replies=["a2a1q"], start_fen=fen, allow_promotion=True)
    h.settle(chess.Board(fen).board_fen())
    h.settle(after(fen, "h2h3"))
    roots = h.think_calls[0][1]
    assert roots is not None                       # under-promotions are still excluded
    assert chess.Move.from_uci("a2a1q") in roots and chess.Move.from_uci("a2a1n") not in roots
    assert h.actions[-1]["actions"] == [{"pick": "a2", "place": DISCARD}, {"pick": "SPARE_Q_B", "place": "a1"}]


# ------------------------------------------------- end-to-end simulated games
def execute_on_physical_board(phys: chess.BaseBoard, actions):
    """Play arm actions on a model of the physical board, asserting the arm-team invariants."""
    for a in actions:
        if a["pick"].startswith("SPARE_"):
            _, sym, col = a["pick"].split("_")
            piece = chess.Piece(chess.PIECE_SYMBOLS.index(sym.lower()), col == "W")
        else:
            piece = phys.remove_piece_at(chess.parse_square(a["pick"]))
            assert piece is not None, f"pick from empty square {a['pick']}"
        if a["place"] != DISCARD:
            sq = chess.parse_square(a["place"])
            assert phys.piece_at(sq) is None, f"place onto occupied square {a['place']}"
            phys.set_piece_at(sq, piece)


@pytest.mark.parametrize("seed", range(12))
def test_simulated_game_vision_and_arm_in_the_loop(seed):
    rng = random.Random(seed)
    robot = chess.BLACK if seed % 2 == 0 else chess.WHITE

    def chooser(game, root_moves):
        return rng.choice(root_moves if root_moves is not None else list(game.board.legal_moves))

    h = Harness(chooser=chooser, robot_color=robot, allow_promotion=True)
    ref = chess.Board()                      # independent referee
    phys = chess.BaseBoard()                 # what the camera would see
    h.settle(phys.board_fen())

    for _ in range(60):
        if referee_outcome(ref) is not None:
            break
        if ref.turn == robot:
            assert h.m.state == State.EXECUTING, (h.m.state, h.status)
            payload = h.actions[-1]
            move = chess.Move.from_uci(payload["move"])
            assert move in ref.legal_moves
            execute_on_physical_board(phys, payload["actions"])
            ref.push(move)
            assert phys.board_fen() == ref.board_fen()
            h.tick(2.0)
            h.arm(done_id=payload["id"])
            h.tick()
            if referee_outcome(ref) is None:
                h.settle(phys.board_fen())   # verification detection
                assert h.status["status"] == "waiting_human" and h.m._verified
        else:
            assert h.m.state == State.WAIT_HUMAN, (h.m.state, h.status)
            legal = list(ref.legal_moves)
            special = [m for m in legal if ref.is_castling(m) or ref.is_en_passant(m) or m.promotion]
            move = rng.choice(special) if special and rng.random() < 0.7 else rng.choice(legal)
            ref.push(move)
            phys = chess.BaseBoard(ref.board_fen())   # the human moves pieces perfectly
            h.settle(phys.board_fen())
        assert h.m.game.board.board_fen() == ref.board_fen() or h.m.state == State.EXECUTING

    if referee_outcome(ref) is not None:
        assert h.m.state == State.GAME_OVER
        assert h.status["result"] == referee_outcome(ref).result()
    assert h.m.game.fen() == ref.fen() or h.m.state == State.EXECUTING
    assert all(s["status"] not in ("illegal_move", "arm_mismatch", "engine_error", "unsupported_move")
               for s in h.topic("status"))
