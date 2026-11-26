"""The state machine driven by a detector that only reports WHICH squares are occupied.

Same harness as test_machine.py (fake clock, scripted engine). What is specific to occupancy:
a capture needs a second frame, a frame from before arm_done counts for less, and moves that
look the same from above are put to the operator (``move <uci>``).
"""

import json
import random

import chess
import pytest

from chess_brain.actions import DISCARD
from chess_brain.machine import State
from test_machine import START, STABLE, Harness, after, execute_on_physical_board, referee_outcome

S = chess.STARTING_FEN
AFTER_E4_D5 = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
TWO_CAPTURES = "4k3/8/8/2p1p3/3P4/8/8/4K3 w - - 0 1"      # the d4 pawn can take on c5 or on e5
PROMOTION = "4k3/P6p/8/8/8/8/7P/4K3 w - - 0 1"           # pawns on the h-file keep K+N from being a draw
BAD_STATUSES = ("illegal_move", "arm_mismatch", "engine_error", "unsupported_move")


def squares(fen, *ucis):
    """Names of the occupied squares after playing ``ucis`` from ``fen``."""
    board = chess.Board(fen)
    for uci in ucis:
        board.push_uci(uci)
    return {chess.square_name(square) for square in chess.SQUARES if board.piece_at(square)}


def occ(fen, *ucis, without=()):
    """The message a one-class detector publishes: a JSON list of occupied squares."""
    return json.dumps(sorted(squares(fen, *ucis) - set(without)))


def see(h, message):
    return h.m.on_occupancy(message, h.now)


def settle(h, message):
    """Report ``message``, let it become stable, and let the machine react."""
    see(h, message)
    h.tick(STABLE + 0.01)
    h.tick(n=3)


def started(start_fen=S, **kw):
    """A machine that has seen the start position (as occupancy) and waits for the human."""
    h = Harness(start_fen=start_fen, **kw)
    h.tick()
    settle(h, occ(start_fen))
    assert h.m.state == State.WAIT_HUMAN
    return h


# ------------------------------------------------------------ the normal turn
def test_whole_turn_from_occupied_squares_only():
    h = started(replies=["e7e5"])
    settle(h, occ(S, "e2e4"))                                     # a quiet move: one frame is enough
    assert h.m.state == State.EXECUTING
    occupied = sorted({f + r for r in "1278" for f in "abcdefgh"} - {"e2"} | {"e4"})
    assert h.actions == [
        {"id": 1, "move": "e7e5", "actions": [{"pick": "e7", "place": "e5"}], "occupied": occupied}
    ]
    h.arm()
    h.tick()
    assert h.m.state == State.WAIT_HUMAN and not h.m._verified
    settle(h, occ(S, "e2e4", "e7e5"))                             # post-move verification
    assert h.status["status"] == "waiting_human" and h.m._verified
    assert not any(s["status"] in BAD_STATUSES for s in h.topic("status"))


def test_en_passant_needs_no_second_frame():
    h = started("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1", replies=["e8g8"])
    settle(h, occ(h.m.game.fen(), "e5d6"))                        # two squares emptied, one filled
    assert h.m.state == State.EXECUTING and h.m.game.board.piece_at(chess.D6) is not None
    assert h.actions[-1]["actions"] == [{"pick": "e8", "place": "g8"}, {"pick": "h8", "place": "f8"}]


@pytest.mark.parametrize("uci,rook", [("e1g1", chess.F1), ("e1c1", chess.D1)])
def test_human_castling_needs_no_second_frame(uci, rook):
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    h = started(fen, replies=["a8b8"])
    settle(h, occ(fen, uci))
    assert h.m.state == State.EXECUTING and "confirming_capture" not in [s["status"] for s in h.topic("status")]
    assert h.m.game.board.piece_at(rook) == chess.Piece(chess.ROOK, chess.WHITE)


# ---------------------------------------------- a capture needs a second frame
def test_capture_is_applied_only_after_a_second_frame():
    h = started(AFTER_E4_D5, replies=["d8d5"])
    settle(h, occ(AFTER_E4_D5, "e4d5"))
    assert h.m.state == State.WAIT_HUMAN and h.status["status"] == "confirming_capture"
    assert "e4d5" in h.status["detail"] and h.topic("detect_request")[-1] == "confirm_capture"
    assert not h.think_calls and not h.actions

    h.tick(10.0)                                                  # no new frame: nothing, however long
    h.tick(n=3)
    assert h.status["status"] == "confirming_capture" and not h.think_calls

    settle(h, occ(AFTER_E4_D5, "e4d5"))                           # the second frame shows the same squares
    assert h.m.state == State.EXECUTING
    assert h.actions[-1]["actions"] == [{"pick": "d5", "place": DISCARD}, {"pick": "d8", "place": "d5"}]


def test_a_piece_in_the_hand_is_not_played_as_a_capture():
    """The e4 pawn lifted (or missed by the detector) is, from above, exactly e4xd5."""
    h = started(AFTER_E4_D5, replies=["c8f5"])
    lifted = occ(AFTER_E4_D5, without=["e4"])
    assert lifted == occ(AFTER_E4_D5, "e4d5")
    settle(h, lifted)
    assert h.status["status"] == "confirming_capture"

    settle(h, occ(AFTER_E4_D5))                                   # the pawn is back on e4
    assert h.status["status"] == "waiting_human" and not h.think_calls and not h.actions
    assert h.m.game.placement() == chess.Board(AFTER_E4_D5).board_fen() and h.m.game.human_to_move

    settle(h, lifted)                                             # lifted again: again only a question,
    assert h.status["status"] == "confirming_capture" and not h.think_calls   # the first frame does not count
    settle(h, occ(AFTER_E4_D5))
    assert h.status["status"] == "waiting_human" and not h.think_calls

    settle(h, occ(AFTER_E4_D5, "e4e5"))                           # and the game goes on
    assert h.m.state == State.EXECUTING and h.actions[-1]["move"] == "c8f5"


def test_each_new_candidate_capture_is_confirmed_on_its_own():
    fen = "4k3/8/8/2p1p3/1P3P2/8/8/4K3 w - - 0 1"                 # b4xc5 and f4xe5 are both possible
    h = started(fen, replies=["e8e7"])
    settle(h, occ(fen, "b4c5"))
    assert h.status["status"] == "confirming_capture" and "b4c5" in h.status["detail"]
    settle(h, occ(fen, "f4e5"))                                   # a different capture: asked again
    assert h.status["status"] == "confirming_capture" and "f4e5" in h.status["detail"]
    assert not h.think_calls
    settle(h, occ(fen, "f4e5"))
    assert h.m.state == State.EXECUTING and h.m.game.board.piece_at(chess.F4) is None


def test_one_frame_never_confirms_itself_even_with_stable_time_zero():
    h = started(AFTER_E4_D5, replies=["d8d5"], stable_time=0.0)
    lifted = occ(AFTER_E4_D5, without=["e4"])
    see(h, lifted)
    h.tick(n=5)                                                   # same clock reading for frame and ticks
    assert h.status["status"] == "confirming_capture" and not h.think_calls
    see(h, lifted)
    h.tick(n=2)
    assert h.m.state == State.EXECUTING


def test_capture_by_the_human_is_confirmed_when_the_robot_plays_white():
    h = Harness(robot_color=chess.WHITE, replies=["e2e4", "b1c3", "c3e4"])
    h.tick()
    settle(h, occ(S))
    for robot, human in (("e2e4", "d7d5"), ("b1c3", "d5e4")):
        assert h.m.state == State.EXECUTING and h.actions[-1]["move"] == robot
        h.arm()
        h.tick()
        settle(h, occ(h.m.game.fen()))                            # verification
        settle(h, occ(h.m.game.fen(), human))
    assert h.m.state == State.WAIT_HUMAN and h.status["status"] == "confirming_capture"
    assert "d5e4" in h.status["detail"] and len(h.actions) == 2
    settle(h, occ(h.m.game.fen(), "d5e4"))
    assert h.m.state == State.EXECUTING and h.actions[-1]["move"] == "c3e4"


def test_reset_and_a_full_placement_end_a_pending_confirmation():
    h = started(AFTER_E4_D5, replies=["d8d5"])
    lifted = occ(AFTER_E4_D5, without=["e4"])
    settle(h, lifted)
    assert h.status["status"] == "confirming_capture"
    h.m.on_cmd("reset")
    h.tick()
    settle(h, occ(AFTER_E4_D5))
    assert h.m.state == State.WAIT_HUMAN
    settle(h, lifted)                                             # nothing is left over from before the reset
    assert h.status["status"] == "confirming_capture" and not h.think_calls
    h.settle(after(AFTER_E4_D5, "e4d5"))                          # a detector that sees pieces says: exd5
    assert h.m.state == State.EXECUTING and h.actions[-1]["move"] == "d8d5"


# -------------------------------------------------- frames around the arm move
def test_frame_from_before_arm_done_is_used_if_it_shows_exactly_the_finished_move():
    h = started(replies=["e7e5"])
    settle(h, occ(S, "e2e4"))
    see(h, occ(S, "e2e4", "e7e5"))                                # single-shot vision fires first...
    h.tick(0.3)
    h.arm()                                                       # ...arm_done arrives a moment later
    h.tick()
    h.tick(STABLE + 0.01)
    assert h.m.state == State.WAIT_HUMAN and h.m._verified


def test_frame_from_before_arm_done_that_reads_as_a_capture_is_not_trusted():
    """Finished robot move with the arm still hiding the e4 pawn: from above that is e4xd5."""
    h = started(replies=["d7d5"])
    settle(h, occ(S, "e2e4"))
    see(h, occ(S, "e2e4", "d7d5", without=["e4"]))
    h.tick(0.3)
    h.arm()
    h.tick()
    h.tick(STABLE + 5.0)
    h.tick(n=3)
    assert h.m.state == State.WAIT_HUMAN and not h.m._verified
    assert len(h.think_calls) == 1 and len(h.actions) == 1       # nothing was inferred from it
    assert h.m.game.placement() == after(S, "e2e4", "d7d5")

    settle(h, occ(S, "e2e4", "d7d5"))                             # a frame taken after arm_done
    assert h.m._verified and h.status["status"] == "waiting_human"


def test_recover_reads_before_and_after_from_occupancy():
    h = started(replies=["e7e5"])
    pre, post = occ(S, "e2e4"), occ(S, "e2e4", "e7e5")
    settle(h, pre)
    h.arm(status="failed", reason="grasp")
    h.tick()
    assert h.m.state == State.RECOVER and h.status["status"] == "arm_failed"
    settle(h, pre)                                                # nothing moved: one automatic re-send
    assert h.m.state == State.EXECUTING and [a["id"] for a in h.actions] == [1, 2]
    h.arm(status="failed", reason="grasp")
    h.tick()
    settle(h, post)                                               # somebody finished the move by hand
    assert h.m.state == State.WAIT_HUMAN and h.m._verified
    assert h.m.game.placement() == after(S, "e2e4", "e7e5")


def robot_capture_failed():
    """Human exd5 (confirmed), robot answers Qxd5, the arm reports failure. Nothing has moved."""
    h = started(AFTER_E4_D5, replies=["d8d5"])
    settle(h, occ(AFTER_E4_D5, "e4d5"))
    settle(h, occ(AFTER_E4_D5, "e4d5"))
    assert h.m.state == State.EXECUTING
    h.arm(status="failed", reason="grasp")
    h.tick()
    assert h.m.state == State.RECOVER
    return h


def test_recover_does_not_take_a_missed_piece_for_a_finished_capture():
    """Qxd5 done is, from above, "the queen is gone from d8": so is a frame that misses her."""
    h = robot_capture_failed()
    settle(h, occ(AFTER_E4_D5, "e4d5", without=["d8"]))
    assert h.m.state == State.RECOVER and "looks done" in h.status["detail"]
    assert h.topic("detect_request")[-1] == "confirm_capture" and h.m.game.human_to_move is False
    settle(h, occ(AFTER_E4_D5, "e4d5"))                           # the next frame sees her: nothing moved
    assert h.m.state == State.EXECUTING and [a["move"] for a in h.actions] == ["d8d5", "d8d5"]


def test_recover_accepts_a_finished_capture_after_a_second_frame():
    h = robot_capture_failed()
    done = occ(AFTER_E4_D5, "e4d5", "d8d5")
    settle(h, done)
    assert h.m.state == State.RECOVER
    settle(h, done)
    assert h.m.state == State.WAIT_HUMAN and h.m._verified and len(h.actions) == 1
    assert h.m.game.board.piece_at(chess.D5) == chess.Piece(chess.QUEEN, chess.BLACK)


@pytest.mark.parametrize(
    "fen,human,reply,done_first",
    [
        (AFTER_E4_D5, "e4d5", "d8d5", {"d5"}),                    # capture: the taken piece is gone, no more
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "a1b1", "e8g8", {"e8"}),   # castling: only the king moved
    ],
)
def test_half_done_robot_move_is_never_sent_again(fen, human, reply, done_first):
    h = started(fen, replies=[reply])
    settle(h, occ(fen, human))
    if h.status["status"] == "confirming_capture":
        settle(h, occ(fen, human))
    assert h.m.state == State.EXECUTING
    half = squares(fen, human) - done_first
    if reply == "e8g8":
        half |= {"g8"}
    half = json.dumps(sorted(half))
    h.arm(status="failed", reason="grasp")
    h.tick()
    settle(h, half)
    assert h.m.state == State.RECOVER and "neither" in h.status["detail"]
    h.m.on_cmd("retry")
    h.tick()
    settle(h, half)
    assert h.m.state == State.RECOVER and len(h.actions) == 1


def test_board_changed_while_thinking_is_seen_from_occupancy():
    h = started(replies=["PENDING"])
    settle(h, occ(S, "e2e4"))
    assert h.m.state == State.THINKING
    see(h, occ(S))                                                # the human takes the move back
    h.pending.finish("e7e5")
    h.tick()
    assert h.m.state == State.RECOVER and h.status["status"] == "board_changed" and not h.actions


# ------------------------------------------------- moves that look the same
def asked(**kw):
    """The d4 pawn took on c5 or on e5: from above the two look the same."""
    h = started(TWO_CAPTURES, replies=["e8e7"], **kw)
    settle(h, occ(TWO_CAPTURES, "d4c5"))
    assert h.m.state == State.WAIT_HUMAN and h.status["status"] == "ambiguous_move"
    assert "d4c5 or d4e5" in h.status["detail"] and not h.think_calls
    return h


def test_two_possible_captures_are_put_to_the_operator():
    h = asked()
    h.m.on_cmd("move d4e5")
    h.tick()
    assert h.m.state == State.EXECUTING and h.actions[-1]["move"] == "e8e7"
    assert h.m.game.board.piece_at(chess.E5) == chess.Piece(chess.PAWN, chess.WHITE)
    assert h.m.game.board.piece_at(chess.C5) == chess.Piece(chess.PAWN, chess.BLACK)


@pytest.mark.parametrize("cmd", ["move d4d5", "move", "move zz", "move 0000", "move d4c5 now", "move e1h1"])
def test_an_answer_that_is_not_a_candidate_is_refused_and_says_why(cmd):
    h = asked()
    h.m.on_cmd(cmd)
    h.tick()
    assert h.m.state == State.WAIT_HUMAN and h.status["status"] == "ambiguous_move"
    assert "refused" in h.status["detail"] and "d4c5 or d4e5" in h.status["detail"]
    assert not h.think_calls
    h.m.on_cmd("MOVE D4C5 ")                                      # the question is still open
    h.tick()
    assert h.m.state == State.EXECUTING


def test_an_answer_is_refused_once_the_board_has_changed():
    h = asked()
    see(h, occ(TWO_CAPTURES))                                     # the pawn is back; not stable yet
    h.m.on_cmd("move d4c5")
    h.tick()
    assert "changed" in h.status["detail"] and not h.think_calls
    h.tick(STABLE + 0.01)
    assert h.status["status"] == "waiting_human"
    h.m.on_cmd("move d4c5")                                       # no question is open any more
    h.tick()
    assert h.m.game.fen() == TWO_CAPTURES and not h.think_calls


def test_move_without_an_open_question_is_ignored():
    h = Harness()
    h.tick()
    h.m.on_cmd("move e2e4")                                       # in SETUP
    h.tick()
    assert h.m.state == State.SETUP
    h = started()
    h.m.on_cmd("move e2e4")                                       # waiting, but nothing was asked
    h.tick()
    assert h.m.state == State.WAIT_HUMAN and h.m.game.fen() == S
    assert any("no ambiguous move" in line for line in h.logs)


def test_status_names_the_candidates_and_says_when_the_robot_move_is_verified():
    h = asked()
    assert h.status["candidates"] == ["d4c5", "d4e5"] and h.status["verified"] is True
    h.m.on_cmd("move d4c5")
    h.tick()
    assert h.status["candidates"] == []
    h.arm()
    h.tick()
    assert h.status["state"] == "wait_human" and h.status["verified"] is False
    settle(h, occ(TWO_CAPTURES, "d4c5", "e8e7"))
    assert h.status["verified"] is True


def test_an_answer_is_ignored_while_vision_is_lost():
    h = asked(vision_timeout=5.0)
    h.tick(6.0)
    assert h.status["status"] == "vision_lost"
    h.m.on_cmd("move d4c5")
    h.tick()
    assert not h.think_calls and h.m.game.fen() == TWO_CAPTURES
    settle(h, occ(TWO_CAPTURES, "d4c5"))                          # vision is back: the question is asked again
    assert h.status["status"] == "ambiguous_move"
    h.m.on_cmd("move d4c5")
    h.tick()
    assert h.m.state == State.EXECUTING


def test_a_full_placement_answers_the_question_itself():
    h = asked()
    h.settle(after(TWO_CAPTURES, "d4e5"))
    assert h.m.state == State.EXECUTING
    assert h.m.game.board.piece_at(chess.E5) == chess.Piece(chess.PAWN, chess.WHITE)


def test_capture_promotion_is_answered_with_target_and_piece():
    fen = "1n1n2k1/2P5/8/8/8/8/7P/4K3 w - - 0 1"
    h = started(fen, replies=["g8g7"])
    settle(h, occ(fen, "c7b8q"))
    assert h.status["status"] == "ambiguous_move" and len(h.status["candidates"]) == 8
    h.m.on_cmd("move c7b8n")
    h.tick()
    assert h.m.game.board.piece_at(chess.B8) == chess.Piece(chess.KNIGHT, chess.WHITE)
    assert h.m.game.board.piece_at(chess.D8) == chess.Piece(chess.KNIGHT, chess.BLACK)


def test_human_promotion_is_asked_and_an_underpromotion_is_kept():
    h = started(PROMOTION, replies=["e8e7"])
    settle(h, occ(PROMOTION, "a7a8q"))
    assert h.status["status"] == "ambiguous_move"
    assert "a7a8b or a7a8n or a7a8q or a7a8r" in h.status["detail"]
    h.m.on_cmd("move a7a8n")
    h.tick()
    assert h.m.game.board.piece_at(chess.A8) == chess.Piece(chess.KNIGHT, chess.WHITE)
    assert h.m.state == State.EXECUTING


# ------------------------------------------------------------------ messages
@pytest.mark.parametrize("text", ["not json", '"e4"', '{"e4": 1}', '["z9"]', "[1, 2]", "", "null", '[["e4"]]',
                                  "[" * 100000])
def test_malformed_occupancy_is_ignored(text):
    h = Harness()
    h.tick()
    assert h.m.on_occupancy(text, h.now) is False
    assert h.m._placement is None and any("malformed occupancy" in line for line in h.logs)


def test_square_names_are_normalised_and_duplicates_folded():
    h = Harness()
    assert h.m.on_occupancy('[" E4", "e4", "a1"]', h.now) is True
    assert h.m._placement == frozenset({"e4", "a1"})


def test_one_observation_slot_alternating_topics_never_settle():
    """A set-up publishes on ONE of the two topics. Both at once keep replacing each other."""
    h = Harness()
    h.tick()
    for _ in range(8):
        h.see(START)
        h.tick(0.3)
        see(h, occ(S))
        h.tick(0.3)
    assert h.m.state == State.SETUP
    settle(h, occ(S))
    assert h.m.state == State.WAIT_HUMAN


# ------------------------------------------- what occupancy cannot tell (limits)
def test_limit_setup_cannot_see_colours_or_piece_types():
    h = Harness()
    h.tick()
    settle(h, occ("RNBKQBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbkqbnr w - - 0 1"))
    assert h.m.state == State.WAIT_HUMAN                          # the operator has to check the set-up


def test_limit_a_failed_capture_can_look_finished():
    """Taken piece left on d5 and the queen lost on the way: the same squares as Qxd5 done.
    The arm side must check that d5 is empty before it places the queen (machine.py docstring)."""
    h = started(AFTER_E4_D5, replies=["d8d5"])
    settle(h, occ(AFTER_E4_D5, "e4d5"))
    settle(h, occ(AFTER_E4_D5, "e4d5"))
    assert h.m.state == State.EXECUTING
    failed = occ(AFTER_E4_D5, "e4d5", without=["d8"])
    assert failed == occ(AFTER_E4_D5, "e4d5", "d8d5")
    h.arm()
    h.tick()
    settle(h, failed)
    assert h.m._verified


# ---------------------------------------------------------------- whole games
def test_simulated_games_on_occupancy_with_an_operator():
    """Random games with the camera reduced to occupied squares. A referee keeps the truth; the
    operator answers every question with the move that was really played."""
    confirmed = questions = 0
    for seed in range(8):
        rng = random.Random(1000 + seed)
        robot = chess.BLACK if seed % 2 == 0 else chess.WHITE

        def chooser(game, root_moves):
            return rng.choice(root_moves if root_moves is not None else list(game.board.legal_moves))

        h = Harness(chooser=chooser, robot_color=robot, allow_promotion=True)
        ref = chess.Board()
        phys = chess.BaseBoard()

        def seen():
            return json.dumps([chess.square_name(sq) for sq in chess.SQUARES if phys.piece_at(sq)])

        settle(h, seen())
        for _ in range(80):
            if referee_outcome(ref) is not None:
                break
            if ref.turn == robot:
                assert h.m.state == State.EXECUTING, (seed, h.m.state, h.status)
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
                    settle(h, seen())
                    assert h.status["status"] == "waiting_human" and h.m._verified
            else:
                assert h.m.state == State.WAIT_HUMAN, (seed, h.m.state, h.status)
                legal = list(ref.legal_moves)
                special = [m for m in legal if ref.is_capture(m) or ref.is_castling(m) or m.promotion]
                move = rng.choice(special) if special and rng.random() < 0.7 else rng.choice(legal)
                ref.push(move)
                phys = chess.BaseBoard(ref.board_fen())           # the human moves pieces perfectly
                settle(h, seen())
                if h.status["status"] == "confirming_capture":
                    confirmed += 1
                    settle(h, seen())
                if h.status["status"] == "ambiguous_move":
                    questions += 1
                    assert move.uci() in h.status["detail"]
                    h.m.on_cmd(f"move {move.uci()}")
                    h.tick(n=2)
            assert h.m.game.board.board_fen() == ref.board_fen() or h.m.state == State.EXECUTING, (seed, h.status)

        if referee_outcome(ref) is not None:
            assert h.m.state == State.GAME_OVER
            assert h.status["result"] == referee_outcome(ref).result()
        assert h.m.game.fen() == ref.fen() or h.m.state == State.EXECUTING
        assert not any(s["status"] in BAD_STATUSES for s in h.topic("status")), seed
    assert confirmed > 20 and questions > 5, (confirmed, questions)


def test_simulated_games_with_missed_pieces_and_a_failing_arm():
    """The camera sometimes misses one piece for a frame and the arm sometimes fails without
    moving anything. No move may be applied that was not played, and the arm may only ever be
    sent the move the referee expects, from the position it stands in."""
    failures = misses = 0
    for seed in range(10):
        rng = random.Random(2000 + seed)
        robot = chess.BLACK if seed % 2 == 0 else chess.WHITE

        def chooser(game, root_moves):
            return rng.choice(root_moves if root_moves is not None else list(game.board.legal_moves))

        h = Harness(chooser=chooser, robot_color=robot, allow_promotion=True)
        ref = chess.Board()
        phys = chess.BaseBoard()

        def look():
            """Vision reports: perhaps one frame with a piece missing, then what is really there
            (again, while the machine asks for a second frame)."""
            nonlocal misses
            names = [chess.square_name(sq) for sq in chess.SQUARES if phys.piece_at(sq)]
            if rng.random() < 0.3:
                misses += 1
                settle(h, json.dumps([name for name in names if name != rng.choice(names)]))
            for _ in range(3):
                asked_before = len(h.topic("detect_request"))
                settle(h, json.dumps(names))
                if "confirm_capture" not in h.topic("detect_request")[asked_before:]:
                    break

        settle(h, occ(S))
        for _ in range(70):
            if referee_outcome(ref) is not None:
                break
            if ref.turn == robot:
                assert h.m.state == State.EXECUTING, (seed, h.m.state, h.status)
                payload = h.actions[-1]
                move = chess.Move.from_uci(payload["move"])
                assert move in ref.legal_moves, (seed, payload)
                if rng.random() < 0.25:                           # the arm fails; nothing has moved
                    failures += 1
                    h.tick(1.0)
                    h.arm(done_id=payload["id"], status="failed", reason="grasp")
                    h.tick()
                    look()
                    assert h.m.state == State.EXECUTING, (seed, h.status)
                    assert h.actions[-1]["move"] == payload["move"] and h.actions[-1]["id"] != payload["id"]
                    payload = h.actions[-1]
                execute_on_physical_board(phys, payload["actions"])
                ref.push(move)
                h.tick(2.0)
                h.arm(done_id=payload["id"])
                h.tick()
                if referee_outcome(ref) is None:
                    look()
                    assert h.status["status"] == "waiting_human" and h.m._verified, (seed, h.status)
            else:
                assert h.m.state == State.WAIT_HUMAN, (seed, h.m.state, h.status)
                legal = list(ref.legal_moves)
                special = [m for m in legal if ref.is_capture(m) or ref.is_castling(m) or m.promotion]
                move = rng.choice(special) if special and rng.random() < 0.7 else rng.choice(legal)
                ref.push(move)
                phys = chess.BaseBoard(ref.board_fen())
                look()
                if h.status["status"] == "ambiguous_move":
                    assert move.uci() in h.status["candidates"]
                    h.m.on_cmd(f"move {move.uci()}")
                    h.tick(n=2)
            assert h.m.game.board.board_fen() == ref.board_fen() or h.m.state == State.EXECUTING, (seed, h.status)
        assert h.m.game.fen() == ref.fen() or h.m.state == State.EXECUTING, seed
    assert failures > 20 and misses > 100, (failures, misses)
