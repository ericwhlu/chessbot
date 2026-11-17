"""Tests for the plug-in adapters: board coordinates, arm command execution, image -> FEN."""

import json
import math

import chess
import pytest

from chess_brain.actions import actions_payload, move_to_actions
from chess_brain.arm_adapter import make_dry_run_executor, plan_command, run_command
from chess_brain.board_frame import BoardFrame, Locations
from chess_brain.plugins import load_callable
from chess_brain.vision_adapter import START_PLACEMENT, example_detector, image_to_bgr, run_detector

try:
    import numpy as np
except ImportError:  # the pure-logic tests still run without numpy
    np = None

Z = 0.00656168


def close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b)) and len(a) == len(b)


# ------------------------------------------------------------------ board frame
@pytest.mark.parametrize(
    "square,expected",
    [("a1", (0.025, 0.025)), ("h1", (0.375, 0.025)), ("a8", (0.025, 0.375)), ("h8", (0.375, 0.375)),
     ("e4", (0.225, 0.175)), ("E2", (0.225, 0.075)), (" d5 ", (0.175, 0.225))],
)
def test_square_centres_follow_board_setup_md(square, expected):
    assert close(BoardFrame().board_xy(square), expected)
    assert close(BoardFrame().world_xyz(square), expected + (Z,))


@pytest.mark.parametrize("bad", ["i1", "a9", "a0", "e", "e44", "", "DISCARD", None, 12])
def test_bad_square_names_are_rejected(bad):
    with pytest.raises(ValueError):
        BoardFrame().board_xy(bad)


def test_board_placed_and_rotated_in_the_world():
    moved = BoardFrame(origin_x=0.30, origin_y=-0.20, surface_z=0.01)
    assert close(moved.world_xyz("a1"), (0.325, -0.175, 0.01))
    turned = BoardFrame(yaw=math.radians(90))            # files now run along world +Y, ranks along -X
    assert close(turned.world_xyz("h1"), (-0.025, 0.375, Z), tol=1e-12)
    flipped = BoardFrame(origin_x=0.4, origin_y=0.4, yaw=math.radians(180))   # robot sits on black's side
    assert close(flipped.world_xyz("a1"), (0.375, 0.375, Z), tol=1e-12)
    assert close(flipped.world_xyz("h8"), (0.025, 0.025, Z), tol=1e-12)


def test_discard_slots_and_spares():
    loc = Locations()
    assert close(loc.place_xyz("DISCARD"), (-0.075, 0.025, Z))
    assert close(loc.place_xyz("DISCARD"), (-0.075, 0.025, Z))        # not used up until committed
    assert close(loc.place_xyz("DISCARD", discard_offset=1), (-0.075, 0.075, Z))
    for _ in range(8):
        loc.commit_discard()
    assert close(loc.place_xyz("discard"), (-0.125, 0.025, Z))        # ninth piece starts a second column
    loc.reset()
    assert loc.discarded == 0 and close(loc.place_xyz("DISCARD"), (-0.075, 0.025, Z))
    assert close(loc.pick_xyz("SPARE_Q_W"), (0.475, 0.025, Z))
    assert close(loc.pick_xyz("SPARE_Q_B"), (0.475, 0.375, Z))
    assert close(loc.pick_xyz("e2"), (0.225, 0.075, Z))
    with pytest.raises(ValueError):
        loc.pick_xyz("DISCARD")
    with pytest.raises(ValueError):
        loc.place_xyz("SPARE_Q_W")
    with pytest.raises(ValueError):
        loc.pick_xyz("SPARE_N_W")                                     # only spare queens exist


# ------------------------------------------------------------------ arm adapter
class Recorder:
    """An executor that records its calls and can be told to fail."""

    def __init__(self, fail_at=None, how=False):
        self.calls, self.fail_at, self.how = [], fail_at, how

    def __call__(self, pick_xyz, place_xyz, info):
        self.calls.append((info["pick"], pick_xyz, info["place"], place_xyz, dict(info)))
        if self.fail_at is not None and info["index"] == self.fail_at:
            if isinstance(self.how, Exception):
                raise self.how
            return self.how
        return None


CAPTURE = {"id": 5, "move": "d5e4", "actions": [{"pick": "e4", "place": "DISCARD"}, {"pick": "d5", "place": "e4"}]}


def test_run_command_calls_the_executor_once_per_action_in_order():
    loc, rec = Locations(), Recorder()
    reply = run_command(CAPTURE, loc, rec)
    assert reply == {"id": 5, "status": "ok"}
    assert [(c[0], c[2]) for c in rec.calls] == [("e4", "DISCARD"), ("d5", "e4")]
    assert close(rec.calls[0][1], (0.225, 0.175, Z)) and close(rec.calls[0][3], (-0.075, 0.025, Z))
    assert close(rec.calls[1][1], (0.175, 0.225, Z)) and close(rec.calls[1][3], (0.225, 0.175, Z))
    first, last = rec.calls[0][4], rec.calls[1][4]
    assert first["index"] == 0 and first["count"] == 2 and first["last"] is False and first["move"] == "d5e4"
    assert last["index"] == 1 and last["last"] is True and last["id"] == 5
    assert loc.discarded == 1                                         # the slot is used up


@pytest.mark.parametrize(
    "how,needle",
    [(False, "returned False"), ("gripper slipped", "gripper slipped"), (RuntimeError("no plan"), "RuntimeError: no plan")],
)
def test_executor_failure_is_reported_and_stops_the_command(how, needle):
    loc, rec = Locations(), Recorder(fail_at=0, how=how)
    reply = run_command(CAPTURE, loc, rec)
    assert reply["status"] == "failed" and reply["failed_action_index"] == 0 and needle in reply["reason"]
    assert len(rec.calls) == 1 and loc.discarded == 0                 # second action skipped, slot not used


def test_failure_on_the_second_action_keeps_the_discard_that_did_happen():
    loc, rec = Locations(), Recorder(fail_at=1)
    reply = run_command(CAPTURE, loc, rec)
    assert reply["failed_action_index"] == 1 and loc.discarded == 1


@pytest.mark.parametrize(
    "payload,needle",
    [
        ({"id": 1, "move": "e7e5", "actions": [{"pick": "e7", "place": "e9"}]}, "not a square name"),
        ({"id": 1, "move": "e7e5", "actions": [{"pick": "DISCARD", "place": "e5"}]}, "not a pick location"),
        ({"id": 1, "move": "e7e5", "actions": [{"pick": "e7", "place": "e5"}, {"pick": "zz", "place": "e5"}]}, "action 1"),
        ({"id": 1, "move": "e7e5", "actions": [{"from": "e7"}]}, "needs 'pick' and 'place'"),
        ({"id": 1, "move": "e7e5", "actions": []}, "no actions"),
        ({"id": 1, "move": "e7e5"}, "no actions"),
        ("not a dict", "JSON object"),
    ],
)
def test_bad_commands_are_rejected_before_anything_moves(payload, needle):
    rec = Recorder()
    reply = run_command(payload, Locations(), rec)
    assert reply["status"] == "failed" and reply["failed_action_index"] == -1 and needle in reply["reason"], reply
    assert rec.calls == []


def test_two_discards_in_one_command_get_two_slots():
    payload = {"id": 2, "move": "b7a8q", "actions": [
        {"pick": "a8", "place": "DISCARD"}, {"pick": "b7", "place": "DISCARD"}, {"pick": "SPARE_Q_W", "place": "a8"}]}
    plan = plan_command(payload, Locations())
    assert close(plan[0][3], (-0.075, 0.025, Z)) and close(plan[1][3], (-0.075, 0.075, Z))
    assert close(plan[2][1], (0.475, 0.025, Z)) and close(plan[2][3], (0.025, 0.375, Z))
    loc = Locations()
    assert run_command(payload, loc, Recorder())["status"] == "ok" and loc.discarded == 2


def test_dry_run_executor_logs_and_succeeds():
    lines = []
    reply = run_command(CAPTURE, Locations(), make_dry_run_executor(0.0, log=lines.append))
    assert reply["status"] == "ok" and len(lines) == 2
    assert "e4" in lines[0] and "DISCARD" in lines[0] and "(0.225, 0.175, 0.007)" in lines[0]


def test_every_command_chess_brain_can_produce_is_executable():
    """Random games: every arm_actions payload from actions.py resolves and runs."""
    import random

    rng = random.Random(4)
    for _ in range(15):
        board, loc = chess.Board(), Locations()
        for _ in range(80):
            legal = [m for m in board.legal_moves if m.promotion in (None, chess.QUEEN)]
            if not legal:
                break
            move = rng.choice(legal)
            payload = actions_payload(1, move, move_to_actions(board, move, allow_promotion=True))
            reply = run_command(json.loads(json.dumps(payload)), loc, Recorder())
            assert reply["status"] == "ok", (board.fen(), move.uci(), reply)
            board.push(move)
        captured = 32 - len(board.piece_map())
        promoted = sum(1 for m in board.move_stack if m.promotion)
        assert loc.discarded == captured + promoted                   # captured pieces + promoted pawns


# ---------------------------------------------------------------------- plugins
def test_load_callable():
    assert load_callable("chess_brain.vision_adapter:example_detector") is example_detector
    assert load_callable(" math:sqrt ")(9) == 3
    for bad in ["", "math", "math:", ":sqrt", "no_such_module_xyz:f", "math:no_such_function", "math:pi"]:
        with pytest.raises(ValueError):
            load_callable(bad)


# --------------------------------------------------------------- vision adapter
def test_run_detector_validates_the_answer():
    assert run_detector(example_detector, None) == (START_PLACEMENT, "ok")
    assert run_detector(lambda img: START_PLACEMENT + " w KQkq - 0 1", None)[0] == START_PLACEMENT
    assert run_detector(lambda img: "  " + START_PLACEMENT + "\n", None)[0] == START_PLACEMENT
    placement, why = run_detector(lambda img: None, None)
    assert placement is None and "None" in why
    placement, why = run_detector(lambda img: "8/8/8", None)
    assert placement is None and "not a placement" in why
    placement, why = run_detector(lambda img: 42, None)
    assert placement is None and "not a placement" in why

    def boom(img):
        raise RuntimeError("corners not found")

    placement, why = run_detector(boom, None)
    assert placement is None and "RuntimeError: corners not found" in why


@pytest.mark.skipif(np is None, reason="numpy not installed")
def test_image_to_bgr_handles_the_common_encodings():
    # one 2x1 image: left pixel pure red, right pixel pure blue
    assert image_to_bgr(1, 2, "rgb8", 6, bytes([255, 0, 0, 0, 0, 255])).tolist() == [[[0, 0, 255], [255, 0, 0]]]
    assert image_to_bgr(1, 2, "bgr8", 6, bytes([0, 0, 255, 255, 0, 0])).tolist() == [[[0, 0, 255], [255, 0, 0]]]
    assert image_to_bgr(1, 2, "rgba8", 8, bytes([255, 0, 0, 9, 0, 0, 255, 9])).tolist() == [[[0, 0, 255], [255, 0, 0]]]
    assert image_to_bgr(1, 2, "BGRA8", 8, bytes([0, 0, 255, 9, 255, 0, 0, 9])).tolist() == [[[0, 0, 255], [255, 0, 0]]]
    assert image_to_bgr(1, 2, "mono8", 2, bytes([7, 200])).tolist() == [[[7, 7, 7], [200, 200, 200]]]
    out = image_to_bgr(2, 1, "rgb8", 3, bytes([1, 2, 3, 4, 5, 6]))
    assert out.shape == (2, 1, 3) and out.dtype == np.uint8 and out.tolist() == [[[3, 2, 1]], [[6, 5, 4]]]


@pytest.mark.skipif(np is None, reason="numpy not installed")
def test_image_to_bgr_handles_row_padding_and_array_data():
    import array

    # 2 rows of 1 rgb pixel, padded to step 4
    data = array.array("B", [10, 20, 30, 99, 40, 50, 60, 99])
    assert image_to_bgr(2, 1, "rgb8", 4, data).tolist() == [[[30, 20, 10]], [[60, 50, 40]]]
    out = image_to_bgr(1, 1, "rgb8", 3, bytes([1, 2, 3]))
    out[0, 0, 0] = 77                                                 # the result is writable (a copy)
    assert out[0, 0, 0] == 77


@pytest.mark.skipif(np is None, reason="numpy not installed")
@pytest.mark.parametrize(
    "args,needle",
    [
        ((1, 2, "yuv422", 4, bytes(4)), "unsupported image encoding"),
        ((0, 2, "rgb8", 6, bytes(6)), "empty image"),
        ((1, 2, "rgb8", 5, bytes(6)), "too small"),
        ((2, 2, "rgb8", 6, bytes(6)), "expected 12"),
    ],
)
def test_image_to_bgr_rejects_bad_images(args, needle):
    with pytest.raises(ValueError, match=needle):
        image_to_bgr(*args)
