"""A vision that only reports the occupied squares, on the real-robot side: what the vision
adapter accepts from such a detector, and the arm adapter's look at the board before it moves."""

import chess
import pytest

from chess_brain.actions import actions_payload, move_to_actions, occupied_squares
from chess_brain.arm_adapter import check_action, run_command
from chess_brain.board_frame import Locations
from chess_brain.vision_adapter import (
    camera_summary,
    run_detector,
    run_occupancy_detector,
    superseded,
    wants_camera,
)

START = sorted(f + r for r in "1278" for f in "abcdefgh")


# ----------------------------------------------------------------------- the vision adapter
def test_a_detector_may_take_the_camera_as_second_parameter():
    assert not wants_camera(lambda image: None)
    assert wants_camera(lambda image, camera: None)
    assert wants_camera(lambda *everything: None)
    assert not wants_camera(lambda image, *, option=None: None)
    assert not wants_camera(len)        # a builtin without a signature we can use: treated as one-parameter


def test_occupied_squares_are_validated_and_sorted():
    squares, why = run_occupancy_detector(lambda image: ["E4", " a1", "h8", "a1"], object())
    assert (squares, why) == (["a1", "e4", "h8"], "ok")
    assert run_occupancy_detector(lambda image: [], object()) == ([], "ok")
    assert run_occupancy_detector(lambda image: frozenset(START), object())[0] == START


@pytest.mark.parametrize(
    "answer, part_of_reason",
    [
        (None, "returned None"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", "not a list of squares"),
        (["a1", "z9"], "not a list of square names"),
        ([1, 2], "not a list of square names"),
        (42, "not a list of squares"),
    ],
)
def test_other_answers_are_not_published(answer, part_of_reason):
    squares, why = run_occupancy_detector(lambda image: answer, object())
    assert squares is None and part_of_reason in why


def test_a_detector_that_raises_does_not_take_the_node_down():
    def broken(image):
        raise RuntimeError("no tags in the picture")

    squares, why = run_occupancy_detector(broken, object())
    assert squares is None and "RuntimeError" in why and "no tags" in why


def test_the_camera_reaches_a_detector_that_asks_for_it():
    got = {}

    def with_camera(image, camera):
        got["camera"] = camera
        return ["a1"]

    camera = camera_summary(640, 480, [600, 0, 320, 0, 600, 240, 0, 0, 1], [0.1, 0, 0, 0, 0], "cam", {"square_size": 0.04})
    assert run_occupancy_detector(with_camera, object(), camera)[0] == ["a1"]
    assert got["camera"]["k"] == [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    assert got["camera"]["d"][0] == 0.1 and got["camera"]["config"] == {"square_size": 0.04}
    assert (got["camera"]["width"], got["camera"]["height"], got["camera"]["frame_id"]) == (640, 480, "cam")
    # a detector written for the image alone is still called with the image alone
    assert run_occupancy_detector(lambda image: ["b2"], object(), camera)[0] == ["b2"]
    assert run_detector(lambda image, cam: "8/8/8/8/8/8/8/8", object(), camera)[0] == "8/8/8/8/8/8/8/8"


def test_a_camera_matrix_must_have_nine_numbers():
    with pytest.raises(ValueError, match="9"):
        camera_summary(640, 480, [1, 2, 3], [])


def test_a_detection_older_than_a_waiting_request_is_out_of_date():
    assert superseded(image_at=10.0, request_waiting=True, request_at=11.0)
    assert not superseded(image_at=12.0, request_waiting=True, request_at=11.0)
    assert not superseded(image_at=10.0, request_waiting=False, request_at=11.0)   # nobody is waiting


# ----------------------------------------------------------------- looking before the arm moves
def command(fen, uci, allow_promotion=False):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    return actions_payload(1, move, move_to_actions(board, move, allow_promotion), occupied_squares(board))


class Table:
    """What the camera would see: the occupied squares, changed by the actions that ran."""

    def __init__(self, fen, stuck=()):
        self.occupied = set(occupied_squares(chess.Board(fen)))
        self.stuck = set(stuck)          # squares whose piece the arm fails to take away
        self.looks, self.infos = 0, []

    def look(self):
        self.looks += 1
        return frozenset(self.occupied)

    def executor(self, pick_xyz, place_xyz, info):
        self.infos.append(dict(info))
        if info["pick"] not in self.stuck:
            self.occupied.discard(info["pick"])
        if len(info["place"]) == 2:
            self.occupied.add(info["place"])


CAPTURE = "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"       # e4 takes d5
CASTLE = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"


def test_check_action():
    locations = Locations()
    seen = {"e2", "d5"}
    assert check_action("e2", "e4", seen, locations) is None
    assert "no piece on e3" in check_action("e3", "e4", seen, locations)
    assert "a piece on d5" in check_action("e2", "d5", seen, locations)
    assert check_action("d5", "DISCARD", seen, locations) is None            # off-board: not the camera's business
    assert check_action("SPARE_Q_W", "e8", seen, locations) is None
    assert "a piece on d5" in check_action("SPARE_Q_W", "d5", seen, locations)
    assert check_action("E2", " e4 ", ["E2 "], locations) is None             # names as loosely as elsewhere


def test_a_quiet_move_is_looked_at_once_before_it_starts():
    table = Table(chess.STARTING_FEN)
    reply = run_command(command(chess.STARTING_FEN, "e2e4"), Locations(), table.executor, look=table.look)
    assert reply["status"] == "ok" and table.looks == 1
    assert table.infos[0]["last"] is True


def test_a_capture_is_looked_at_again_before_the_capturing_piece_is_put_down():
    table = Table(CAPTURE)
    reply = run_command(command(CAPTURE, "e4d5"), Locations(), table.executor, look=table.look)
    assert reply["status"] == "ok" and table.looks == 2
    # the arm parks after taking the captured piece away, so that the camera can see d5
    assert [info["last"] for info in table.infos] == [True, True]


def test_castling_needs_no_second_look_and_no_extra_parking():
    table = Table(CASTLE)
    reply = run_command(command(CASTLE, "e1g1"), Locations(), table.executor, look=table.look)
    assert reply["status"] == "ok" and table.looks == 1
    assert [info["last"] for info in table.infos] == [False, True]


def test_without_a_look_nothing_changes():
    table = Table(CAPTURE)
    reply = run_command(command(CAPTURE, "e4d5"), Locations(), table.executor)
    assert reply["status"] == "ok" and table.looks == 0
    assert [info["last"] for info in table.infos] == [False, True]


def test_the_capturing_piece_is_not_put_on_a_captured_piece_that_is_still_there():
    locations = Locations()
    table = Table(CAPTURE, stuck={"d5"})     # the arm reports success, but the pawn never left d5
    reply = run_command(command(CAPTURE, "e4d5"), locations, table.executor, look=table.look)
    assert reply["status"] == "failed" and reply["failed_action_index"] == 1
    assert "a piece on d5" in reply["reason"] and "not started" in reply["reason"]
    assert len(table.infos) == 1             # the second action never reached the arm
    assert "e4" in table.occupied            # the capturing piece is still on its square


def test_a_move_whose_piece_is_not_where_the_brain_thinks_does_not_start():
    table = Table(chess.STARTING_FEN)
    table.occupied.discard("e2")
    reply = run_command(command(chess.STARTING_FEN, "e2e4"), Locations(), table.executor, look=table.look)
    assert reply["status"] == "failed" and reply["failed_action_index"] == 0
    assert "no piece on e2" in reply["reason"] and table.infos == []


@pytest.mark.parametrize("look", [lambda: None, lambda: 1 / 0])
def test_no_picture_no_move(look):
    table = Table(chess.STARTING_FEN)
    reply = run_command(command(chess.STARTING_FEN, "e2e4"), Locations(), table.executor, look=look)
    assert reply["status"] == "failed" and table.infos == []
    assert "not started" in reply["reason"]
