"""Tests for move inference from occupied squares only (no piece types, no colours)."""

import random

import chess
import pytest

from chess_brain.occupancy import (
    describe_difference,
    infer_from_occupancy,
    normalise,
    occupancy_after,
    placement_after,
    squares_of,
)


def seen_after(board, uci):
    return occupancy_after(board, chess.Move.from_uci(uci))


def test_squares_of_the_starting_position():
    squares = squares_of(chess.Board())
    assert len(squares) == 32
    assert {"a1", "h1", "e2", "a7", "h8"} <= squares and "e4" not in squares


def test_nothing_moved():
    board = chess.Board()
    assert infer_from_occupancy(board, squares_of(board)).kind == "no_change"


@pytest.mark.parametrize("uci", ["e2e4", "e2e3", "g1f3", "b1a3", "h2h4"])
def test_quiet_moves_are_unique(uci):
    board = chess.Board()
    result = infer_from_occupancy(board, seen_after(board, uci))
    assert result.kind == "move" and result.move == chess.Move.from_uci(uci)
    assert board == chess.Board()                           # the board was not modified


def test_capture_with_one_possible_target():
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    result = infer_from_occupancy(board, seen_after(board, "e4d5"))
    assert result.kind == "move" and result.move.uci() == "e4d5"


def test_capture_with_two_possible_targets_is_ambiguous():
    # The d4 pawn can take on c5 or on e5: either way d4 empties and c5, e5 stay occupied.
    board = chess.Board("4k3/8/8/2p1p3/3P4/8/8/4K3 w - - 0 1")
    for uci in ("d4c5", "d4e5"):
        result = infer_from_occupancy(board, seen_after(board, uci))
        assert result.kind == "ambiguous"
        assert [move.uci() for move in result.moves] == ["d4c5", "d4e5"]
        assert result.move is None
    assert infer_from_occupancy(board, seen_after(board, "d4d5")).move.uci() == "d4d5"   # the push is not


def test_en_passant():
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2")
    result = infer_from_occupancy(board, seen_after(board, "e5d6"))
    assert result.kind == "move" and result.move.uci() == "e5d6"
    assert "d5" not in seen_after(board, "e5d6")            # the captured pawn is gone from d5


@pytest.mark.parametrize("uci", ["e1g1", "e1c1"])
def test_castling(uci):
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    result = infer_from_occupancy(board, seen_after(board, uci))
    assert result.kind == "move" and result.move.uci() == uci


def test_a_promotion_is_ambiguous_between_the_four_pieces():
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    for uci in ("a7a8q", "a7a8n", "a7a8r", "a7a8b"):
        result = infer_from_occupancy(board, seen_after(board, uci))
        assert result.kind == "ambiguous"
        assert [move.uci() for move in result.moves] == ["a7a8b", "a7a8n", "a7a8q", "a7a8r"]


def test_promotion_by_capture_lists_every_target_and_piece():
    board = chess.Board("1n1n2k1/2P5/8/8/8/8/8/4K3 w - - 0 1")
    result = infer_from_occupancy(board, seen_after(board, "c7b8q"))
    assert result.kind == "ambiguous"
    assert [move.uci() for move in result.moves] == [
        "c7b8b", "c7b8n", "c7b8q", "c7b8r", "c7d8b", "c7d8n", "c7d8q", "c7d8r"]


def test_an_illegal_change_is_not_a_move():
    board = chess.Board()
    seen = (squares_of(board) - {"e2"}) | {"e5"}            # a pawn jumped three squares
    assert infer_from_occupancy(board, seen).kind == "illegal_or_misread"
    assert infer_from_occupancy(board, squares_of(board) - {"e2"}).kind == "illegal_or_misread"   # a piece vanished
    assert infer_from_occupancy(board, squares_of(board) | {"e4"}).kind == "illegal_or_misread"   # one appeared


def test_a_move_of_the_wrong_side_is_not_accepted():
    board = chess.Board()
    seen = (squares_of(board) - {"e7"}) | {"e5"}            # black moved, but it is white's turn
    assert infer_from_occupancy(board, seen).kind == "illegal_or_misread"


@pytest.mark.parametrize("bad", [["e9"], ["zz"], [None], 5, ["e2", ""]])
def test_garbage_never_raises(bad):
    assert infer_from_occupancy(chess.Board(), bad).kind == "illegal_or_misread"


def test_square_names_are_normalised():
    assert normalise([" E2 ", "e4"]) == frozenset({"e2", "e4"})
    board = chess.Board()
    seen = [name.upper() for name in seen_after(board, "e2e4")]
    assert infer_from_occupancy(board, seen).move.uci() == "e2e4"


def test_placement_after_is_what_a_full_detector_would_report():
    assert placement_after(chess.Board(), chess.Move.from_uci("e2e4")) == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"


def test_difference_is_readable():
    board = chess.Board()
    assert describe_difference(board, seen_after(board, "e2e4")) == "now empty: e2; now occupied: e4"
    assert describe_difference(board, squares_of(board)) == "no difference"
    assert "not a list of squares" in describe_difference(board, ["q9"])


def test_every_legal_move_of_random_games_is_recovered():
    """The move actually played is always the answer or among the ambiguous candidates."""
    rng = random.Random(2026)
    counts = {"move": 0, "ambiguous": 0}
    for _ in range(15):
        board = chess.Board()
        for _ in range(70):
            moves = list(board.legal_moves)
            if not moves:
                break
            for move in moves:
                result = infer_from_occupancy(board, occupancy_after(board, move))
                assert result.kind in counts, (board.fen(), move.uci(), result.kind)
                counts[result.kind] += 1
                if result.kind == "move":
                    assert result.move == move, (board.fen(), move.uci())
                else:
                    assert move in result.moves, (board.fen(), move.uci())
                    # ambiguity only ever comes from one piece: several possible captures, or
                    # what a pawn promotes to
                    assert len({m.from_square for m in result.moves}) == 1
                    assert all(board.is_capture(m) or m.promotion for m in result.moves)
            board.push(rng.choice(moves))
    assert counts["move"] > 10 * counts["ambiguous"] > 0
    print(f"occupancy inference over random games: {counts}")
