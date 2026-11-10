"""Fixture-driven tests for GameState.infer_human_move and game-over handling."""

import json
import pathlib

import chess
import pytest

from chess_brain.game import GameState, canonical_placement

FIXTURES = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "infer_moves.json").read_text()
)

START_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


@pytest.mark.parametrize("case", FIXTURES, ids=[c["name"] for c in FIXTURES])
def test_fixture(case):
    game = GameState(case["before_fen"])
    before = game.board.fen()

    result = game.infer_human_move(case["after_placement"])

    assert result.kind == case["expect"]["kind"], result
    assert game.board.fen() == before, "inference must not mutate the board"

    if result.kind == "move":
        assert result.move.uci() == case["expect"]["uci"]
        game.apply(result.move)
        assert game.placement() == canonical_placement(case["after_placement"])
        if "game_over_after" in case["expect"]:
            assert game.is_game_over()
            assert game.outcome().result() == case["expect"]["game_over_after"]
    else:
        assert result.move is None


# ---------------------------------------------------------------- start_fen trap
def test_partial_start_fen_rejected():
    with pytest.raises(ValueError, match="6-field"):
        GameState(START_PLACEMENT)


def test_garbage_start_fen_rejected():
    with pytest.raises(ValueError):
        GameState("this is not a fen at all x y")


def test_full_start_fen_keeps_castling_rights():
    game = GameState(chess.STARTING_FEN)
    assert game.board.castling_rights == chess.BB_CORNERS


# --------------------------------------------------------------- canonicalise
@pytest.mark.parametrize(
    "text",
    [
        START_PLACEMENT,
        "  " + START_PLACEMENT + "  ",
        START_PLACEMENT + " w KQkq - 0 1",
        START_PLACEMENT + "\n",
    ],
)
def test_canonical_placement_accepts_variants(text):
    assert canonical_placement(text) == START_PLACEMENT


@pytest.mark.parametrize("text", ["", "   ", "hello", "8/8/8/8/8/8/8", "9/8/8/8/8/8/8/8", None, 42])
def test_canonical_placement_rejects_garbage(text):
    assert canonical_placement(text) is None


# ------------------------------------------------------------------ apply / status
def test_apply_rejects_illegal_move():
    game = GameState()
    with pytest.raises(ValueError):
        game.apply(chess.Move.from_uci("e2e5"))


def test_status_fields_after_one_move():
    game = GameState(robot_color=chess.BLACK)
    assert game.human_to_move
    game.apply(chess.Move.from_uci("e2e4"))
    s = game.status()
    assert s["turn"] == "black"
    assert s["human_to_move"] is False
    assert s["placement"] == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    assert s["game_over"] is False and s["result"] is None


def test_checkmate_detected_after_human_move():
    game = GameState("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", robot_color=chess.BLACK)
    res = game.infer_human_move("R5k1/5ppp/8/8/8/8/5PPP/6K1")
    game.apply(res.move)
    assert game.is_game_over()
    assert game.outcome().termination == chess.Termination.CHECKMATE
    assert game.status()["result"] == "1-0"


def test_stalemate_detected():
    # Black king a8, white queen b6, white king c6. After Kc7 the black king is
    # not in check and has no square: stalemate.
    game = GameState("k7/8/1QK5/8/8/8/8/8 w - - 0 1")
    game.apply(chess.Move.from_uci("c6c7"))  # Kc7: black king a8 has no moves, not in check
    assert game.outcome().termination == chess.Termination.STALEMATE


def test_threefold_repetition_ends_game_on_the_third_occurrence_not_before():
    # K+R vs K (not K+N: that is insufficient material and ends the game at once)
    game = GameState("4k3/8/8/8/8/8/8/4K2R w - - 0 1")
    cycle = ["h1g1", "e8d8", "g1h1", "d8e8"]
    for uci in cycle + cycle[:3]:
        game.apply(chess.Move.from_uci(uci))
        # python-chess would already let a player *claim* the draw one ply early;
        # the robot only stops when the position has really occurred three times.
        assert not game.is_game_over(), uci
    game.apply(chess.Move.from_uci(cycle[3]))      # start position, third occurrence
    assert game.is_game_over()
    assert game.outcome().termination == chess.Termination.THREEFOLD_REPETITION
    assert game.status()["result"] == "1/2-1/2"


def test_fifty_move_rule_ends_game_at_clock_100():
    game = GameState("4k3/8/8/8/8/8/8/4K2R w - - 99 80")
    assert not game.is_game_over()                 # claimable by python-chess, but not yet reached
    game.apply(chess.Move.from_uci("h1g1"))
    assert game.is_game_over()
    assert game.outcome().termination == chess.Termination.FIFTY_MOVES
