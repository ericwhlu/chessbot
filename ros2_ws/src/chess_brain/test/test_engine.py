"""Engine wrapper tests. Most run against the fake UCI engine; one needs real Stockfish."""

import os
import pathlib
import shutil
import sys

import chess
import pytest

from chess_brain.game import Engine, GameState

FAKE = str(pathlib.Path(__file__).parent / "fake_uci_engine.py")


def fake_engine(**env):
    """Start the fake engine with extra environment variables."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return Engine([sys.executable, FAKE], think_time=0.01, skill_level=3)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_fake_engine_replies_with_a_legal_move():
    with fake_engine() as eng:
        assert eng.name == "FakeEngine"
        assert eng.skill_level_supported
        game = GameState(robot_color=chess.WHITE)
        move = eng.reply(game)
        assert move is not None and move in game.board.legal_moves


def test_reply_refuses_on_humans_turn():
    with fake_engine() as eng:
        game = GameState(robot_color=chess.BLACK)  # white (human) to move
        with pytest.raises(RuntimeError):
            eng.reply(game)


def test_reply_is_none_when_game_is_over():
    with fake_engine() as eng:
        # Black is checkmated; it is black's turn; robot plays black.
        game = GameState("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1", robot_color=chess.BLACK)
        assert game.is_game_over()
        assert eng.reply(game) is None


def test_engine_under_promotion_is_replaced_by_queen():
    with fake_engine(FAKE_ENGINE_PROMOTE="n") as eng:
        game = GameState("8/P6k/8/8/8/8/8/K7 w - - 0 1", robot_color=chess.WHITE)
        move = eng.reply(game)
        assert move == chess.Move.from_uci("a7a8q")
        game.apply(move)
        assert game.placement() == "Q7/7k/8/8/8/8/8/K7"


def test_engine_plays_many_plies_without_error():
    """Engine plays both sides for up to 200 plies; every reply must be legal."""
    with fake_engine(FAKE_ENGINE_SEED="7") as eng:
        game = GameState(robot_color=chess.WHITE)
        plies = 0
        while plies < 200 and not game.is_game_over():
            game.robot_color = game.board.turn  # pretend the robot is whoever moves
            move = eng.reply(game)
            assert move is not None and move in game.board.legal_moves
            game.apply(move)
            plies += 1
        assert plies > 10
        if game.is_game_over():
            assert game.status()["result"] in ("1-0", "0-1", "1/2-1/2")


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="stockfish not installed")
def test_real_stockfish_replies_legally():
    with Engine("stockfish", think_time=0.1, skill_level=5) as eng:
        assert eng.skill_level_supported
        game = GameState(robot_color=chess.WHITE)
        move = eng.reply(game)
        assert move in game.board.legal_moves
        game.apply(move)
        # now human's turn; engine must refuse
        with pytest.raises(RuntimeError):
            eng.reply(game)
