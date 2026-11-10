"""Engine wrapper tests. Most run against the fake UCI engine; one needs real Stockfish."""

import os
import pathlib
import shutil
import sys

import chess
import pytest

from chess_brain.game import Engine, GameState, resolve_engine_command

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


def test_reply_respects_root_moves():
    with fake_engine(FAKE_ENGINE_SEED="3") as eng:
        game = GameState(robot_color=chess.WHITE)
        only = [chess.Move.from_uci("a2a3")]
        for _ in range(5):
            assert eng.reply(game, root_moves=only) == only[0]


def test_reply_rejects_empty_root_moves():
    with fake_engine() as eng:
        with pytest.raises(ValueError):
            eng.reply(GameState(robot_color=chess.WHITE), root_moves=[])


def test_game_copy_is_independent_and_keeps_game_id():
    game = GameState()
    snap = game.copy()
    snap.apply(chess.Move.from_uci("e2e4"))
    assert game.fen() == chess.STARTING_FEN
    assert snap.game_id == game.game_id and snap.robot_color == game.robot_color
    assert GameState().game_id != game.game_id


def test_resolve_engine_command_leaves_unknown_commands_alone():
    cmd = [sys.executable, FAKE]
    assert resolve_engine_command(cmd) is cmd
    assert resolve_engine_command("/no/such/engine") == "/no/such/engine"
    assert os.path.isabs(resolve_engine_command(sys.executable))


def test_machine_with_engine_in_a_worker_thread():
    """The node's wiring: the search runs in a thread pool and the machine polls the Future."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    from chess_brain.machine import BrainMachine, Config, State

    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        with fake_engine(FAKE_ENGINE_SEED="5") as eng:
            machine = BrainMachine(lambda game, roots: pool.submit(eng.reply, game, roots), Config(stable_time=0.0))
            out = []
            machine.on_detection(start, time.monotonic())
            out += machine.tick(time.monotonic())
            assert machine.state == State.WAIT_HUMAN
            machine.on_detection("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR", time.monotonic())
            deadline = time.monotonic() + 10.0
            while machine.state != State.EXECUTING and time.monotonic() < deadline:
                out += machine.tick(time.monotonic())
                time.sleep(0.01)
            assert machine.state == State.EXECUTING, machine.status()
            (payload,) = [p for topic, p in out if topic == "arm_actions"]
            move = chess.Move.from_uci(payload["move"])
            assert move in chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").legal_moves
            machine.on_arm_done('{"id": %d, "status": "ok"}' % payload["id"], time.monotonic())
            machine.tick(time.monotonic())
            assert machine.state == State.WAIT_HUMAN and machine.game.board.turn == chess.WHITE
    finally:
        pool.shutdown(wait=True)


@pytest.mark.skipif(shutil.which(resolve_engine_command("stockfish")) is None, reason="stockfish not installed")
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
