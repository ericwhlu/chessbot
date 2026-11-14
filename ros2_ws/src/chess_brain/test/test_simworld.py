"""Tests for the simulated world (fake human + camera + arm) and the operator console parsing."""

import json
import pathlib
import random
import sys

import chess
import pytest

from chess_brain.actions import actions_payload, move_to_actions
from chess_brain.console import classify
from chess_brain.game import Engine
from chess_brain.machine import BrainMachine, Config
from chess_brain.simworld import SimWorld, engine_chooser, random_chooser, scripted_chooser

FAKE = str(pathlib.Path(__file__).parent / "fake_uci_engine.py")
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def robot_payload(world, uci, action_id=1, allow_promotion=True):
    move = chess.Move.from_uci(uci)
    return actions_payload(action_id, move, move_to_actions(world.board, move, allow_promotion=allow_promotion))


# ---------------------------------------------------------------------- arm
def test_arm_executes_a_valid_command_and_updates_board_and_referee():
    world = SimWorld(choose=scripted_chooser(["e2e4"]))
    assert world.placement() == START and world.human_to_move
    assert world.play_human_move().uci() == "e2e4"
    assert world.robot_to_move
    reply = world.execute_arm_actions(robot_payload(world, "e7e5", action_id=7))
    assert reply == {"id": 7, "status": "ok"}
    assert world.placement() == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR"
    assert world.board.turn == chess.WHITE and world.history == ["human e2e4", "robot e7e5"]


@pytest.mark.parametrize(
    "fen,uci,captures",
    [
        ("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", "e4d5", 1),              # capture
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "e1g1", 0),        # castling
        ("rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", "e5d6", 1),              # en passant
        ("r6k/1P6/8/8/8/8/8/1K6 w - - 0 1", "b7a8q", 2),                                           # capture-promotion
    ],
    ids=["capture", "castling", "en-passant", "capture-promotion"],
)
def test_arm_handles_every_move_type(fen, uci, captures):
    world = SimWorld(human_color=chess.BLACK, start_fen=fen)   # the robot is white here
    reply = world.execute_arm_actions(robot_payload(world, uci))
    assert reply["status"] == "ok", reply
    assert world.placement() == world.board.board_fen()
    assert len(world.discarded) == captures


@pytest.mark.parametrize(
    "payload,needle",
    [
        ({"id": 1, "move": "e7e5", "actions": [{"pick": "e6", "place": "e5"}]}, "pick from empty square e6"),
        ({"id": 1, "move": "e7e5", "actions": [{"pick": "e7", "place": "d7"}]}, "place onto occupied square d7"),
        ({"id": 1, "move": "e7e5", "actions": [{"pick": "e7", "place": "e6"}]}, "expected"),
        ({"id": 1, "move": "e7e4", "actions": [{"pick": "e7", "place": "e4"}]}, "not a legal move"),
        ({"id": 1, "move": "e7e5", "actions": [{"pick": "z9", "place": "e5"}]}, "unknown pick location"),
        ({"id": 1, "move": "e7e5", "actions": [{"pick": "e7", "place": "nowhere"}]}, "unknown place location"),
        ({"id": 1, "move": "e7e5", "actions": [{"pick": "SPARE_X_W", "place": "e5"}]}, "unknown spare"),
        ({"id": 1, "move": "e7e5", "actions": [{"take": "e7"}]}, "malformed"),
        ({"id": 1, "actions": []}, "malformed arm_actions"),
        ({"id": 1, "move": "e7e5", "actions": []}, "expected"),
    ],
)
def test_arm_rejects_bad_commands_without_changing_anything(payload, needle):
    world = SimWorld(choose=scripted_chooser(["e2e4"]))
    world.play_human_move()
    before = (world.placement(), world.board.fen())
    reply = world.execute_arm_actions(payload)
    assert reply["status"] == "failed" and needle in reply["reason"], reply
    assert (world.placement(), world.board.fen()) == before and world.discarded == []


def test_arm_refuses_when_it_is_the_humans_turn_and_when_told_to_fail():
    world = SimWorld()
    reply = world.execute_arm_actions({"id": 1, "move": "e7e5", "actions": [{"pick": "e7", "place": "e5"}]})
    assert reply["status"] == "failed" and "not the robot's turn" in reply["reason"]
    world.play_human_move()
    move = next(iter(world.board.legal_moves))
    reply = world.execute_arm_actions(robot_payload(world, move.uci()), fail=True)
    assert reply["status"] == "failed" and world.board.turn == chess.BLACK
    assert world.arm_commands == 2


# -------------------------------------------------------------------- humans
def test_random_chooser_is_reproducible_and_legal():
    a, b = SimWorld(choose=random_chooser(3)), SimWorld(choose=random_chooser(3))
    assert a.play_human_move() == b.play_human_move()
    assert a.board.move_stack[0] in chess.Board().legal_moves


def test_scripted_chooser_follows_the_script_then_falls_back():
    world = SimWorld(choose=scripted_chooser(["e2e4", "g1f3"], then=random_chooser(1)))
    assert world.play_human_move().uci() == "e2e4"
    world.execute_arm_actions(robot_payload(world, "e7e5"))
    assert world.play_human_move().uci() == "g1f3"
    world.execute_arm_actions(robot_payload(world, "b8c6"))
    assert world.play_human_move() is not None        # script exhausted: the fallback chooser plays
    assert len(world.board.move_stack) == 5


def test_scripted_chooser_drops_the_script_when_the_game_leaves_it():
    choose = scripted_chooser(["e2e5", "g1f3"])       # e2e5 is illegal: the script is abandoned
    world = SimWorld(choose=choose)
    assert world.play_human_move() is None
    assert world.board.move_stack == []


def test_human_does_not_move_out_of_turn_or_after_the_game():
    world = SimWorld(human_color=chess.BLACK)
    assert world.play_human_move() is None            # white (the robot) is to move
    mated = SimWorld(start_fen="R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1", human_color=chess.BLACK)
    assert mated.game_over and mated.result() == "1-0" and mated.play_human_move() is None
    assert "1-0" in mated.summary()


def test_engine_chooser_plays_for_either_colour():
    eng = Engine([sys.executable, FAKE], think_time=0.01)
    try:
        world = SimWorld(choose=engine_chooser(eng))
        assert world.play_human_move() is not None                       # white human
        black = SimWorld(human_color=chess.BLACK, choose=engine_chooser(eng),
                         start_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        assert black.play_human_move() is not None                       # black human
    finally:
        eng.close()


# ------------------------------------------- brain + world: whole games, no ROS
class Done:
    def __init__(self, value):
        self._value = value

    def done(self):
        return True

    def result(self):
        return self._value


def play_game(seed, robot=chess.BLACK, arm_fail_every=0, max_ticks=40000):
    """The loop sim_world_node runs, without ROS: brain and world talk until the game ends."""
    rng = random.Random(seed)
    world = SimWorld(human_color=not robot, choose=random_chooser(seed + 1000))

    def think(game, roots):
        return Done(rng.choice(roots if roots is not None else list(game.board.legal_moves)))

    machine = BrainMachine(think, Config(robot_color=robot, stable_time=0.5, allow_promotion=True))
    now, brain, statuses, detects, queue = 0.0, None, [], [], []
    for _ in range(max_ticks):
        now += 0.25
        for topic, payload in machine.tick(now):
            if topic == "status":
                brain = payload
                statuses.append(payload["status"])
            elif topic == "arm_actions":
                queue.append(payload)
            elif topic == "detect_request":
                detects.append(payload)
                machine.on_detection(world.placement(), now)          # vision answers the request
        if brain["state"] in ("game_over", "error"):
            break
        if queue:
            payload = queue.pop(0)
            fail = arm_fail_every >= 2 and (world.arm_commands + 1) % arm_fail_every == 0
            machine.on_arm_done(json.dumps(world.execute_arm_actions(payload, fail=fail)), now)
        elif (
            brain["state"] == "wait_human"
            and brain["expected_placement"] == world.placement()
            and world.human_to_move
        ):
            world.play_human_move()
            machine.on_detection(world.placement(), now)              # the human presses the button
    return machine, world, brain, statuses, detects


@pytest.mark.parametrize("seed", range(6))
def test_brain_and_world_play_a_whole_game(seed):
    robot = chess.BLACK if seed % 2 == 0 else chess.WHITE
    machine, world, brain, statuses, detects = play_game(seed, robot=robot)
    assert brain["state"] == "game_over", brain
    assert world.game_over and brain["result"] == world.result()
    assert machine.game.fen() == world.board.fen()
    assert world.placement() == world.board.board_fen()
    assert not {"illegal_move", "arm_mismatch", "engine_error", "unsupported_move", "arm_failed"} & set(statuses)
    assert detects[0] == "setup" and "after_robot_move" in detects


def test_whole_game_with_an_arm_that_fails_every_third_command():
    machine, world, brain, statuses, detects = play_game(11, arm_fail_every=3)
    assert brain["state"] == "game_over" and brain["result"] == world.result()
    assert machine.game.fen() == world.board.fen()
    robot_moves = sum(1 for h in world.history if h.startswith("robot"))
    assert "arm_failed" in statuses and "arm_failed" in detects
    assert world.arm_commands > robot_moves >= 2                       # every failure was re-sent and then succeeded
    assert not {"illegal_move", "arm_mismatch", "engine_error", "unsupported_move"} & set(statuses)


# ----------------------------------------------------------- operator console
@pytest.mark.parametrize(
    "line,expected",
    [
        ("\n", ("detect", "button")),
        ("   ", ("detect", "button")),
        ("start\n", ("cmd", "start")),
        (" RESET ", ("cmd", "reset")),
        ("retry", ("cmd", "retry")),
        ("q", ("quit", "")),
        ("help", ("help", "")),
        (START + "\n", ("fen", START)),
        (START + " w KQkq - 0 1", ("fen", START)),
        ("8/8/8", ("unknown", "8/8/8")),
        ("dance", ("unknown", "dance")),
    ],
)
def test_operator_input_classification(line, expected):
    assert classify(line) == expected
