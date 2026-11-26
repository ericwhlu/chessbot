"""The simulated side of a game (WorldSession) against the real chess_brain state machine.

No ROS and no MuJoCo: a ``Rig`` passes the messages between the two the way the topics would,
on a fake clock, and a ``FakeWorld`` stands in for the simulation.
"""

import json
import random

import chess
import pytest

from chess_brain.actions import DISCARD
from chess_brain.machine import BrainMachine, Config, State
from chessbot_sim_ros.session import SessionConfig, WorldSession

AFTER_E4_D5 = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
TWO_CAPTURES = "4k3/7p/8/2p1p3/3P4/8/7P/4K3 w - - 0 1"      # the d4 pawn can take on c5 or on e5
STABLE = 0.5


class Done:
    """A finished engine search (the shape chess_brain expects from its think function)."""

    def __init__(self, move=None, error=None):
        self.move, self.error = move, error

    def done(self):
        return True

    def result(self):
        if self.error is not None:
            raise self.error
        return self.move


class FakeWorld:
    """A physical board without physics. Counts what was asked of it."""

    def __init__(self, fen=chess.STARTING_FEN, script=(), seed=3, fail_arm=(), blind_to=()):
        self.phys = chess.BaseBoard(chess.Board(fen).board_fen())
        self.script = [chess.Move.from_uci(uci) for uci in script]
        self.rng = random.Random(seed)
        self.fail_arm = set(fail_arm)      # 1-based numbers of the arm commands that fail
        self.blind_to = set(blind_to)      # squares the camera never reports
        self.unreadable = 0                # how many of the next scans cannot read the board at all
        self.scans = 0
        self.executed, self.by_hand, self.human, self.records = [], [], [], []
        self.before_human = None           # hook: called just before the opponent's hand moves

    def scan(self):
        self.scans += 1
        if self.unreadable > 0:
            self.unreadable -= 1
            return None
        seen = {chess.square_name(square) for square in chess.SQUARES if self.phys.piece_at(square)}
        return frozenset(seen - self.blind_to)

    def choose_human(self, board):
        if self.script:
            return self.script.pop(0)
        moves = sorted(board.legal_moves, key=lambda move: move.uci())
        captures = [move for move in moves if board.is_capture(move)]
        return self.rng.choice(captures if captures and self.rng.random() < 0.6 else moves) if moves else None

    def _set(self, board, move):
        after = board.copy(stack=False)
        after.push(move)
        self.phys = chess.BaseBoard(after.board_fen())

    def play_human(self, board, move):
        if self.before_human is not None:
            self.before_human()
        self.human.append(move.uci())
        self._set(board, move)

    def execute(self, board, move, actions):
        self.executed.append(move.uci())
        if len(self.executed) in self.fail_arm:
            return "simulated arm failure"                 # nothing was moved
        for action in actions:
            piece = self.phys.remove_piece_at(chess.parse_square(action["pick"]))
            assert piece is not None, f"pick from empty square {action['pick']}"
            if action["place"] != DISCARD:
                target = chess.parse_square(action["place"])
                assert self.phys.piece_at(target) is None, f"place onto occupied square {action['place']}"
                self.phys.set_piece_at(target, piece)
        after = board.copy(stack=False)
        after.push(move)
        assert self.phys.board_fen() == after.board_fen()
        return None

    def finish_by_hand(self, board, move):
        self.by_hand.append(move.uci())
        self._set(board, move)

    def record(self, entry):
        self.records.append(dict(entry))


class Rig:
    """chess_brain's state machine and a WorldSession, wired together on a fake clock.

    By default every message arrives at once and in order. ``lose`` drops messages
    ({"arm_done": {2}} loses the second arm_done), ``status_delay`` delivers chess_brain's
    status that many steps late (so that other topics overtake it), and ``brain_up = False``
    keeps chess_brain switched off for a while.
    """

    def __init__(self, world=None, robot=chess.BLACK, fen=chess.STARTING_FEN, engine_seed=11,
                 engine_error=None, lose=None, status_delay=0, stable_time=STABLE, arm_timeout=120.0, **session):
        self.now = 0.0
        self.world = world or FakeWorld(fen)
        self.rng = random.Random(engine_seed)
        self.engine_error = engine_error
        self.lose = {topic: set(numbers) for topic, numbers in (lose or {}).items()}
        self.count = {}                                     # messages seen so far, per topic
        self.status_delay, self.late = status_delay, []     # [(steps left, status text)]
        self.brain_up = True
        self.brain_log, self.session_log = [], []
        self.sent = []                                      # (sender, topic, payload), in order; lost ones too
        self.machine = BrainMachine(
            self._think, Config(robot_color=robot, start_fen=fen, stable_time=stable_time, arm_timeout=arm_timeout),
            log=self.brain_log.append)
        session.setdefault("plies", 12)
        self.session = WorldSession(self.world, SessionConfig(robot_color=robot, start_fen=fen, **session),
                                    clock=lambda: self.now, log=self.session_log.append)

    def _think(self, game, root_moves):
        if self.engine_error is not None:
            return Done(error=self.engine_error)
        moves = root_moves if root_moves is not None else list(game.board.legal_moves)
        return Done(self.rng.choice(sorted(moves, key=lambda move: move.uci())))

    def _arrives(self, sender, topic, payload):
        self.sent.append((sender, topic, payload))
        self.count[topic] = self.count.get(topic, 0) + 1
        return self.count[topic] not in self.lose.get(topic, ())

    def step(self, dt=0.1):
        self.now += dt
        self.late = [(steps - 1, text) for steps, text in self.late]
        while self.late and self.late[0][0] <= 0:
            self.session.on_status(self.late.pop(0)[1])
        for topic, payload in (self.machine.tick(self.now) if self.brain_up else ()):
            if not self._arrives("brain", topic, payload):
                continue
            if topic == "status" and self.status_delay:
                self.late.append((self.status_delay, json.dumps(payload)))
            elif topic == "status":
                self.session.on_status(json.dumps(payload))
            elif topic == "arm_actions":
                self.session.on_arm_actions(json.dumps(payload))
            elif topic == "detect_request":
                self.session.on_detect_request(payload)
        for topic, payload in self.session.tick():
            if not self._arrives("world", topic, payload) or not self.brain_up:
                continue
            if topic == "detected_occupancy":
                self.machine.on_occupancy(json.dumps(payload), self.now)
            elif topic == "arm_done":
                self.machine.on_arm_done(json.dumps(payload), self.now)
            elif topic == "cmd":
                self.machine.on_cmd(payload)

    def run(self, seconds=600.0):
        """Step until the session ends (or the time is up)."""
        end = self.now + seconds
        while self.session.finished is None and self.now < end:
            self.step()
        return self.session

    def topic(self, sender, name):
        return [payload for who, topic, payload in self.sent if who == sender and topic == name]

    def statuses(self):
        return [status["status"] for status in self.topic("brain", "status")]


def outcomes(session, who):
    return [ply["outcome"] for ply in session.plies if ply["who"] == who]


# ---------------------------------------------------------------- whole games
@pytest.mark.parametrize("robot", [chess.BLACK, chess.WHITE])
@pytest.mark.parametrize("scan_mode", ["continuous", "on_request"])
def test_twelve_plies_in_both_scan_modes_and_colours(robot, scan_mode):
    rig = Rig(robot=robot, scan_mode=scan_mode)
    session = rig.run()
    assert session.finished == "12 plies played" and session.clean
    assert [ply["who"] for ply in session.plies][:2] == (["human", "robot"] if robot == chess.BLACK else ["robot", "human"])
    assert set(outcomes(session, "robot")) == {"ok"} and len(session.plies) == 12
    assert set(outcomes(session, "human")) <= {"ok", "ambiguous"}
    assert rig.machine.game.fen() == session.board.fen()
    # one command per robot move; playing white, chess_brain is already on its 7th when the 12th ply ends the game
    assert len(rig.world.executed) == 6
    assert [c["move"] for c in rig.topic("brain", "arm_actions")][:6] == rig.world.executed
    assert [reply["status"] for reply in rig.topic("world", "arm_done")] == ["ok"] * 6
    assert rig.world.records == session.plies and not rig.world.by_hand
    assert not {"illegal_move", "arm_mismatch", "arm_failed", "arm_timeout"} & set(rig.statuses())


def test_long_random_games_stay_in_step():
    for seed in range(6):
        rig = Rig(FakeWorld(seed=seed), engine_seed=100 + seed, robot=chess.BLACK if seed % 2 else chess.WHITE, plies=0)
        session = rig.run(seconds=4000.0)
        assert session.finished is not None and not session.failed, (seed, session.finished, rig.session_log[-3:])
        assert session.finished.startswith("game over")
        assert rig.machine.state == State.GAME_OVER and rig.machine.game.fen() == session.board.fen()
        assert not {"illegal_move", "arm_mismatch"} & set(rig.statuses()), seed


# ------------------------------------------------------------------ start-up
@pytest.mark.parametrize("scan_mode", ["continuous", "on_request"])
@pytest.mark.parametrize("first", ["brain", "world"])
def test_game_starts_whoever_is_up_first_and_although_the_first_request_is_lost(scan_mode, first):
    rig = Rig(scan_mode=scan_mode, lose={"detect_request": {1}})
    if first == "brain":
        for _ in range(40):                   # chess_brain has asked for its first scan; nobody heard
            rig.now += 0.1
            rig.machine.tick(rig.now)
    else:
        rig.brain_up = False                  # the world scans, and nobody listens yet
        for _ in range(40):
            rig.step()
        rig.brain_up = True
    assert rig.run().finished == "12 plies played"


def test_on_request_mode_scans_only_when_asked():
    """Half a minute in which the opponent thinks: the heartbeat scans, on_request stays quiet."""
    counts = {}
    for scan_mode in ("on_request", "continuous"):
        rig = Rig(scan_mode=scan_mode, human_delay=1000.0, idle_timeout=2000.0)
        for _ in range(300):
            rig.step()
        assert rig.machine.state == State.WAIT_HUMAN and rig.session.finished is None
        counts[scan_mode] = rig.world.scans
    assert counts["continuous"] >= 12
    assert counts["on_request"] <= 2          # the first scan, and chess_brain's own request


@pytest.mark.parametrize("settings", [{"human_delay": 3.0}, {"rescan_period": 1.0}, {"stable_time": 2.5}])
def test_timing_settings_do_not_starve_the_opponent(settings):
    """A slow opponent, a fast heartbeat, a chess_brain that wants long-stable frames."""
    session = Rig(**settings).run()
    assert session.finished == "12 plies played" and session.clean


def test_game_that_never_starts_is_reported():
    world = FakeWorld(blind_to={"e2"})        # the camera never sees the start position
    session = Rig(world, start_timeout=30.0).run()
    assert session.failed and "did not start" in session.finished and "setup" in session.finished
    assert not world.human and not world.executed and not session.clean


@pytest.mark.parametrize("scan_mode", ["continuous", "on_request"])
def test_frames_the_camera_cannot_read_are_taken_again(scan_mode):
    rig = Rig(scan_mode=scan_mode)
    while len(rig.session.plies) < 3:
        rig.step()
    rig.world.unreadable = 2                  # the next two scans read nothing
    session = rig.run()
    assert session.finished == "12 plies played" and session.clean and rig.world.unreadable == 0
    assert any("could not read the board" in line for line in rig.session_log)


# --------------------------------------------------- the opponent and the arm
@pytest.mark.parametrize("settings", [{}, {"stable_time": 3.0}, {"status_delay": 4}, {"lose": {"detected_occupancy": {3, 6, 9}}}])
def test_opponent_moves_only_after_the_robot_move_is_verified(settings):
    """Guaranteed by chess_brain's ``verified``, not by a delay that happens to be long enough."""
    rig = Rig(**settings)
    seen = []
    rig.world.before_human = lambda: seen.append((rig.machine.state, rig.machine._verified))
    assert rig.run().finished == "12 plies played"
    assert len(seen) == 6 and set(seen) == {(State.WAIT_HUMAN, True)}


def test_arm_command_that_overtakes_the_status_waits_for_it():
    """Topics are not ordered among themselves: the command can arrive before the status that
    says chess_brain has seen the opponent's move. It is then executed, not refused."""
    rig = Rig(status_delay=4)
    session = rig.run()
    assert session.finished == "12 plies played" and session.clean
    assert [reply["status"] for reply in rig.topic("world", "arm_done")] == ["ok"] * 6
    assert len(rig.topic("brain", "arm_actions")) == 6 == len(rig.world.executed)


def test_duplicate_and_stale_arm_commands_run_once():
    rig = Rig(FakeWorld(script=["e2e4"]), plies=4)
    while not rig.topic("brain", "arm_actions"):
        rig.step()
    command = json.dumps(rig.topic("brain", "arm_actions")[0])
    assert rig.session.finished is None       # the arm executed it in the same step
    assert len(rig.world.executed) == 1
    rig.session.on_arm_actions(command)       # the same message again
    rig.step()
    assert len([reply for reply in rig.topic("world", "arm_done") if reply["id"] == 1]) == 1   # not answered twice
    rig.session.on_arm_actions(command.replace('"id": 1', '"id": 99'))   # same move, new id
    rig.step()
    assert len(rig.world.executed) == 1
    refused = [reply for reply in rig.topic("world", "arm_done") if reply["id"] == 99]
    assert refused and refused[0]["status"] == "failed" and "not a move for the robot" in refused[0]["reason"]
    assert rig.run().finished == "4 plies played" and len(rig.world.executed) == 2


def test_arm_command_for_a_move_that_is_not_legal_is_refused_untouched():
    rig = Rig(robot=chess.WHITE)
    rig.brain_up = False                      # only the world: it is the robot's turn, nothing is open
    rig.session.on_status(json.dumps({"state": "thinking", "status": "thinking", "fen": chess.STARTING_FEN}))
    rig.session.on_arm_actions(json.dumps({"id": 50, "move": "e2e5", "actions": [{"pick": "e2", "place": "e5"}]}))
    rig.step()
    assert rig.topic("world", "arm_done") == [
        {"id": 50, "status": "failed", "reason": f"e2e5 is not a move for the robot in {chess.STARTING_FEN}"}]
    assert not rig.world.executed and rig.session.board.fen() == chess.STARTING_FEN


def test_a_newer_arm_command_replaces_one_that_has_not_run():
    rig = Rig(FakeWorld(script=["e2e4"]), plies=4)
    while not rig.topic("brain", "arm_actions"):
        rig.step()
    reply = rig.topic("brain", "arm_actions")[0]["move"]
    assert rig.world.executed == [reply]
    rig.session.on_arm_actions(json.dumps({"id": 7, "move": "a2a3", "actions": [{"pick": "a2", "place": "a3"}]}))
    rig.session.on_arm_actions(json.dumps({"id": 8, "move": "h2h3", "actions": [{"pick": "h2", "place": "h3"}]}))
    rig.step()
    assert any("was replaced by 8" in line for line in rig.session_log)
    assert [r["id"] for r in rig.topic("world", "arm_done")] == [1, 8] and rig.world.executed == [reply]


@pytest.mark.parametrize("text", ["nonsense", "{}", '{"id": "x", "move": "e7e5", "actions": []}', '{"id": 1, "move": "e7e5"}',
                                  '{"id": 1e999, "move": "e7e5", "actions": []}', "[" * 100000])
def test_malformed_arm_commands_are_ignored(text):
    rig = Rig()
    rig.session.on_arm_actions(text)
    rig.session.on_status(text)
    for _ in range(5):
        rig.step()
    assert not rig.world.executed and not rig.topic("world", "arm_done")


# ----------------------------------------------------------------- arm failure
def test_failed_arm_move_is_finished_by_hand_and_sent_only_once():
    world = FakeWorld(fail_arm={2})
    rig = Rig(world)
    session = rig.run()
    assert session.finished == "12 plies played" and not session.failed and not session.clean
    assert outcomes(session, "robot") == ["ok", "operator", "ok", "ok", "ok", "ok"]
    assert session.plies[3]["detail"] == "simulated arm failure"
    assert len(world.by_hand) == 1 and world.by_hand[0] == session.plies[3]["move"]
    # the failed command was reported, never repeated: one command per robot move
    commands = rig.topic("brain", "arm_actions")
    assert [c["move"] for c in commands] == world.executed and len(commands) == 6
    assert [r["status"] for r in rig.topic("world", "arm_done")] == ["ok", "failed", "ok", "ok", "ok", "ok"]
    assert "arm_failed" in rig.statuses() and rig.machine.game.fen() == session.board.fen()


def test_every_arm_move_failing_still_finishes_the_game():
    world = FakeWorld(fail_arm=set(range(1, 50)))
    rig = Rig(world, plies=8)
    session = rig.run()
    assert session.finished == "8 plies played" and set(outcomes(session, "robot")) == {"operator"}
    assert len(rig.topic("brain", "arm_actions")) == 4 == len(world.by_hand)


# ------------------------------------------------------------- lost messages
def test_lost_arm_done_is_made_up_for_by_the_camera():
    """chess_brain never hears that the arm finished; after its arm_timeout a frame shows the move."""
    rig = Rig(lose={"arm_done": {2}}, arm_timeout=15.0, ply_timeout=60.0)
    session = rig.run()
    assert session.finished == "12 plies played" and session.clean
    assert "arm_timeout" in rig.statuses() and len(rig.world.executed) == 6


def test_robot_move_chess_brain_never_arrives_at_stops_the_session():
    rig = Rig(lose={"arm_done": {2}}, arm_timeout=500.0, ply_timeout=20.0)
    session = rig.run()
    assert session.failed and not session.clean and "did not arrive at robot move" in session.finished
    last = session.plies[-1]
    assert last["who"] == "robot" and last["outcome"] == "operator" and "not confirmed" in last["detail"]


@pytest.mark.parametrize("scan_mode", ["continuous", "on_request"])
def test_lost_arm_command_ends_the_session_instead_of_hanging(scan_mode):
    rig = Rig(lose={"arm_actions": {2}}, arm_timeout=15.0, idle_timeout=40.0, scan_mode=scan_mode)
    session = rig.run()
    assert session.failed and "nothing happened for 40 s" in session.finished
    assert "arm_timeout" in session.finished and len(rig.world.executed) == 1


@pytest.mark.parametrize("scan_mode", ["continuous", "on_request"])
@pytest.mark.parametrize("lost", [{"detected_occupancy": {4}}, {"detected_occupancy": {5, 6}}, {"detect_request": {2, 3}}])
def test_a_lost_frame_or_request_does_not_stop_the_game(scan_mode, lost):
    rig = Rig(scan_mode=scan_mode, lose=lost)
    session = rig.run()
    assert session.finished == "12 plies played" and session.clean, rig.session_log[-3:]


def test_lost_confirming_frame_of_a_capture_is_taken_again():
    for scan_mode in ("continuous", "on_request"):
        for lost in ({"detected_occupancy": {3}}, {"detect_request": {2}}, {"detected_occupancy": {2, 3}}):
            rig = Rig(FakeWorld(AFTER_E4_D5, script=["e4d5"]), fen=AFTER_E4_D5, plies=2, scan_mode=scan_mode, lose=lost)
            session = rig.run()
            assert session.finished == "2 plies played" and session.clean, (scan_mode, lost, rig.session_log[-3:])


# ----------------------------------------------- captures and ambiguous moves
def test_capture_is_confirmed_by_a_second_scan():
    rig = Rig(FakeWorld(AFTER_E4_D5, script=["e4d5"]), fen=AFTER_E4_D5, plies=2)
    session = rig.run()
    assert session.finished == "2 plies played" and session.clean
    assert session.plies[0] == {**session.plies[0], "who": "human", "move": "e4d5", "outcome": "ok"}
    assert "confirming_capture" in rig.statuses()
    assert "confirm_capture" in rig.topic("brain", "detect_request")


@pytest.mark.parametrize("scan_mode", ["continuous", "on_request"])
def test_operator_answers_an_ambiguous_capture(scan_mode):
    rig = Rig(FakeWorld(TWO_CAPTURES, script=["d4e5"]), fen=TWO_CAPTURES, plies=2, scan_mode=scan_mode)
    session = rig.run()
    assert session.finished == "2 plies played" and session.clean
    first = session.plies[0]
    assert (first["who"], first["move"], first["outcome"]) == ("human", "d4e5", "ambiguous")
    assert "d4c5 or d4e5" in first["detail"]
    assert "move d4e5" in rig.topic("world", "cmd")
    assert rig.machine.game.board.piece_at(chess.E5) == chess.Piece(chess.PAWN, chess.WHITE)


def test_operator_repeats_an_answer_that_came_too_early():
    """An answer given while the observation is being replaced is refused; the next one lands."""
    rig = Rig(FakeWorld(TWO_CAPTURES, script=["d4e5"]), fen=TWO_CAPTURES, plies=2)
    while "ambiguous_move" not in rig.statuses():
        rig.step()
    rig.machine.on_occupancy(json.dumps(["a1"]), rig.now)      # a stray frame replaces the observation
    rig.step()
    assert any("refusing cmd" in line for line in rig.brain_log)
    assert rig.run().finished == "2 plies played"
    assert rig.topic("world", "cmd").count("move d4e5") >= 2


def test_operator_does_not_answer_a_question_about_another_frame():
    """A misread frame left chess_brain asking "d4c5 or d4e5?" while the opponent played h2h3."""
    world = FakeWorld(TWO_CAPTURES, script=["h2h3"])
    clock = [0.0]
    session = WorldSession(world, SessionConfig(start_fen=TWO_CAPTURES), clock=lambda: clock[0])
    calm = {"state": "wait_human", "status": "waiting_human", "verified": True, "candidates": [], "fen": TWO_CAPTURES}
    session.on_status(json.dumps(calm))
    out = []
    for _ in range(40):
        clock[0] += 0.1
        out += session.tick()
    assert world.human == ["h2h3"]
    question = {**calm, "status": "ambiguous_move", "candidates": ["d4c5", "d4e5"]}
    session.on_status(json.dumps(question))
    for _ in range(40):
        clock[0] += 0.1
        out += session.tick()
    assert not [payload for topic, payload in out if topic == "cmd"]
    session.on_status(json.dumps({**question, "candidates": ["h2h3", "h2h4"]}))
    clock[0] += 0.1
    assert ("cmd", "move h2h3") in session.tick()


# ------------------------------------------------------------- things go wrong
def test_misread_human_move_stops_the_game_with_a_reason():
    world = FakeWorld(script=["e2e4", "g1f3"])
    rig = Rig(world, ply_timeout=20.0)
    while len(rig.session.plies) < 2:
        rig.step()
    world.blind_to = {"f3"}                   # from now on the camera misses the knight on f3
    session = rig.run()
    assert session.failed and "did not arrive at human move g1f3" in session.finished
    assert session.plies[-1]["outcome"] == "not recognised" and not session.clean
    assert len(world.executed) == 1           # the arm was not sent anywhere on a board it does not understand


def test_engine_error_stops_the_session():
    session = Rig(engine_error=RuntimeError("engine died")).run()
    assert session.failed and "engine_error" in session.finished


def test_unknown_scan_mode_is_refused():
    with pytest.raises(ValueError):
        WorldSession(FakeWorld(), SessionConfig(scan_mode="sometimes"))
