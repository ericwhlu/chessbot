"""ROS 2 node that stands in for the human, the camera and the arm (see simworld.py).

Run it next to brain_node and the two of them play a whole game with no hardware:

    ros2 launch chess_brain sim_game.launch.py

What it does, on the same topics the real components will use:

* publishes ``chessbot/detected_fen`` — the placement of its simulated physical board:
  once after the "human" moves, whenever anyone publishes ``chessbot/detect_request``,
  and every ``republish_period`` seconds as a safety net. Nothing while the arm is moving.
* subscribes to ``chessbot/arm_actions``, takes ``arm_action_time`` seconds per action,
  executes them on the simulated board and answers on ``chessbot/arm_done``. A command
  that breaks the arm's rules (pick from an empty square, place onto an occupied one,
  wrong final position) is answered with ``failed`` and a reason.
* plays the human side: a weak Stockfish (``human:=engine``), random moves, or a script.

It only moves for the human once the brain reports ``wait_human`` and agrees on the board.
"""

from __future__ import annotations

import json
import time

import chess
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .game import Engine
from .simworld import SimWorld, engine_chooser, random_chooser, scripted_chooser

LATCHED = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class SimWorldNode(Node):
    def __init__(self):
        super().__init__("sim_world")

        def declare_number(name, default):
            self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))

        self.declare_parameter("robot_color", "black")          # the human plays the other colour
        self.declare_parameter("start_fen", chess.STARTING_FEN)
        self.declare_parameter("human", "engine")               # engine | random | script
        self.declare_parameter("human_moves", "")               # UCI moves for human:=script, e.g. "e2e4 g1f3"
        self.declare_parameter("stockfish_path", "stockfish")
        self.declare_parameter("exit_on_game_over", True)
        declare_number("human_skill", 0)                        # Stockfish Skill Level of the simulated human
        declare_number("human_think_time", 0.1)
        declare_number("human_delay", 1.5)                      # s the human waits before moving
        declare_number("arm_action_time", 1.0)                  # s per pick-and-place
        declare_number("arm_fail_every", 0)                     # N >= 2: every Nth arm command fails (recovery demo)
        declare_number("republish_period", 2.0)                 # s; 0 disables the periodic detection
        declare_number("max_plies", 0)                          # stop after this many half-moves; 0 = play to the end
        declare_number("seed", 0)
        declare_number("tick_period", 0.1)

        def param(name):
            return self.get_parameter(name).value

        robot = str(param("robot_color")).strip().lower()
        if robot not in ("white", "black"):
            raise ValueError(f"robot_color must be 'white' or 'black', got {robot!r}")
        human_color = chess.BLACK if robot == "white" else chess.WHITE

        self._human_delay = float(param("human_delay"))
        self._arm_action_time = float(param("arm_action_time"))
        self._arm_fail_every = int(param("arm_fail_every"))
        self._republish_period = float(param("republish_period"))
        self._max_plies = int(param("max_plies"))
        self._exit_on_game_over = bool(param("exit_on_game_over"))
        self._start_fen = str(param("start_fen"))

        self._engine = None
        kind = str(param("human")).strip().lower()
        chooser = random_chooser(int(param("seed")))
        if kind in ("engine", "script"):
            try:
                self._engine = Engine(
                    str(param("stockfish_path")),
                    think_time=float(param("human_think_time")),
                    skill_level=int(param("human_skill")),
                )
                chooser = engine_chooser(self._engine)
            except OSError as exc:
                self.get_logger().warning(f"no engine for the simulated human ({exc}); playing random moves")
        elif kind != "random":
            raise ValueError(f"human must be engine, random or script, got {kind!r}")
        if kind == "script":
            ucis = str(param("human_moves")).replace(",", " ").split()
            try:
                for uci in ucis:
                    chess.Move.from_uci(uci)
            except ValueError as exc:
                self.shutdown()
                raise ValueError(f"human_moves: {exc}") from exc
            chooser = scripted_chooser(ucis, then=chooser)

        try:
            self.world = SimWorld(human_color=human_color, start_fen=self._start_fen, choose=chooser)

            self.finished = False
            self._brain = None            # last status dict from the brain
            self._arm_queue = []          # arm_actions payloads waiting to be executed
            self._arm_job = None          # (payload, finish_time) while the arm is "moving"
            self._human_due = None        # when the human will make the next move
            self._last_pub = 0.0
            self._failures = 0
            self._reported = False

            self._pub_fen = self.create_publisher(String, "chessbot/detected_fen", 10)
            self._pub_done = self.create_publisher(String, "chessbot/arm_done", 10)
            self.create_subscription(String, "chessbot/arm_actions", self._on_arm_actions, 10)
            self.create_subscription(String, "chessbot/detect_request", self._on_detect_request, 10)
            self.create_subscription(String, "chessbot/status", self._on_status, LATCHED)
            self.create_timer(
                float(param("tick_period")), self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME)
            )
        except Exception:
            self.shutdown()
            raise
        who = self._engine.name if self._engine is not None else "random mover"
        self.get_logger().info(
            f"sim_world ready: human plays {'white' if human_color else 'black'} ({kind}: {who}), "
            f"arm takes {self._arm_action_time:g}s per action, waiting for chess_brain ..."
        )

    # ----------------------------------------------------------------- callbacks
    def _on_status(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
        except ValueError:
            return
        if isinstance(status, dict) and "state" in status and "expected_placement" in status:
            self._brain = status

    def _on_arm_actions(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("not a JSON object")
        except ValueError as exc:
            self.get_logger().error(f"malformed arm_actions ignored: {exc}")
            return
        self._arm_queue.append(payload)

    def _on_detect_request(self, msg: String) -> None:
        if self._arm_job is None and not self._arm_queue:  # the camera is blind while the arm moves
            self._publish_placement(f"detect_request:{msg.data}")

    # ---------------------------------------------------------------------- tick
    def _publish_placement(self, why: str) -> None:
        self._last_pub = time.monotonic()
        self._pub_fen.publish(String(data=self.world.placement()))
        self.get_logger().debug(f"detected_fen ({why}): {self.world.placement()}")

    def _tick(self) -> None:
        if self.finished:
            return
        now = time.monotonic()

        # 1. The arm is moving: finish the job when its time is up.
        if self._arm_job is not None:
            payload, finish_at = self._arm_job
            if now < finish_at:
                return
            self._arm_job = None
            n = self._arm_fail_every
            fail = n >= 2 and (self.world.arm_commands + 1) % n == 0
            reply = self.world.execute_arm_actions(payload, fail=fail)
            if reply["status"] == "ok":
                self.get_logger().info(f"arm: executed {payload.get('move')} (id {reply['id']})")
            else:
                self._failures += 1
                text = f"arm: FAILED id {reply['id']}: {reply['reason']}"
                if fail:  # two call sites: rclpy forbids changing the severity at one call site
                    self.get_logger().warning(text)
                else:
                    self.get_logger().error(text)
            self._pub_done.publish(String(data=json.dumps(reply)))
            return

        # 2. Start the next queued arm command.
        if self._arm_queue:
            payload = self._arm_queue.pop(0)
            actions = payload.get("actions")
            count = len(actions) if isinstance(actions, list) else 1
            self._arm_job = (payload, now + self._arm_action_time * max(1, count))
            self._human_due = None
            self.get_logger().info(f"arm: moving for {payload.get('move')} ({count} action(s)) ...")
            return

        # 3. Nothing to do until the brain is up.
        brain = self._brain
        if brain is None:
            return

        # 4. Game over (as the brain sees it), or the demo limit is reached.
        plies = len(self.world.board.move_stack)
        caught_up = brain["state"] == "wait_human" and brain["expected_placement"] == self.world.placement()
        limit_hit = self._max_plies > 0 and plies >= self._max_plies and self.world.human_to_move and caught_up
        if brain["state"] == "game_over":
            self._report("game over")
            return
        if brain["state"] == "error":
            self._report(f"chess_brain stopped with an error ({brain.get('status')}: {brain.get('detail')})")
            return
        if limit_hit:
            self._report("demo limit reached")
            return

        # 5. Safety-net detection (also covers a start position sent before the brain was listening).
        if self._republish_period > 0 and now - self._last_pub >= self._republish_period:
            self._publish_placement("periodic")

        # 6. The human moves once the brain is waiting and agrees on the board.
        in_sync = brain["state"] == "wait_human" and brain["expected_placement"] == self.world.placement()
        if in_sync and self.world.human_to_move:
            if self._human_due is None:
                self._human_due = now + self._human_delay
            elif now >= self._human_due:
                self._human_due = None
                move = self.world.play_human_move()
                if move is not None:
                    self.get_logger().info(f"human: played {move.uci()}")
                    self._publish_placement("human moved")  # the human presses the detect button
        else:
            self._human_due = None

    def _report(self, why: str) -> None:
        if self._reported:
            return
        self._reported = True
        world = self.world
        try:
            moves = chess.Board(self._start_fen).variation_san(world.board.move_stack)
        except ValueError:
            moves = " ".join(m.uci() for m in world.board.move_stack)
        brain_result = (self._brain or {}).get("result")
        agree = "brain and referee agree" if brain_result == world.result() else (
            f"MISMATCH: brain says {brain_result}, referee says {world.result()}"
        )
        self.get_logger().info(
            f"=== {why}: {world.summary()}; arm commands {world.arm_commands}, failures {self._failures}; {agree}"
        )
        self.get_logger().info(f"=== moves: {moves}")
        if self._exit_on_game_over:
            self.finished = True

    # ------------------------------------------------------------------ shutdown
    def shutdown(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SimWorldNode()
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
