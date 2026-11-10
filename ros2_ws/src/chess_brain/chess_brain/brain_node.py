"""ROS 2 node for chess_brain: a thin shell around ``BrainMachine`` (machine.py).

Topics (all ``std_msgs/String``; names are relative, so they can be remapped):

  subscribes
    chessbot/detected_fen   piece-placement FEN from vision
    chessbot/arm_done       {"id": 7, "status": "ok"} | {"id": 7, "status": "failed", "reason": "..."}
    chessbot/cmd            start | reset | retry
  publishes
    chessbot/arm_actions    {"id": 7, "move": "e7e5", "actions": [{"pick": "e7", "place": "e5"}]}
                            NOT latched: a command must never be replayed to an arm that (re)starts
                            later. An arm that missed it shows up as arm_timeout -> RECOVER.
    chessbot/status         {"state": ..., "status": ..., "detail": ..., "fen": ..., ...}   (latched)
    chessbot/board_state    full 6-field FEN after every applied move                       (latched)

Design rules:

* Subscription callbacks only store data; a 10 Hz timer runs the state machine.
* Stockfish runs in a single worker thread so a search never blocks the executor.
* Wall-clock (monotonic) time is used for stability, timeouts and the tick timer,
  so pausing the simulation (or use_sim_time without /clock) does not freeze the game logic.

Run:  ros2 run chess_brain brain_node --ros-args -p robot_color:=black -p think_time:=0.5
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import chess
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .game import Engine
from .machine import BrainMachine, Config

LATCHED = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
# Best-effort matches both reliable and sensor-data (best-effort) publishers.
# A reliable subscriber would silently never connect to a best-effort publisher.
DETECTION = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


class BrainNode(Node):
    def __init__(self):
        super().__init__("chess_brain")

        def declare_number(name, default):
            # Dynamic typing: accept `-p think_time:=1` as well as `1.0` (rclpy does not
            # coerce int <-> double and would refuse to start). Values are cast below.
            self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))

        self.declare_parameter("stockfish_path", "stockfish")
        self.declare_parameter("robot_color", "black")
        self.declare_parameter("start_fen", chess.STARTING_FEN)
        self.declare_parameter("allow_promotion", False)
        declare_number("think_time", 0.5)
        declare_number("skill_level", 5)
        declare_number("stable_time", 1.0)
        declare_number("arm_timeout", 120.0)
        declare_number("vision_timeout", 0.0)
        declare_number("max_arm_retries", 1)
        declare_number("tick_period", 0.1)
        declare_number("first_action_id", -1)  # -1: seed from the clock so ids differ between runs

        def param(name):
            return self.get_parameter(name).value

        color = str(param("robot_color")).strip().lower()
        if color not in ("white", "black"):
            raise ValueError(f"robot_color must be 'white' or 'black', got {color!r}")

        config = Config(
            robot_color=chess.WHITE if color == "white" else chess.BLACK,
            start_fen=str(param("start_fen")),
            stable_time=float(param("stable_time")),
            arm_timeout=float(param("arm_timeout")),
            vision_timeout=float(param("vision_timeout")),
            allow_promotion=bool(param("allow_promotion")),
            max_arm_retries=int(param("max_arm_retries")),
        )
        first_id = int(param("first_action_id"))
        # A stale arm_done from a previous run of this node must not match a new command.
        config.first_action_id = first_id if first_id >= 0 else (int(time.time()) % 10000) * 100

        self._engine = None
        self._pool = None
        # Validate the configuration (start_fen) before anything is started.
        self._machine = BrainMachine(self._think, config, log=self.get_logger().info)
        try:
            # Start the engine before advertising topics: if Stockfish is missing, fail loudly.
            try:
                self._engine = Engine(
                    str(param("stockfish_path")),
                    think_time=float(param("think_time")),
                    skill_level=int(param("skill_level")),
                )
            except OSError as exc:
                self.get_logger().fatal(
                    f"could not start Stockfish ({exc}). Run scripts/setup_deps.sh (or apt-get install "
                    "stockfish), or pass -p stockfish_path:=/full/path/to/stockfish"
                )
                raise
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stockfish")

            self._pub_actions = self.create_publisher(String, "chessbot/arm_actions", 10)
            self._pub_status = self.create_publisher(String, "chessbot/status", LATCHED)
            self._pub_board = self.create_publisher(String, "chessbot/board_state", LATCHED)

            self.create_subscription(String, "chessbot/detected_fen", self._on_detection, DETECTION)
            self.create_subscription(String, "chessbot/arm_done", self._on_arm_done, 10)
            self.create_subscription(String, "chessbot/cmd", self._on_cmd, 10)

            self.create_timer(
                float(param("tick_period")), self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME)
            )
        except Exception:
            self.shutdown()  # python-chess's engine thread is non-daemon: never leave it running
            raise
        self.get_logger().info(
            f"chess_brain ready: engine={self._engine.name} ({self._engine.command}), robot={color}, "
            f"stable_time={config.stable_time:g}s, allow_promotion={config.allow_promotion}, "
            f"first arm_actions id={config.first_action_id + 1}"
        )

    # ------------------------------------------------------------- engine thread
    def _think(self, game, root_moves):
        """Called by the machine; returns a Future so the search never blocks the executor."""
        return self._pool.submit(self._engine.reply, game, root_moves)

    # ----------------------------------------------------------------- callbacks
    def _on_detection(self, msg: String) -> None:
        self._machine.on_detection(msg.data, time.monotonic())

    def _on_arm_done(self, msg: String) -> None:
        self._machine.on_arm_done(msg.data, time.monotonic())

    def _on_cmd(self, msg: String) -> None:
        self._machine.on_cmd(msg.data)

    def _tick(self) -> None:
        for topic, payload in self._machine.tick(time.monotonic()):
            if topic == "arm_actions":
                self.get_logger().info(f"arm_actions: {payload}")
                self._pub_actions.publish(String(data=json.dumps(payload)))
            elif topic == "status":
                self.get_logger().info(f"status: {payload['state']} / {payload['status']} {payload['detail']}")
                self._pub_status.publish(String(data=json.dumps(payload)))
            elif topic == "board_state":
                self._pub_board.publish(String(data=payload))

    # ------------------------------------------------------------------ shutdown
    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        if self._engine is not None:
            self._engine.close()
            self._engine = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = BrainNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
