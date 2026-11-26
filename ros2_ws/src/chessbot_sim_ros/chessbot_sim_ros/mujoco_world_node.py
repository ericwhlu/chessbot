"""ROS 2 node that owns the MuJoCo simulation: the camera, the arm, the opponent and an operator
for a game played by chess_brain.

    ros2 launch chessbot_sim_ros mujoco_game.launch.py plies:=12
    ros2 run chessbot_sim_ros mujoco_world_node --ros-args -p repo_root:=/chessbot

Topics (all ``std_msgs/String``; the same ones a real camera node and a real arm node use):

  subscribes
    chessbot/arm_actions    {"id": 7, "move": "e7e5", "actions": [{"pick": "e7", "place": "e5"}]}
    chessbot/detect_request anything: "look at the board now" (chess_brain, or a button)
    chessbot/status         what chess_brain is doing (latched)
  publishes
    chessbot/detected_occupancy  JSON list of the squares the camera sees a piece on
    chessbot/arm_done            {"id": 7, "status": "ok"} | {"id": 7, "status": "failed", "reason": "..."}
    chessbot/cmd                 "move <uci>": the simulated operator answering ambiguous_move

What happens when is decided by ``WorldSession`` (session.py; read its docstring), which is
tested without ROS. This file only connects it to the topics and to ``chessbot_sim.world``.

One thread on purpose: callbacks store, a timer works. MuJoCo's renderer and its OpenGL
context must stay on the thread that created them, and the vision and the arm share one
simulation, so there is nothing to run in parallel. The node is busy while the arm moves
(10 to 40 s); chess_brain sends nothing during that time and accepts a late reply.

The node exits when the game does: 0 if no move needed the operator's hands, 1 otherwise.
Images and ``summary.json`` go to ``out_dir``.
"""

from __future__ import annotations

import json
import sys

import chess
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .paths import use_repo
from .session import SessionConfig, WorldSession

LATCHED = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class MujocoWorldNode(Node):
    def __init__(self):
        super().__init__("mujoco_world")

        def declare_number(name, default):
            # Dynamic typing: accept `-p plies:=12.0` as well as `12` (as brain_node does).
            self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))

        self.declare_parameter("repo_root", "")              # where chessbot_sim/ is; "" = find it
        self.declare_parameter("robot_color", "black")
        self.declare_parameter("scan_mode", "continuous")    # continuous | on_request
        self.declare_parameter("script", "")                 # opponent moves to play first: "e2e4,d2d4"
        self.declare_parameter("out_dir", "game_run_ros")
        self.declare_parameter("video", False)
        declare_number("plies", 12)                          # 0 = play to the end
        declare_number("seed", 7)                            # of the simulated opponent
        declare_number("human_delay", 1.5)                   # s; must exceed chess_brain's stable_time
        declare_number("rescan_period", 2.0)
        declare_number("start_timeout", 120.0)
        declare_number("ply_timeout", 60.0)
        declare_number("tick_period", 0.1)

        def param(name):
            return self.get_parameter(name).value

        color = str(param("robot_color")).strip().lower()
        if color not in ("white", "black"):
            raise ValueError(f"robot_color must be 'white' or 'black', got {color!r}")
        config = SessionConfig(
            robot_color=chess.WHITE if color == "white" else chess.BLACK,
            plies=int(param("plies")),
            scan_mode=str(param("scan_mode")).strip().lower(),
            rescan_period=float(param("rescan_period")),
            human_delay=float(param("human_delay")),
            start_timeout=float(param("start_timeout")),
            ply_timeout=float(param("ply_timeout")),
        )

        self.exit_code = None
        self.world = None
        root = use_repo(str(param("repo_root")))
        from chessbot_sim.world import MujocoWorld  # needs the repository on sys.path (use_repo)

        self.get_logger().info(f"loading the MuJoCo scene and the detector from {root} ...")
        self.world = MujocoWorld(out=str(param("out_dir")), robot=color, seed=int(param("seed")),
                                 script=str(param("script")), video=bool(param("video")))
        self._session = WorldSession(self.world, config, log=self.get_logger().info)

        self._out = {  # not _publishers: rclpy's Node keeps its own list under that name
            "detected_occupancy": self.create_publisher(String, "chessbot/detected_occupancy", 10),
            "arm_done": self.create_publisher(String, "chessbot/arm_done", 10),
            "cmd": self.create_publisher(String, "chessbot/cmd", 10),
        }
        self.create_subscription(String, "chessbot/arm_actions", self._on_arm_actions, 10)
        self.create_subscription(String, "chessbot/detect_request", self._on_detect_request, 10)
        self.create_subscription(String, "chessbot/status", self._on_status, LATCHED)
        self.create_timer(float(param("tick_period")), self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f"mujoco_world ready: robot={color}, plies={config.plies}, scan_mode={config.scan_mode}"
        )

    # callbacks only store; the timer does the work
    def _on_arm_actions(self, msg: String) -> None:
        self._session.on_arm_actions(msg.data)

    def _on_detect_request(self, msg: String) -> None:
        self._session.on_detect_request(msg.data)

    def _on_status(self, msg: String) -> None:
        self._session.on_status(msg.data)

    def _tick(self) -> None:
        if self.exit_code is not None:
            return
        for topic, payload in self._session.tick():
            text = payload if isinstance(payload, str) else json.dumps(payload)
            if topic != "detected_occupancy":
                self.get_logger().info(f"{topic}: {text}")
            self._out[topic].publish(String(data=text))
        session = self._session
        if session.finished is not None:
            for line in self.world.finish(session.finished, session.failed, session.board.fen()):
                self.get_logger().info(line)
            self.exit_code = 0 if session.clean else 1

    def close(self) -> None:
        if self.world is not None:
            self.world.close()
            self.world = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    code = 1
    try:
        node = MujocoWorldNode()
        while rclpy.ok() and node.exit_code is None:
            rclpy.spin_once(node, timeout_sec=0.2)
        if node.exit_code is not None:
            code = node.exit_code
    except (KeyboardInterrupt, ExternalShutdownException):
        code = 130
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
