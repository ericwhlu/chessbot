"""Operator console for chess_brain: the "detect button" plus a few commands.

    ros2 run chess_brain operator_node

Type in the terminal, then press Enter:

    (just Enter)        press the detect button: asks vision for one detection
                        (publishes "button" on chessbot/detect_request)
    start               start the game without waiting for the start position
    reset               start a new game
    retry               re-send the robot's move after an arm failure / retry the engine
    <placement FEN>     act as vision yourself: publish this placement on
                        chessbot/detected_fen (e.g. pasted from the detector's web demo)
    help                show this text
    quit                leave

It also prints every status change of chess_brain, so one terminal shows what the
robot is waiting for.
"""

from __future__ import annotations

import json
import queue
import sys
import threading

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .console import HELP, classify

LATCHED = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class OperatorNode(Node):
    def __init__(self):
        super().__init__("chess_operator")
        self.finished = False
        self._lines: "queue.Queue[str | None]" = queue.Queue()
        self._last_line = None

        self._pub_detect = self.create_publisher(String, "chessbot/detect_request", 10)
        self._pub_cmd = self.create_publisher(String, "chessbot/cmd", 10)
        self._pub_fen = self.create_publisher(String, "chessbot/detected_fen", 10)
        self.create_subscription(String, "chessbot/status", self._on_status, LATCHED)
        self.create_timer(0.1, self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

        threading.Thread(target=self._read_stdin, name="stdin", daemon=True).start()
        print(HELP, flush=True)

    def _read_stdin(self) -> None:
        for line in sys.stdin:
            self._lines.put(line)
        self._lines.put(None)  # EOF (Ctrl-D, or no terminal attached)

    def _on_status(self, msg: String) -> None:
        try:
            s = json.loads(msg.data)
            line = f"[brain] {s['state']} / {s['status']}"
            if s.get("detail"):
                line += f" - {s['detail']}"
            line += f"   ({s['turn']} to move{', check' if s.get('check') else ''})"
        except (ValueError, TypeError, KeyError):
            return
        if line != self._last_line:
            self._last_line = line
            print(line, flush=True)

    def _tick(self) -> None:
        while True:
            try:
                line = self._lines.get_nowait()
            except queue.Empty:
                return
            if line is None:
                self.finished = True
                return
            kind, payload = classify(line)
            if kind == "detect":
                self._pub_detect.publish(String(data=payload))
                print("[you] detect button pressed", flush=True)
            elif kind == "cmd":
                self._pub_cmd.publish(String(data=payload))
                print(f"[you] cmd: {payload}", flush=True)
            elif kind == "fen":
                self._pub_fen.publish(String(data=payload))
                print(f"[you] published placement {payload}", flush=True)
            elif kind == "help":
                print(HELP, flush=True)
            elif kind == "quit":
                self.finished = True
                return
            else:
                print(f"[you] not understood: {payload!r} (type help)", flush=True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = OperatorNode()
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
