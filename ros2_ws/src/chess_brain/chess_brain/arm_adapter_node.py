"""ROS 2 node between chess_brain and the arm: ``arm_actions`` in, one function call per action.

    ros2 run chess_brain arm_adapter_node                                      # dry run: logs, waits, says ok
    ros2 run chess_brain arm_adapter_node --ros-args -p executor:=my_pkg.motion:pick_and_place

The arm team plugs in ONE function (see arm_adapter.py for its contract). This node does the rest:

* subscribes to ``chessbot/arm_actions`` and answers each command it runs on ``chessbot/arm_done``;
* converts square names and DISCARD / SPARE_* to world coordinates (board_frame.py, following
  docs/board_setup.md); where the board sits in the world is set by the ``board_*`` parameters;
* runs the motion in a worker thread, one command at a time, so a slow arm never blocks ROS;
* a newer command replaces older ones that have not started yet; a repeated id is ignored;
* the same move re-sent while its first copy is still running is never executed twice: the second
  copy is answered with the first copy's result;
* ``reset`` on ``chessbot/cmd`` frees the discard slots for a new game.

Do not run this together with sim_world_node: both answer ``arm_actions``.
"""

from __future__ import annotations

import json
import math
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from .arm_adapter import make_dry_run_executor, run_command
from .board_frame import BoardFrame, Locations
from .plugins import load_callable


class ArmAdapterNode(Node):
    def __init__(self):
        super().__init__("arm_adapter")

        def declare_number(name, default):
            self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))

        self.declare_parameter("executor", "")     # "package.module:function"; empty = dry run
        declare_number("dry_run_time", 1.0)        # s per action in dry-run mode
        declare_number("board_x", 0.0)             # world position of the board's a1 corner (m)
        declare_number("board_y", 0.0)
        declare_number("board_yaw_deg", 0.0)       # rotation of the board about +Z
        declare_number("surface_z", 0.00656168)    # height of the playing surface (m)
        declare_number("square_size", 0.05)
        declare_number("discard_x", -0.075)        # first discard slot, in the board frame (m)
        declare_number("discard_y", 0.025)
        declare_number("spare_q_w_x", 0.475)       # spare queens, in the board frame (m)
        declare_number("spare_q_w_y", 0.025)
        declare_number("spare_q_b_x", 0.475)
        declare_number("spare_q_b_y", 0.375)

        def number(name):
            return float(self.get_parameter(name).value)

        frame = BoardFrame(
            origin_x=number("board_x"),
            origin_y=number("board_y"),
            yaw=math.radians(number("board_yaw_deg")),
            surface_z=number("surface_z"),
            square=number("square_size"),
        )
        self._locations = Locations(
            frame=frame,
            discard_first=(number("discard_x"), number("discard_y")),
            spares={
                "SPARE_Q_W": (number("spare_q_w_x"), number("spare_q_w_y")),
                "SPARE_Q_B": (number("spare_q_b_x"), number("spare_q_b_y")),
            },
        )

        spec = str(self.get_parameter("executor").value).strip()
        if spec:
            self._pick_and_place = load_callable(spec)   # ValueError with a clear message if it cannot be loaded
            mode = f"executor {spec}"
        else:
            self._pick_and_place = make_dry_run_executor(number("dry_run_time"), log=self.get_logger().info)
            mode = "DRY RUN (no arm motion; set -p executor:=package.module:function)"

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arm")
        self._pending = None            # newest command not started yet
        self._running = None            # Future of the command in progress
        self._replies: "queue.Queue[tuple]" = queue.Queue()   # (command payload, arm_done reply)
        self._stop = threading.Event()  # set at shutdown: no further actions are started
        self._seen_ids = []             # recently accepted ids, to ignore repeats
        self._reset_wanted = False      # a reset arrived while the arm was moving

        self._pub_done = self.create_publisher(String, "chessbot/arm_done", 10)
        self.create_subscription(String, "chessbot/arm_actions", self._on_arm_actions, 10)
        self.create_subscription(String, "chessbot/cmd", self._on_cmd, 10)
        self.create_timer(0.05, self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

        a1, h8 = frame.world_xyz("a1"), frame.world_xyz("h8")
        self.get_logger().info(
            f"arm_adapter ready: {mode}; a1 at ({a1[0]:.3f}, {a1[1]:.3f}, {a1[2]:.4f}), "
            f"h8 at ({h8[0]:.3f}, {h8[1]:.3f}, {h8[2]:.4f})"
        )

    # ----------------------------------------------------------------- callbacks
    def _on_arm_actions(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("not a JSON object")
        except ValueError as exc:
            self.get_logger().error(f"malformed arm_actions ignored: {exc}")
            return
        command_id = payload.get("id")
        if command_id in self._seen_ids:
            self.get_logger().warning(f"arm_actions id {command_id} already received; ignoring the repeat")
            return
        self._seen_ids = (self._seen_ids + [command_id])[-50:]
        if self._pending is not None:
            self.get_logger().warning(
                f"arm_actions id {self._pending.get('id')} replaced by newer id {command_id} before it started"
            )
        self._pending = payload

    def _on_cmd(self, msg: String) -> None:
        if msg.data.strip().lower() == "reset":
            self._pending = None
            self._reset_wanted = True   # applied in _tick, never while the worker is using the slots

    # ---------------------------------------------------------------------- tick
    def _work(self, payload: dict) -> None:
        """Runs in the worker thread."""

        def guarded(pick_xyz, place_xyz, info):
            if self._stop.is_set():
                return "arm_adapter is shutting down"
            return self._pick_and_place(pick_xyz, place_xyz, info)

        self._replies.put((payload, run_command(payload, self._locations, guarded)))

    def _tick(self) -> None:
        while True:
            try:
                done, reply = self._replies.get_nowait()
            except queue.Empty:
                break
            self._pub_done.publish(String(data=json.dumps(reply)))
            if reply["status"] == "ok":
                self.get_logger().info(f"arm_done: id {reply['id']} ok")
            else:
                self.get_logger().error(
                    f"arm_done: id {reply['id']} FAILED at action {reply['failed_action_index']}: {reply['reason']}"
                )
            again = self._pending
            if again is not None and again.get("actions") == done.get("actions"):
                # The same move was re-sent (new id) while this copy was still running, e.g. after
                # an arm_timeout and an operator retry. Never run it twice: answer the second
                # copy with this result. ok completes the move; failed lets a detection decide.
                self._pending = None
                self._pub_done.publish(String(data=json.dumps(dict(reply, id=again.get("id")))))
                self.get_logger().warning(
                    f"arm_actions id {again.get('id')} not executed: same move as id {reply['id']} ({reply['status']})"
                )

        if self._running is not None and self._running.done():
            error = self._running.exception()
            self._running = None
            if error is not None:  # run_command catches executor errors, so this is a bug in the adapter
                self.get_logger().error(f"arm worker crashed: {error!r}")

        if self._running is None and self._reset_wanted:
            self._reset_wanted = False
            self._locations.reset()
            self.get_logger().info("reset: discard slots cleared")

        if self._running is None and self._pending is not None:
            payload, self._pending = self._pending, None
            self.get_logger().info(
                f"arm_actions: id {payload.get('id')} move {payload.get('move')} "
                f"({len(payload.get('actions') or [])} action(s))"
            )
            self._running = self._pool.submit(self._work, payload)

    # ------------------------------------------------------------------ shutdown
    def shutdown(self) -> None:
        self._stop.set()  # the running pick-and-place finishes; no further action is started
        self._pool.shutdown(wait=False, cancel_futures=True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ArmAdapterNode()
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
