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
* ``reset`` on ``chessbot/cmd`` frees the discard slots for a new game;
* when chess_brain says which squares are occupied, it chooses for every action the direction
  the gripper should open along so that its fingers stay clear of the other pieces
  (grip_direction.py) and hands it to the executor; ``-p grip_direction:=fixed`` turns that off;
* with ``-p check_squares:=occupancy`` it looks before it moves: it asks for a detection
  (``chessbot/detect_request``), waits for the occupied squares on ``chessbot/detected_occupancy``
  and only starts if the pick square holds a piece and the place square is empty. It does so
  before the first action of a command and again before the second half of a capture (after
  parking the arm), so that with a vision that cannot tell the pieces apart a capturing piece is
  never put down on a captured piece that is still there. The parked arm must not hide the board
  from the camera (executor parameter ``park``);
* if the executor function has a ``prepare`` attribute (a function without arguments), it is
  called once at start-up, before any command: the place to bring the arm up;
* if it has an ``operator`` attribute, every command on ``chessbot/cmd`` is passed to it.

Do not run this together with sim_world_node: both answer ``arm_actions``.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from .arm_adapter import make_dry_run_executor, run_command
from .board_params import declare_board_parameters, locations_from_parameters
from .grip_direction import Fingers
from .plugins import load_callable


class ArmAdapterNode(Node):
    def __init__(self):
        super().__init__("arm_adapter")

        def declare_number(name, default):
            self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))

        self.declare_parameter("executor", "")     # "package.module:function"; empty = dry run
        declare_number("dry_run_time", 1.0)        # s per action in dry-run mode
        self.declare_parameter("grip_direction", "auto")   # auto = chosen per action; fixed = the arm's default
        declare_number("grip_gap", Fingers.gap)                    # m between the open fingers (model)
        declare_number("grip_piece_radius", Fingers.piece_radius)  # m, another piece at finger height (model)
        self.declare_parameter("check_squares", "off")   # off | occupancy: look at the board before moving
        declare_number("check_timeout", 15.0)             # s to wait for the camera's answer
        declare_board_parameters(self)             # board_x, board_y, board_yaw_deg, surface_z, ...

        def number(name):
            return float(self.get_parameter(name).value)

        self._locations = locations_from_parameters(self)
        frame = self._locations.frame
        how = str(self.get_parameter("grip_direction").value).strip().lower()
        if how not in ("auto", "fixed"):
            raise ValueError(f"grip_direction must be auto or fixed, not {how!r}")
        self._grips = how == "auto"
        self._fingers = Fingers(gap=number("grip_gap"), piece_radius=number("grip_piece_radius"))
        check = str(self.get_parameter("check_squares").value).strip().lower()
        if check not in ("off", "occupancy"):
            raise ValueError(f"check_squares must be off or occupancy, not {check!r}")
        self._check = check == "occupancy"
        self._check_timeout = number("check_timeout")
        self._seen = None               # (when it arrived, the squares) of the latest occupancy frame
        self._seen_lock = threading.Lock()

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
        if self._check:
            self._pub_look = self.create_publisher(String, "chessbot/detect_request", 10)
            self.create_subscription(String, "chessbot/detected_occupancy", self._on_occupancy, 10)
        self.create_timer(0.05, self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

        prepare = getattr(self._pick_and_place, "prepare", None)
        if callable(prepare):
            self.get_logger().warning("preparing the executor now: THE ARM MAY MOVE")
            self._running = self._pool.submit(self._prepare, prepare)   # commands wait behind it

        a1, h8 = frame.world_xyz("a1"), frame.world_xyz("h8")
        self.get_logger().info(
            f"arm_adapter ready: {mode}; a1 at ({a1[0]:.3f}, {a1[1]:.3f}, {a1[2]:.4f}), "
            f"h8 at ({h8[0]:.3f}, {h8[1]:.3f}, {h8[2]:.4f}); grip direction: {how}; "
            f"looking at the board before moving: {check}"
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

    def _on_occupancy(self, msg: String) -> None:
        try:
            squares = json.loads(msg.data)
            if not isinstance(squares, list):
                raise ValueError("not a list")
            seen = frozenset(str(name).strip().lower() for name in squares)
        except ValueError as exc:
            self.get_logger().warning(f"malformed detected_occupancy ignored: {exc}", throttle_duration_sec=10.0)
            return
        with self._seen_lock:
            self._seen = (time.monotonic(), seen)

    def _look(self):
        """Runs in the worker thread: ask for a detection and wait for a frame that arrived after
        the request. Returns the occupied squares, or None after ``check_timeout``."""
        asked = time.monotonic()
        self._pub_look.publish(String(data="arm_check"))
        while time.monotonic() - asked < self._check_timeout and not self._stop.is_set():
            with self._seen_lock:
                latest = self._seen
            if latest is not None and latest[0] > asked:
                return latest[1]
            time.sleep(0.05)
        return None

    def _on_cmd(self, msg: String) -> None:
        command = msg.data.strip().lower()
        if command == "reset":
            self._pending = None
            self._reset_wanted = True   # applied in _tick, never while the worker is using the slots
        operator = getattr(self._pick_and_place, "operator", None)
        if callable(operator):
            try:
                operator(command)
            except Exception as exc:
                self.get_logger().error(f"executor's operator hook failed on '{command}': {exc!r}")

    # ---------------------------------------------------------------------- tick
    def _prepare(self, prepare) -> None:
        """Runs in the worker thread, once, before the first command."""
        try:
            problem = prepare()
        except Exception as exc:
            problem = f"{type(exc).__name__}: {exc}"
        if problem is None or problem is True:
            self.get_logger().info("executor prepared: the arm is ready")
        else:
            self.get_logger().error(f"executor could not prepare: {problem} (it will try again on the first move)")

    def _work(self, payload: dict) -> None:
        """Runs in the worker thread."""

        def guarded(pick_xyz, place_xyz, info):
            if self._stop.is_set():
                return "arm_adapter is shutting down"
            return self._pick_and_place(pick_xyz, place_xyz, info)

        look = self._look if self._check else None
        self._replies.put((payload, run_command(payload, self._locations, guarded, self._fingers, self._grips, look)))

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
    def motion_in_progress(self) -> bool:
        return self._running is not None and not self._running.done()

    def shutdown(self) -> None:
        self._stop.set()  # the running pick-and-place finishes; no further action is started
        self._pool.shutdown(wait=False, cancel_futures=True)


def main(args=None) -> None:
    code, busy, node = 0, False, None
    try:
        rclpy.init(args=args)
        node = ArmAdapterNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except BaseException:  # a crash: show it even if we have to leave through os._exit below
        traceback.print_exc()
        code = 1
    finally:
        try:
            if node is not None:
                busy = node.motion_in_progress()
                node.shutdown()
                node.destroy_node()
            rclpy.try_shutdown()
        except Exception:
            traceback.print_exc()
            code = code or 1
        if busy:
            # The worker thread is still inside the executor (waiting for the arm) and Python would
            # wait for it forever. Leaving does NOT stop a motion the arm has already been sent.
            print("arm_adapter: stopped while the arm was moving; exiting without waiting for it", flush=True)
            sys.stderr.flush()
            os._exit(code)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
