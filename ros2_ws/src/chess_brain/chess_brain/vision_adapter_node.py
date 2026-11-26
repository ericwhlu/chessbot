"""ROS 2 node between the camera and chess_brain: image in, one detector call, placement FEN (or the
occupied squares) out.

    ros2 run chess_brain vision_adapter_node --ros-args \\
        -p detector:=my_pkg.detector:detect -p image_topic:=/camera/image_raw

The vision team plugs in ONE function (see vision_adapter.py for its contract). This node does the rest:

* subscribes to the camera (``sensor_msgs/Image`` on ``image_topic``) and keeps the latest frame;
* on every message on ``chessbot/detect_request`` (the operator's button, or chess_brain asking
  after the robot's move) it runs the detector once on the latest frame and publishes the result
  on ``chessbot/detected_fen``; with ``output:=occupancy`` the detector returns the occupied squares
  and they go out as a JSON list on ``chessbot/detected_occupancy`` instead;
* with ``camera_info_topic`` set, a detector that takes two parameters also gets the camera
  (intrinsics from ``sensor_msgs/CameraInfo``, plus the ``detector_config`` parameter, a JSON object);
* with ``period`` > 0 it also detects continuously, every ``period`` seconds;
* a result whose picture is older than a request that is still waiting is not published: whoever
  asked (chess_brain after the robot's move, the arm adapter before a capture) wants the board as
  it is now, and gets the next detection instead;
* publishes nothing if there is no fresh image or the detector cannot read the board
  (chess_brain then simply keeps waiting), and says why in the log;
* runs the detector in a worker thread, so a slow detector never blocks ROS.
"""

from __future__ import annotations

import json
import queue
import time
from concurrent.futures import ThreadPoolExecutor

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from .plugins import load_callable
from .vision_adapter import (
    camera_summary,
    image_to_bgr,
    run_detector,
    run_occupancy_detector,
    superseded,
    wants_camera,
)


class VisionAdapterNode(Node):
    def __init__(self):
        super().__init__("vision_adapter")

        def declare_number(name, default):
            self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))

        self.declare_parameter("detector", "")                        # "package.module:function"
        self.declare_parameter("image_topic", "chessbot/camera/image")
        declare_number("period", 0.0)                                 # s; > 0 = continuous detection
        declare_number("max_image_age", 2.0)                          # s; older frames are not used
        self.declare_parameter("output", "fen")                       # fen | occupancy
        self.declare_parameter("camera_info_topic", "")               # sensor_msgs/CameraInfo, optional
        self.declare_parameter("detector_config", "")                 # JSON object handed to the detector

        self._period = float(self.get_parameter("period").value)
        self._max_age = float(self.get_parameter("max_image_age").value)
        topic = str(self.get_parameter("image_topic").value)

        spec = str(self.get_parameter("detector").value).strip()
        self._detect = load_callable(spec) if spec else None          # ValueError if it cannot be loaded
        self._output = str(self.get_parameter("output").value).strip().lower()
        if self._output not in ("fen", "occupancy"):
            raise ValueError(f"output must be fen or occupancy, not {self._output!r}")
        text = str(self.get_parameter("detector_config").value).strip()
        try:
            self._config = json.loads(text) if text else {}
            if not isinstance(self._config, dict):
                raise ValueError("not a JSON object")
        except ValueError as exc:
            raise ValueError(f"detector_config must be a JSON object: {exc}") from None
        info_topic = str(self.get_parameter("camera_info_topic").value).strip()
        self._needs_camera = bool(info_topic) and self._detect is not None and wants_camera(self._detect)
        self._camera = None             # latest camera summary (camera_summary), if a topic is given

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detector")
        self._image = None              # latest sensor_msgs/Image
        self._image_at = 0.0            # when it arrived (monotonic)
        self._request = None            # reason of a pending detection request
        self._request_at = 0.0          # when it arrived (monotonic)
        self._running = None            # Future of the detection in progress
        self._results: "queue.Queue[tuple]" = queue.Queue()
        self._last_periodic = 0.0

        self._pub_fen = self.create_publisher(String, "chessbot/detected_fen", 10)
        self._pub_occupancy = self.create_publisher(String, "chessbot/detected_occupancy", 10)
        if info_topic:
            self.create_subscription(CameraInfo, info_topic, self._on_camera_info, qos_profile_sensor_data)
        self.create_subscription(Image, topic, self._on_image, qos_profile_sensor_data)
        self.create_subscription(String, "chessbot/detect_request", self._on_request, 10)
        self.create_timer(0.05, self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

        mode = f"detector {spec}" if spec else "NO DETECTOR (set -p detector:=package.module:function)"
        how = f"every {self._period:g}s and on request" if self._period > 0 else "on request"
        extra = f"; camera info from '{info_topic}'" if info_topic else ""
        self.get_logger().info(
            f"vision_adapter ready: {mode}; output {self._output}; images from '{topic}'{extra}; detecting {how}"
        )

    # ----------------------------------------------------------------- callbacks
    def _on_image(self, msg: Image) -> None:
        if self._image is None and self._request is None:
            # chess_brain's start-up request usually arrives before this node is discovered;
            # detect once on the first frame so the game can start without a button press.
            self._request = "first_image"
        self._image = msg
        self._image_at = time.monotonic()

    def _on_camera_info(self, msg: CameraInfo) -> None:
        try:
            self._camera = camera_summary(msg.width, msg.height, msg.k, msg.d, msg.header.frame_id, self._config)
        except ValueError as exc:
            self.get_logger().warning(f"camera info not usable: {exc}", throttle_duration_sec=10.0)

    def _on_request(self, msg: String) -> None:
        self._request = msg.data.strip() or "request"
        self._request_at = time.monotonic()

    # ---------------------------------------------------------------------- tick
    def _work(self, msg: Image, reason: str, camera, image_at: float) -> None:
        """Runs in the worker thread. ``image_at``: when the picture arrived."""
        try:
            image = image_to_bgr(msg.height, msg.width, msg.encoding, msg.step, msg.data)
        except Exception as exc:
            self._results.put((None, f"cannot use the camera image: {exc}", reason, image_at))
            return
        if camera is None and self._config:
            camera = {"config": dict(self._config)}   # no camera info topic: the detector's settings only
        if self._output == "occupancy":
            found, why = run_occupancy_detector(self._detect, image, camera)
        else:
            found, why = run_detector(self._detect, image, camera)
        self._results.put((found, why, reason, image_at))

    def _tick(self) -> None:
        now = time.monotonic()

        while True:
            try:
                placement, why, reason, image_at = self._results.get_nowait()
            except queue.Empty:
                break
            if superseded(image_at, self._request is not None, self._request_at):
                self.get_logger().info(f"detection ({reason}) not published: a newer request is waiting")
            elif placement is None:
                self.get_logger().warning(f"no detection published ({reason}): {why}")
            elif self._output == "occupancy":
                self._pub_occupancy.publish(String(data=json.dumps(placement)))
                self.get_logger().info(f"detected_occupancy ({reason}): {len(placement)} squares: {' '.join(placement)}")
            else:
                self._pub_fen.publish(String(data=placement))
                self.get_logger().info(f"detected_fen ({reason}): {placement}")

        if self._running is not None and self._running.done():
            error = self._running.exception()
            self._running = None
            if error is not None:
                self.get_logger().error(f"detector worker crashed: {error!r}")

        if self._request is None and self._period > 0 and now - self._last_periodic >= self._period:
            self._request = "periodic"
        if self._request is None or self._running is not None:
            return
        waited = now - self._request_at
        if self._detect is not None and self._image is not None and self._image_at < self._request_at and waited < self._max_age:
            return  # prefer a frame taken AFTER the request (the arm may still have been in view before)

        reason, self._request = self._request, None
        self._last_periodic = now
        if self._detect is None:
            self.get_logger().warning(f"detection requested ({reason}) but no detector is configured")
        elif self._image is None:
            self.get_logger().warning(f"detection requested ({reason}) but no camera image has arrived yet")
        elif now - self._image_at > self._max_age:
            age = now - self._image_at
            self.get_logger().warning(f"detection requested ({reason}) but the last camera image is {age:.1f}s old")
        elif self._needs_camera and self._camera is None:
            self.get_logger().warning(f"detection requested ({reason}) but no camera info has arrived yet")
        else:
            self._running = self._pool.submit(self._work, self._image, reason, self._camera, self._image_at)

    # ------------------------------------------------------------------ shutdown
    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = VisionAdapterNode()
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
