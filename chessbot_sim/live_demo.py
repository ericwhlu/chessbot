"""Standalone desktop demo for RGB perception from a movable MuJoCo camera."""
import argparse
import json
from pathlib import Path
from time import monotonic

import cv2
import glfw
import numpy as np
from OpenGL import GL

from .checks import verify, verify_pieces
from .piece_pose import WEIGHTS, BoardPlane, PieceDetector, SimPieceLayout
from .scene import HEIGHT, WIDTH
from .simulation import ARM_CONFIG, TAG_CONFIG, Camera, Simulation
from .vision import inverse_pose, rotation_about


PANEL_WIDTH = 540
WINDOW = "Chessbot | MuJoCo AprilTag camera pose"


def dashboard(frame, intrinsics):
    panel = np.full((HEIGHT, PANEL_WIDTH, 3), (28, 25, 23), dtype=np.uint8)
    text = (231, 235, 237)
    muted = (166, 167, 167)
    board_color = (150, 226, 154)
    base_color = (238, 212, 128)

    def label(value, x, y, size=0.56, color=text):
        cv2.putText(panel, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

    def matrix(values, x, y, step=61):
        if values is None:
            label("--", x, y, color=muted)
            return
        for row, row_values in enumerate(values):
            for col, value in enumerate(row_values):
                label(f"{value: .3f}", x + step * col, y + row * 24, 0.49)

    def show_error(name, pair, y, color, observation=None):
        label(name, 20, y, color=color)
        if observation is not None:
            label(f"{len(observation.ids)}/4 tags", 228, y, 0.50, color=muted)
        if pair.estimate is None:
            label("Unavailable", 327, y, color=muted)
            return
        delta, angle = pair.errors()
        label(f"{np.linalg.norm(delta) * 1000:.2f} mm   {angle:.3f} deg", 327, y, color=color)

    camera_to_board = frame.board.pose.inverse()
    label("Live ground-truth error", 20, 40, 0.85)
    show_error("Board in camera", frame.board.pose, 67, board_color, frame.board)
    show_error("Camera in board", camera_to_board, 94, board_color)
    show_error("Arm base in camera", frame.base.pose, 121, base_color, frame.base)
    show_error("Arm base in board", frame.base_to_board, 148, text)

    label("AprilTag estimate", 20, 187, 0.66)
    label("MuJoCo ground truth", 289, 187, 0.66)
    pairs = [
        ("Camera to board", camera_to_board),
        ("Board to camera", frame.board.pose),
        ("Arm base to camera", frame.base.pose),
        ("Arm base to board", frame.base_to_board),
    ]
    for index, (title, pair) in enumerate(pairs):
        top = 222 + index * 106
        label(title, 20, top, color=muted)
        matrix(pair.estimate[:3] if pair.estimate is not None else None, 20, top + 25)
        matrix(pair.truth[:3], 289, top + 25)

    label("K   Camera intrinsics (px)", 20, 648, color=muted)
    matrix(intrinsics, 20, 676, step=84)
    cv2.line(panel, (20, 745), (PANEL_WIDTH - 20, 745), (74, 71, 69), 1)
    label("Left drag  Orbit    Right drag  Free look    Wheel  Dolly", 20, 772, 0.51)
    label("WASD  Fly    Q/E  Down/up    Z/X  Roll", 20, 797, 0.51)
    switch = "Tab  Piece positions    " if frame.pieces is not None else ""
    label(switch + "R  Reset    P  Print    Esc  Quit", 20, 822, 0.51)
    return np.hstack((frame.bgr, panel))


def piece_dashboard(frame):
    panel = np.full((HEIGHT, PANEL_WIDTH, 3), (28, 25, 23), dtype=np.uint8)
    text = (231, 235, 237)
    muted = (166, 167, 167)
    accent = (118, 226, 252)

    def label(value, x, y, size=0.56, color=text):
        cv2.putText(panel, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

    label("Live piece positions", 20, 42, 0.88)
    label("Board coordinates (mm)    |    Piece base center", 20, 72, 0.54, muted)
    detections = frame.pieces["detections"]
    metrics = frame.pieces["metrics"]
    if metrics is None:
        label("Board pose unavailable", 20, 111, 0.71, accent)
        label("Bring the board and its AprilTags into view.", 20, 139, color=muted)
    else:
        label(f"{len(detections)} detected   |   {metrics['matched']}/{metrics['expected']} squares matched",
              20, 111, 0.65, accent)
        error = metrics["mean_error_mm"]
        if error is not None:
            label(f"Mean position error vs MuJoCo  {error:.2f} mm", 20, 140, color=muted)

    relative = frame.base_to_board
    if relative.estimate is not None:
        delta, angle = relative.errors()
        label(f"Arm base in board error  {np.linalg.norm(delta) * 1000:.2f} mm  {angle:.3f} deg",
              20, 166, 0.52, muted)

    rows_per_column = max(16, (len(detections) + 1) // 2)
    row_height = min(29, 508 // rows_per_column)
    if len(detections) > rows_per_column:
        cv2.line(panel, (266, 195), (266, 725), (74, 71, 69), 1)

    def table_row(values, base_x, y, size, color):
        label(values[0], base_x, y, size, color)
        for edge, value in zip((100, 155, 197, 238), values[1:]):
            width = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0]
            label(value, base_x + edge - width, y, size, color)

    for column, base_x in enumerate((20, 282)):
        if column * rows_per_column >= len(detections):
            break
        table_row(("Square", "X", "Y", "Z", "Conf."), base_x, 211, 0.48, muted)
        start = column * rows_per_column
        for row, piece in enumerate(detections[start:start + rows_per_column]):
            numbers = [piece["square"], *(f"{1000 * value:.1f}" for value in piece["xyz_m"]),
                       f"{100 * piece['confidence']:.0f}%"]
            table_row(numbers, base_x, 238 + row * row_height, 0.50, text)

    cv2.line(panel, (20, 745), (PANEL_WIDTH - 20, 745), (74, 71, 69), 1)
    label("Left drag  Orbit    Right drag  Free look    Wheel  Dolly", 20, 772, 0.51)
    label("WASD  Fly    Q/E  Down/up    Z/X  Roll", 20, 797, 0.51)
    label("Tab  Calibration    R  Reset    P  Print    Esc  Quit", 20, 822, 0.51)
    return np.hstack((frame.bgr, panel))


def print_matrices(frame, intrinsics):
    def observation_dict(observation, target):
        pair = observation.pose
        inverse = pair.inverse()
        to_camera = f"{target}_to_camera"
        from_camera = f"camera_to_{target}"
        result = {
            "tag_ids": observation.ids,
            "ground_truth": {to_camera: pair.truth.tolist(), from_camera: inverse.truth.tolist()},
            "estimate": None,
        }
        if pair.estimate is not None:
            error_pose = inverse if target == "board" else pair
            delta, angle = error_pose.errors()
            result["estimate"] = {
                to_camera: pair.estimate.tolist(),
                from_camera: inverse.estimate.tolist(),
                "reprojection_rms_px": observation.reprojection_px,
                "position_error_m": float(np.linalg.norm(delta)),
                f"position_delta_{'board' if target == 'board' else 'camera'}_xyz_m": delta.tolist(),
                "rotation_error_deg": angle,
            }
        return result

    relative = frame.base_to_board
    base_in_board = {"ground_truth": relative.truth.tolist(), "estimate": None}
    if relative.estimate is not None:
        delta, angle = relative.errors()
        base_in_board.update(
            estimate=relative.estimate.tolist(),
            position_error_m=float(np.linalg.norm(delta)),
            position_delta_board_xyz_m=delta.tolist(),
            rotation_error_deg=angle,
        )
    print(json.dumps({
        "K": intrinsics.tolist(),
        **observation_dict(frame.board, "board"),
        "arm_base": observation_dict(frame.base, "base"),
        "base_to_board": base_in_board,
        **({"pieces": frame.pieces} if frame.pieces is not None else {}),
    }, indent=2), flush=True)


def interactive(sim, detector=None, layout=None):
    # The offscreen renderer has its own context. Use a visible GLFW window for input and display.
    if not glfw.init():
        raise RuntimeError("Could not initialize the GLFW viewer")
    glfw.default_window_hints()
    total_width = WIDTH + PANEL_WIDTH
    window_width = 1300
    title = "Chessbot | Live YOLO piece localization" if detector is not None else WINDOW
    window = glfw.create_window(window_width, round(HEIGHT * window_width / total_width),
                                title, None, None)
    if window is None:
        raise RuntimeError("Could not open the GLFW viewer")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    texture = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glEnable(GL.GL_TEXTURE_2D)

    camera = Camera.initial(sim.view_center)
    frame = None
    dirty = True
    panel_dirty = False
    show_pieces = detector is not None
    print_requested = False
    last_mouse = None

    def on_cursor(_window, x, y):
        nonlocal last_mouse, dirty
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if (left or right) and last_mouse is not None:
            last_x, last_y = last_mouse
            pivot = sim.view_center if left and not right else None
            camera.drag(x - last_x, y - last_y, pivot)
            dirty = True
        last_mouse = (x, y)

    def on_scroll(_window, _dx, dy):
        nonlocal dirty
        camera.position -= camera.rotation[:, 2] * dy * 0.04
        dirty = True

    def on_key(_window, key, _scancode, action, _mods):
        nonlocal camera, dirty, panel_dirty, print_requested, show_pieces
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_R:
                camera = Camera.initial(sim.view_center)
                dirty = True
            elif key == glfw.KEY_P:
                print_requested = True
            elif key == glfw.KEY_TAB and detector is not None:
                show_pieces = not show_pieces
                panel_dirty = True

    glfw.set_cursor_pos_callback(window, on_cursor)
    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_key_callback(window, on_key)

    def pressed(key):
        return int(glfw.get_key(window, key) == glfw.PRESS)

    last_time = monotonic()
    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            now = monotonic()
            dt = min(now - last_time, 0.05)
            last_time = now
            movement = np.array([
                pressed(glfw.KEY_D) - pressed(glfw.KEY_A),
                pressed(glfw.KEY_E) - pressed(glfw.KEY_Q),
                pressed(glfw.KEY_S) - pressed(glfw.KEY_W),
            ])
            if movement.any():
                camera.position += camera.rotation @ movement * (0.32 * dt / np.linalg.norm(movement))
                dirty = True
            roll = pressed(glfw.KEY_Z) - pressed(glfw.KEY_X)
            if roll:
                camera.rotation = camera.rotation @ rotation_about([0, 0, 1], roll * 1.2 * dt)
                dirty = True

            if dirty:
                frame = sim.capture(camera, detector, layout)
            if dirty or panel_dirty:
                pixels = piece_dashboard(frame) if show_pieces else dashboard(frame, sim.intrinsics)
                glfw.make_context_current(window)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D, 0, GL.GL_RGB, pixels.shape[1], pixels.shape[0],
                    0, GL.GL_BGR, GL.GL_UNSIGNED_BYTE, pixels,
                )
                dirty = False
                panel_dirty = False
            if print_requested:
                print_matrices(frame, sim.intrinsics)
                print_requested = False

            width, height = glfw.get_framebuffer_size(window)
            scale = min(width / total_width, height / HEIGHT)
            view_width, view_height = int(total_width * scale), int(HEIGHT * scale)
            GL.glViewport((width - view_width) // 2, (height - view_height) // 2,
                          view_width, view_height)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glBegin(GL.GL_TRIANGLE_STRIP)
            for u, v, x, y in [(0, 1, -1, -1), (1, 1, 1, -1), (0, 0, -1, 1), (1, 0, 1, 1)]:
                GL.glTexCoord2f(u, v)
                GL.glVertex2f(x, y)
            GL.glEnd()
            glfw.swap_buffers(window)
    finally:
        glfw.make_context_current(window)
        GL.glDeleteTextures([texture])
        glfw.destroy_window(window)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", type=Path, default=TAG_CONFIG, help="tag layout JSON")
    parser.add_argument("--arm", type=Path, default=ARM_CONFIG, help="arm layout JSON")
    parser.add_argument("--fovy", type=float, default=49, help="vertical field of view in degrees")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="verify both targets using the default scene and FOV")
    mode.add_argument("--verify-pieces", action="store_true", help="check YOLO 3D positions against MuJoCo")
    parser.add_argument("--piece-weights", type=Path, default=WEIGHTS, help="trained YOLO pose checkpoint")
    parser.add_argument("--piece-confidence", type=float, default=0.25, help="YOLO box confidence threshold")
    args = parser.parse_args()
    if not 0 < args.fovy < 180:
        parser.error("--fovy must lie between 0 and 180 degrees")
    if not 0 < args.piece_confidence < 1:
        parser.error("--piece-confidence must lie between 0 and 1")
    if args.verify_pieces and not args.piece_weights.is_file():
        parser.error(f"YOLO checkpoint does not exist: {args.piece_weights}")
    with Simulation.from_files(args.tags, args.arm, args.fovy) as sim:
        if args.verify:
            verify(sim)
        elif args.verify_pieces:
            detector = PieceDetector(BoardPlane.from_model(sim.model), args.piece_weights, args.piece_confidence)
            verify_pieces(sim, detector)
        else:
            detector = None
            layout = None
            if args.piece_weights.is_file():
                detector = PieceDetector(BoardPlane.from_model(sim.model), args.piece_weights, args.piece_confidence)
                layout = SimPieceLayout(sim.model)
            interactive(sim, detector, layout)


if __name__ == "__main__":
    main()
