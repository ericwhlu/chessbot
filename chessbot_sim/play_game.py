"""Play a chess game in the MuJoCo scene: chess_brain chooses, the arm moves, the camera checks.

    python -m chessbot_sim.play_game                # 12 plies, robot plays black
    python -m chessbot_sim.play_game --plies 30 --video
    python -m chessbot_sim.play_game --script e2e4,d2d4,e4d5 --seed 3

One ply of the loop:

* human's turn - the simulated opponent picks a move and a simulated hand puts the piece on
  its new square. The camera then scans the board (AprilTags + YOLO, as in the other demos) and
  chess_brain works out which move was played from the occupied squares alone.
* robot's turn - chess_brain asks Stockfish for a move and turns it into pick-and-place
  actions (a capture is two: the captured piece to the side first). For each action the camera
  locates the piece, ``plan_pick_place`` plans the motion and the arm executes it through the
  MuJoCo actuators. A final scan must show exactly the squares the move should leave occupied.

The robot only uses what the camera reports. MuJoCo's ground truth is used for two things a
real game has people for: the opponent's hand, and an operator who finishes a move by hand when
the arm or the camera fails (the game then continues, and the summary says so).

Results go to ``--out`` (default ``game_run/``): one annotated camera image per ply,
``summary.json``, and ``game.mp4`` with ``--video``.
"""
import argparse
import json
from pathlib import Path
import sys

import chess
import cv2
import mujoco
import numpy as np

from .gameplay import STARTING_SQUARES, HumanPlayer, PieceBook, Report, discard_slots
from .manipulation import CLOSED_GRIP, FPS, OPEN_GRIP, Arm, plan_pick_place
from .piece_pose import BoardPlane, PieceDetector
from .scene import HEIGHT, WIDTH
from .simulation import Camera, Simulation

# chess_brain is the ROS 2 package of this repository; its game logic is plain Python.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "chess_brain"))
from chess_brain.actions import DISCARD, UnsupportedMove, move_to_actions  # noqa: E402
from chess_brain.game import Engine, GameState  # noqa: E402
from chess_brain.occupancy import infer_from_occupancy, occupancy_after, squares_of  # noqa: E402


def quiet(_label):
    pass


class Hand:
    """The simulated opponent's hand, and the operator's: moves piece bodies directly.

    Simulation only. It is the one place that writes MuJoCo state for the pieces.
    """

    def __init__(self, sim, plane, book):
        self.sim, self.plane, self.book = sim, plane, book
        self.buried = 0

    def put(self, key, xyz):
        joint = self.sim.model.joint(f"free_{key}").id
        q = int(self.sim.model.jnt_qposadr[joint])
        v = int(self.sim.model.jnt_dofadr[joint])
        self.sim.data.qpos[q:q + 7] = [xyz[0], xyz[1], xyz[2], 1, 0, 0, 0]
        self.sim.data.qvel[v:v + 6] = 0

    def bury(self, key):
        """Take a captured piece off the table (far from the board and the camera)."""
        self.put(key, (-1.0 - 0.07 * self.buried, -1.0, 0.0))
        self.buried += 1

    def play(self, changes):
        """Carry out one move: only the pieces it concerns are touched."""
        for key in changes.removed:
            self.bury(key)
        for key, square in changes.moved:
            self.put(key, self.plane.square_center(square))
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def tidy(self, removed=()):
        """Put every piece upright on the square the book says, and remove ``removed``."""
        for key in removed:
            self.bury(key)
        for square, key in self.book.at.items():
            self.put(key, self.plane.square_center(square))
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def truth(self):
        """Squares that really hold an upright piece, from MuJoCo (for evaluation only)."""
        squares = set()
        for key in STARTING_SQUARES:
            body = self.sim.data.body(f"object_{key}")
            if abs(body.xpos[2] - self.plane.surface) < 0.004 and body.xmat[8] > 0.9:
                name = self.plane.square_name(body.xpos)
                if name is not None:
                    squares.add(name)
        return frozenset(squares)


class Recorder:
    """Saves an annotated camera image per ply and, optionally, a video of the whole game."""

    def __init__(self, directory, sim, camera, video):
        self.directory, self.sim, self.camera = directory, sim, camera
        self.caption = ""
        directory.mkdir(parents=True, exist_ok=True)
        self.writer = None
        if video:
            self.writer = cv2.VideoWriter(str(directory / "game.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                          FPS, (WIDTH, HEIGHT))
            if not self.writer.isOpened():
                print("could not open the video writer; continuing without video", flush=True)
                self.writer = None

    @staticmethod
    def label(image, text):
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (28, 26, 24), -1)
        cv2.putText(image, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (226, 232, 231), 1, cv2.LINE_AA)

    def motion(self, phase):
        """Called by the arm once per video frame while it moves."""
        if self.writer is None:
            return
        rgb, _ = self.sim.render_camera(self.camera)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.label(image, f"{self.caption}   |   {phase}")
        self.writer.write(image)

    def snapshot(self, name, bgr, text):
        image = bgr.copy()
        self.label(image, text)
        cv2.imwrite(str(self.directory / f"{name}.png"), image)
        if self.writer is not None:
            for _ in range(round(0.7 * FPS)):
                self.writer.write(image)

    def close(self):
        if self.writer is not None:
            self.writer.release()


def run_plan(arm, plan, record):
    """Execute a pick-and-place plan: the sequence of pick_place_demo, with the times of the
    long motions scaled to their length so that far squares are not reached faster."""
    reach = float(np.max(np.abs(np.asarray(plan.approach) - arm.parked)))
    travel = max(1.55, reach / 0.9)
    carry = max(0.90, 0.09 * (len(plan.transfer) - 1))
    back = max(0.50, 0.05 * (len(plan.transfer) - 1))
    arm.play([arm.parked, plan.approach], travel, OPEN_GRIP, record, "Move above the piece")
    arm.play(plan.descent, 1.05, OPEN_GRIP, record, "Lower gripper")
    arm.hold(0.24, OPEN_GRIP, record, "Settle above piece")
    arm.hold(0.52, CLOSED_GRIP, record, "Close gripper")
    arm.play(plan.descent[::-1], 1.0, CLOSED_GRIP, record, "Lift piece")
    arm.hold(0.12, CLOSED_GRIP, record, "Lift piece")
    arm.play(plan.transfer, carry, CLOSED_GRIP, record, "Carry")
    arm.play(plan.placement, 1.02, CLOSED_GRIP, record, "Lower piece")
    arm.hold(0.36, OPEN_GRIP, record, "Open gripper")
    arm.play(plan.placement[::-1], 0.70, OPEN_GRIP, record, "Retract")
    arm.play(plan.transfer[::-1], back, OPEN_GRIP, record, "Return")
    arm.play([plan.approach, arm.parked], travel, OPEN_GRIP, record, "Return to scan pose")
    arm.hold(0.25, OPEN_GRIP, record, "Settle")


def difference(expected, seen):
    missing, extra = sorted(expected - seen), sorted(seen - expected)
    parts = []
    if missing:
        parts.append("no piece seen on " + " ".join(missing))
    if extra:
        parts.append("unexpected piece on " + " ".join(extra))
    return "; ".join(parts) or "no difference"


class Match:
    def __init__(self, args, sim, engine):
        self.args, self.sim, self.engine = args, sim, engine
        self.plane = BoardPlane.from_model(sim.model)
        self.detector = PieceDetector(self.plane)
        self.arm = Arm(sim)
        self.camera = Camera.initial(sim.view_center)
        self.book = PieceBook()
        self.hand = Hand(sim, self.plane, self.book)
        self.recorder = Recorder(Path(args.out).resolve(), sim, self.camera, args.video)
        self.game = GameState(robot_color=chess.WHITE if args.robot == "white" else chess.BLACK)
        self.human = HumanPlayer(seed=args.seed, script=args.script.split(",") if args.script else ())
        self.slots = discard_slots()
        self.discarded = 0
        self.report = Report()

    # ------------------------------------------------------------------ camera
    def scan(self):
        """One camera frame through the perception: (frame, {square: detection}) or (frame, None)."""
        frame = self.sim.capture(self.camera, self.detector)
        pieces = frame.pieces
        if pieces is None or pieces["state"] != "ready":
            return frame, None
        return frame, {detection["square"]: detection for detection in pieces["detections"]}

    def look_for(self, expected):
        """Scan, and once more after a short wait if the camera does not see ``expected``."""
        frame, found = self.scan()
        if found is None or frozenset(found) != expected:
            self.arm.hold(0.3, OPEN_GRIP, quiet, "")
            frame, found = self.scan()
        return frame, found

    # ------------------------------------------------------------------- turns
    def human_turn(self, ply):
        board = self.game.board
        move = self.human.choose(board)
        self.recorder.caption = f"ply {ply}  human  {move.uci()}"
        expected = occupancy_after(board, move)
        self.hand.play(self.book.apply(board, move))
        self.arm.hold(0.4, OPEN_GRIP, self.recorder.motion, "Opponent moves")

        frame, found = self.look_for(expected)
        if found is None:
            outcome, detail = "not recognised", "the camera could not locate the board"
        else:
            result = infer_from_occupancy(board, found)
            if result.kind == "move" and result.move == move:
                outcome, detail = "ok", ""
            elif result.kind == "ambiguous" and move in result.moves:
                outcome = "ambiguous"
                detail = "occupied squares fit " + " or ".join(m.uci() for m in result.moves)
            else:
                outcome = "not recognised"
                detail = f"inference said {result.kind}; " + difference(expected, frozenset(found))
        self.game.apply(move)
        self.finish(ply, "human", move, outcome, detail, frame)

    def robot_turn(self, ply):
        board = self.game.board
        doable = []
        for candidate in board.legal_moves:
            try:
                move_to_actions(board, candidate)
                doable.append(candidate)
            except UnsupportedMove:
                pass
        if not doable:
            raise SystemExit("the robot has only moves the arm cannot execute (promotions); stopping")
        restricted = len(doable) < board.legal_moves.count()
        move = self.engine.reply(self.game, root_moves=doable if restricted else None)
        if move is None:
            raise SystemExit("the engine returned no move; stopping")
        actions = move_to_actions(board, move)
        self.recorder.caption = f"ply {ply}  robot  {move.uci()}"
        expected = occupancy_after(board, move)

        problem = None
        for action in actions:
            problem = self.pick_and_place(action)
            if problem is not None:
                break
        frame = None
        if problem is None:
            frame, found = self.look_for(expected)
            if found is None:
                problem = "the camera could not locate the board after the move"
            elif frozenset(found) != expected:
                problem = "the camera does not confirm the move: " + difference(expected, frozenset(found))

        changes = self.book.apply(board, move)
        self.game.apply(move)
        if problem is None:
            self.finish(ply, "robot", move, "ok", "", frame)
            return
        really_done = self.hand.truth() == expected
        problem += " (the pieces really were in place)" if really_done else " (the pieces really were not in place)"
        self.hand.tidy(changes.removed)             # the operator finishes the move by hand
        self.arm.hold(0.4, OPEN_GRIP, quiet, "")
        frame, _ = self.scan()
        self.finish(ply, "robot", move, "operator", problem, frame)

    def pick_and_place(self, action):
        """One arm action, planned from a fresh camera frame. Returns None, or what went wrong."""
        frame, found = self.scan()
        if found is None or frame.base_to_board.estimate is None:
            return "the camera could not locate the board and the arm"
        if action.pick not in found:
            return f"the camera sees no piece on {action.pick}"
        if action.place == DISCARD:
            if self.discarded >= len(self.slots):
                return "no free place left for captured pieces"
            x, y = self.slots[self.discarded]
            destination = np.array([x, y, 0.0])     # on the table beside the board
        elif action.place in found:
            return f"the camera sees a piece on {action.place}, which should be empty"
        else:
            destination = self.plane.square_center(action.place)
        pickup = np.array(found[action.pick]["xyz_m"])
        try:
            plan = plan_pick_place(self.arm, pickup, destination, frame.base_to_board.estimate)
        except RuntimeError as error:
            return f"no arm path for {action.pick} -> {action.place}: {error}"
        print(f"         arm: {action.pick} -> {action.place}", flush=True)
        run_plan(self.arm, plan, self.recorder.motion)
        if action.place == DISCARD:
            self.discarded += 1
        return None

    def finish(self, ply, who, move, outcome, detail, frame):
        truth_ok = self.hand.truth() == squares_of(self.game.board)
        self.report.add(ply=ply, who=who, move=move.uci(), outcome=outcome, detail=detail,
                        pieces_match_game=truth_ok, fen=self.game.fen())
        note = "" if outcome == "ok" else f"  <- {outcome}: {detail}"
        if not truth_ok:
            note += "  [simulation: the pieces do not match the game]"
        print(f"ply {ply:>2}  {who:5}  {move.uci():6}{note}", flush=True)
        self.recorder.snapshot(f"{ply:02d}_{who}_{move.uci()}", frame.bgr, f"ply {ply}  {who}  {move.uci()}  {outcome}")

    # -------------------------------------------------------------------- game
    def run(self):
        self.arm.hold(0.8, OPEN_GRIP, quiet, "")
        frame, found = self.scan()
        start = squares_of(self.game.board)
        if found is None or frozenset(found) != start:
            seen = "nothing" if found is None else difference(start, frozenset(found))
            raise SystemExit(f"the camera does not see the starting position ({seen}); not starting")
        self.recorder.snapshot("00_start", frame.bgr, "start: 32 pieces seen on their squares")
        print(f"start: the camera sees all 32 pieces; robot plays {self.args.robot}", flush=True)

        for ply in range(1, self.args.plies + 1):
            if self.game.is_game_over():
                break
            if self.game.human_to_move:
                self.human_turn(ply)
            else:
                self.robot_turn(ply)
        self.recorder.close()

        status = self.game.status()
        lines = self.report.lines()
        lines.append(f"final position: {status['fen']}")
        if status["game_over"]:
            lines.append(f"game over: {status['result']} ({status['termination']})")
        print("\n" + "\n".join(lines), flush=True)
        summary = {"arguments": vars(self.args), "plies": self.report.plies, "status": status, "summary": lines}
        (self.recorder.directory / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"images and summary.json are in {self.recorder.directory}", flush=True)
        return self.report.clean()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plies", type=int, default=12, help="half-moves to play (default 12)")
    parser.add_argument("--robot", choices=("black", "white"), default="black", help="the robot's colour")
    parser.add_argument("--seed", type=int, default=7, help="seed of the simulated opponent")
    parser.add_argument("--script", default="", help="opponent moves to play first, e.g. e2e4,d2d4")
    parser.add_argument("--engine", default="stockfish", help="UCI engine for the robot")
    parser.add_argument("--skill", type=int, default=5, help="Stockfish skill level 0-20")
    parser.add_argument("--think-time", type=float, default=0.3, help="engine seconds per move")
    parser.add_argument("--out", default="game_run", help="directory for images and summary.json")
    parser.add_argument("--video", action="store_true", help="also record game.mp4")
    args = parser.parse_args()

    try:
        engine = Engine(args.engine, think_time=args.think_time, skill_level=args.skill)
    except (OSError, chess.engine.EngineError) as error:
        raise SystemExit(f"could not start the chess engine '{args.engine}': {error}\n"
                         "install Stockfish (macOS: brew install stockfish) or pass --engine /path/to/it")
    with engine, Simulation.from_files(movable_pieces=True) as sim:
        clean = Match(args, sim, engine).run()
    raise SystemExit(0 if clean else 1)


if __name__ == "__main__":
    main()
