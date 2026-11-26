"""The ROS 2 game without ROS 2: chess_brain's state machine, a WorldSession and the MuJoCo
world in one process, exchanging the same messages the topics would carry.

    cd ros2_ws/src/chessbot_sim_ros
    python -m chessbot_sim_ros.dry_run --plies 12            # needs mujoco etc. and Stockfish

Use it where there is no ROS 2 (a laptop) to check the order of events against the real
simulation. brain_node and mujoco_world_node do the same thing as two processes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import chess

from .paths import use_repo
from .session import SCAN_MODES, SessionConfig, WorldSession


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plies", type=int, default=12, help="half-moves to play; 0 = to the end")
    parser.add_argument("--robot", choices=("black", "white"), default="black")
    parser.add_argument("--seed", type=int, default=7, help="seed of the simulated opponent")
    parser.add_argument("--script", default="", help="opponent moves to play first, e.g. e2e4,d2d4")
    parser.add_argument("--scan-mode", choices=SCAN_MODES, default="continuous")
    parser.add_argument("--out", default="game_run_dry", help="directory for images and summary.json")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--engine", default="stockfish")
    parser.add_argument("--skill", type=int, default=5)
    parser.add_argument("--think-time", type=float, default=0.3)
    parser.add_argument("--stable-time", type=float, default=0.5)
    parser.add_argument("--repo-root", default="")
    args = parser.parse_args(argv)

    use_repo(args.repo_root)
    from chess_brain.game import Engine
    from chess_brain.machine import BrainMachine, Config
    from chessbot_sim.world import MujocoWorld

    robot = chess.WHITE if args.robot == "white" else chess.BLACK
    engine = Engine(args.engine, think_time=args.think_time, skill_level=args.skill)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        machine = BrainMachine(
            lambda game, root_moves: pool.submit(engine.reply, game, root_moves),
            Config(robot_color=robot, stable_time=args.stable_time, arm_timeout=300.0),
            log=lambda message: print(f"   [brain] {message}", flush=True),
        )
        with MujocoWorld(out=args.out, robot=args.robot, seed=args.seed, script=args.script, video=args.video) as world:
            session = WorldSession(
                world, SessionConfig(robot_color=robot, plies=args.plies, scan_mode=args.scan_mode),
                log=lambda message: print(f"   [world] {message}", flush=True),
            )
            while session.finished is None:
                for topic, payload in machine.tick(time.monotonic()):
                    if topic == "status":
                        session.on_status(json.dumps(payload))
                    elif topic == "arm_actions":
                        print(f"   [brain] arm_actions {json.dumps(payload)}", flush=True)
                        session.on_arm_actions(json.dumps(payload))
                    elif topic == "detect_request":
                        session.on_detect_request(payload)
                for topic, payload in session.tick():
                    if topic == "detected_occupancy":
                        machine.on_occupancy(json.dumps(payload), time.monotonic())
                    elif topic == "arm_done":
                        machine.on_arm_done(json.dumps(payload), time.monotonic())
                    elif topic == "cmd":
                        machine.on_cmd(payload)
                time.sleep(0.05)
            print("\n" + "\n".join(world.finish(session.finished, session.failed, session.board.fen())), flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        engine.close()
    return 0 if session.clean else 1


if __name__ == "__main__":
    sys.exit(main())
