# chessbot_sim_ros

`chess_brain` playing a whole game in the MuJoCo simulation (`chessbot_sim/`) over ROS 2.

```
brain_node  (chess_brain)                      mujoco_world_node  (this package)
  chess rules, Stockfish, turn logic             owns the MuJoCo scene: camera + perception,
                                                 arm, simulated opponent, simulated operator
        chessbot/arm_actions      ───────────▶
        chessbot/detect_request   ───────────▶
        chessbot/status (latched) ───────────▶
                                  ◀───────────   chessbot/detected_occupancy
                                  ◀───────────   chessbot/arm_done
                                  ◀───────────   chessbot/cmd   ("move <uci>")
```

These are the topics a real camera node and a real arm node use. On the real arm
`mujoco_world_node` is replaced by those two nodes; nothing in `chess_brain` changes.

| File | What it is |
|---|---|
| `chessbot_sim_ros/session.py` | `WorldSession`: who does what, when. Pure Python, tested against the real state machine. Read its docstring first. |
| `chessbot_sim_ros/mujoco_world_node.py` | the ROS 2 node: connects `WorldSession` to the topics and to `chessbot_sim/world.py` |
| `chessbot_sim_ros/dry_run.py` | the same game in one process, without ROS 2 |
| `launch/mujoco_game.launch.py` | `brain_node` + `mujoco_world_node` |
| `scripts/setup_sim_deps.sh` | Python packages and EGL libraries for a ROS 2 Jazzy container |

## Run it in the course container (VM)

Every new container needs the two set-up scripts once (they install for `/usr/bin/python3`,
the interpreter `ros2 run` uses).

```bash
bash /chessbot/ros2_ws/src/chess_brain/scripts/setup_deps.sh
bash /chessbot/ros2_ws/src/chessbot_sim_ros/scripts/setup_sim_deps.sh
cd /chessbot/ros2_ws
colcon build --symlink-install --packages-select chess_brain chessbot_sim_ros
source install/setup.bash
ros2 launch chessbot_sim_ros mujoco_game.launch.py plies:=12
```

If `colcon build` stops with `can't copy '.../build/chess_brain/launch/...'`, the build directory
was left by another branch: `rm -rf build/chess_brain install/chess_brain` and build again.

The launch ends when the game does. The last lines are the summary; one camera image per ply
and `summary.json` are in `out_dir` (default `game_run_ros/` in the directory you launched from).
To watch from a second terminal: `ros2 run chess_brain operator_node`.

Launch arguments (`ros2 launch chessbot_sim_ros mujoco_game.launch.py --show-args`):
`plies`, `robot_color` (`black` | `white`), `seed`, `script` (opponent moves to play first, e.g.
`e2e4,d2d4`), `scan_mode`, `out_dir`, `repo_root`, `video`, and chess_brain's `think_time`,
`skill_level`, `stable_time`, `arm_timeout`.

Rendering is offscreen through EGL (`MUJOCO_GL=egl`, set by the node if the variable is unset).

## The two ways the brain learns that the opponent has moved

- `scan_mode:=continuous` (default): the camera reports every 2 s. chess_brain accepts a move
  when the occupied squares have been unchanged for `stable_time` and fit exactly one legal move.
- `scan_mode:=on_request`: the camera reports only when asked on `chessbot/detect_request`, by
  chess_brain or by a button (`operator_node`: press Enter). In the simulation the opponent
  "presses the button" after each move. There is no physical button yet.

## On a machine without the course image (Apple Silicon)

The course image is amd64 only. `docker/jazzy-sim/Dockerfile` builds a native mirror from the
same two scripts:

```bash
docker build -f docker/jazzy-sim/Dockerfile -t chessbot-jazzy-sim ros2_ws/src
docker run --rm -it -v "$PWD":/chessbot chessbot-jazzy-sim
# inside: colcon build ... ; source install/setup.bash ; ros2 launch ...   (as above)
```

## Without ROS 2

```bash
cd ros2_ws/src/chessbot_sim_ros
python -m chessbot_sim_ros.dry_run --plies 12        # an environment with mujoco etc., and Stockfish
```

Same state machine, same `WorldSession`, same simulation, one process.

## What is simulated, and what the robot is allowed to know

The robot's side uses only the camera: scans go through the AprilTag + YOLO perception, and an
arm move is planned from the piece position the camera reports. MuJoCo's ground truth is used
only for the opponent's hand, for the operator who finishes a move the arm failed, and for the
`pieces_match_game` column of the summary.

The perception reports which squares hold a piece, not which piece. What that costs is written
in `chess_brain/occupancy.py` and in the docstring of `chess_brain/machine.py`.

## Tests

```bash
cd ros2_ws/src/chessbot_sim_ros && python -m pytest -q      # needs python-chess and pytest only
```
