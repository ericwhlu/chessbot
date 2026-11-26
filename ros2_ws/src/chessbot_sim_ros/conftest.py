# Makes `import chessbot_sim_ros` and `import chess_brain` work when running `pytest` from this
# directory, before the packages are installed by colcon.
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
for path in (HERE, HERE.parent / "chess_brain"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
