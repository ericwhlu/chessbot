"""Find the repository's plain-Python packages from inside a ROS 2 package.

``chessbot_sim`` (the MuJoCo simulation) is not a ROS package and locates its assets next to
its own files, so it is imported from the repository itself, wherever that is mounted.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_repo_root(hint: str = "") -> Path:
    """The directory that holds ``chessbot_sim/``: ``hint`` (the node's ``repo_root``
    parameter), then $CHESSBOT_ROOT, then the directories above this file (a source checkout,
    or an install made with --symlink-install), then /chessbot (the mount point in the
    containers)."""
    here = Path(__file__).resolve()
    candidates = [hint, os.environ.get("CHESSBOT_ROOT", ""), *map(str, here.parents), "/chessbot"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "chessbot_sim" / "simulation.py").is_file():
            return Path(candidate).resolve()
    raise RuntimeError(
        "cannot find the chessbot repository (the directory with chessbot_sim/). Pass "
        "repo_root:=/path/to/chessbot or set CHESSBOT_ROOT."
    )


def use_repo(hint: str = "") -> Path:
    """Make ``chessbot_sim`` (and ``chess_brain``, if it is not installed) importable, and
    choose offscreen rendering on Linux, where MuJoCo would otherwise try to open a window."""
    root = find_repo_root(hint)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import chess_brain  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(root / "ros2_ws" / "src" / "chess_brain"))
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    return root
