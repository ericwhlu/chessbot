#!/usr/bin/env bash
# Install chess_brain's two non-ROS dependencies inside the course container:
# python-chess and Stockfish. The container is started with --rm, so run this
# once per container start (until the team bakes them into an image).
#
#   bash /chessbot/ros2_ws/src/chess_brain/scripts/setup_deps.sh
#
# Note: in the course image `python3` on PATH is a virtualenv (/opt/ros_python),
# but ROS entry points (ros2 run / ros2 launch) start nodes with /usr/bin/python3.
# python-chess must be importable by THAT interpreter, so it is installed into
# the system interpreter's site-packages directory, whichever pip is available.
set -u

NODE_PY=/usr/bin/python3
[ -x "$NODE_PY" ] || NODE_PY="$(command -v python3)"

have_chess()     { "$NODE_PY" -c "import chess" 2>/dev/null; }
have_stockfish() { command -v stockfish >/dev/null 2>&1 || [ -x /usr/games/stockfish ]; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

export DEBIAN_FRONTEND=noninteractive
if ! have_stockfish; then
  echo "[setup_deps] installing stockfish with apt-get ..."
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq stockfish
fi

if ! have_chess; then
  # python-chess is NOT packaged for Ubuntu 24.04 (no python3-chess), so use pip.
  # It is pure Python, so any pip can install it for $NODE_PY via --target.
  target="$("$NODE_PY" -c 'import site; print(site.getsitepackages()[0])')"
  echo "[setup_deps] installing python-chess for $NODE_PY into $target ..."
  if python3 -m pip --version >/dev/null 2>&1; then
    $SUDO python3 -m pip install -q --upgrade --target "$target" "chess>=1.10,<2"
  elif "$NODE_PY" -m pip --version >/dev/null 2>&1; then
    $SUDO "$NODE_PY" -m pip install -q --break-system-packages "chess>=1.10,<2"
  else
    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq python3-pip
    $SUDO "$NODE_PY" -m pip install -q --break-system-packages "chess>=1.10,<2"
  fi
fi

if have_chess; then
  echo "[setup_deps] python-chess OK for $NODE_PY: $("$NODE_PY" -c 'import chess; print(chess.__version__, chess.__file__)')"
else
  echo "[setup_deps] python-chess MISSING for $NODE_PY"
fi
if have_stockfish; then
  echo "[setup_deps] stockfish OK: $(command -v stockfish || echo /usr/games/stockfish)"
else
  echo "[setup_deps] stockfish MISSING"
fi
have_chess && have_stockfish
