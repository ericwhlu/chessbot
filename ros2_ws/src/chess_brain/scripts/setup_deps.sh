#!/usr/bin/env bash
# Install chess_brain's two non-ROS dependencies inside the course container:
# python-chess and Stockfish. The container is started with --rm, so run this
# once per container start (until the team bakes them into an image).
#
#   bash /chessbot/ros2_ws/src/chess_brain/scripts/setup_deps.sh
set -u

have_chess()     { python3 -c "import chess" 2>/dev/null; }
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
  echo "[setup_deps] installing python-chess with pip ..."
  python3 -m pip --version >/dev/null 2>&1 || { $SUDO apt-get update -qq && $SUDO apt-get install -y -qq python3-pip; }
  $SUDO python3 -m pip install --break-system-packages "chess>=1.10,<2"
fi

if have_chess; then
  echo "[setup_deps] python-chess OK: $(python3 -c 'import chess; print(chess.__version__)')"
else
  echo "[setup_deps] python-chess MISSING"
fi
if have_stockfish; then
  echo "[setup_deps] stockfish OK: $(command -v stockfish || echo /usr/games/stockfish)"
else
  echo "[setup_deps] stockfish MISSING"
fi
have_chess && have_stockfish
