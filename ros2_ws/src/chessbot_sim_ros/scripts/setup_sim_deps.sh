#!/usr/bin/env bash
# Install what the MuJoCo simulation (chessbot_sim) needs inside a ROS 2 Jazzy container:
# mujoco, ultralytics (YOLO) with CPU-only torch, opencv, scipy, and the EGL libraries for
# offscreen rendering. Run it once per container start, after chess_brain's setup_deps.sh:
#
#   bash /chessbot/ros2_ws/src/chess_brain/scripts/setup_deps.sh
#   bash /chessbot/ros2_ws/src/chessbot_sim_ros/scripts/setup_sim_deps.sh
#
# Everything is installed for /usr/bin/python3, the interpreter ROS entry points (ros2 run,
# ros2 launch) start nodes with; in the course image `python3` on PATH is a virtualenv.
# numpy and setuptools are kept at the versions already installed: the ROS message bindings are
# built against Ubuntu's numpy 1.26 (pip would pull numpy 2), and colcon needs setuptools < 80.
# torch comes from the CPU-only index: the default wheel brings gigabytes of CUDA libraries.
set -u

NODE_PY=/usr/bin/python3
[ -x "$NODE_PY" ] || NODE_PY="$(command -v python3)"
SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
export DEBIAN_FRONTEND=noninteractive

have_modules() { "$NODE_PY" -c "import mujoco, cv2, scipy, torch, ultralytics, chess" 2>/dev/null; }
have_egl()     { ldconfig -p 2>/dev/null | grep -q 'libEGL\.so\.1'; }

if ! have_egl || ! "$NODE_PY" -m pip --version >/dev/null 2>&1 || ! "$NODE_PY" -c "import numpy" 2>/dev/null; then
  echo "[setup_sim_deps] installing pip, numpy and the EGL / GL libraries with apt-get ..."
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq --no-install-recommends \
      python3-pip python3-numpy python3-setuptools libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libgles2 libglib2.0-0
fi

if ! have_modules; then
  constraints="$(mktemp)"
  "$NODE_PY" -c '
import numpy, setuptools
print("numpy==" + numpy.__version__)
print("setuptools==" + setuptools.__version__)' > "$constraints" 2>/dev/null
  echo "[setup_sim_deps] installing torch (CPU only) for $NODE_PY, keeping $(tr '\n' ' ' < "$constraints")..."
  $SUDO "$NODE_PY" -m pip install -q --no-cache-dir --break-system-packages -c "$constraints" \
      --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
      torch torchvision
  echo "[setup_sim_deps] installing mujoco, ultralytics, opencv, scipy ..."
  $SUDO "$NODE_PY" -m pip install -q --no-cache-dir --break-system-packages -c "$constraints" \
      "mujoco>=3.3,<4" "opencv-python>=4.10,<5" "scipy>=1.11,<2" "ultralytics>=8.4,<9" \
      "chess>=1.10,<2" "glfw>=2.7,<3" "PyOpenGL>=3.1,<4"
  rm -f "$constraints"
fi

if have_modules; then
  echo "[setup_sim_deps] python packages OK for $NODE_PY: $("$NODE_PY" -c '
import numpy, mujoco, cv2, torch, ultralytics
print("numpy", numpy.__version__, "mujoco", mujoco.__version__, "opencv", cv2.__version__,
      "torch", torch.__version__, "ultralytics", ultralytics.__version__)' 2>/dev/null | tail -1)"
else
  echo "[setup_sim_deps] python packages MISSING for $NODE_PY"
fi
if have_egl; then echo "[setup_sim_deps] EGL OK"; else echo "[setup_sim_deps] EGL MISSING"; fi
have_modules && have_egl
