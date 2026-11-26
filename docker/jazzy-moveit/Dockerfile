# The simulation image plus MoveIt and the Kinova description: runs the Gen3 Lite arm code on a
# machine that cannot run the course image (Apple Silicon), with ros2_control mock hardware
# instead of Gazebo. See ros2_ws/src/chess_brain/README.md, "Without Gazebo".
#
#   docker build -f docker/jazzy-sim/Dockerfile -t chessbot-jazzy-sim ros2_ws/src
#   docker build -t chessbot-jazzy-moveit docker/jazzy-moveit
FROM chessbot-jazzy-sim
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ros-jazzy-moveit ros-jazzy-kortex-description ros-jazzy-kinova-gen3-lite-moveit-config \
      ros-jazzy-pymoveit2 ros-jazzy-gripper-controllers ros-jazzy-joint-trajectory-controller \
      ros-jazzy-joint-state-broadcaster ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
      ros-jazzy-controller-manager ros-jazzy-xacro ros-jazzy-robot-state-publisher \
 && rm -rf /var/lib/apt/lists/*
