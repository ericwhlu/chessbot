# ROS 2 Jazzy with what chess_brain and the MuJoCo simulation need: a mirror of the course
# container for machines where that image does not run natively (it is amd64 only).
# The repository is NOT copied in; mount it at /chessbot.
#
#   docker build -f docker/jazzy-sim/Dockerfile -t chessbot-jazzy-sim ros2_ws/src
#   docker run --rm -it -v "$PWD":/chessbot chessbot-jazzy-sim
#
# The two scripts are the ones a fresh course container runs by hand.
FROM ros:jazzy-ros-base
ENV DEBIAN_FRONTEND=noninteractive
COPY chess_brain/scripts/setup_deps.sh chessbot_sim_ros/scripts/setup_sim_deps.sh /tmp/setup/
RUN bash /tmp/setup/setup_deps.sh \
    && bash /tmp/setup/setup_sim_deps.sh \
    && apt-get install -y -qq --no-install-recommends python3-pytest \
    && rm -rf /var/lib/apt/lists/* /tmp/setup /root/.cache
WORKDIR /chessbot/ros2_ws
