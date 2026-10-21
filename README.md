Chessbot — Gazebo Simulation

Chessbot is a chessboard simulation using Gazebo Harmonic and ROS 2 Jazzy.

1. Requirements

Ubuntu 24.04 (Linux VM)

Docker

Graphical desktop with X11 support (e.g., FastX)

Kinova Jazzy Docker image:

ghcr.io/mems-intro-to-robotics/mems-robotics-toolkit:kinova-jazzy-latest

Gazebo is already included in the Docker image. No separate Gazebo or ROS 2 installation is required.

2. Clone the Repository

Open a Linux terminal outside Docker:

cd ~
git clone https://github.com/ericwhlu/chessbot.git
cd chessbot

Repository structure:

chessbot/
├── README.md
├── chessboard.sdf
└── models/
    └── chessbot/
        └── meshes/
            └── chessboard_v1.stl

3. Launch Docker

Make sure you are connected to your Linux graphical desktop.

Run the following command in your Linux terminal:

docker run --rm -it \
  --hostname "$(hostname)" \
  -e DISPLAY="$DISPLAY" \
  -e XAUTHORITY=/tmp/.Xauthority \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$XAUTHORITY:/tmp/.Xauthority:ro" \
  -v "$HOME/chessbot:/chessbot" \
  --entrypoint /bin/bash \
  ghcr.io/mems-intro-to-robotics/mems-robotics-toolkit:kinova-jazzy-latest

This command:

Opens the Kinova Jazzy environment.

Enables graphical display forwarding from Docker to the Linux desktop.

Mounts the local chessbot repository at /chessbot inside Docker.

4. Launch Gazebo

Inside the Docker terminal, run:

gz sim /chessbot/chessboard.sdf

Gazebo should open and display the chessboard.

The current model is static and intended for visualization. Robot joints, actuators, and control are not yet implemented.

5. Updating the Repository

When changes are pushed to GitHub, update the local repository from the Linux host terminal:

cd ~/chessbot
git pull

Restart Gazebo to load the updated simulation.

6. Troubleshooting

Error: gz: command not found

Gazebo is installed inside Docker, not on the host. Make sure you have entered the Kinova container.

Error: qt.qpa.xcb: could not connect to display

Docker cannot access the graphical display. Exit Docker and verify that $DISPLAY and $XAUTHORITY are configured on the Linux host:

echo $DISPLAY
echo $XAUTHORITY

Then relaunch Docker using the command in Section 3.

Chessboard does not appear

Verify the model files exist inside Docker:

ls -R /chessbot

Check that both chessboard.sdf and chessboard_v1.stl are present.
