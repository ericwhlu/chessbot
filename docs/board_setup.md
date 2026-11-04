# Chessbot Board

## Dimensions
- Playing area: 400 x 400 mm.
- Each square: 50 x 50 mm.
- Board origin: bottom corner of the board.
- +X: files a to h. +Y: ranks 1 to 8. +Z: up.
- a1 is dark; h1 is light.
- Square centre X: (file_index + 0.5) * 0.05 m.
- Square centre Y: (rank - 0.5) * 0.05 m.
- File index: a=0 through h=7.
- Collision surface height: 0.00656168 m.
- Visual tiles extend 0.1 mm above the collision surface.

## Gazebo
Use the course kinova-jazzy-latest Docker image.
Mount the repository at /chessbot and enable X11 display forwarding.
Inside the container, run:

    export GZ_SIM_RESOURCE_PATH="/chessbot/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
    gz sdf -k /chessbot/chessboard.sdf
    gz sim -r /chessbot/chessboard.sdf

The scene contains the ground and static chessboard.
Robot, camera and chess pieces are not included.
Robot reachability must be verified after board placement.

## Printing
File: Chessbot_400mm_Print.pdf in this directory.

- Page 1: full board, 440 x 440 mm including margins.
  Use A1 or custom stock.
- Page 2: printing and assembly instructions.
- Pages 3-8: six A4 portrait tiles with 20 mm overlaps.

Print at Actual size / 100%. Disable Fit and Shrink.
For A4, print pages 3-8 only, single-sided.
Follow the assembly map on page 2.
Overlap matching patterns; do not join trimmed edges edge-to-edge.

Check that each square measures 50 mm.
Check that the playing area measures 400 x 400 mm.
A 100 mm scale check is included.
Mount flat on rigid backing; matte paper is preferred.
Measure the physical board surface height before robot testing.

## Validation
The board-only SDF passed Gazebo format validation.
The checkerboard was visually inspected in Gazebo.
PDF page dimensions and layout were checked digitally.
Physical print dimensions and robot reachability remain to be verified.
