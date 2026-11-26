"""The ROS parameters that say where the board is, shared by arm_adapter_node and reach_check.

``board_x`` / ``board_y`` / ``board_yaw_deg`` place the board's a1 corner in the world (for the
Kinova arm the world is ``base_link``); ``surface_z`` is the height of the playing surface there.
The discard column and the spare queens are given in the board frame (see board_frame.py);
``offboard_drop`` is how far below the playing surface they are (the board's thickness when
they are on the table beside the board).
"""

from __future__ import annotations

import math

from rcl_interfaces.msg import ParameterDescriptor

from .board_frame import BoardFrame, Locations

# name -> default
BOARD_PARAMETERS = {
    "board_x": 0.0,             # world position of the board's a1 corner (m)
    "board_y": 0.0,
    "board_yaw_deg": 0.0,       # rotation of the board about +Z
    "surface_z": 0.00656168,    # height of the playing surface (m)
    "square_size": 0.05,
    "discard_x": -0.075,        # first discard slot, in the board frame (m)
    "discard_y": 0.025,
    "spare_q_w_x": 0.475,       # spare queens, in the board frame (m)
    "spare_q_w_y": 0.025,
    "spare_q_b_x": 0.475,
    "spare_q_b_y": 0.375,
    "offboard_drop": 0.0,       # discard slots and spares are this far below the surface (m)
}


def declare_board_parameters(node) -> None:
    for name, default in BOARD_PARAMETERS.items():
        node.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))


def locations_from_parameters(node) -> Locations:
    """Build the board frame and the off-board locations from the node's parameters."""

    def number(name):
        return float(node.get_parameter(name).value)

    frame = BoardFrame(
        origin_x=number("board_x"),
        origin_y=number("board_y"),
        yaw=math.radians(number("board_yaw_deg")),
        surface_z=number("surface_z"),
        square=number("square_size"),
    )
    return Locations(
        frame=frame,
        discard_first=(number("discard_x"), number("discard_y")),
        spares={
            "SPARE_Q_W": (number("spare_q_w_x"), number("spare_q_w_y")),
            "SPARE_Q_B": (number("spare_q_b_x"), number("spare_q_b_y")),
        },
        offboard_drop=number("offboard_drop"),
    )
