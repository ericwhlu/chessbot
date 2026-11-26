"""Which squares can the arm reach? Asks MoveIt for every square; the arm does not move.

    ros2 run chess_brain reach_check --ros-args \\
        -p board_x:=0.60 -p board_y:=-0.20 -p board_yaw_deg:=90 -p surface_z:=-0.056

Run it (with the arm's bring-up and MoveIt running, in Gazebo or on the real arm) before
deciding where the board goes: it uses the same board parameters as arm_adapter_node and the
same heights as kinova_executor, and prints a map of the board:

    #  reachable at the carrying height and at the gripping height
    ^  only at the carrying height (the arm cannot get down to the piece)
    .  not reachable

"Reachable" means MoveIt accepts joint angles for the top-down gripper pose: inside the joint
limits, touching neither the table nor the arm. It is a necessary check, not a guarantee that
every straight descent can be planned. The exit code is 0 only if everything is fully reachable.

A second line says how many squares have joint angles at all (gen3_lite_kinematics, no collision
check, no ROS): where the two disagree, the square is within the arm's length but something is
in the way, or the two models differ.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from typing import Callable, Dict, List, Tuple

from .board_frame import FILES, RANKS, Locations
from .gen3_lite_kinematics import top_down_solutions

Reach = Tuple[bool, bool]  # (at the carrying height, at the gripping height)


def reach_table(
    locations: Locations,
    travel_z: Callable[[float], float],
    grip_z: Callable[[float], float],
    can_reach: Callable[[float, float, float], bool],
    discard_slots: int = 8,
) -> Dict[str, Reach]:
    """Check every square and off-board location. ``travel_z`` / ``grip_z`` map the surface
    height to the end-effector height (kinova_executor.Settings provides both)."""
    points = {f + r: locations.frame.world_xyz(f + r) for r in RANKS for f in FILES}
    for index in range(discard_slots):
        points[f"DISCARD[{index}]"] = locations.discard_slot_xyz(index)
    for name in sorted(locations.spares):
        points[name] = locations.pick_xyz(name)

    table = {}
    for name, (x, y, z) in points.items():
        high = bool(can_reach(x, y, travel_z(z)))
        low = bool(can_reach(x, y, grip_z(z)))
        table[name] = (high, low)
    return table


def _mark(reach: Reach) -> str:
    high, low = reach
    return "#" if high and low else "^" if high else "."


def format_reach(table: Dict[str, Reach]) -> List[str]:
    """The board as text (rank 8 at the top, a-file on the left), then the other locations."""
    lines = []
    for rank in reversed(RANKS):
        lines.append(f"  {rank}  " + " ".join(_mark(table[f + rank]) for f in FILES))
    lines.append("     " + " ".join(FILES))
    squares = [f + r for r in RANKS for f in FILES]
    good = [name for name in squares if table[name] == (True, True)]
    bad = [name for name in squares if table[name] != (True, True)]
    lines.append(f"  squares fully reachable: {len(good)} of {len(squares)}")
    if bad:
        lines.append("  not fully reachable: " + " ".join(bad))
    others = [name for name in table if name not in squares]
    if others:
        lines.append("  off-board: " + "  ".join(f"{name} {_mark(table[name])}" for name in others))
    return lines


def kinematic_reach(x: float, y: float, z: float) -> bool:
    """Do joint angles exist for the top-down pose, by this package's own kinematics?"""
    return bool(top_down_solutions(x, y, z))


def compare_with_kinematics(table: Dict[str, Reach], kinematic: Dict[str, Reach]) -> List[str]:
    """How MoveIt's answer relates to the arm's geometry alone (same keys in both tables)."""
    squares = [f + r for r in RANKS for f in FILES]
    full = (True, True)
    lines = [
        "  by joint angles alone (no collision check): "
        f"{sum(kinematic[name] == full for name in squares)} of {len(squares)} squares"
    ]
    # compared height by height: (carrying, gripping)
    blocked = [n for n in table if any(k and not m for m, k in zip(table[n], kinematic[n]))]
    extra = [n for n in table if any(m and not k for m, k in zip(table[n], kinematic[n]))]
    if blocked:
        lines.append("  joint angles exist but MoveIt refuses them: " + " ".join(blocked))
    if extra:
        lines.append("  MoveIt accepts but the kinematics here found nothing: " + " ".join(extra))
    if not blocked and not extra:
        lines.append("  MoveIt and the kinematics agree on every location, at both heights")
    return lines


def main(args=None) -> None:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    from .board_params import declare_board_parameters, locations_from_parameters
    from .kinova_arm import KinovaArm, settings_from_parameters
    from .kinova_executor import _spin

    code = 1  # 0 only if every square and off-board spot is fully reachable
    try:
        rclpy.init(args=args)
        node = KinovaArm("chess_reach_check")
        declare_board_parameters(node)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        threading.Thread(target=_spin, args=(executor,), daemon=True).start()

        settings = settings_from_parameters(node)
        locations = locations_from_parameters(node)
        if node.wait_until_ready():
            node.add_table()
            a1, h8 = locations.frame.world_xyz("a1"), locations.frame.world_xyz("h8")
            node.get_logger().info(
                f"checking the board: a1 at ({a1[0]:.3f}, {a1[1]:.3f}), h8 at ({h8[0]:.3f}, {h8[1]:.3f}), "
                f"surface z {a1[2]:.4f}; carrying at z {settings.travel_z(a1[2]):.3f}, "
                f"gripping at z {settings.grip_z(a1[2]):.3f} (end_effector_link in base_link); "
                f"asking {node.reach_method()}. This takes several minutes."
            )
            total = 2 * (len(FILES) * len(RANKS) + 8 + len(locations.spares))
            checked = [0]

            def can_reach(x: float, y: float, z: float) -> bool:
                answer = node.can_reach(x, y, z)
                checked[0] += 1
                if checked[0] % 16 == 0 or checked[0] == total:
                    print(f"  ... {checked[0]} of {total} poses checked", flush=True)
                return answer

            table = reach_table(locations, settings.travel_z, settings.grip_z, can_reach)
            kinematic = reach_table(locations, settings.travel_z, settings.grip_z, kinematic_reach)
            print("\n".join(format_reach(table) + compare_with_kinematics(table, kinematic)), flush=True)
            if not any(high or low for high, low in table.values()):
                print("  NOTHING is reachable: is MoveIt answering? Check its terminal.", flush=True)
            code = 0 if all(reach == (True, True) for reach in table.values()) else 2
    except KeyboardInterrupt:
        pass
    except BaseException:
        traceback.print_exc()
    finally:
        try:
            rclpy.try_shutdown()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)  # do not wait for the MoveIt clients' threads


if __name__ == "__main__":
    main()
