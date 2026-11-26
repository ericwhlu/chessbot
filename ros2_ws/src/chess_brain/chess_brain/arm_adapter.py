"""Turn an ``arm_actions`` message into calls of one pick-and-place function. Pure Python.

This is the arm team's plug-in point. chess_brain sends

    {"id": 7, "move": "d5e4", "actions": [{"pick": "e4", "place": "DISCARD"},
                                          {"pick": "d5", "place": "e4"}]}

and ``run_command`` resolves every name to world coordinates (board_frame.py), then calls
the *executor* once per action, in order, and builds the ``arm_done`` reply.

The executor is the ONE function the arm team writes:

    def pick_and_place(pick_xyz, place_xyz, info):
        '''Move the piece standing at pick_xyz to place_xyz and return when it is there.

        pick_xyz, place_xyz : (x, y, z) in metres, world frame; z is the board surface.
        info : {"id", "move", "index", "count", "last", "pick", "place"} - names and position
               of this action in the command. When info["last"] is True, finish by parking
               the arm out of the camera's view: the board is photographed right after.
               Optional: "pick_grip" and "place_grip" - the angle (rad, world frame, measured
               from +x) of the line a two-finger gripper should open along at the pick and at
               the place so that its fingers stay clear of the other pieces (a line has no
               sign: the angle is defined modulo pi), and "grip_clearance" (m), what the
               plan-view model says that choice leaves. Present only when the command says
               which squares are occupied; an executor may ignore them.

        Return None or True on success. Return False, return a string (the reason), or
        raise an exception to report failure; the remaining actions are then skipped.
        If the piece DID arrive but something failed afterwards (e.g. parking), return
        ``PlacedButFailed(reason)`` so that a discard slot it now occupies is not reused.
        '''

Two optional attributes on the function are used by arm_adapter_node if present:
``prepare()`` is called once at start-up, and ``operator(command)`` receives every operator
command from ``chessbot/cmd`` (``start`` / ``reset`` / ``retry``).

Rules ``run_command`` guarantees to the executor:

* every location in the command is valid before the first motion starts
  (an unknown name fails the whole command with nothing moved);
* a ``place`` square is empty when its action starts (chess_brain orders captures that way);
* captured pieces get consecutive discard slots; a slot is only used up when the
  action that fills it succeeded;
* if the command carries ``occupied`` (the squares holding a piece before its first action),
  every action gets a grip direction chosen by grip_direction.py for the pieces that are on the
  board while that action runs, plus the discard slots already filled.
"""

from __future__ import annotations

import math
import time
from typing import Callable, List, Optional, Tuple

from .board_frame import DISCARD, Locations, XYZ
from .grip_direction import Fingers, choose

Executor = Callable[[XYZ, XYZ, dict], object]


class PlacedButFailed(str):
    """A failure reason for an action whose piece nevertheless reached its place."""

    placed = True


def make_dry_run_executor(seconds: float = 1.0, log: Optional[Callable[[str], None]] = None) -> Executor:
    """A stand-in arm: logs what it would do, waits ``seconds`` per action, always succeeds."""

    def dry_run(pick_xyz: XYZ, place_xyz: XYZ, info: dict) -> None:
        if log is not None:
            log(
                f"dry run {info['index'] + 1}/{info['count']}: {info['pick']} "
                f"({pick_xyz[0]:.3f}, {pick_xyz[1]:.3f}, {pick_xyz[2]:.3f}) -> {info['place']} "
                f"({place_xyz[0]:.3f}, {place_xyz[1]:.3f}, {place_xyz[2]:.3f})"
            )
        if seconds > 0:
            time.sleep(seconds)

    return dry_run


def plan_command(payload: dict, locations: Locations) -> List[Tuple[str, XYZ, str, XYZ]]:
    """Resolve every action to ``(pick_name, pick_xyz, place_name, place_xyz)``.

    Raises ``ValueError`` if the payload is malformed or names an unknown location.
    Does not use up discard slots.
    """
    if not isinstance(payload, dict):
        raise ValueError("arm_actions must be a JSON object")
    actions = payload.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("arm_actions has no actions")
    plan = []
    discards = 0
    for index, action in enumerate(actions):
        try:
            pick_name, place_name = str(action["pick"]), str(action["place"])
        except (KeyError, TypeError):
            raise ValueError(f"action {index} needs 'pick' and 'place'") from None
        try:
            pick_xyz = locations.pick_xyz(pick_name)
            place_xyz = locations.place_xyz(place_name, discard_offset=discards)
        except ValueError as exc:
            raise ValueError(f"action {index}: {exc}") from None
        if place_name.strip().upper() == DISCARD:
            discards += 1
        plan.append((pick_name, pick_xyz, place_name, place_xyz))
    return plan


def plan_grips(
    payload: dict, locations: Locations, fingers: Optional[Fingers] = None
) -> Optional[List[Tuple[float, float, float]]]:
    """For every action: the world-frame angles (rad) of the lines the gripper should open along
    at the pick and at the place, and the plan-view clearance (m) that choice leaves.

    None if the command does not say which squares are occupied. Call after ``plan_command``
    has accepted the payload. Raises ``ValueError`` if ``occupied`` is not a list of squares.
    """
    listed = payload.get("occupied")
    if listed is None:
        return None
    if not isinstance(listed, (list, tuple)):
        raise ValueError("'occupied' must be a list of square names")
    frame = locations.frame
    occupied = set()
    for name in listed:
        key = str(name).strip().lower()
        try:
            frame.board_xy(key)
        except ValueError:
            raise ValueError(f"'occupied' has {name!r}, which is not a square name") from None
        occupied.add(key)

    fingers = fingers or Fingers()
    grips = []
    filled = 0   # discard slots filled by earlier actions of this command
    for action in payload["actions"]:
        pick_name, place_name = str(action["pick"]).strip(), str(action["place"]).strip()
        pick_xy = locations.board_xy(pick_name)
        place_xy = locations.board_xy(place_name, discard_offset=filled)
        occupied.discard(pick_name.lower())   # the piece in the gripper is not in its own way
        others = [frame.board_xy(name) for name in sorted(occupied)]
        others += [locations.discard_slot_board_xy(i) for i in range(locations.discarded + filled)]
        at_pick, at_place, clear = choose(pick_xy, others, place_xy, others, fingers)
        grips.append((math.radians(at_pick) + frame.yaw, math.radians(at_place) + frame.yaw, clear))
        if place_name.upper() == DISCARD:
            filled += 1
        elif place_name.upper() not in locations.spares:
            occupied.add(place_name.lower())
    return grips


def run_command(
    payload: dict, locations: Locations, executor: Executor, fingers: Optional[Fingers] = None, grips: bool = True
) -> dict:
    """Execute one ``arm_actions`` payload and return the ``arm_done`` reply.

    ``grips=False`` never passes grip directions to the executor; ``fingers`` sizes the model
    that chooses them (defaults in grip_direction.py).
    """
    done_id = payload.get("id") if isinstance(payload, dict) else None
    try:
        plan = plan_command(payload, locations)
        chosen = plan_grips(payload, locations, fingers) if grips else None
    except ValueError as exc:
        return {"id": done_id, "status": "failed", "failed_action_index": -1, "reason": f"rejected: {exc}"}

    count = len(plan)
    for index, (pick_name, pick_xyz, place_name, place_xyz) in enumerate(plan):
        info = {
            "id": done_id,
            "move": payload.get("move"),
            "index": index,
            "count": count,
            "last": index == count - 1,
            "pick": pick_name,
            "place": place_name,
        }
        if chosen is not None:
            info["pick_grip"], info["place_grip"], info["grip_clearance"] = chosen[index]
        try:
            result = executor(pick_xyz, place_xyz, info)
        except Exception as exc:  # the arm code failed: report it, never crash the node
            result = f"{type(exc).__name__}: {exc}"
        if result is None or result is True:
            if place_name.strip().upper() == DISCARD:
                locations.commit_discard()
            continue
        if getattr(result, "placed", False) and place_name.strip().upper() == DISCARD:
            locations.commit_discard()  # the slot is physically occupied even though the action failed
        reason = "executor returned False" if result is False else str(result)
        return {"id": done_id, "status": "failed", "failed_action_index": index, "reason": reason}
    return {"id": done_id, "status": "ok"}
