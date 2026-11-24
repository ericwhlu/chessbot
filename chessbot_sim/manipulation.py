"""Kinematic planning and actuator-driven manipulation in MuJoCo."""
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares

from .vision import inverse_pose, rotation_about


FPS = 24
HIGH_Z = 0.19
GRIP_HEIGHT = 0.045
OPEN_GRIP = 125
CLOSED_GRIP = 235


@dataclass
class PickPlan:
    approach: np.ndarray
    descent: np.ndarray
    transfer: np.ndarray
    placement: np.ndarray


class Arm:
    def __init__(self, sim):
        self.model, self.data = sim.model, sim.data
        self.ik = mujoco.MjData(self.model)
        joints = [self.model.joint(f"joint_{i}").id for i in range(1, 8)]
        self.qids = self.model.jnt_qposadr[joints]
        self.actuators = [self.model.actuator(f"joint_{i}").id for i in range(1, 8)]
        self.fingers = self.model.actuator("fingers_actuator").id
        ranges = self.model.jnt_range[joints]
        limited = self.model.jnt_limited[joints]
        self.low = np.where(limited, ranges[:, 0], -2 * np.pi)
        self.high = np.where(limited, ranges[:, 1], 2 * np.pi)
        self.parked = self.data.qpos[self.qids].copy()
        self.command = self.parked.copy()
        self.rng = np.random.default_rng(17)

    def solve(self, position, orientation, seed):
        def residual(q):
            loc, rot = self.forward_in_base(q)
            rotation = cv2.Rodrigues(rot @ orientation.T)[0].ravel()
            return np.r_[loc - position, 0.16 * rotation]

        result = least_squares(residual, np.clip(seed, self.low + 1e-6, self.high - 1e-6),
                               bounds=(self.low, self.high), max_nfev=100, gtol=1e-8)
        return result.x, np.linalg.norm(result.fun)

    def forward_in_base(self, q):
        self.ik.qpos[self.qids] = q
        mujoco.mj_kinematics(self.model, self.ik)
        base = self.ik.body("base_link")
        tool = self.ik.site("pinch_site")
        world_to_base = base.xmat.reshape(3, 3).T
        return world_to_base @ (tool.xpos - base.xpos), world_to_base @ tool.xmat.reshape(3, 3)

    def safe_approaches(self, position, orientation, board_to_base):
        candidates = []
        for seed in [self.parked, *self.rng.uniform(self.low, self.high, size=(8, 7))]:
            q, error = self.solve(position, orientation, seed)
            if error > 0.001:
                continue
            for t in np.linspace(0, 1, 18):
                location, _ = self.forward_in_base(self.parked + t * (q - self.parked))
                height = (location - board_to_base[:3, 3]) @ board_to_base[:3, 2]
                if height < 0.105:
                    break
            else:
                candidates.append(q)
        return sorted(candidates, key=lambda q: np.linalg.norm(q - self.parked))

    def line(self, start, end, orientation, seed):
        # Cartesian waypoints keep approach and retreat vertical near the pieces.
        count = max(2, int(np.ceil(np.linalg.norm(end - start) / 0.010)))
        path = [seed]
        for position in np.linspace(start, end, count + 1)[1:]:
            q, error = self.solve(position, orientation, path[-1])
            if error > 0.001 or np.linalg.norm(q - path[-1]) > 0.55:
                raise RuntimeError(f"IK continuity failed at {position}, {error * 1000:.2f} mm residual")
            path.append(q)
        return np.asarray(path)

    def play(self, path, seconds, grip, record, label):
        frame_count = max(1, round(seconds * FPS))
        steps = round(1 / (FPS * self.model.opt.timestep))
        initial_grip = self.data.ctrl[self.fingers]
        for i in range(frame_count):
            for k in range(steps):
                t = (i * steps + k + 1) / (frame_count * steps)
                t = t * t * (3 - 2 * t)
                segment = t * (len(path) - 1)
                left = min(int(segment), len(path) - 2)
                u = segment - left
                q = path[left] * (1 - u) + path[left + 1] * u

                self.data.ctrl[self.actuators] = q
                self.data.ctrl[self.fingers] = initial_grip + t * (grip - initial_grip)
                self.command = q.copy()
                mujoco.mj_step(self.model, self.data)
            record(label)

    def hold(self, seconds, grip, record, label):
        self.play([self.command, self.command], seconds, grip, record, label)

    def jaw_forces(self, piece):
        """Measure normal contact forces from each jaw for simulation verification."""
        forces = {"left": 0.0, "right": 0.0}
        wrench = np.zeros(6)
        for i, contact in enumerate(self.data.contact):
            if contact.efc_address < 0:
                continue
            g1, g2 = contact.geom1, contact.geom2
            if self.model.geom_bodyid[g1] == piece.id:
                jaw = g2
            elif self.model.geom_bodyid[g2] == piece.id:
                jaw = g1
            else:
                continue
            name = self.model.geom(jaw).name
            for side in forces:
                if name.startswith(f"{side}_pad"):
                    mujoco.mj_contactForce(self.model, self.data, i, wrench)
                    forces[side] += float(wrench[0])
        return tuple(forces.values())


def plan_pick_place(arm, pickup_board_m, destination_board_m, base_to_board):
    """Plan a top-down pawn move from calibrated, metric board-frame inputs."""
    board_to_base = inverse_pose(base_to_board)
    rotation, shift = board_to_base[:3, :3], board_to_base[:3, 3]
    pickup = np.asarray(pickup_board_m) + [0, 0, GRIP_HEIGHT]
    release = np.asarray(destination_board_m) + [0, 0, GRIP_HEIGHT + 0.002]
    above_pickup = np.array([*pickup[:2], HIGH_Z])
    above_release = np.array([*release[:2], HIGH_Z])
    in_air, grab, above_dest, drop = [
        rotation @ point + shift for point in (above_pickup, pickup, above_release, release)
    ]
    for yaw in (40, 130, 220, 310):
        tool_down = rotation_about([0, 0, 1], np.deg2rad(yaw)) @ np.diag([1., -1., -1.])
        orientation = board_to_base[:3, :3] @ tool_down
        for first in arm.safe_approaches(in_air, orientation, board_to_base):
            try:
                descent = arm.line(in_air, grab, orientation, first)
                transfer = arm.line(in_air, above_dest, orientation, first)
                place = arm.line(above_dest, drop, orientation, transfer[-1])
            except RuntimeError:
                continue
            return PickPlan(first, descent, transfer, place)
    raise RuntimeError(f"No continuous pick-and-place path to {grab} and {drop}")
