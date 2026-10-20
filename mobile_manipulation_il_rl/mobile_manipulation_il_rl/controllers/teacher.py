from __future__ import annotations

import math

import mujoco
import numpy as np

from ..envs.fsm import Phase
from ..envs.mobile_pick_place import wrap_angle


class AnalyticTeacher:
    """Non-learning proportional base controller plus damped-least-squares IK."""

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg["teacher"]
        self._last = np.zeros(7, np.float32)
        self.last_ik = {"target": None, "residual": None, "condition": None, "limit_joints": []}

    def reset(self) -> None:
        self._last.fill(0)

    def _base_action(self) -> np.ndarray:
        base, yaw = self.env._base_pose()
        delta = self.env._nav_target() - base[:2]
        rho = float(np.linalg.norm(delta))
        bearing = math.atan2(delta[1], delta[0])
        alpha = wrap_angle(bearing - yaw)
        v = self.cfg["base_k_rho"] * rho * max(0.0, math.cos(alpha))
        omega = self.cfg["base_k_alpha"] * alpha
        if rho < self.env.cfg["task"]["position_tolerance"] * 1.5:
            v *= rho / (self.env.cfg["task"]["position_tolerance"] * 1.5)
        r = self.env.cfg["robot"]
        return np.array([v / r["base_max_linear_velocity"], omega / r["base_max_angular_velocity"]], np.float32)

    def _ik_action(self, target: np.ndarray) -> np.ndarray:
        m, d = self.env.model, self.env.data
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self.env.ee_site)
        dofs = np.array([m.jnt_dofadr[j] for j in self.env.arm_joints])
        j = jacp[:, dofs]
        error = np.asarray(target) - self.env._ee_pos()
        lam = self.cfg["ik_damping"]
        dq = j.T @ np.linalg.solve(j @ j.T + lam * lam * np.eye(3), self.cfg["ik_gain"] * error)
        # Null-space centering reduces limit encounters and singular postures.
        q = np.array([self.env._qpos(x) for x in self.env.arm_joints])
        centers = np.array([np.mean(m.jnt_range[x]) for x in self.env.arm_joints])
        dq += 0.05 * (centers - q)
        margins = np.array([min(q[i] - m.jnt_range[x, 0], m.jnt_range[x, 1] - q[i]) for i, x in enumerate(self.env.arm_joints)])
        self.last_ik = {
            "target": np.asarray(target).tolist(), "residual": float(np.linalg.norm(error)),
            "condition": float(np.linalg.cond(j @ j.T + lam * lam * np.eye(3))),
            "limit_joints": np.flatnonzero(margins < 0.05).astype(int).tolist(),
        }
        scale = self.env.cfg["robot"]["joint_delta_rad"] / self.env.dt
        return np.clip(dq / scale, -1, 1).astype(np.float32)

    def act(self, observation=None) -> np.ndarray:
        del observation
        phase = self.env.fsm.phase
        action = np.zeros(7, np.float32)
        obj = self.env._object_pos()
        base = self.env._base_pose()[0]
        approach_xy = obj[:2] - base[:2]
        approach_xy /= max(np.linalg.norm(approach_xy), 1e-6)
        approach = np.r_[approach_xy, 0.0]
        table_z = self.env.cfg["task"]["table_size"][2]
        if phase in (Phase.NAVIGATE_TO_PICK, Phase.NAVIGATE_TO_PLACE):
            action[:2] = self._base_action()
            if phase == Phase.NAVIGATE_TO_PLACE:
                action[:2] *= 0.20
        elif phase == Phase.REACH:
            action[2:6] = self._ik_action(obj - 0.05 * approach + np.array([0, 0, 0.15]))
            action[6] = 1.0
        elif phase == Phase.GRASP:
            # Finger pads are centred 20 mm ahead of the EE site and 25 mm below it.
            action[2:6] = self._ik_action(obj - 0.020 * approach + np.array([0, 0, 0.025]))
            action[6] = -1.0
        elif phase == Phase.LIFT:
            action[2:6] = 0.20 * self._ik_action(self.env._lift_target)
            action[6] = -1.0
        elif phase == Phase.PLACE:
            goal = np.r_[self.env._episode_targets["place_xy"], table_z + 0.075]
            action[2:6] = self._ik_action(goal)
            horizontal = np.linalg.norm(obj[:2] - goal[:2])
            assisted_release = self.env.cfg["task"]["grasp_mode"] == "assisted" and self.env.fsm.phase_time > 1.0
            action[6] = 1.0 if assisted_release or (horizontal < 0.07 and obj[2] < table_z + 0.11) else -1.0
        # Low-pass transitions avoid command discontinuities at phase boundaries.
        result = np.clip(0.65 * self._last + 0.35 * action, -1, 1).astype(np.float32)
        self._last = result
        return result

