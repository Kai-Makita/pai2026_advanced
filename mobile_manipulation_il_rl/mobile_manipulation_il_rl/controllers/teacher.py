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
        self.last_ik = {"target": None, "residual": None, "condition": None, "limit_joints": [],
                        "target_pitch": None, "pitch_error": None}
        self._phase = None
        self._grasp_start = None
        self._grasp_target = None
        self._close_started_at = None
        self._lift_start = None
        self._lift_approach = None
        self._retract_start = None

    def reset(self) -> None:
        self._last.fill(0)
        self._phase = None; self._grasp_start = None; self._grasp_target = None
        self._close_started_at = None; self._lift_start = None; self._lift_approach = None
        self._retract_start = None

    def _base_action(self) -> np.ndarray:
        base, yaw = self.env._base_pose()
        delta = self.env._nav_target() - base[:2]
        rho = float(np.linalg.norm(delta))
        bearing = math.atan2(delta[1], delta[0])
        alpha = wrap_angle(bearing - yaw)
        task = self.env.cfg["task"]
        active = np.asarray(self.env._episode_targets["active_xy"], float)
        final_heading_error = wrap_angle(math.atan2(active[1] - base[1], active[0] - base[0]) - yaw)
        if rho < task["position_tolerance"]:
            omega = 0.0 if abs(final_heading_error) < math.radians(task["yaw_tolerance_deg"]) else self.cfg["base_k_yaw"] * final_heading_error
            r = self.env.cfg["robot"]
            return np.clip(np.array([0.0, omega / r["base_max_angular_velocity"]], np.float32), -1, 1)
        v = self.cfg["base_k_rho"] * rho * max(0.0, math.cos(alpha))
        omega = self.cfg["base_k_alpha"] * alpha
        if rho < self.env.cfg["task"]["position_tolerance"] * 1.5:
            v *= rho / (self.env.cfg["task"]["position_tolerance"] * 1.5)
        r = self.env.cfg["robot"]
        return np.clip(np.array([v / r["base_max_linear_velocity"],
                                 omega / r["base_max_angular_velocity"]], np.float32), -1, 1)

    def _ik_action(self, target: np.ndarray, target_pitch: float | None = None, max_action: float = 1.0) -> np.ndarray:
        m, d = self.env.model, self.env.data
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self.env.ee_site)
        dofs = np.array([m.jnt_dofadr[j] for j in self.env.arm_joints])
        j = jacp[:, dofs]
        error = np.asarray(target) - self.env._ee_pos()
        q = np.array([self.env._qpos(x) for x in self.env.arm_joints])
        pitch_error = None
        if target_pitch is not None:
            pitch_error = wrap_angle(target_pitch - float(np.sum(q[1:])))
            j = np.vstack((j, np.array([0.0, 1.0, 1.0, 1.0])))
            error = np.r_[error, self.cfg["ik_pitch_weight"] * pitch_error]
        lam = self.cfg["ik_damping"]
        regularizer = lam * lam * np.eye(j.shape[0])
        dq = j.T @ np.linalg.solve(j @ j.T + regularizer, self.cfg["ik_gain"] * error)
        # Null-space centering reduces limit encounters and singular postures.
        # matrix dimension follows the position-only or position+pitch task
        centers = np.array([np.mean(m.jnt_range[x]) for x in self.env.arm_joints])
        dq += 0.05 * (centers - q)
        margins = np.array([min(q[i] - m.jnt_range[x, 0], m.jnt_range[x, 1] - q[i]) for i, x in enumerate(self.env.arm_joints)])
        self.last_ik = {
            "target": np.asarray(target).tolist(), "residual": float(np.linalg.norm(np.asarray(target) - self.env._ee_pos())),
            "condition": float(np.linalg.cond(j @ j.T + regularizer)),
            "limit_joints": np.flatnonzero(margins < 0.05).astype(int).tolist(),
            "target_pitch": target_pitch, "pitch_error": pitch_error,
        }
        if self.env.debug_target_mocap >= 0:
            self.env.data.mocap_pos[self.env.debug_target_mocap] = target
        scale = self.env.cfg["robot"]["joint_delta_rad"] / self.env.dt
        return np.clip(dq / scale, -max_action, max_action).astype(np.float32)

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def act(self, observation=None) -> np.ndarray:
        del observation
        phase = self.env.fsm.phase
        action = np.zeros(7, np.float32)
        obj = self.env._object_pos()
        base = self.env._base_pose()[0]
        approach_xy = obj[:2] - base[:2]
        approach_xy /= max(np.linalg.norm(approach_xy), 1e-6)
        approach = np.r_[approach_xy, 0.0]
        # Place the pad centre on the object centre.  The EE site is ahead of
        # the palm while the pad is lower, so both offsets must be rotated by
        # the selected (kinematically feasible) tool pitch.
        pitch = self.cfg["grasp_pitch_rad"]
        pad_local_x = self.env.cfg["robot"]["finger_pad_forward_offset"] - 0.045
        pad_local_z = self.env.cfg["robot"]["finger_pad_vertical_offset"]
        pad_forward = math.cos(pitch) * pad_local_x + math.sin(pitch) * pad_local_z
        pad_vertical = -math.sin(pitch) * pad_local_x + math.cos(pitch) * pad_local_z
        grasp_target = obj - pad_forward * approach - np.array([0.0, 0.0, pad_vertical])
        table_z = self.env.cfg["task"]["table_size"][2]
        if phase != self._phase:
            self._phase = phase
            # Incremental actions can leave a queued position-actuator target
            # behind the actual joints.  Anchor it at phase boundaries so the
            # first command follows the new Cartesian trajectory smoothly.
            if phase in (Phase.GRASP, Phase.LIFT, Phase.PLACE):
                self.env._joint_targets[:] = [self.env._qpos(j) for j in self.env.arm_joints]
            if phase == Phase.GRASP:
                self._grasp_start = self.env._ee_pos()
                self._grasp_target = grasp_target.copy()
                self._close_started_at = None
            elif phase == Phase.LIFT:
                self._lift_start = self.env._ee_pos()
                self._lift_approach = approach.copy()
                self._retract_start = None
        if phase in (Phase.NAVIGATE_TO_PICK, Phase.NAVIGATE_TO_PLACE):
            action[:2] = self._base_action()
            if phase == Phase.NAVIGATE_TO_PLACE:
                action[0] *= self.cfg["transport_linear_scale"]
                action[1] *= self.cfg["transport_angular_scale"]
        elif phase == Phase.REACH:
            # The high pre-grasp point lies near maximum reach. Constraining
            # pitch there over-constrains the four-DOF arm; acquire pitch while
            # descending through the GRASP approach instead.
            action[2:6] = self._ik_action(obj - 0.05 * approach + np.array([0, 0, 0.15]),
                                            None, 0.45)
            action[6] = 1.0
        elif phase == Phase.GRASP:
            duration = self.cfg["grasp_approach_seconds"]
            blend = self._smoothstep(self.env.fsm.phase_time / duration)
            target = (1 - blend) * self._grasp_start + blend * self._grasp_target
            action[2:6] = self._ik_action(target, self.cfg["grasp_pitch_rad"], 0.40)
            aligned = np.linalg.norm(self.env._ee_pos() - self._grasp_target) < self.cfg["ik_tolerance"]
            pitch_aligned = abs(self.last_ik["pitch_error"]) < self.cfg["grasp_pitch_tolerance_rad"]
            if blend >= 1.0 and aligned and pitch_aligned and self._close_started_at is None:
                self._close_started_at = self.env.fsm.phase_time
            if self._close_started_at is None:
                action[6] = 1.0
            else:
                close = self._smoothstep((self.env.fsm.phase_time - self._close_started_at) /
                                         self.cfg["gripper_close_seconds"])
                action[6] = 1.0 - 2.0 * close
        elif phase == Phase.LIFT:
            if self.env._hold_active:
                action[6] = -1.0
                self._last = action.copy()
                return action
            blend = self._smoothstep(self.env.fsm.phase_time / self.cfg["lift_seconds"])
            target = self._lift_start + np.array([0, 0, self.cfg["lift_height"] * blend])
            retract = self._smoothstep((self.env.fsm.phase_time - self.cfg["lift_seconds"]) /
                                       self.cfg["lift_retract_seconds"])
            target -= self.cfg["lift_retract_distance"] * retract * self._lift_approach
            # Once opposed contacts are established, prioritize the reachable
            # Cartesian lift. A simultaneous fixed-pitch constraint drives the
            # four-DOF arm away from the object near its workspace boundary.
            action[2:6] = self._ik_action(target, None, 0.35)
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

