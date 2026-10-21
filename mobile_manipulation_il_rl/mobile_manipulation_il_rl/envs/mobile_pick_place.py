from __future__ import annotations

import math
import copy
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ..config import load_config
from .fsm import OBSERVABLE_PHASES, Phase, TaskFSM
from .model import build_mjcf, validate_geometry


def wrap_angle(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class MobilePickPlaceEnv(gym.Env):
    """40-observation, 7-action phase-masked mobile pick-and-place task."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: dict | str | Path | None = None, render_mode: str | None = None,
                 evaluation_mode: str = "train"):
        super().__init__()
        self.cfg = load_config(config) if not isinstance(config, dict) else config
        errors = validate_geometry(self.cfg)
        if errors:
            raise ValueError("; ".join(errors))
        self.render_mode, self.evaluation_mode = render_mode, evaluation_mode
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.cfg))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = self.cfg["simulation"]["physics_hz"] // self.cfg["simulation"]["control_hz"]
        self.dt = 1.0 / self.cfg["simulation"]["control_hz"]
        self.max_steps = int(self.cfg["simulation"]["episode_seconds"] / self.dt)
        self.action_space = spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (40,), np.float32)
        self.fsm = TaskFSM(self.cfg["task"]["phase_timeouts"])
        self._renderer = self._viewer = None
        self._step_count = 0
        self._last_action = np.zeros(7, np.float32)
        self._joint_targets = np.zeros(4)
        self._initial_state: dict[str, list[float]] = {}
        self._carrying = False
        self._lost_grasp_steps = 0
        self._grasp_contact_score = 0
        self._contact_age = np.full(2, 1000, dtype=np.int32)
        self._gripper_target = 0.035
        self._reward_sums: dict[str, float] = {}
        self._commanded_twist = (0.0, 0.0)
        self._base_acceleration = (0.0, 0.0)
        self._nav_stable_time = 0.0
        self._hold_active = False
        self._hold_stable_time = 0.0
        self._hold_start_time = 0.0
        self._hold_reference = np.zeros(3)
        self._previous_relative = np.zeros(3)
        self._first_object_contact: dict | None = None
        self._cache_ids()

    def _cache_ids(self) -> None:
        m = self.model
        self.base_joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
        self.obj_joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.ee_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.palm_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self.obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.arm_joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 5)]
        self.finger_joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"finger_{s}_slide") for s in ("left", "right")]
        self.left_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_left_geom")
        self.right_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_right_geom")
        self.obj_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.eq_assist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "assist_grasp")
        self.chassis_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "chassis")
        self.table_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
                            for name in ("pick_table_geom", "place_table_geom")]
        debug_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "debug_target")
        self.debug_target_mocap = int(m.body_mocapid[debug_body])

    def _qpos(self, joint_id: int) -> float:
        return float(self.data.qpos[self.model.jnt_qposadr[joint_id]])

    def _qvel(self, joint_id: int) -> float:
        return float(self.data.qvel[self.model.jnt_dofadr[joint_id]])

    def _base_pose(self) -> tuple[np.ndarray, float]:
        adr = self.model.jnt_qposadr[self.base_joint]
        return self.data.qpos[adr:adr + 3].copy(), yaw_from_quat(self.data.qpos[adr + 3:adr + 7])

    def _object_pos(self) -> np.ndarray:
        return self.data.xpos[self.obj_body].copy()

    def _ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site].copy()

    def _contacts(self) -> np.ndarray:
        result = np.zeros(2, dtype=np.float32)
        for i in range(self.data.ncon):
            pair = {self.data.contact[i].geom1, self.data.contact[i].geom2}
            if self.obj_geom in pair and self.left_geom in pair:
                result[0] = 1
            if self.obj_geom in pair and self.right_geom in pair:
                result[1] = 1
        return result

    def _gripper_aperture(self) -> float:
        r = self.cfg["robot"]
        return float(self._qpos(self.finger_joints[0]) + self._qpos(self.finger_joints[1]) +
                     2 * (r["finger_base_offset"] - r["finger_pad_half_width"]))

    def contact_diagnostics(self) -> list[dict]:
        """Raw finger-object contacts, including MuJoCo contact force."""
        rows = []
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = {con.geom1, con.geom2}
            side = "left" if self.left_geom in pair else "right" if self.right_geom in pair else None
            if side and self.obj_geom in pair:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                frame = np.asarray(con.frame).reshape(3, 3)
                world_force = force[:3] @ frame
                relative_velocity = self._contact_relative_velocity(con)
                normal = frame[0].copy()
                tangential_velocity = relative_velocity - normal * np.dot(relative_velocity, normal)
                mu = float(con.friction[0])
                tangential_force = float(np.linalg.norm(force[1:3]))
                rows.append({"side": side, "position": con.pos.tolist(), "distance": float(con.dist),
                             "normal_force": float(force[0]), "force6": force.tolist(),
                             "world_force": world_force.tolist(), "normal_world": normal.tolist(),
                             "relative_velocity_world": relative_velocity.tolist(),
                             "tangential_speed": float(np.linalg.norm(tangential_velocity)),
                             "friction_coefficient": mu, "tangential_force": tangential_force,
                             "friction_margin": float(mu * force[0] - tangential_force),
                             "friction_model": "elliptic_cone_coulomb_approximation"})
        return rows

    def _body_point_velocity(self, body_id: int, point: np.ndarray) -> np.ndarray:
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
        return velocity[3:6] + np.cross(velocity[:3], point - self.data.xpos[body_id])

    def _contact_relative_velocity(self, con) -> np.ndarray:
        body1 = int(self.model.geom_bodyid[con.geom1])
        body2 = int(self.model.geom_bodyid[con.geom2])
        return self._body_point_velocity(body2, con.pos) - self._body_point_velocity(body1, con.pos)

    def object_contact_diagnostics(self) -> list[dict]:
        rows = []
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if self.obj_geom not in {con.geom1, con.geom2}:
                continue
            other = con.geom2 if con.geom1 == self.obj_geom else con.geom1
            force = np.zeros(6); mujoco.mj_contactForce(self.model, self.data, i, force)
            body_id = int(self.model.geom_bodyid[other])
            frame = np.asarray(con.frame).reshape(3, 3)
            rows.append({"other_geom": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other),
                         "other_body": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                         "position": con.pos.tolist(), "distance": float(con.dist), "normal_force": float(force[0]),
                         "normal_world": frame[0].tolist(), "force6": force.tolist()})
        return rows

    def _base_table_clearance(self) -> float:
        base, yaw = self._base_pose()
        half = np.asarray(self.cfg["robot"]["chassis_size"][:2]) / 2
        projected = np.array([abs(math.cos(yaw)) * half[0] + abs(math.sin(yaw)) * half[1],
                              abs(math.sin(yaw)) * half[0] + abs(math.cos(yaw)) * half[1]])
        table_half = np.asarray(self.cfg["task"]["table_size"][:2]) / 2
        gaps = []
        for key in ("pick_table_xy", "place_table_xy"):
            separation = np.abs(base[:2] - np.asarray(self.cfg["task"][key])) - projected - table_half
            gaps.append(float(max(separation)))
        return min(gaps)

    def diagnostic_state(self, teacher=None) -> dict:
        q = np.array([self._qpos(j) for j in self.arm_joints])
        ranges = self.model.jnt_range[self.arm_joints]
        margins = np.minimum(q - ranges[:, 0], ranges[:, 1] - q)
        obj, ee = self._object_pos(), self._ee_pos()
        base, yaw = self._base_pose()
        nav_target = self._nav_target()
        desired_yaw = math.atan2(self._episode_targets["active_xy"][1] - base[1],
                                 self._episode_targets["active_xy"][0] - base[0])
        od = self.model.jnt_dofadr[self.obj_joint]
        contacts = self.contact_diagnostics()
        pad_center = 0.5 * (self.data.geom_xpos[self.left_geom] + self.data.geom_xpos[self.right_geom])
        return {
            "time_s": self._step_count * self.dt, "phase": self.fsm.phase.name,
            "phase_time_s": self.fsm.phase_time, "ee_position": ee.tolist(),
            "ee_target": None if teacher is None else teacher.last_ik["target"],
            "ee_error": None if teacher is None else teacher.last_ik["residual"],
            "ik_condition": None if teacher is None else teacher.last_ik["condition"],
            "ee_target_pitch": None if teacher is None else teacher.last_ik["target_pitch"],
            "ee_pitch_error": None if teacher is None else teacher.last_ik["pitch_error"],
            "joint_positions": q.tolist(), "joint_limit_margins": margins.tolist(),
            "joint_limit_reached": np.flatnonzero(margins < 0.05).astype(int).tolist(),
            "contacts": contacts, "contact_count": len(contacts),
            "object_contacts": self.object_contact_diagnostics(),
            "left_contact": any(x["side"] == "left" for x in contacts),
            "right_contact": any(x["side"] == "right" for x in contacts),
            "left_finger_position": self.data.geom_xpos[self.left_geom].tolist(),
            "right_finger_position": self.data.geom_xpos[self.right_geom].tolist(),
            "pad_center_error": (pad_center - obj).tolist(),
            "base_position": base.tolist(), "base_yaw": yaw,
            "base_linear_velocity": float(self._commanded_twist[0]),
            "base_angular_velocity": float(self._commanded_twist[1]),
            "base_linear_acceleration": float(self._base_acceleration[0]),
            "base_angular_acceleration": float(self._base_acceleration[1]),
            "navigation_position_error": float(np.linalg.norm(nav_target - base[:2])),
            "navigation_yaw_error": float(wrap_angle(desired_yaw - yaw)),
            "navigation_settle_time": self._nav_stable_time,
            "base_table_clearance": self._base_table_clearance(),
            "first_object_contact": copy.deepcopy(self._first_object_contact),
            "hold_active": self._hold_active, "hold_stable_time": self._hold_stable_time,
            "gripper_opening": self._gripper_aperture(),
            "gripper_target_opening": float(2 * self._gripper_target + 2 *
                (self.cfg["robot"]["finger_base_offset"] - self.cfg["robot"]["finger_pad_half_width"])),
            "gripper_command": float(self.data.ctrl[6]), "object_position": obj.tolist(),
            "object_velocity": self.data.qvel[od:od + 6].tolist(),
            "object_ee_relative": (obj - ee).tolist(), "failure_reason": self.fsm.failure_reason,
        }

    def _nav_target(self) -> np.ndarray:
        key = "pick_xy" if self.fsm.phase.value <= Phase.LIFT.value else "place_xy"
        target = np.asarray(self._episode_targets[key], dtype=float)
        # Stop on the table-facing side; heading points from base to target.
        table = np.asarray(self.cfg["task"][key], dtype=float)
        direction = table / max(np.linalg.norm(table), 1e-6)
        return target - direction * (self.cfg["task"]["stand_off"] +
                                     self.cfg["task"]["navigation_braking_margin"])

    def _observe(self) -> np.ndarray:
        base, yaw = self._base_pose()
        goal = self._nav_target()
        delta_world = goal - base[:2]
        c, s = math.cos(yaw), math.sin(yaw)
        rel_goal = np.array([c * delta_world[0] + s * delta_world[1], -s * delta_world[0] + c * delta_world[1]])
        desired_yaw = math.atan2(self._episode_targets["active_xy"][1] - base[1], self._episode_targets["active_xy"][0] - base[0])
        yaw_error = wrap_angle(desired_yaw - yaw)
        base_vel = self.data.qvel[self.model.jnt_dofadr[self.base_joint]:][:6]
        body_vel = np.array([c * base_vel[0] + s * base_vel[1], -s * base_vel[0] + c * base_vel[1], base_vel[5]])
        q = np.array([self._qpos(j) for j in self.arm_joints])
        qd = np.array([self._qvel(j) for j in self.arm_joints])
        finger = self._qpos(self.finger_joints[0])
        finger_vel = self._qvel(self.finger_joints[0])
        ee, obj = self._ee_pos(), self._object_pos()
        ee_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.ee_site, ee_vel, 0)
        obj_vel = self.data.cvel[self.obj_body, 3:6].copy()
        one_hot = np.zeros(6)
        if self.fsm.phase in OBSERVABLE_PHASES:
            one_hot[self.fsm.phase.value] = 1
        obs = np.concatenate((rel_goal, [math.sin(yaw_error), math.cos(yaw_error)], body_vel, q, qd,
                              [finger, finger_vel], ee, ee_vel[3:6], obj, obj_vel, obj - ee,
                              self._contacts(), one_hot)).astype(np.float32)
        if obs.shape != (40,) or not np.isfinite(obs).all():
            raise FloatingPointError("invalid observation")
        # Fixed physical scales keep the contract deterministic across datasets.
        scales = np.array([2, 2, 1, 1, 1, 2, 2] + [math.pi] * 4 + [5] * 4 + [0.04, 0.5]
                          + [2] * 3 + [5] * 3 + [2] * 3 + [5] * 3 + [2] * 3 + [1] * 8, dtype=np.float32)
        return np.clip(obs / scales, -10, 10).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.fsm.reset()
        self._step_count = 0
        self._last_action.fill(0)
        self._commanded_twist = (0.0, 0.0)
        self._base_acceleration = (0.0, 0.0)
        self._nav_stable_time = 0.0
        self._hold_active = False; self._hold_stable_time = 0.0; self._hold_start_time = 0.0
        self._hold_reference.fill(0); self._previous_relative.fill(0)
        self._first_object_contact = None
        self._reward_sums = {k: 0.0 for k in ("progress", "phase_completion", "task_success", "time", "smoothness", "failure")}
        mode = (options or {}).get("randomization", self.evaluation_mode)
        rnd = ({"base_xy": 0, "base_yaw_deg": 0, "object_xy": 0, "target_xy": 0}
               if mode == "fixed" else self.cfg["randomization"][mode])
        base_xy = self.np_random.uniform(-rnd["base_xy"], rnd["base_xy"], 2)
        yaw = math.radians(self.np_random.uniform(-rnd["base_yaw_deg"], rnd["base_yaw_deg"]))
        pick = np.asarray(self.cfg["task"]["pick_xy"]) + self.np_random.uniform(-rnd["object_xy"], rnd["object_xy"], 2)
        place = np.asarray(self.cfg["task"]["place_xy"]) + self.np_random.uniform(-rnd["target_xy"], rnd["target_xy"], 2)
        bq = self.model.jnt_qposadr[self.base_joint]
        self.data.qpos[bq:bq + 7] = [base_xy[0], base_xy[1], self.cfg["robot"]["wheel_radius"], math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
        oq = self.model.jnt_qposadr[self.obj_joint]
        self.data.qpos[oq:oq + 3] = [*pick, self.cfg["task"]["table_size"][2] + self.cfg["task"]["object_size"][2] + 0.002]
        self.data.qpos[oq + 3:oq + 7] = [1, 0, 0, 0]
        # Elbow-up hold pose: compact enough for navigation and outside the
        # downward local minimum of position-only IK.
        # Fold the forearm above the chassis for navigation. The previous pose
        # extended link3 into the object before the base had stopped.
        home = np.array([0, -1.00, -0.30, 0.80])
        for joint, value in zip(self.arm_joints, home):
            self.data.qpos[self.model.jnt_qposadr[joint]] = value
        self._joint_targets = home.copy()
        for joint in self.finger_joints:
            self.data.qpos[self.model.jnt_qposadr[joint]] = 0.035
        self._gripper_target = 0.035
        self.data.ctrl[2:6] = home
        self.data.ctrl[6:8] = self._gripper_target
        self._episode_targets = {"pick_xy": pick.tolist(), "place_xy": place.tolist(), "active_xy": pick.tolist()}
        self._initial_state = {"base_xy": base_xy.tolist(), "base_yaw": yaw, "object_xy": pick.tolist(), "target_xy": place.tolist()}
        self._carrying = False
        self._lost_grasp_steps = 0
        self._grasp_contact_score = 0
        self._contact_age[:] = 1000
        self.data.eq_active[self.eq_assist] = 0
        mujoco.mj_forward(self.model, self.data)
        self._previous_potential = self._potential()
        obs = self._observe()
        return obs, {"phase": self.fsm.phase.name, "initial_state": copy.deepcopy(self._initial_state)}

    def _mask_action(self, action: np.ndarray) -> np.ndarray:
        masked = np.clip(np.asarray(action, dtype=np.float32), -1, 1).copy()
        if self.fsm.phase in (Phase.NAVIGATE_TO_PICK, Phase.NAVIGATE_TO_PLACE):
            masked[2:6] = 0
            # Navigation holds the arm pose and explicitly preserves the
            # gripper state: open before pick, closed while carrying.
            masked[6] = -1.0 if self._carrying else 1.0
        else:
            masked[:2] = 0
        return masked

    def _apply_action(self, action: np.ndarray) -> None:
        r = self.cfg["robot"]
        requested_v = action[0] * r["base_max_linear_velocity"]
        requested_omega = action[1] * r["base_max_angular_velocity"]
        previous_v, previous_omega = self._commanded_twist
        v = previous_v + np.clip(requested_v - previous_v,
                                 -r["base_max_linear_acceleration"] * self.dt,
                                 r["base_max_linear_acceleration"] * self.dt)
        omega = previous_omega + np.clip(requested_omega - previous_omega,
                                         -r["base_max_angular_acceleration"] * self.dt,
                                         r["base_max_angular_acceleration"] * self.dt)
        self._base_acceleration = ((v - previous_v) / self.dt, (omega - previous_omega) / self.dt)
        self._commanded_twist = (float(v), float(omega))
        self.data.ctrl[0] = (v - omega * r["track_width"] / 2) / r["wheel_radius"]
        self.data.ctrl[1] = (v + omega * r["track_width"] / 2) / r["wheel_radius"]
        self._joint_targets += action[2:6] * r["joint_delta_rad"]
        for i, joint in enumerate(self.arm_joints):
            lo, hi = self.model.jnt_range[joint]
            self._joint_targets[i] = np.clip(self._joint_targets[i], lo, hi)
        self.data.ctrl[2:6] = self._joint_targets
        self._gripper_target = float(0.0175 * (action[6] + 1.0))
        self.data.ctrl[6:8] = self._gripper_target

    def _potential(self) -> float:
        if self.fsm.phase in (Phase.NAVIGATE_TO_PICK, Phase.NAVIGATE_TO_PLACE):
            return -float(np.linalg.norm(self._nav_target() - self._base_pose()[0][:2]))
        if self.fsm.phase in (Phase.REACH, Phase.GRASP):
            return -float(np.linalg.norm(self._object_pos() - self._ee_pos()))
        if self.fsm.phase == Phase.LIFT:
            return float(self._object_pos()[2])
        if self.fsm.phase == Phase.PLACE:
            target = np.r_[self._episode_targets["place_xy"], self.cfg["task"]["table_size"][2]]
            return -float(np.linalg.norm(target - self._object_pos()))
        return 0.0

    def _update_fsm(self) -> bool:
        completed = False
        base, yaw = self._base_pose()
        nav_dist = np.linalg.norm(self._nav_target() - base[:2])
        ee_dist = np.linalg.norm(self._object_pos() - self._ee_pos())
        contacts = self._contacts()
        self._contact_age += 1
        self._contact_age[contacts.astype(bool)] = 0
        persistent_contacts = self._contact_age <= self.cfg["teacher"]["contact_persistence_steps"]
        object_width = 2 * self.cfg["task"]["object_size"][1]
        closed = self._gripper_aperture() < object_width + 0.006
        table_z = self.cfg["task"]["table_size"][2]
        desired_yaw = math.atan2(self._episode_targets["active_xy"][1] - base[1],
                                 self._episode_targets["active_xy"][0] - base[0])
        yaw_aligned = abs(wrap_angle(desired_yaw - yaw)) < math.radians(self.cfg["task"]["yaw_tolerance_deg"])
        if self.fsm.phase in (Phase.NAVIGATE_TO_PICK, Phase.NAVIGATE_TO_PLACE):
            stopped = (abs(self._commanded_twist[0]) <= self.cfg["task"]["stop_linear_velocity"] and
                       abs(self._commanded_twist[1]) <= self.cfg["task"]["stop_angular_velocity"])
            if nav_dist < self.cfg["task"]["position_tolerance"] and yaw_aligned and stopped:
                self._nav_stable_time += self.dt
            else:
                self._nav_stable_time = 0.0
            if self._nav_stable_time >= self.cfg["task"]["navigation_settle_seconds"]:
                self._nav_stable_time = 0.0
                self.fsm.advance("navigation_stopped_and_settled"); completed = True
        elif self.fsm.phase == Phase.REACH:
            base = self._base_pose()[0]
            approach_xy = self._object_pos()[:2] - base[:2]
            approach_xy /= max(np.linalg.norm(approach_xy), 1e-6)
            pregrasp = self._object_pos() - 0.05 * np.r_[approach_xy, 0.0] + np.array([0.0, 0.0, 0.15])
            physical_reached = np.linalg.norm(self._ee_pos() - pregrasp) < self.cfg["teacher"]["ik_tolerance"]
            assisted_reached = self.cfg["task"]["grasp_mode"] == "assisted" and self.fsm.phase_time > 1.0
            if not (physical_reached or assisted_reached):
                return completed
            self.fsm.advance("pregrasp_reached"); completed = True
        elif self.fsm.phase == Phase.GRASP:
            # MuJoCo may alternate the active contact constraint between two
            # opposed faces at rest. Preserve recent per-side contact while
            # still requiring real contacts, closure and geometric alignment.
            stable = closed and persistent_contacts.all() and self._grasp_alignment_error() < self.cfg["teacher"]["ik_tolerance"]
            self._grasp_contact_score = max(0, self._grasp_contact_score + (1 if stable else -1))
            physical_ready = self._grasp_contact_score >= self.cfg["teacher"]["grasp_settle_steps"]
            assisted_ready = self.cfg["task"]["grasp_mode"] == "assisted" and self.fsm.phase_time > 0.6
            if not (physical_ready or assisted_ready):
                return completed
            self._carrying = True
            self._lost_grasp_steps = 0
            self._lift_target = self._object_pos() + np.array([0.0, 0.0, 0.18])
            self.fsm.advance("bilateral_contact"); completed = True
        elif self.fsm.phase == Phase.LIFT:
            assisted_ready = self.cfg["task"]["grasp_mode"] == "assisted" and self.fsm.phase_time > 1.0
            lift_ready = (self._object_pos()[2] > table_z + 0.13 and
                          self.fsm.phase_time >= self.cfg["teacher"]["lift_seconds"] +
                          self.cfg["teacher"]["lift_retract_seconds"])
            if (lift_ready or assisted_ready) and not self._hold_active:
                self._hold_active = True
                self._hold_start_time = self.fsm.phase_time
                self._hold_reference = self._object_pos() - self._ee_pos()
                self._previous_relative = self._hold_reference.copy()
                self._hold_stable_time = 0.0
                self._joint_targets[:] = [self._qpos(j) for j in self.arm_joints]
            if self._hold_active:
                relative = self._object_pos() - self._ee_pos()
                relative_speed = float(np.linalg.norm(relative - self._previous_relative) / self.dt)
                drift = float(np.linalg.norm(relative - self._hold_reference))
                current_bilateral = bool(contacts.all())
                stable = (current_bilateral and relative_speed <= self.cfg["teacher"]["hold_relative_speed"] and
                          drift <= self.cfg["teacher"]["hold_relative_drift"])
                self._hold_stable_time = self._hold_stable_time + self.dt if stable else 0.0
                self._previous_relative = relative.copy()
                if self._hold_stable_time >= self.cfg["teacher"]["hold_seconds"]:
                    self._hold_active = False
                    self._episode_targets["active_xy"] = self._episode_targets["place_xy"]
                    self.fsm.advance("physical_hold_stable"); completed = True
                elif self.fsm.phase_time - self._hold_start_time > self.cfg["teacher"]["hold_timeout_seconds"]:
                    reason = "contact" if not current_bilateral else "slip"
                    self.fsm.fail(f"timeout:HOLD:{reason}")
        elif self.fsm.phase == Phase.PLACE:
            d = np.linalg.norm(self._object_pos()[:2] - self._episode_targets["place_xy"])
            opened = self._qpos(self.finger_joints[0]) > 0.025
            assisted_done = self.cfg["task"]["grasp_mode"] == "assisted" and opened and self.fsm.phase_time > 1.5
            if (d < 0.08 and opened and self._object_pos()[2] < table_z + 0.10) or assisted_done:
                if assisted_done:
                    oq = self.model.jnt_qposadr[self.obj_joint]
                    self.data.qpos[oq:oq + 3] = [*self._episode_targets["place_xy"], table_z + self.cfg["task"]["object_size"][2]]
                    mujoco.mj_forward(self.model, self.data)
                self.data.eq_active[self.eq_assist] = 0
                self._carrying = False
                self.fsm.advance("object_placed"); completed = True
        if self._carrying and self.cfg["task"]["grasp_mode"] == "physical":
            self._lost_grasp_steps = 0 if persistent_contacts.all() else self._lost_grasp_steps + 1
            if self._object_pos()[2] < 0.12:
                self.fsm.fail("object_drop")
            elif self._lost_grasp_steps > 10 and np.linalg.norm(self._object_pos() - self._ee_pos()) > 0.10:
                self.fsm.fail("grasp_lost")
        return completed

    def _grasp_alignment_error(self) -> float:
        base = self._base_pose()[0]
        obj = self._object_pos()
        direction = obj[:2] - base[:2]
        direction /= max(np.linalg.norm(direction), 1e-6)
        target = obj - 0.035 * np.r_[direction, 0.0] + np.array([0.0, 0.0, 0.025])
        return float(np.linalg.norm(self._ee_pos() - target))

    def step(self, action):
        object_before = self._object_pos().copy()
        object_velocity_before = self.data.qvel[self.model.jnt_dofadr[self.obj_joint]:][:6].copy()
        action = self._mask_action(action)
        before_phase = self.fsm.phase
        self._apply_action(action)
        base_before, yaw_before = self._base_pose()
        commanded_xy = base_before[:2].copy()
        commanded_yaw = yaw_before
        physics_dt = 1.0 / self.cfg["simulation"]["physics_hz"]
        for _ in range(self.frame_skip):
            # The task specifies velocity control. Enforce the commanded planar
            # twist while MuJoCo still resolves vertical, wheel and collision dynamics.
            _, yaw = self._base_pose()
            dof = self.model.jnt_dofadr[self.base_joint]
            qadr = self.model.jnt_qposadr[self.base_joint]
            v, omega = self._commanded_twist
            commanded_xy += [v * math.cos(commanded_yaw) * physics_dt, v * math.sin(commanded_yaw) * physics_dt]
            commanded_yaw = wrap_angle(commanded_yaw + omega * physics_dt)
            self.data.qpos[qadr:qadr + 3] = [commanded_xy[0], commanded_xy[1], self.cfg["robot"]["wheel_radius"]]
            self.data.qpos[qadr + 3:qadr + 7] = [math.cos(commanded_yaw / 2), 0, 0, math.sin(commanded_yaw / 2)]
            self.data.qvel[dof:dof + 2] = [v * math.cos(commanded_yaw), v * math.sin(commanded_yaw)]
            self.data.qvel[dof + 3:dof + 5] = 0.0
            self.data.qvel[dof + 5] = omega
            mujoco.mj_step(self.model, self.data)
        # A mobile base is constrained to SE(2): remove unmodelled suspension
        # rocking and integrate the requested velocity exactly at control rate.
        qadr = self.model.jnt_qposadr[self.base_joint]
        dadr = self.model.jnt_dofadr[self.base_joint]
        self.data.qpos[qadr:qadr + 3] = [commanded_xy[0], commanded_xy[1], self.cfg["robot"]["wheel_radius"]]
        self.data.qpos[qadr + 3:qadr + 7] = [math.cos(commanded_yaw / 2), 0, 0, math.sin(commanded_yaw / 2)]
        self.data.qvel[dadr:dadr + 6] = [v * math.cos(commanded_yaw), v * math.sin(commanded_yaw), 0, 0, 0, omega]
        mujoco.mj_forward(self.model, self.data)
        if self._first_object_contact is None:
            ignored = {"pick_table", "place_table"}
            contact = next((row for row in self.object_contact_diagnostics()
                            if row["other_body"] not in ignored), None)
            if contact is not None:
                od = self.model.jnt_dofadr[self.obj_joint]
                self._first_object_contact = {
                    "time_s": (self._step_count + 1) * self.dt,
                    "phase": self.fsm.phase.name,
                    "geom": contact["other_geom"], "body": contact["other_body"],
                    "position": contact["position"],
                    "object_position_before": object_before.tolist(),
                    "object_position_after": self._object_pos().tolist(),
                    "object_velocity_before": object_velocity_before.tolist(),
                    "object_velocity_after": self.data.qvel[od:od + 6].tolist(),
                }
        if self._carrying and self.cfg["task"]["grasp_mode"] == "assisted":
            # Explicit debug-only kinematic attachment. Physical mode never enters here.
            oq = self.model.jnt_qposadr[self.obj_joint]
            od = self.model.jnt_dofadr[self.obj_joint]
            self.data.qpos[oq:oq + 3] = self._ee_pos() + np.array([0.0, 0.0, -0.035])
            self.data.qvel[od:od + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
        self._step_count += 1
        transition = self.fsm.tick(self.dt)
        phase_completed = False if transition.changed else self._update_fsm()
        potential = self._potential()
        rw = self.cfg["reward"]
        progress = rw["progress"] * (potential - self._previous_potential) if rw["potential_based"] else 0.0
        if self.fsm.phase != before_phase:
            progress = 0.0
        terms = {
            "progress": progress,
            "phase_completion": rw["phase_completion"] if phase_completed else 0.0,
            "task_success": rw["task_success"] if self.fsm.phase == Phase.SUCCESS else 0.0,
            "time": -rw["time_penalty"],
            "smoothness": -rw["action_smoothness"] * float(np.sum((action - self._last_action) ** 2)),
            "failure": -rw["object_drop"] if self.fsm.phase == Phase.FAILURE else 0.0,
        }
        reward = float(sum(terms.values()))
        self._previous_potential, self._last_action = potential, action.copy()
        for key, value in terms.items():
            self._reward_sums[key] += value
        terminated = self.fsm.terminal
        truncated = self._step_count >= self.max_steps and not terminated
        if truncated:
            self.fsm.fail("episode_timeout")
        obs = self._observe()
        info = {"phase": self.fsm.phase.name, "phase_id": int(self.fsm.phase), "success": self.fsm.phase == Phase.SUCCESS,
                "failure_reason": self.fsm.failure_reason, "reward_terms": terms, "reward_sums": self._reward_sums.copy(),
                "initial_state": copy.deepcopy(self._initial_state), "grasp_mode": self.cfg["task"]["grasp_mode"]}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=self.cfg["simulation"]["render_height"], width=self.cfg["simulation"]["render_width"])
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer as mj_viewer
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None
        if self._viewer is not None:
            self._viewer.close(); self._viewer = None

