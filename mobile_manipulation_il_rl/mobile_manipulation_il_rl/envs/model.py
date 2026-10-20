from __future__ import annotations

import math
from typing import Any


def _v(values: list[float]) -> str:
    return " ".join(f"{x:.8g}" for x in values)


def build_mjcf(cfg: dict[str, Any]) -> str:
    """Build an original, parameterized educational robot MJCF (no external assets)."""
    r, t, sim = cfg["robot"], cfg["task"], cfg["simulation"]
    lx = r["arm_link_lengths"]
    lm = r["arm_link_masses"]
    jr = [[math.radians(x) for x in pair] for pair in r["joint_ranges_deg"]]
    chassis_half = [x / 2 for x in r["chassis_size"]]
    table_half = [x / 2 for x in t["table_size"]]
    obj_half = t["object_size"]
    return f'''<mujoco model="course_mobile_manipulator">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{1/sim['physics_hz']:.8f}" gravity="0 0 -9.81" integrator="implicitfast" iterations="50"/>
  <size nconmax="200" njmax="1000"/>
  <default>
    <geom condim="4" friction="0.9 0.01 0.001" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    <joint armature="{r['joint_armature']}"/>
    <position kp="{r['position_kp']}" kv="{r['position_kv']}"/>
  </default>
  <asset>
    <material name="body" rgba="0.18 0.36 0.55 1"/><material name="arm" rgba="0.9 0.55 0.12 1"/>
    <material name="table" rgba="0.55 0.38 0.23 1"/><material name="object" rgba="0.82 0.16 0.12 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3"/><geom name="floor" type="plane" size="3 3 0.1" rgba="0.75 0.78 0.8 1"/>
    <body name="pick_table" pos="{t['pick_table_xy'][0]} {t['pick_table_xy'][1]} {table_half[2]}">
      <geom type="box" size="{_v(table_half)}" material="table" mass="30"/>
    </body>
    <body name="place_table" pos="{t['place_table_xy'][0]} {t['place_table_xy'][1]} {table_half[2]}">
      <geom type="box" size="{_v(table_half)}" material="table" mass="30"/>
    </body>
    <site name="place_target" pos="{t['place_xy'][0]} {t['place_xy'][1]} {t['table_size'][2]+0.003}" type="cylinder" size="0.07 0.003" rgba="0.1 0.8 0.2 0.5"/>
    <body name="object" pos="{t['pick_xy'][0]} {t['pick_xy'][1]} {t['table_size'][2]+obj_half[2]+0.002}">
      <freejoint name="object_free"/><geom name="object_geom" type="box" size="{_v(obj_half)}" mass="{t['object_mass']}" material="object" friction="1.2 0.02 0.002"/>
    </body>
    <body name="base" pos="0 0 {r['wheel_radius']}">
      <freejoint name="base_free"/>
      <geom name="chassis" type="box" pos="0 0 {chassis_half[2]}" size="{_v(chassis_half)}" mass="{r['chassis_mass']}" material="body"/>
      <body name="left_wheel" pos="0 {r['track_width']/2} 0"><joint name="left_wheel_hinge" axis="0 1 0" damping="0.05"/><geom type="cylinder" euler="1.5708 0 0" size="{r['wheel_radius']} {r['wheel_width']/2}" mass="{r['wheel_mass']}" friction="3.0 0.02 0.002"/></body>
      <body name="right_wheel" pos="0 {-r['track_width']/2} 0"><joint name="right_wheel_hinge" axis="0 1 0" damping="0.05"/><geom type="cylinder" euler="1.5708 0 0" size="{r['wheel_radius']} {r['wheel_width']/2}" mass="{r['wheel_mass']}" friction="3.0 0.02 0.002"/></body>
      <geom name="caster_rear" type="sphere" pos="{-chassis_half[0]+0.035} 0 {-r['wheel_radius']+0.028}" size="0.025" mass="0.08" friction="0.03 0.003 0.0001"/>
      <geom name="caster_front" type="sphere" pos="{chassis_half[0]-0.035} 0 {-r['wheel_radius']+0.028}" size="0.025" mass="0.08" friction="0.03 0.003 0.0001"/>
      <body name="arm0" pos="0 0 {r['chassis_size'][2]+0.02}">
        <joint name="joint1" type="hinge" axis="0 0 1" range="{_v(jr[0])}" damping="{r['joint_damping'][0]}"/>
        <geom type="cylinder" size="0.045 0.045" mass="0.45" material="arm"/>
        <body name="link1" pos="0 0 0.045"><joint name="joint2" axis="0 1 0" range="{_v(jr[1])}" damping="{r['joint_damping'][1]}"/><geom type="capsule" fromto="0 0 0 {lx[0]} 0 0" size="0.025" mass="{lm[0]}" material="arm"/>
          <body name="link2" pos="{lx[0]} 0 0"><joint name="joint3" axis="0 1 0" range="{_v(jr[2])}" damping="{r['joint_damping'][2]}"/><geom type="capsule" fromto="0 0 0 {lx[1]} 0 0" size="0.022" mass="{lm[1]}" material="arm"/>
            <body name="link3" pos="{lx[1]} 0 0"><joint name="joint4" axis="0 1 0" range="{_v(jr[3])}" damping="{r['joint_damping'][3]}"/><geom type="capsule" fromto="0 0 0 {lx[2]} 0 0" size="0.019" mass="{lm[2]}" material="arm"/>
              <body name="palm" pos="{lx[2]} 0 0"><geom type="box" size="0.02 0.05 0.018" mass="0.18" material="arm"/><site name="ee_site" pos="0.045 0 0" size="0.012"/>
                <body name="finger_left" pos="0 {r['finger_base_offset']} 0"><joint name="finger_left_slide" type="slide" axis="0 1 0" range="0 0.035" damping="1"/><geom name="finger_left_geom" type="box" pos="0.065 0 -0.025" size="0.04 {r['finger_pad_half_width']} 0.045" mass="0.06" friction="1.5 0.02 0.002"/></body>
                <body name="finger_right" pos="0 {-r['finger_base_offset']} 0"><joint name="finger_right_slide" type="slide" axis="0 -1 0" range="0 0.035" damping="1"/><geom name="finger_right_geom" type="box" pos="0.065 0 -0.025" size="0.04 {r['finger_pad_half_width']} 0.045" mass="0.06" friction="1.5 0.02 0.002"/></body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <contact><exclude body1="base" body2="arm0"/></contact>
  <equality><weld name="assist_grasp" body1="palm" body2="object" active="false" solref="0.01 1"/></equality>
  <actuator>
    <velocity name="left_wheel" joint="left_wheel_hinge" kv="8" ctrlrange="-12 12" forcerange="-40 40"/>
    <velocity name="right_wheel" joint="right_wheel_hinge" kv="8" ctrlrange="-12 12" forcerange="-40 40"/>
    <position name="joint1_pos" joint="joint1" forcerange="-{r['joint_force_limits'][0]} {r['joint_force_limits'][0]}"/>
    <position name="joint2_pos" joint="joint2" forcerange="-{r['joint_force_limits'][1]} {r['joint_force_limits'][1]}"/>
    <position name="joint3_pos" joint="joint3" forcerange="-{r['joint_force_limits'][2]} {r['joint_force_limits'][2]}"/>
    <position name="joint4_pos" joint="joint4" forcerange="-{r['joint_force_limits'][3]} {r['joint_force_limits'][3]}"/>
    <position name="finger_left" joint="finger_left_slide" kp="250" forcerange="-{r['finger_force']} {r['finger_force']}"/>
    <position name="finger_right" joint="finger_right_slide" kp="250" forcerange="-{r['finger_force']} {r['finger_force']}"/>
  </actuator>
  <sensor>
    <framepos name="ee_pos" objtype="site" objname="ee_site"/><framelinvel name="ee_vel" objtype="site" objname="ee_site"/>
    <touch name="left_touch" site="ee_site"/><jointpos name="finger_pos" joint="finger_left_slide"/><jointvel name="finger_vel" joint="finger_left_slide"/>
  </sensor>
</mujoco>'''


def validate_geometry(cfg: dict[str, Any]) -> list[str]:
    """Conservative reachability checks for both table workspaces."""
    r, t = cfg["robot"], cfg["task"]
    reach = sum(r["arm_link_lengths"]) + 0.045  # EE site offset from the last link
    errors = []
    if t["stand_off"] > reach - 0.02:
        errors.append("stand_off exceeds conservative arm reach")
    table_front = t["pick_table_xy"][0] - t["table_size"][0] / 2
    nominal_base_x = t["pick_xy"][0] - t["stand_off"]
    if nominal_base_x + r["chassis_size"][0] / 2 > table_front:
        errors.append("pick stop pose intersects the pick table")
    for key in ("pick_xy", "place_xy"):
        if len(t[key]) != 2:
            errors.append(f"{key} must be planar")
    if t["table_size"][2] > r["wheel_radius"] + r["chassis_size"][2] + reach:
        errors.append("table top is above maximum arm reach")
    return errors
