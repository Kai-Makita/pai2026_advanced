from pathlib import Path

import yaml

from mobile_manipulation_il_rl.envs.model import build_mjcf, validate_geometry


ROOT = Path(__file__).resolve().parents[1]


def test_model_is_complete_and_originally_generated():
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    xml = build_mjcf(cfg)
    assert '<mujoco model="course_mobile_manipulator">' in xml
    assert xml.count('<joint name="joint') == 4
    assert "object_geom" in xml and "assist_grasp" in xml
    assert not validate_geometry(cfg)

