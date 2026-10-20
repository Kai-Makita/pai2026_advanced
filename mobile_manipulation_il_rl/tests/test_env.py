import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env
from mobile_manipulation_il_rl.envs import MobilePickPlaceEnv


def test_api_dimensions_and_finiteness():
    env = MobilePickPlaceEnv()
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=7)
    assert obs.shape == (40,)
    assert env.action_space.shape == (7,)
    for _ in range(10):
        obs, reward, terminated, truncated, _ = env.step(np.zeros(7, np.float32))
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if terminated or truncated:
            break
    env.close()


def test_seed_reproducibility():
    env = MobilePickPlaceEnv()
    a, ia = env.reset(seed=123)
    b, ib = env.reset(seed=123)
    np.testing.assert_array_equal(a, b)
    assert ia["initial_state"] == ib["initial_state"]


def test_diagnostics_and_pick_stop_have_no_table_penetration():
    env = MobilePickPlaceEnv()
    env.reset(seed=1, options={"randomization": "fixed"})
    from mobile_manipulation_il_rl.evaluation.physical_debug import _fixed_pick_setup
    _fixed_pick_setup(env)
    pick_table = env.model.body("pick_table").id
    base = env.model.body("base").id
    assert all({env.model.geom_bodyid[c.geom1], env.model.geom_bodyid[c.geom2]} != {pick_table, base}
               for c in env.data.contact[:env.data.ncon])
    diagnostic = env.diagnostic_state()
    assert {"contacts", "joint_limit_margins", "object_ee_relative", "gripper_opening"} <= diagnostic.keys()


@pytest.mark.physics
def test_model_stable_under_gravity():
    env = MobilePickPlaceEnv()
    env.reset(seed=0)
    z0 = env._base_pose()[0][2]
    for _ in range(40):
        _, _, term, trunc, _ = env.step(np.zeros(7, np.float32))
        assert not term and not trunc
    assert abs(env._base_pose()[0][2] - z0) < 0.04

