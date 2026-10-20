import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("mujoco")
from mobile_manipulation_il_rl.envs import MobilePickPlaceEnv
from mobile_manipulation_il_rl.learning.bc import Actor
from mobile_manipulation_il_rl.learning.ppo import make_ppo, transfer_bc_actor


def test_bc_to_ppo_action_mean_exact():
    actor = Actor()
    cfg = {"learning_rate": 3e-4, "gamma": .99, "gae_lambda": .95, "clip_range": .2, "n_envs": 1, "log_std_init": -1}
    model = make_ppo(MobilePickPlaceEnv, cfg, 1)
    assert transfer_bc_actor(model, actor.state_dict(), -1) < 1e-6
    assert torch.allclose(model.policy.log_std, torch.full_like(model.policy.log_std, -1))


def test_ppo_save_reload(tmp_path):
    cfg = {"learning_rate": 3e-4, "gamma": .99, "gae_lambda": .95, "clip_range": .2, "n_envs": 1, "log_std_init": -1}
    model = make_ppo(MobilePickPlaceEnv, cfg, 2)
    path = tmp_path / "ppo_smoke"; model.save(path)
    from stable_baselines3 import PPO
    loaded = PPO.load(path)
    obs, _ = MobilePickPlaceEnv().reset(seed=2)
    assert loaded.predict(obs, deterministic=True)[0].shape == (7,)

