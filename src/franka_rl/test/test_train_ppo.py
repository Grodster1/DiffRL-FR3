"""Tests for `train_ppo.py` - resume point round trip on a toy env, no ROS."""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from franka_rl.train_ppo import (RESUME_MODEL, RESUME_VECNORM, PPOResumePointCallback,
                                 resolve_resume)


class ToyEnv(gym.Env):
    """Same spaces as FrankaPickPlaceEnv, reward of the same order of magnitude."""
    observation_space = gym.spaces.Box(-1.0, 1.0, (17,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(17, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        return np.zeros(17, dtype=np.float32), -1.5, False, self.t >= 20, {}


def test_resume_point_restores_model_and_reward_statistics(tmp_path):
    venv = VecNormalize(DummyVecEnv([ToyEnv]), norm_obs=False, norm_reward=True)
    model = PPO("MlpPolicy", venv, n_steps=64, batch_size=32, n_epochs=1, device="cpu")
    model.learn(128, callback=PPOResumePointCallback(1000, tmp_path))

    model_path, vecnorm_path, run_dir = resolve_resume(tmp_path)
    assert model_path == tmp_path / RESUME_MODEL
    assert vecnorm_path == tmp_path / RESUME_VECNORM
    assert run_dir == tmp_path

    restored = VecNormalize.load(str(vecnorm_path), DummyVecEnv([ToyEnv]))
    assert restored.ret_rms.count == venv.ret_rms.count
    assert PPO.load(str(model_path), env=restored, device="cpu").num_timesteps == 128


def test_resume_without_statistics_still_resumes(tmp_path):
    (tmp_path / RESUME_MODEL).touch()
    _, vecnorm_path, _ = resolve_resume(tmp_path)
    assert vecnorm_path is None
