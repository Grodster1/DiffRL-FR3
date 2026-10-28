from pathlib import Path

import numpy as np
import pytest
import torch

from franka_diffusion.dataset import DemonstrationDataset

DEMOS = Path(__file__).resolve().parents[3] / "data" / "demos"
HORIZON, N_OBS, N_ACT = 16, 2, 8


def _write_episode(path, T, offset=0.0):
    """ Synthetic episode with every step identifiable by value:
        obs[t] = t + offset, action[t, :3] = (t + 1) / 100 (never zero), gripper alternates
        and the last action keeps the gripper closed, so padding has to copy it. """
    obs = np.tile(np.arange(T + 1, dtype=np.float32)[:, None] + offset, (1, 17))
    action = np.zeros((T, 4), dtype=np.float32)
    action[:, :3] = (np.arange(T, dtype=np.float32)[:, None] + 1) / 100
    action[:, 3] = np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    action[-1, 3] = -1.0
    np.savez_compressed(path, obs=obs, action=action,
                        reward=np.zeros(T, dtype=np.float32),
                        goal_pos=np.zeros(3, dtype=np.float32))


@pytest.fixture
def demo_dir(tmp_path):
    # Episode lengths differ so a window taken from the wrong episode shows up
    _write_episode(tmp_path / "ep_0000.npz", T=20, offset=0.0)
    _write_episode(tmp_path / "ep_0001.npz", T=25, offset=1000.0)
    return tmp_path


def _dataset(path, **kwargs):
    return DemonstrationDataset(path, horizon=HORIZON, n_obs_steps=N_OBS,
                                n_action_steps=N_ACT, **kwargs)


def _now(sample):
    """ Episode step the window treats as the present, read back from the obs value. """
    return int(sample["obs"][-1, 0].item()) % 1000


def test_window_count_is_T_minus_action_steps_plus_one(demo_dir):
    ds = _dataset(demo_dir)
    assert len(ds) == (20 - N_ACT + 1) + (25 - N_ACT + 1)


def test_sample_shapes_and_dtypes(demo_dir):
    sample = _dataset(demo_dir)[0]
    assert sample["obs"].shape == (N_OBS, 17)
    assert sample["action"].shape == (HORIZON, 4)
    assert sample["obs"].dtype == torch.float32
    assert sample["action"].dtype == torch.float32


def test_first_window_pads_history_with_the_initial_obs(demo_dir):
    sample = _dataset(demo_dir)[0]
    assert torch.all(sample["obs"] == 0.0)
    # The action executed "now" (index n_obs_steps - 1) is the first action of the demo
    assert sample["action"][N_OBS - 1, 0].item() == pytest.approx(1 / 100)


def test_current_obs_and_first_executed_action_share_a_time_step(demo_dir):
    ds = _dataset(demo_dir)
    for i in range(len(ds)):
        sample = ds[i]
        now = _now(sample)
        assert sample["action"][N_OBS - 1, 0].item() == pytest.approx((now + 1) / 100)


def test_executed_actions_never_come_from_padding(demo_dir):
    ds = _dataset(demo_dir)
    executed = slice(N_OBS - 1, N_OBS - 1 + N_ACT)
    for i in range(len(ds)):
        assert torch.all(ds[i]["action"][executed, :3] != 0.0), f"window {i}"


def test_last_window_ends_exactly_at_the_end_of_the_demo(demo_dir):
    ds = _dataset(demo_dir)
    last_of_ep0 = 20 - N_ACT + 1 - 1
    sample = ds[last_of_ep0]
    executed = sample["action"][N_OBS - 1:N_OBS - 1 + N_ACT]
    assert executed[-1, 0].item() == pytest.approx(20 / 100)  # action[T-1]
    # The next window already belongs to the second episode
    assert ds[last_of_ep0 + 1]["obs"][0, 0].item() == 1000.0


def test_tail_padding_holds_position_and_keeps_the_last_gripper_command(demo_dir):
    ds = _dataset(demo_dir)
    sample = ds[20 - N_ACT]  # last window of the first episode
    now = _now(sample)
    n_real = 20 - (now - (N_OBS - 1))
    tail = sample["action"][n_real:]
    assert len(tail) > 0
    assert torch.all(tail[:, :3] == 0.0)
    assert torch.all(tail[:, 3] == -1.0)


def test_getitem_does_not_modify_the_stored_episodes(demo_dir):
    ds = _dataset(demo_dir)
    before = ds.episodes[0][1].copy()
    for i in range(len(ds)):
        ds[i]
    np.testing.assert_array_equal(ds.episodes[0][1], before)


def test_episode_ids_index_the_sorted_files(demo_dir):
    ds = _dataset(demo_dir, episode_ids=[1])
    assert len(ds) == 25 - N_ACT + 1
    assert ds[0]["obs"][0, 0].item() == 1000.0


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _dataset(tmp_path / "does_not_exist")


def test_directory_without_episodes_raises(tmp_path):
    with pytest.raises(RuntimeError):
        _dataset(tmp_path)


@pytest.mark.skipif(not (DEMOS / "demo_L1_seed0_20261008_1035").is_dir(),
                    reason="L1 demos from 08.10 not present")
def test_real_L1_demos_give_the_expected_window_count():
    # summary.json: 10448 steps in 100 episodes -> 10448 - 100 * (N_ACT - 1)
    assert len(_dataset(DEMOS / "demo_L1_seed0_20261008_1035")) == 9748
