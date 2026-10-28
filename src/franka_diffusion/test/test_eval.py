import numpy as np
import pytest
import torch

from franka_diffusion.eval import DiffusionPolicyRunner

N_OBS, N_ACT, OBS_DIM = 2, 8, 17


class FakePolicy:
    """ Records the observation history of every query and plans actions that encode
        (query number, position in the chunk), so their order can be checked. """

    n_obs_steps = N_OBS
    n_action_steps = N_ACT

    def __init__(self):
        self.queries = []

    def predict_action(self, obs):
        assert obs.shape == (1, N_OBS, OBS_DIM) and obs.dtype == torch.float32
        self.queries.append(obs[0, :, 0].tolist())
        q = len(self.queries)
        action = torch.zeros(1, N_ACT, 4)
        action[0, :, 0] = q
        action[0, :, 1] = torch.arange(N_ACT, dtype=torch.float32)
        return action


def _obs(t):
    return np.full(OBS_DIM, float(t), dtype=np.float32)


@pytest.fixture
def runner():
    r = DiffusionPolicyRunner(FakePolicy(), "cpu")
    r.reset()
    return r


def test_first_step_pads_the_history_with_the_initial_obs(runner):
    runner(_obs(0))
    assert runner.policy.queries == [[0.0, 0.0]]


def test_policy_is_queried_once_per_action_chunk(runner):
    for t in range(2 * N_ACT + 1):
        runner(_obs(t))
    assert len(runner.policy.queries) == 3
    assert len(runner.latencies) == 3


def test_history_follows_every_step_not_only_the_queries(runner):
    for t in range(N_ACT + 1):
        runner(_obs(t))
    # The second query happens at step N_ACT and must see the two latest observations
    assert runner.policy.queries[1] == [float(N_ACT - 1), float(N_ACT)]


def test_actions_come_out_in_planned_order_as_float32_vectors(runner):
    actions = [runner(_obs(t)) for t in range(N_ACT + 1)]
    assert all(isinstance(a, np.ndarray) and a.shape == (4,) and a.dtype == np.float32
               for a in actions)
    assert [(a[0], a[1]) for a in actions[:N_ACT]] == [(1.0, float(i)) for i in range(N_ACT)]
    assert actions[N_ACT][0] == 2.0 and actions[N_ACT][1] == 0.0


def test_reset_drops_the_previous_episode(runner):
    for t in range(3):
        runner(_obs(t))
    runner.reset()
    action = runner(_obs(100))
    # Leftover planned actions are discarded and the history starts over
    assert runner.policy.queries[-1] == [100.0, 100.0]
    assert action[0] == 2.0 and action[1] == 0.0
