import numpy as np
import pytest

from franka_rl.evaluate_sac import resolve_model, run_episode, summarize, wilson_interval


def test_wilson_interval_stays_inside_unit_range_at_the_edges():
    low, high = wilson_interval(0, 50)
    assert low == 0.0 and 0.0 < high < 0.1

    low, high = wilson_interval(50, 50)
    assert 0.9 < low < 1.0 and high == 1.0


def test_wilson_interval_contains_the_point_estimate():
    low, high = wilson_interval(9, 50)
    assert low < 9 / 50 < high


def test_resolve_model_uses_latest_zip_in_a_run_dir(tmp_path):
    (tmp_path / "latest.zip").touch()
    assert resolve_model(tmp_path) == tmp_path / "latest.zip"


def test_resolve_model_accepts_an_explicit_zip(tmp_path):
    model = tmp_path / "rl_model_1000_steps.zip"
    model.touch()
    assert resolve_model(model) == model


def test_resolve_model_raises_when_nothing_is_there(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model(tmp_path)


class FakeEnv:
    """ Holds the cube for the first two steps, then lets go; ends after `n` steps. """

    def __init__(self, n=4):
        self.n = n
        self.t = 0

    @property
    def obs_dict(self):
        return {"is_grasped": self.t < 2}

    def _info(self):
        return {"is_success": False, "is_grasped": True, "dropped": False,
                "curriculum_start": False, "min_ee_to_cube": 0.01,
                "min_cube_to_goal": 0.07, "cube_to_goal": np.array([0.06, 0.0, 0.0]),
                "ik_failures": 0}

    def reset(self):
        self.t = 0
        return np.zeros(17), self._info()

    def step(self, action):
        self.t += 1
        return np.zeros(17), -1.0, False, self.t >= self.n, self._info()


def test_release_attempts_count_only_open_commands_while_holding():
    # open on every step: 2 steps holding, 2 steps after the grasp is gone
    record = run_episode(FakeEnv(n=4), lambda obs: np.array([0.0, 0.0, 0.0, 1.0]))
    assert record["release_attempts"] == 2
    assert record["length"] == 4
    assert record["return"] == -4.0
    assert record["final_cube_to_goal_xy"] == pytest.approx(0.06)


def test_closed_gripper_never_counts_as_a_release_attempt():
    record = run_episode(FakeEnv(n=4), lambda obs: np.array([0.0, 0.0, 0.0, -1.0]))
    assert record["release_attempts"] == 0


def _record(success=False, dropped=False, xy=0.06, release_attempts=0):
    return {"return": -10.0, "length": 200, "is_success": success, "is_grasped": True,
            "dropped": dropped, "release_attempts": release_attempts,
            "final_cube_to_goal_xy": xy, "ik_failures": 0}


def test_summarize_takes_the_missed_median_only_from_undropped_failures():
    records = [_record(success=True, xy=0.01), _record(dropped=True, xy=0.5),
               _record(xy=0.06), _record(xy=0.08, release_attempts=3)]
    stats = summarize(records)

    assert stats["success_rate"] == 0.25
    assert stats["drop_rate"] == 0.25
    assert stats["release_attempt_rate"] == 0.25
    assert stats["missed_final_xy_median"] == pytest.approx(0.07)
