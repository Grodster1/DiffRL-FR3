from pathlib import Path

import pytest

from franka_rl.evaluate_rl import infer_algo, resolve_model


def test_resolve_model_uses_latest_zip_in_a_run_dir(tmp_path):
    (tmp_path / "latest.zip").touch()
    assert resolve_model(tmp_path) == tmp_path / "latest.zip"


def test_resolve_model_accepts_an_explicit_zip(tmp_path):
    model = tmp_path / "rl_model_1000_steps.zip"
    model.touch()
    assert resolve_model(model) == model


@pytest.mark.parametrize("path, algo", [
    ("data/runs/ppo_L2_seed0_20261012_1400/latest.zip", "ppo"),
    ("data/runs/sac_L2_seed0_20261012_1400_wp/checkpoints/rl_model_20000_steps.zip", "sac"),
])
def test_infer_algo_reads_the_run_name(path, algo):
    assert infer_algo(Path(path)) == algo


def test_infer_algo_raises_on_an_unknown_run_name():
    with pytest.raises(ValueError):
        infer_algo(Path("somewhere/model.zip"))


def test_resolve_model_raises_when_nothing_is_there(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model(tmp_path)
