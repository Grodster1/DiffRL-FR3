import sys

import pytest

from franka_diffusion import eval as dp_eval
from franka_diffusion import train as dp_train


def _parse(module, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["prog"] + argv)
    return module.parse_args()


def test_train_dp_parses_the_cheatsheet_command(monkeypatch):
    args = _parse(dp_train, monkeypatch,
                  ["--demos", "data/demos/demo_L1_seed0_20261008_1035", "--level", "L1"])
    assert args.demos == ["data/demos/demo_L1_seed0_20261008_1035"]
    assert args.level == "L1" and args.epochs == 2000 and args.val_ratio == 0.1


def test_train_dp_accepts_several_demo_dirs(monkeypatch):
    args = _parse(dp_train, monkeypatch, ["--demos", "a", "b"])
    assert args.demos == ["a", "b"]


def test_train_dp_rejects_a_val_ratio_that_leaves_no_training_data(monkeypatch):
    with pytest.raises(ValueError):
        _parse(dp_train, monkeypatch, ["--demos", "a", "--val-ratio", "1.0"])


def test_eval_dp_parses_the_cheatsheet_command(monkeypatch):
    args = _parse(dp_eval, monkeypatch,
                  ["data/runs/dp_L1_seed0_20261028_1129", "--level", "L1", "--episodes", "50"])
    assert args.checkpoint == "data/runs/dp_L1_seed0_20261028_1129"
    assert args.level == "L1" and args.episodes == 50
    # Same scenes as evaluate_rl
    assert args.seed == 1000


def test_eval_dp_requires_a_perturbation_on_L3(monkeypatch):
    with pytest.raises(ValueError):
        _parse(dp_eval, monkeypatch, ["run", "--level", "L3"])


def test_eval_dp_rejects_a_perturbation_on_L1(monkeypatch):
    with pytest.raises(ValueError):
        _parse(dp_eval, monkeypatch, ["run", "--level", "L1", "--perturbation", "displace"])
