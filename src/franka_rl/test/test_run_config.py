"""Tests for `run_config.py` - no ROS."""

import argparse
import json
import subprocess

from franka_rl.run_config import RUN_CONFIG, git_blob_hash, source_hashes, write_run_config


def test_blob_hash_of_empty_file_matches_git():
    assert git_blob_hash(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_blob_hash_matches_git_hash_object(tmp_path):
    f = tmp_path / "a.py"
    f.write_bytes(b"print('hello')\n")
    expected = subprocess.run(["git", "hash-object", str(f)], capture_output=True,
                              text=True, check=True).stdout.strip()
    assert git_blob_hash(f.read_bytes()) == expected


def test_source_hashes_cover_the_training_code():
    hashes = source_hashes()
    assert {"gym_env.py", "config.py", "train_sac.py", "train_ppo.py"} <= set(hashes)


def test_resume_appends_a_launch_instead_of_overwriting(tmp_path):
    args = argparse.Namespace(level="L2", transport_waypoint=True)
    write_run_config(tmp_path, "sac", args, 0)
    write_run_config(tmp_path, "sac", args, 390000)

    launches = json.loads((tmp_path / RUN_CONFIG).read_text())["launches"]
    assert [l["start_timesteps"] for l in launches] == [0, 390000]
    assert launches[0]["args"] == {"level": "L2", "transport_waypoint": True}
