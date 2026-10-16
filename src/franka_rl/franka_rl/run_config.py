""" run_config.json - what a run was launched with, one entry per launch (start + resumes)."""

import hashlib
import json
import socket
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

RUN_CONFIG = "run_config.json"
LIBRARIES = ("stable-baselines3", "torch", "gymnasium", "numpy")


def git_blob_hash(data: bytes) -> str:
    """ Same value as `git hash-object <file>`. """
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def source_hashes(package_dir=None):
    """ Blob hash of every .py file of the package, keyed by file name. """
    package_dir = Path(package_dir or Path(__file__).parent)

    return {p.name: git_blob_hash(p.read_bytes()) for p in sorted(package_dir.glob("*.py"))}


def library_versions():
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None

    return versions


def launch_record(algo, args, start_timesteps):
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "algo": algo,
        "argv": sys.argv[1:],
        "args": vars(args),
        "start_timesteps": int(start_timesteps),
        "host": socket.gethostname(),
        "code": source_hashes(),
        "libraries": library_versions(),
    }


def write_run_config(run_dir, algo, args, start_timesteps=0):
    """ Appends this launch to <run_dir>/run_config.json. A resume never overwrites
        the original launch - it is the record of what produced the first part of the run. """
    path = Path(run_dir) / RUN_CONFIG
    config = json.loads(path.read_text()) if path.exists() else {"launches": []}
    config["launches"].append(launch_record(algo, args, start_timesteps))
    path.write_text(json.dumps(config, indent=2) + "\n")

    return path
