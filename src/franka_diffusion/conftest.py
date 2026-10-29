import sys
from pathlib import Path

# franka_task is a sibling source package: tests import it from src/, not from the
# install/ copy, so they keep running without a colcon build (same as this package)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "franka_task"))
