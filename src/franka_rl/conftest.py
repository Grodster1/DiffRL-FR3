import sys
import time
from pathlib import Path

import pytest

# franka_task is a sibling source package: tests import it from src/, not from the
# install/ copy, so they keep running without a colcon build (same as this package)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "franka_task"))

SKIP_REASON = "no /clock messages - start bringup.launch.py"

def _sim_is_running(timeout=3.0):
    """Waits for an actual /clock message, not just a publisher."""
    import rclpy
    from rosgraph_msgs.msg import Clock

    started_context = False
    try:
        if not rclpy.ok():
            rclpy.init()
            started_context = True

        node = rclpy.create_node("pytest_sim_probe")
        ticks = []
        node.create_subscription(Clock, "/clock", ticks.append, 10)
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and not ticks:
                rclpy.spin_once(node, timeout_sec=0.05)
            return bool(ticks)
        finally:
            node.destroy_node()
    except Exception:
        return False
    finally:
        if started_context and rclpy.ok():
            rclpy.shutdown()
            
def pytest_configure(config):
    config.addinivalue_line("markers", "sim: requires working bringup.launch.py")  
   
def pytest_collection_modifyitems(config, items):
    needs_sim = [item for item in items if item.get_closest_marker("sim")]
    if not needs_sim or _sim_is_running():
        return

    skip = pytest.mark.skip(reason=SKIP_REASON)
    for item in needs_sim:
        item.add_marker(skip)