"""Testy `ros_bridge.py` - wylacznie to, co nie dotyka symulacji.

Kontakt z zywym Gazebo sprawdza sekcja 6 w `test_gym_env.py` (znacznik `sim`), bo
tam przechodzi przez cala petle srodowiska. Tutaj zostaje parsowanie wiadomosci
i budowa komend JTC: callbacki wolamy wprost z recznie zlozonych wiadomosci, a
publikacje przechwytujemy podmieniajac `publish`, wiec nic nie spina i nic nie czeka.

Poprzednia wersja tego pliku byla skryptem wykonywanym przy imporcie (`rclpy.init()`
+ `reserve_t()` na poziomie modulu), przez co `pytest src/franka_rl/test` bez Gazebo
wisial i padal juz na KOLEKCJI - zanim cokolwiek zdazylo sie uruchomic.
"""

import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState

import franka_rl.config as cfg
from franka_rl.ros_bridge import SimInterface


@pytest.fixture(scope="module")
def node():
    if not rclpy.ok():
        rclpy.init()
    sim = SimInterface()
    yield sim
    sim.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def captured(node, monkeypatch):
    """Przechwytuje publikacje zamiast wystawiac je na DDS - test jest deterministyczny
    i nie zalezy od obecnosci subskrybenta."""
    box = {}
    monkeypatch.setattr(node._arm_pub, "publish", lambda msg: box.__setitem__("arm", msg))
    monkeypatch.setattr(node._gripper_pub, "publish", lambda msg: box.__setitem__("gripper", msg))
    return box


# ---------------------------------------------------------------------------
# Callbacki - parsowanie wiadomosci
# ---------------------------------------------------------------------------

def test_joint_states_are_mapped_by_name_not_by_order(node):
    """`JointState` nie gwarantuje uporzadkowania. Mapowanie pozycyjne dawaloby ciche
    przestawienie stawow - blad niewidoczny az do dziwnie zachowujacego sie IK."""
    shuffled = ["fr3_joint4", "fr3_finger_joint1", "fr3_joint1", "fr3_joint7",
                "fr3_joint2", "fr3_joint6", "fr3_joint3", "fr3_finger_joint2", "fr3_joint5"]
    positions = {name: 0.1 * (i + 1) for i, name in enumerate(shuffled)}
    velocities = {name: -0.01 * (i + 1) for i, name in enumerate(shuffled)}

    msg = JointState()
    msg.name = shuffled
    msg.position = [positions[n] for n in shuffled]
    msg.velocity = [velocities[n] for n in shuffled]

    node._on_joint_states_callback(msg)

    np.testing.assert_allclose(node._q_arm, [positions[n] for n in cfg.ARM_JOINTS])
    np.testing.assert_allclose(node._dq_arm, [velocities[n] for n in cfg.ARM_JOINTS])
    assert node._q_arm.shape == (7,) and node._dq_arm.shape == (7,)


def test_gripper_opening_is_sum_of_both_fingers(node):
    """Palce jada niezaleznie od osi symetrii, a podzial miedzy nie zalezy od tego,
    gdzie bocznie lezy kostka (zmierzone: 0.040 / 0.010 przy poprawnym chwycie).
    Odczyt jednego palca nie jest miara rozwarcia - byl przyczyna grasp_rate = 0."""
    msg = JointState()
    msg.name = cfg.ARM_JOINTS + cfg.GRIPPER_JOINTS
    msg.position = [0.0] * 7 + [0.040, 0.010]
    msg.velocity = [0.0] * 9

    node._on_joint_states_callback(msg)

    assert node._gripper_opening == pytest.approx(0.050)



def test_cube_callback_extracts_position_and_linear_velocity(node):
    msg = Odometry()
    msg.pose.pose.position = Point(x=0.5, y=-0.1, z=0.425)
    msg.twist.twist.linear = Vector3(x=0.01, y=0.02, z=-0.03)
    msg.twist.twist.angular = Vector3(x=9.0, y=9.0, z=9.0)   # nieuzywane przez env

    node._on_cube_callback(msg)

    np.testing.assert_allclose(node._cube_pos, [0.5, -0.1, 0.425])
    np.testing.assert_allclose(node._cube_vel, [0.01, 0.02, -0.03])


def test_clock_callback_combines_seconds_and_nanoseconds(node):
    node._on_clock_callback(Clock(clock=Time(sec=12, nanosec=500_000_000)))

    assert node._sim_time == pytest.approx(12.5)


def test_get_state_reports_every_field_the_env_consumes(node):
    """Klucze musza sie zgadzac z tym, po czym `wait_for_state` wykrywa braki."""
    assert set(node.get_state()) == {
        "gripper_opening", "q_arm", "dq_arm", "cube_pos", "cube_vel"
    }


# ---------------------------------------------------------------------------
# Budowa komend JTC
# ---------------------------------------------------------------------------

def test_arm_command_names_all_seven_joints_in_config_order(node, captured):
    q = np.linspace(0.1, 0.7, 7)
    node.publish_arm_command(q, dt=0.05)

    msg = captured["arm"]
    assert msg.joint_names == cfg.ARM_JOINTS
    assert len(msg.points) == 1
    np.testing.assert_allclose(msg.points[0].positions, q)


def test_gripper_command_drives_both_fingers_explicitly(node, captured):
    """Mimic jest niewspierany przez DART - drugi palec musi dostac wlasna komende,
    inaczej chwytak zamyka sie jednostronnie (decyzja: Opcja A).
    Argument jest rozwarciem CALKOWITYM, wiec kazdy palec dostaje polowe."""
    node.publish_gripper_command(cfg.GRIPPER_RANGE[1], dt=0.05)

    msg = captured["gripper"]
    assert msg.joint_names == cfg.GRIPPER_JOINTS
    assert len(msg.joint_names) == 2
    np.testing.assert_allclose(msg.points[0].positions, [0.04, 0.04])


def test_time_from_start_matches_dt(node, captured):
    """JTC odrzuca trajektorie z `time_from_start = 0`, a przy zlym dt interpolacja
    rozjezdza sie z krokiem srodowiska."""
    node.publish_arm_command(np.zeros(7), dt=0.05)

    tfs = captured["arm"].points[0].time_from_start
    assert tfs.sec == 0
    assert tfs.nanosec == 50_000_000
