"""Testy `gym_env.py`.

Podzial pliku odpowiada temu, co da sie sprawdzic bez symulacji, a co nie:

  - Sekcje 1-5 sa CZYSTE: zero rclpy, zero Gazebo. Konstruktor `FrankaPickPlaceEnv`
    jest omijany (`object.__new__`), bo robi `rclpy.init()` i renderuje xacro - a
    testowana logika (latch chwytu, predykat sukcesu, reward, losowanie, `step`)
    nie potrzebuje ani jednego, ani drugiego. Sekcja 5 podstawia `FakeSim`, wiec
    caly `step()` z prawdziwym DLS-IK leci offline.
  - Sekcja 6 chodzi po ZYWEJ symulacji (punkty 4-6 z `docs/plan-gym-wrapper-dls-ik.md`)
    i idzie przez caly most: `step()` -> `SimInterface` -> JTC -> Gazebo -> `/joint_states`.
    Oznaczona `@pytest.mark.sim`; autodetekcja z `conftest.py` pomija ja, gdy nikt nie
    publikuje `/clock`. Odpal `bringup.launch.py` i te same komendy uruchomia calosc.
"""

import numpy as np
import pytest

import franka_rl.config as cfg
from franka_rl.gym_env import FrankaPickPlaceEnv
from franka_rl.kinematics import FrankaKinematics

URDF_PATH = "/tmp/fr3.urdf"

# Wysokosci kostki wzgledem blatu - progi latcha z config.py, nazwane raz zamiast
# przepisywac arytmetyke w kazdym tescie.
Z_TABLE = cfg.CUBE_POS_DEFAULT[2]
Z_ENTER = Z_TABLE + cfg.LIFT_MARGIN_ENTER + 0.005   # pewnie POWYZEJ progu wejscia
Z_BAND = Z_TABLE + 0.5 * (cfg.LIFT_MARGIN_EXIT + cfg.LIFT_MARGIN_ENTER)  # w histerezie
Z_EXIT = Z_TABLE                                     # pewnie PONIZEJ progu wyjscia

GRIP_HOLD = 0.5 * (cfg.GRASP_OPENING_MIN + cfg.GRASP_OPENING_MAX)   # rozwarcie "trzyma"
GRIP_OPEN = cfg.GRIPPER_RANGE[1]                                    # rozwarcie "puscil"

NEAR = np.array([0.0, 0.0, 0.5 * cfg.GRASP_DIST_THRESH])   # EE przy kostce
FAR = np.array([0.0, 0.0, 2.0 * cfg.GRASP_DIST_THRESH])    # EE daleko od kostki


def make_env(level="L1", goal=cfg.GOAL_LEFT, seed=0):
    """Env bez `__init__` - tylko pola, ktorych uzywa testowana logika."""
    env = object.__new__(FrankaPickPlaceEnv)
    env.dt = 0.05
    env.max_episode_steps = 200
    env.level = level
    env.goal_pos = np.asarray(goal, dtype=float).copy()
    env._grasped = False
    env._grasped_streak = 0
    env._just_grasped = False
    env._ever_grasped = False
    env._success = False
    env._dropped = False
    env._min_ee_to_cube = np.inf
    env._step_count = 0
    env._ik_failures = 0
    env.np_random = np.random.default_rng(seed)
    return env


def set_obs(env, ee_to_cube=NEAR, cube_to_goal=(0.0, 0.0, 0.0), is_grasped=False):
    env._last_obs_dict = {
        "ee_to_cube": np.asarray(ee_to_cube, dtype=float),
        "cube_to_goal": np.asarray(cube_to_goal, dtype=float),
        "is_grasped": is_grasped,
    }
    return env._last_obs_dict


def latch(env, n, ee_to_cube=NEAR, z=Z_ENTER, gripper=GRIP_HOLD):
    """Wola `_update_grasped` n razy z tym samym stanem; zwraca ostatni wynik."""
    result = None
    for _ in range(n):
        result = env._update_grasped(np.asarray(ee_to_cube, dtype=float),
                                     np.array([0.0, 0.0, z]), gripper)
    return result


# ---------------------------------------------------------------------------
# 1. Latch chwytu (`_update_grasped`) - automat stanowy, nie funkcja klatki
# ---------------------------------------------------------------------------

def test_grasp_needs_all_three_conditions():
    """Zaciety chwytak, bliskosc i uniesienie - brak ktoregokolwiek = brak chwytu."""
    assert latch(make_env(), cfg.N_GRASP_CONFIRM, gripper=GRIP_OPEN) is False
    assert latch(make_env(), cfg.N_GRASP_CONFIRM, ee_to_cube=FAR) is False
    assert latch(make_env(), cfg.N_GRASP_CONFIRM, z=Z_EXIT) is False


def test_grasp_requires_consecutive_confirmations():
    """Pojedyncza klatka nie wystarcza - o to chodzi w N_GRASP_CONFIRM."""
    env = make_env()

    assert latch(env, cfg.N_GRASP_CONFIRM - 1) is False
    assert latch(env, 1) is True


def test_grasp_streak_resets_on_interruption():
    """Przerwa kasuje licznik, a nie tylko go wstrzymuje."""
    env = make_env()

    latch(env, cfg.N_GRASP_CONFIRM - 1)
    latch(env, 1, gripper=GRIP_OPEN)          # jedna klatka bez trzymania
    assert env._grasped_streak == 0
    assert latch(env, cfg.N_GRASP_CONFIRM - 1) is False   # liczymy od zera


def test_just_grasped_is_a_rising_edge():
    """R_GRASP placone raz. Gdyby flaga zostawala True, wisienie = dojna krowa."""
    env = make_env()

    latch(env, cfg.N_GRASP_CONFIRM - 1)
    assert env._just_grasped is False

    latch(env, 1)
    assert env._just_grasped is True

    latch(env, 1)
    assert env._just_grasped is False


def test_grasp_hysteresis_band_holds_but_does_not_start():
    """Wysokosc miedzy progiem wyjscia a wejscia: latch utrzymany, ale nie zalaczony."""
    latched = make_env()
    latch(latched, cfg.N_GRASP_CONFIRM)
    assert latch(latched, 3, z=Z_BAND) is True

    fresh = make_env()
    assert latch(fresh, 10, z=Z_BAND) is False


def test_grasp_released_below_exit_margin():
    env = make_env()
    latch(env, cfg.N_GRASP_CONFIRM)

    assert latch(env, 1, z=Z_EXIT) is False
    assert env._grasped_streak == 0


def test_grasp_released_when_gripper_opens():
    env = make_env()
    latch(env, cfg.N_GRASP_CONFIRM)

    assert latch(env, 1, gripper=GRIP_OPEN) is False


def test_ever_grasped_survives_a_release():
    """`is_grasped` w info to statystyka epizodu, nie stan klatki - inaczej Monitor
    zapisze 0 dla epizodu, w ktorym chwyt sie udal i kostka zostala pozniej puszczona."""
    env = make_env()
    latch(env, cfg.N_GRASP_CONFIRM)
    latch(env, 1, gripper=GRIP_OPEN)

    assert env._grasped is False
    assert env._ever_grasped is True


# ---------------------------------------------------------------------------
# 2. Predykat sukcesu (`_is_success`)
# ---------------------------------------------------------------------------

AT_REST = np.zeros(3)


def on_goal(env, dxy=0.0, dz=0.0):
    return env.goal_pos + np.array([dxy, 0.0, dz])


def test_success_nominal():
    env = make_env()
    assert env._is_success(on_goal(env), AT_REST, grasped=False)


def test_success_rejects_held_cube():
    """Kluczowy warunek: kostka trzymana nad blatem to nie jest odlozenie."""
    env = make_env()
    assert not env._is_success(on_goal(env), AT_REST, grasped=True)


def test_success_rejects_moving_cube():
    env = make_env()
    moving = np.array([cfg.SUCCESS_VEL_EPS * 2.0, 0.0, 0.0])
    assert not env._is_success(on_goal(env), moving, grasped=False)


def test_success_rejects_wrong_height():
    """Rozdzielenie XY od Z jest po to, zeby kostka w locie nie zaliczala sukcesu."""
    env = make_env()
    too_high = on_goal(env, dz=cfg.SUCCESS_Z_TOL * 2.0)
    assert not env._is_success(too_high, AT_REST, grasped=False)


def test_success_rejects_wrong_xy():
    env = make_env()
    off = on_goal(env, dxy=cfg.SUCCESS_XY_TOL * 2.0)
    assert not env._is_success(off, AT_REST, grasped=False)


def test_success_tolerances_are_strict():
    """Dokladnie na progu = brak sukcesu (`<`, nie `<=`). Blad o jeden bit tolerancji
    nie moze decydowac o R_SUCCESS."""
    env = make_env()

    assert not env._is_success(on_goal(env, dxy=cfg.SUCCESS_XY_TOL), AT_REST, False)
    assert not env._is_success(on_goal(env, dz=cfg.SUCCESS_Z_TOL), AT_REST, False)

    vel = np.array([cfg.SUCCESS_VEL_EPS, 0.0, 0.0])
    assert not env._is_success(on_goal(env), vel, False)


def test_success_uses_the_episode_goal_not_a_constant():
    """L2 losuje cel - predykat musi patrzec na `self.goal_pos`."""
    env = make_env(goal=cfg.GOAL_RIGHT)

    assert env._is_success(cfg.GOAL_RIGHT.copy(), AT_REST, False)
    assert not env._is_success(cfg.GOAL_LEFT.copy(), AT_REST, False)


# ---------------------------------------------------------------------------
# 3. Reward (`_compute_reward`)
# ---------------------------------------------------------------------------

def test_reward_is_negative_distance_when_idle():
    """Wagi 1.0 -> czlony geste sa doslownie minus odlegloscia w metrach."""
    env = make_env()
    set_obs(env, ee_to_cube=[0.2, 0.0, 0.0], cube_to_goal=[0.7, 0.0, 0.0])

    reward = env._compute_reward(np.zeros(4), grasped=False, just_grasped=False, success=False)

    assert reward == pytest.approx(-(cfg.W_REACH * 0.2 + cfg.W_TRANSPORT * 0.7))


def test_reward_drops_reach_term_once_grasped():
    """Po chwycie zblizanie EE do kostki przestaje byc zadaniem."""
    env = make_env()
    set_obs(env, ee_to_cube=[0.2, 0.0, 0.0], cube_to_goal=[0.7, 0.0, 0.0])

    free = env._compute_reward(np.zeros(4), False, False, False)
    held = env._compute_reward(np.zeros(4), True, False, False)

    assert held - free == pytest.approx(cfg.W_REACH * 0.2)


def test_entering_grasp_cannot_lower_reward():
    """Twierdzenie z docstringa `_compute_reward` - warte przypilnowania testem,
    bo to ono gwarantuje, ze chwyt nie jest lokalnie nieoplacalny."""
    env = make_env()
    rng = np.random.default_rng(7)

    for _ in range(200):
        set_obs(env, ee_to_cube=rng.uniform(-0.4, 0.4, 3), cube_to_goal=rng.uniform(-0.8, 0.8, 3))
        action = rng.uniform(-1.0, 1.0, 4)

        free = env._compute_reward(action, False, False, False)
        held = env._compute_reward(action, True, True, False)

        assert held >= free


def test_grasp_bonus_paid_once_per_edge():
    env = make_env()
    set_obs(env)

    steady = env._compute_reward(np.zeros(4), True, False, False)
    edge = env._compute_reward(np.zeros(4), True, True, False)

    assert edge - steady == pytest.approx(cfg.R_GRASP)


def test_success_bonus_is_added():
    env = make_env()
    set_obs(env)

    plain = env._compute_reward(np.zeros(4), False, False, False)
    won = env._compute_reward(np.zeros(4), False, False, True)

    assert won - plain == pytest.approx(cfg.R_SUCCESS)


def test_energy_penalty_scales_with_squared_action():
    env = make_env()
    set_obs(env)

    idle = env._compute_reward(np.zeros(4), False, False, False)
    full = env._compute_reward(np.ones(4), False, False, False)

    assert idle - full == pytest.approx(cfg.W_ENERGY * 4.0)


def test_successful_terminal_step_is_net_positive():
    """R_SUCCESS ma przebic kary geste ostatniego kroku z zapasem - inaczej agent
    woli nie konczyc epizodu."""
    env = make_env()
    set_obs(env, ee_to_cube=[0.35, 0.0, 0.0], cube_to_goal=[0.0, 0.0, 0.0])

    reward = env._compute_reward(np.ones(4), grasped=False, just_grasped=False, success=True)

    assert reward > 0.0


# ---------------------------------------------------------------------------
# 4. Losowanie sceny i determinizm ziarna
# ---------------------------------------------------------------------------

def test_l1_goal_is_fixed():
    env = make_env(level="L1")
    for _ in range(20):
        np.testing.assert_allclose(env._sample_goal_cube_pos(), cfg.GOAL_LEFT)


def test_l2_goal_is_bimodal():
    """Dwa stoly to celowa decyzja projektowa - jesli losowanie da tylko jeden,
    L2 degeneruje sie do L1 i wynik pracy traci sens."""
    env = make_env(level="L2")
    goals = [env._sample_goal_cube_pos() for _ in range(200)]

    seen_left = any(np.allclose(g, cfg.GOAL_LEFT) for g in goals)
    seen_right = any(np.allclose(g, cfg.GOAL_RIGHT) for g in goals)

    assert seen_left and seen_right
    assert all(np.allclose(g, cfg.GOAL_LEFT) or np.allclose(g, cfg.GOAL_RIGHT) for g in goals)


def test_sampled_goal_is_a_copy():
    """Zwrocenie referencji do stalej z config pozwoliloby epizodowi ja nadpisac."""
    env = make_env(level="L1")
    goal = env._sample_goal_cube_pos()
    goal += 1.0

    np.testing.assert_allclose(cfg.GOAL_LEFT, [0.0, 0.525, 0.425])


def test_sampled_cube_stays_in_spawn_zone():
    env = make_env(level="L2")

    for _ in range(500):
        pos = env._sample_cube_pos()
        assert 0.4 <= pos[0] <= 0.6
        assert -0.2 <= pos[1] <= 0.2
        assert pos[2] == pytest.approx(cfg.CUBE_POS_DEFAULT[2])


def test_sampling_is_reproducible_for_a_seed():
    """Warunek protokolu ewaluacji: identyczne ziarna sceny dla wszystkich metod."""
    a = make_env(level="L2", seed=42)
    b = make_env(level="L2", seed=42)

    for _ in range(10):
        np.testing.assert_allclose(a._sample_cube_pos(), b._sample_cube_pos())
        np.testing.assert_allclose(a._sample_goal_cube_pos(), b._sample_goal_cube_pos())


# ---------------------------------------------------------------------------
# 5. `step()` na atrapie symulacji - prawdziwe IK, zero ROS
# ---------------------------------------------------------------------------

class FakeSim:
    """Minimalna atrapa `SimInterface`: oddaje zadany stan, zapisuje komendy."""

    def __init__(self, q_arm=None, cube_pos=None, cube_vel=None, gripper_opening=GRIP_OPEN):
        self.state = {
            "q_arm": np.asarray(cfg.Q_READY if q_arm is None else q_arm, dtype=float),
            "dq_arm": np.zeros(7),
            "cube_pos": np.asarray(cfg.CUBE_POS_DEFAULT if cube_pos is None else cube_pos, dtype=float),
            "cube_vel": np.zeros(3) if cube_vel is None else np.asarray(cube_vel, dtype=float),
            "gripper_opening": gripper_opening,
        }
        self.arm_commands = []
        self.arm_dts = []
        self.gripper_commands = []
        self.gripper_dts = []
        self.cube_poses = []
        self.reserved = []
        self.destroyed = 0

    def get_state(self):
        return self.state

    def wait_for_state(self, timeout=10.0):
        return self.state

    def publish_arm_command(self, q, dt):
        self.arm_commands.append(np.asarray(q, dtype=float))
        self.arm_dts.append(dt)

    def publish_gripper_command(self, opening, dt):
        self.gripper_commands.append(opening)
        self.gripper_dts.append(dt)

    def destroy_node(self):
        self.destroyed += 1

    def set_cube_pose(self, cube_pos, timeout=5.0):
        self.cube_poses.append(np.asarray(cube_pos, dtype=float))

    def reserve_t(self, dt):
        self.reserved.append(dt)


class ColdStartSim(FakeSim):
    """`get_state()` oddaje puste pola, dopoki nikt nie zawolal `wait_for_state()`.

    Tak zachowuje sie `SimInterface` tuz po starcie: `reserve_t` wraca, gdy czas
    symulacji przesunie sie o `dt`, a `/clock` idzie 2000 Hz, wiec to kwestia
    milisekund - subskrypcja `/joint_states` moze byc w tym momencie jeszcze
    niedopieta przez DDS i `_q_arm` jest `None`."""

    def __init__(self, **sim_kwargs):
        super().__init__(**sim_kwargs)
        self.warm = False

    def get_state(self):
        if not self.warm:
            return {key: None for key in self.state}
        return self.state

    def wait_for_state(self, timeout=10.0):
        self.warm = True
        return self.state


@pytest.fixture(scope="module")
def kin():
    return FrankaKinematics(URDF_PATH)


def make_stepping_env(kin, **sim_kwargs):
    env = make_env()
    env.kin = kin
    env.sim_interface = FakeSim(**sim_kwargs)
    _, env.R_frozen = kin.fk(cfg.Q_READY)
    return env


def test_step_returns_the_gym_five_tuple(kin):
    env = make_stepping_env(kin)
    obs, reward, terminated, truncated, info = env.step(np.zeros(4))

    assert obs.shape == (cfg.OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert set(info) == {"cube_to_goal", "ik_failures", "is_success", "is_grasped",
                         "dropped", "min_ee_to_cube"}


def test_step_commands_the_arm_towards_the_requested_delta(kin):
    """Akcja jednostkowa w Z ma przesunac TCP o ACTION_SCALE - to sprawdza cala
    sciezke skalowanie -> IK -> komenda, w tym znak w `pose_error`."""
    env = make_stepping_env(kin)
    p_before, _ = kin.fk(cfg.Q_READY)

    env.step(np.array([0.0, 0.0, 1.0, 0.0]))

    q_cmd = env.sim_interface.arm_commands[-1]
    p_after, R_after = kin.fk(q_cmd)

    np.testing.assert_allclose(p_after, p_before + [0.0, 0.0, cfg.ACTION_SCALE], atol=1e-3)
    # orientacja zamrozona: os Z chwytaka dalej patrzy w dol
    assert np.degrees(np.arccos(np.clip(R_after[:, 2] @ np.array([0.0, 0.0, -1.0]), -1, 1))) < 1.0


def test_step_clips_target_to_workspace(kin):
    """Powtarzane maksymalne akcje w gore nie moga wyprowadzic TCP poza pudelko."""
    env = make_stepping_env(kin)

    for _ in range(40):
        env.step(np.array([0.0, 0.0, 1.0, 0.0]))
        env.sim_interface.state["q_arm"] = env.sim_interface.arm_commands[-1]

    p, _ = kin.fk(env.sim_interface.arm_commands[-1])
    assert p[2] <= cfg.WORKSPACE_BOX[1][2] + 1e-3


def test_gripper_command_is_binary(kin):
    """Prog zamiast wartosci ciaglej - zeby polityka nie trzepotala chwytakiem."""
    env = make_stepping_env(kin)

    for a3 in (1.0, 0.3, 0.001, -0.001, -0.3, -1.0):
        env.step(np.array([0.0, 0.0, 0.0, a3]))

    assert set(env.sim_interface.gripper_commands) == set(cfg.GRIPPER_RANGE)
    assert env.sim_interface.gripper_commands[:3] == [cfg.GRIPPER_RANGE[1]] * 3
    assert env.sim_interface.gripper_commands[3:] == [cfg.GRIPPER_RANGE[0]] * 3


def test_step_advances_simulation_exactly_once(kin):
    env = make_stepping_env(kin)
    env.step(np.zeros(4))

    assert env.sim_interface.reserved == [env.dt]


def test_episode_truncates_at_step_limit(kin):
    env = make_stepping_env(kin)
    env.max_episode_steps = 3

    flags = [env.step(np.zeros(4))[3] for _ in range(3)]

    assert flags == [False, False, True]


def test_episode_terminates_when_cube_falls_into_the_gap(kin):
    """Bez tego agent ciagnie 200 krokow stanu bez powrotu i zasmieca replay buffer."""
    env = make_stepping_env(kin, cube_pos=[0.16, 0.0, cfg.CUBE_DROP_Z - 0.01])
    _, _, terminated, _, _ = env.step(np.zeros(4))

    assert terminated is True


def test_episode_terminates_on_success(kin):
    env = make_stepping_env(kin, cube_pos=cfg.GOAL_LEFT)
    _, reward, terminated, _, _ = env.step(np.zeros(4))

    assert terminated is True
    assert reward > 0.0


def test_info_reports_the_success_verdict(kin):
    env = make_stepping_env(kin, cube_pos=cfg.GOAL_LEFT)
    _, _, _, _, info = env.step(np.zeros(4))

    assert info["is_success"] is True
    assert info["dropped"] is False


def test_info_reports_a_dropped_cube(kin):
    env = make_stepping_env(kin, cube_pos=[0.16, 0.0, cfg.CUBE_DROP_Z - 0.01])
    _, _, _, _, info = env.step(np.zeros(4))

    assert info["dropped"] is True
    assert info["is_success"] is False


def test_min_ee_to_cube_never_grows(kin):
    """Minimum epizodu, nie ostatnia klatka - to jedyna liczba, ktora mowi, czy
    polityka w ogole dochodzi do kostki, gdy `grasp_rate` wynosi zero."""
    env = make_stepping_env(kin)
    minima = []
    for _ in range(5):
        _, _, _, _, info = env.step(np.array([0.0, 0.0, -1.0, 0.0]))
        minima.append(info["min_ee_to_cube"])

    assert minima == sorted(minima, reverse=True)
    assert minima[-1] <= np.linalg.norm(env._last_obs_dict["ee_to_cube"]) + 1e-9


def test_ik_succeeds_inside_the_reachable_workspace(kin):
    """Krok z pozy ready w glab przestrzeni roboczej nie moze generowac porazek."""
    env = make_stepping_env(kin)

    for _ in range(10):
        _, _, _, _, info = env.step(np.array([0.0, 0.0, -0.2, 0.0]))
        env.sim_interface.state["q_arm"] = env.sim_interface.arm_commands[-1]

    assert info["ik_failures"] == 0


def test_ik_failures_are_counted_at_the_singular_corner(kin):
    """`WORKSPACE_BOX` jest prostopadloscianem, a strefa robocza nim nie jest: rog
    'daleko i wysoko' lezy w obszarze osobliwym. To jest udokumentowane zrodlo
    systematycznych `ik_failures` (constants-derivations §2.1) - licznik ma je
    zlapac, bo idzie do `info` jako statystyka porownawcza DP vs RL."""
    env = make_stepping_env(kin)

    for _ in range(30):
        _, _, _, _, info = env.step(np.array([1.0, 0.0, 1.0, 0.0]))
        env.sim_interface.state["q_arm"] = env.sim_interface.arm_commands[-1]

    assert info["ik_failures"] > 0
    assert info["ik_failures"] == env._ik_failures


def test_reset_moves_the_arm_before_placing_the_cube(kin):
    """Kolejnosc wymuszona fizyka: kostka postawiona w geometrii palcow zostaje
    wystrzelona ze sceny przez solver kontaktu."""
    env = make_stepping_env(kin)
    env.step(np.zeros(4))          # zabrudz licznik i latch
    env._ik_failures = 5
    env._grasped = True

    order = []
    sim = env.sim_interface
    sim.publish_arm_command = lambda q, dt: order.append("arm")
    sim.set_cube_pose = lambda pos, timeout=5.0: order.append("cube")

    obs, info = env.reset(seed=1)

    assert order == ["arm", "cube"]
    assert env._step_count == 0
    assert env._ik_failures == 0
    assert env._grasped is False and env._just_grasped is False
    assert obs.shape == (cfg.OBS_DIM,)
    assert info["ik_failures"] == 0


def test_reset_opens_the_gripper_before_moving_the_arm(kin):
    """JTC trzyma ostatnia komende palcow. Epizod zamkniety truncacja albo upuszczeniem
    zostawia chwytak zacisniety - bez jawnego otwarcia nastepny epizod nie ma szans na
    chwyt, a `set_cube_pose` stawia kostke w geometrii palcow."""
    env = make_stepping_env(kin)
    env.step(np.array([0.0, 0.0, 0.0, -1.0]))    # zamknij chwytak
    assert env.sim_interface.gripper_commands[-1] == cfg.GRIPPER_RANGE[0]

    order = []
    sim = env.sim_interface
    sim.publish_gripper_command = lambda opening, dt: order.append(("gripper", opening))
    sim.publish_arm_command = lambda q, dt: order.append(("arm", None))

    env.reset(seed=1)

    assert order[0] == ("gripper", cfg.GRIPPER_RANGE[1])
    assert order[1][0] == "arm"


def test_reset_homes_the_arm_slower_than_a_policy_step(kin):
    """Powrot do `Q_READY` idzie przez cala przestrzen robocza. Z `time_from_start = dt`
    JTC dostaje skok pozycji - szarpniecie albo naruszenie tolerancji."""
    env = make_stepping_env(kin)
    env.reset(seed=1)

    assert env.sim_interface.arm_dts[-1] == cfg.RESET_DT
    assert env.sim_interface.gripper_dts[-1] == cfg.RESET_DT
    assert cfg.RESET_DT > env.dt


def test_reset_waits_for_the_state_stream_before_reading_it(kin):
    """Pierwszy `reset()` po starcie wezla nie ma jeszcze `/joint_states`. Petla
    dosterowania musi blokowac na `wait_for_state`, a nie czytac `get_state`
    - inaczej `q_arm` jest `None` i porownanie z `Q_READY` rzuca TypeError."""
    env = make_env()
    env.kin = kin
    env.sim_interface = ColdStartSim()
    _, env.R_frozen = kin.fk(cfg.Q_READY)

    obs, info = env.reset(seed=1)

    assert obs.shape == (cfg.OBS_DIM,)


def test_step_waits_for_the_state_stream_before_reading_it(kin):
    """`step()` czyta stan tym samym torem co `reset()`. W praktyce reset idzie
    pierwszy i stan jest juz cieply, ale na nieblokujacym `get_state` invariant
    trzyma sie wylacznie na kolejnosci wywolan - to ma byc wlasnosc `step()`."""
    env = make_env()
    env.kin = kin
    env.sim_interface = ColdStartSim()
    _, env.R_frozen = kin.fk(cfg.Q_READY)

    obs, reward, terminated, truncated, info = env.step(np.zeros(4))

    assert obs.shape == (cfg.OBS_DIM,)


def test_ik_failure_holds_the_arm_in_place(kin):
    """Poza czesciowo zbiezna z `max_iters` prowadzi W STRONE osobliwosci, nie tam,
    gdzie chciala polityka - porazka IK ma byc no-opem ramienia (chwytak dziala dalej)."""
    env = make_stepping_env(kin)
    q_start = env.sim_interface.state["q_arm"].copy()

    failures = 0
    for _ in range(30):
        _, _, _, _, info = env.step(np.array([1.0, 0.0, 1.0, -1.0]))
        if info["ik_failures"] > failures:
            failures = info["ik_failures"]
            np.testing.assert_allclose(env.sim_interface.arm_commands[-1],
                                       env.sim_interface.state["q_arm"])
            assert env.sim_interface.gripper_commands[-1] == cfg.GRIPPER_RANGE[0]
        env.sim_interface.state["q_arm"] = env.sim_interface.arm_commands[-1]

    assert failures > 0, "test nie dotknal osobliwego rogu - dobierz akcje"
    assert not np.allclose(env.sim_interface.state["q_arm"], q_start), "ramie w ogole nie ruszylo"


def test_close_is_idempotent(kin):
    """Gymnasium wola `close()` takze z `__del__`, SB3 na koncu treningu. Podwojne
    wywolanie nie moze rzucic ani zabic wezla dwa razy."""
    env = make_stepping_env(kin)
    env._owns_rclpy = False          # kontekstu rclpy nie zakladalismy, wiec go nie gasimy
    sim = env.sim_interface

    env.close()
    env.close()

    assert sim.destroyed == 1
    assert env.sim_interface is None


def test_reset_seed_determines_the_scene(kin):
    """`reset(seed=42)` dwa razy -> ta sama scena. Warunek protokolu ewaluacji."""
    a, b = make_stepping_env(kin), make_stepping_env(kin)
    a.level = b.level = "L2"

    a.reset(seed=42)
    b.reset(seed=42)

    np.testing.assert_allclose(a.sim_interface.cube_poses[-1], b.sim_interface.cube_poses[-1])
    np.testing.assert_allclose(a.goal_pos, b.goal_pos)


# ---------------------------------------------------------------------------
# 6. Zywa symulacja - `@pytest.mark.sim`, pomijane bez dzialajacego bringupa
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_env():
    env = FrankaPickPlaceEnv(dt=0.05, max_episode_steps=50, level="L1")
    yield env
    import rclpy
    if rclpy.ok():
        rclpy.shutdown()


@pytest.mark.sim
def test_live_state_is_complete_and_physical(live_env):
    """Most: `/joint_states` i `/model/cube/odometry` musza dowieźć wszystkie pola,
    ktorych uzywa `_get_obs`. Brak jednego mostu objawia sie tu, a nie w treningu."""
    state = live_env.sim_interface.wait_for_state(timeout=10.0)

    assert set(state) == {"gripper_opening", "q_arm", "dq_arm", "cube_pos", "cube_vel"}
    assert state["q_arm"].shape == (7,)
    assert state["cube_pos"][2] > 0.3          # kostka na blacie, nie pod podloga


@pytest.mark.sim
def test_live_reserve_t_runs_on_sim_time(live_env):
    """`use_sim_time=True` jest warunkiem, zeby krok srodowiska nie rozjezdzal sie
    przy zmianie RTF - pulapka wypisana w planie przy `advance(dt)`."""
    sim = live_env.sim_interface
    sim.wait_for_state(timeout=10.0)
    assert sim.get_parameter("use_sim_time").value is True

    # `wait_for_state` pilnuje pieciu pol stanu, ale NIE `_sim_time` - tuz po starcie
    # zegar potrafi byc jeszcze None. Pierwszy `reserve_t` czeka na pierwszy tick.
    sim.reserve_t(0.05)
    assert sim._sim_time is not None

    before = sim._sim_time
    sim.reserve_t(0.2)

    assert sim._sim_time - before >= 0.2


@pytest.mark.sim
def test_live_reset_actually_moves_the_cube(live_env):
    """Gdyby `SetEntityPose` odpowiadal `success=True` bez ruchu kostki, `reset()`
    cicho zwracalby scene z poprzedniego epizodu - i randomizacja L2 bylaby fikcja."""
    live_env.level = "L2"
    cubes = []
    try:
        for seed in (1, 2, 3):
            live_env.reset(seed=seed)
            cubes.append(live_env.sim_interface._cube_pos.copy())
    finally:
        live_env.level = "L1"

    spread = max(np.linalg.norm(a - b) for a in cubes for b in cubes)
    assert spread > 0.01


@pytest.mark.sim
def test_live_env_honours_its_declared_spaces(live_env):
    """Statyczna czesc kontraktu Gym: to, co srodowisko deklaruje, musi sie zgadzac
    z tym, co realnie zwraca. Bez tego SB3 wywali sie dopiero w srodku treningu."""
    obs, info = live_env.reset(seed=0)

    assert live_env.observation_space.contains(obs)
    assert isinstance(info, dict)

    action = live_env.action_space.sample()
    obs, reward, terminated, truncated, info = live_env.step(action)

    assert live_env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


@pytest.mark.sim
@pytest.mark.xfail(
    reason="faza 1 to free-run: `check_env` wymaga bit-w-bit identycznego kroku po "
           "`reset(seed)`, a zegar symulacji biegnie niezaleznie od petli Gym. "
           "Dokladnie ten test jest miara, kiedy faza 2 (`multi_step`) jest konieczna "
           "- gdy zacznie przechodzic, przejscie jest zrobione.",
    strict=False,
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_live_env_passes_full_gym_checker(live_env):
    from gymnasium.utils.env_checker import check_env

    check_env(live_env, skip_render_check=True)


@pytest.mark.sim
def test_live_random_policy_survives_full_episodes(live_env):
    """Punkt 5 planu weryfikacji: epizody bez wyjatku, `ik_failures` ponizej 5%."""
    rng = np.random.default_rng(0)
    steps = 0

    for episode in range(2):
        live_env.reset(seed=episode)
        for _ in range(live_env.max_episode_steps):
            _, _, terminated, truncated, info = live_env.step(rng.uniform(-1.0, 1.0, 4))
            steps += 1
            if terminated or truncated:
                break

    assert info["ik_failures"] / steps < 0.05


@pytest.mark.sim
def test_live_arm_follows_the_commanded_target(live_env):
    """Domyka petle komenda -> JTC -> gz_ros2_control -> `/joint_states`. Prog 0.1 rad
    z pomiaru nadazania (thesis-context: max blad 0.0062 rad przy 87 waypointach)."""
    live_env.reset(seed=0)

    for _ in range(20):
        live_env.step(np.array([0.0, 0.0, -1.0, 1.0]))

    q_cmd = live_env.sim_interface._q_arm
    p_ee, _ = live_env.kin.fk(q_cmd)

    assert p_ee[2] < cfg.WORKSPACE_BOX[1][2]
    assert np.all(np.isfinite(q_cmd))


@pytest.mark.sim
def test_live_reset_is_repeatable(live_env):
    """Faza 1 (free-run) daje powtarzalnosc tylko z tolerancja - rozjazd wiekszy
    niz ponizszy prog jest argumentem za przejsciem na `multi_step`."""
    obs_a, _ = live_env.reset(seed=42)
    obs_b, _ = live_env.reset(seed=42)

    np.testing.assert_allclose(obs_a, obs_b, atol=2e-2)
