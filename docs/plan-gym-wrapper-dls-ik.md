# Plan: wrapper Gymnasium ↔ ROS 2 + DLS-IK (`franka_rl`)

## Kontekst

Środowisko symulacyjne działa: FR3 spawnuje się w `fr3_world.sdf` (stół + kostka), trzy kontrolery
są `active`, chwyt i podniesienie kostki zweryfikowane ręcznie przez `rqt_joint_trajectory_controller`.
Docker build naprawiony (pin `setuptools<81` — `rosdep` wymaga `pkg_resources`).

Brakuje warstwy, która zamienia tę symulację w środowisko uczenia: polityka ma wystawiać
akcję `(Δx, Δy, Δz, g)` i dostawać wektor obserwacji, a nie mówić do JTC bezpośrednio.
To jest krok 4–5 z „Następnych kroków" w `thesis-project-context.md` i wspólny fundament
dla SAC/PPO, Diffusion Policy **i** skryptowanego eksperta — zasada uczciwości porównania
wymaga, żeby wszystkie trzy szły dokładnie tym samym torem sterowania.

Cel: pakiet `franka_rl` z `FrankaPickPlaceEnv(gymnasium.Env)`, który przechodzi
`gymnasium.utils.env_checker` i pozwala uruchomić losową politykę na 100 kroków bez wywrotki.
Uczenie (SAC/PPO) to dopiero kolejny etap — tu budujemy tylko środowisko.

**Uwaga o wykonaniu:** implementacja własna. Plan celowo podaje granice modułów,
sygnatury, kolejność i pułapki — nie gotowy kod.

## Decyzje podjęte przed planem

| Decyzja | Wybór | Konsekwencja |
|---|---|---|
| Kinematyka | **Pinocchio** | `pin.computeFrameJacobian`, zero własnego wyprowadzania FK |
| Orientacja w IK | **Pełne 6D DLS z zamrożoną R** | jakobian 6×7, orientacja nie dryfuje przez epizod |
| Krokowanie sim | **Etapowo: najpierw free-run** | faza 1 działa szybko, faza 2 podmienia `_advance()` |

Odchylenie od `thesis-project-context.md`: dokument mówił „tylko wiersze pozycyjne jakobianu".
Odrzucone — przy 7 DOF i 3 więzach nadmiarowe stopnie swobody pozwalają orientacji dryfować
przez epizod, chwytak przestaje patrzeć w dół i chwyt się rozjeżdża. Do zaktualizowania
w dokumencie przy okazji.

---

## Krok 0 — weryfikacja założeń (15 min, przed pisaniem czegokolwiek)

Blokujące, jeśli którekolwiek padnie:

1. **Pinocchio jest w obrazie:**
   ```bash
   docker exec franka_sim python3 -c "import pinocchio; print(pinocchio.__version__)"
   ```
   Jeśli brak → `apt-get install -y ros-jazzy-pinocchio` (dopisać do Dockerfile, sekcja system deps).
2. **Nazwa frame'u końcówki.** Wygeneruj URDF i sprawdź, co faktycznie istnieje:
   ```bash
   docker exec franka_sim bash -c "ros2 run xacro xacro /ws/src/franka_sim/urdf/fr3_gazebo.urdf.xacro > /tmp/fr3.urdf; grep -o 'link name=\"[^\"]*\"' /tmp/fr3.urdf"
   ```
   Spodziewane: `fr3_hand_tcp` (punkt między palcami) — to ma być frame IK, **nie** `fr3_link8`.
3. **Pinocchio parsuje nasz URDF i widzi ten frame** (`model.getFrameId(...)` ≠ `model.nframes`).
   Uwaga: model buduje się z URDF **po** `strip_finger_mimic` i zawiera `world` + fixed joint,
   więc `model.nq` = 9 (7 ramienia + 2 palce), a nie 7. Indeksowanie stawów rób przez
   `model.getJointId(name)`, nigdy przez pozycję w tablicy.

## Krok 1 — szkielet pakietu `franka_rl` (ament_python)

Pliki: `src/franka_rl/{package.xml,setup.py,resource/franka_rl,franka_rl/__init__.py}`.
Wzoruj się na `franka_sim/package.xml`, ale buildtool → `ament_python`.
Zależności w `package.xml`: `rclpy`, `sensor_msgs`, `trajectory_msgs`, `tf2_msgs`,
`ros_gz_interfaces`, `pinocchio`.

Twórz pliki **z hosta** (reguła 8 z CLAUDE.md — inaczej colcon zrobi je root-owe).

Docelowa struktura:
```
franka_rl/
├── kinematics.py     # czysty Python + pinocchio, ZERO ROS
├── ik.py             # DLS + clip do workspace, ZERO ROS
├── ros_bridge.py     # cała rozmowa z ROS/Gazebo
├── gym_env.py        # FrankaPickPlaceEnv — skleja powyższe
└── test/             # pytest, uruchamialne bez Gazebo
```
Podział jest celowy: `kinematics.py` i `ik.py` muszą dać się testować bez odpalania symulacji.
To one są „modułem testowalnym jednostkowo" z thesis-context.

## Krok 2 — `kinematics.py`

Klasa `FrankaKinematics(urdf_string, ee_frame="fr3_hand_tcp")`:
- `fk(q_arm) -> (p, R)` — pozycja 3D + macierz rotacji 3×3 końcówki w ramce `base`
- `jacobian(q_arm) -> J` — 6×7, `pin.ReferenceFrame.LOCAL_WORLD_ALIGNED`
  (prędkości wyrażone w osiach świata, punkt odniesienia w EE — to konwencja,
  w której błąd pozycji liczony jako `p_des - p_cur` jest bezpośrednio kompatybilny z `J`)
- wewnętrznie: mapowanie 7 kątów ramienia → pełny wektor `q` modelu (palce ustaw na stałe)

**Pułapka:** `computeFrameJacobian` wymaga wcześniejszego `pin.framesForwardKinematics`.
Nie licz na to, że stan `data` z poprzedniego wywołania jest aktualny.

## Krok 3 — `ik.py`

Dwie czyste funkcje, bez stanu:

```python
def dls_step(J, err6, lam=0.05) -> dq        # dq = Jᵀ(JJᵀ + λ²I)⁻¹ · err6
def pose_error(p_cur, R_cur, p_des, R_des) -> err6
```
- Błąd orientacji: `R_err = R_des @ R_cur.T`, potem log SO(3) → wektor 3D
  (`pin.log3` — nie wymyślaj tego sam, to gotowe i przetestowane).
- Skalowanie członu orientacji osobną wagą (`w_rot`, start ~1.0) — pozycja i orientacja
  mają różne jednostki, jedna gałka pozwala regulować, jak agresywnie prostować chwytak.

Plus dwie funkcje pomocnicze:
```python
def clip_to_workspace(p, box) -> p           # box: x∈[-0.15,0.7], y∈[-0.6,0.6], z∈[0.42,0.65]
def clip_to_joint_limits(q, lower, upper) -> (q_clipped, hit_limit: bool)
```

**Kontrakt IK z resztą systemu:** funkcja wyższego rzędu w `gym_env` robi
`p_des = clip_to_workspace(p_cur + Δ)` → `err6` → `dq` → `q_cmd = q + dq` →
`clip_to_joint_limits`. Jeśli `hit_limit` albo `‖dq‖` przekracza próg (bliskość osobliwości)
→ **no-op** (nie wysyłaj komendy) + inkrementacja licznika `ik_failures`.
Ten licznik idzie do `info` — thesis-context wprost chce go raportować jako statystykę
porównawczą DP vs RL.

**Testy (pytest, bez Gazebo)** — to jest miejsce, gdzie TDD faktycznie się opłaca:
- FK dla pozy `ready` daje sensowną pozycję (z > 0.4, x > 0.2)
- jakobian numeryczny (różnice skończone po `fk`) ≈ analityczny z pinocchio, tol. 1e-5
- pętla: 50 iteracji DLS do celu oddalonego o 5 cm zbiega poniżej 1 mm
- `clip_to_workspace` faktycznie przycina poza boxem
- osobliwość (staw 4 wyprostowany, `q4 ≈ 0`) nie wysadza `dq` do nieskończoności

## Krok 4 — mosty Gazebo w `bringup.launch.py`

Do istniejącego `clock_bridge` dołóż (ten sam wzorzec `parameter_bridge`):

| Temat / serwis | Typ | Po co |
|---|---|---|
| `/world/fr3_world/dynamic_pose/info` | `tf2_msgs/msg/TFMessage[gz.msgs.Pose_V` | poza kostki (oracle) |
| `/world/fr3_world/set_pose` | `ros_gz_interfaces/srv/SetEntityPose` | reset kostki |
| `/world/fr3_world/control` | `ros_gz_interfaces/srv/ControlWorld` | faza 2: multi_step |

Serwisy w `parameter_bridge` idą przez plik konfiguracyjny YAML, nie przez argumenty
pozycyjne jak tematy — sprawdź składnię dla Jazzy przed pisaniem.
Po edycji launch → **rebuild** `franka_sim` (reguła 2 z CLAUDE.md).

Sanity check przed pisaniem env:
```bash
ros2 topic echo /world/fr3_world/dynamic_pose/info --once   # czy kostka tam jest
ros2 service list | grep fr3_world
```

## Krok 5 — `ros_bridge.py`

Klasa `SimInterface(Node)` — jedyne miejsce, gdzie w ogóle występuje `rclpy`:
- sub `/joint_states` → cache `q`, `dq` (mapowane **po nazwach**, nie po kolejności —
  `JointState` nie gwarantuje uporządkowania)
- sub `dynamic_pose/info` → cache pozy kostki (szukaj po `child_frame_id == "cube"`)
- pub `/fr3_arm_controller/joint_trajectory` — jednopunktowa trajektoria, `time_from_start = dt`
- pub `/fr3_gripper_controller/joint_trajectory` — oba palce jawnie, ta sama wartość
- klient `SetEntityPose` (reset kostki)
- `advance(dt)` — **jedyny** punkt sterowania czasem. Faza 1: pętla
  `spin_once` aż `/clock` przesunie się o `dt`. Faza 2: podmiana na `ControlWorld(multi_step=N)`,
  reszta env bez zmian.

**Pułapka:** `use_sim_time=True` na tym węźle, inaczej `advance()` mierzy czas ścienny
i przy zmianie RTF wszystko się rozjeżdża.

**Pułapka:** JTC odrzuca trajektorię z `time_from_start = 0`. Przy dt = 0.05 s to nie problem,
ale nie schodź niżej bez sprawdzenia.

## Krok 6 — `gym_env.py`

`FrankaPickPlaceEnv(gymnasium.Env)`, `dt = 0.05` (20 Hz), `max_episode_steps ≈ 200` (10 s).

**Action space:** `Box(-1, 1, (4,))`. `a[:3] × 0.05 m` = Δpozycja, `a[3] > 0` → chwytak zamknięty
(binarnie z progiem — FurnitureBench; wartość „zamknięte" to jednocześnie gałka docisku
z pkt. 5 sekcji o chwytaku w thesis-context: kostka 5 cm → palec nominalnie 0.025 m,
komenda niższa = przesterowanie = większa siła normalna. Start: 0.020).

**Observation space:** `Box(-1, 1, (N,))`, wszystko znormalizowane. Zawartość wg thesis-context:
poz. EE (3) + orientacja 6D (6) + rozwarcie chwytaka (1) + wektor EE→kostka (3) +
wektor kostka→cel (3) [+ opcjonalnie q (7)]. Względne wektory, nie absolutne pozycje — jak panda-gym.
Statystyki normalizacji trzymaj w **jednym module**, bo DP i SB3 muszą użyć identycznych.

**`reset(seed)`:** cel → poza domowa ramienia przez JTC + `advance` aż `‖q − q_home‖` mała;
kostka → `SetEntityPose` na pozycję z `self.np_random` (L1: stała; L2: losowa w boxie).
Kolejność ma znaczenie: najpierw odsuń ramię, potem stawiaj kostkę.

**`step(a)`:** clip akcji → IK (krok 3) → publikacja JTC → `advance(dt)` → obserwacja → reward.

**Reward (shaped, wg thesis-context):** `−‖p_ee − p_cube‖` + bonus za chwyt +
`−‖p_cube − p_goal‖` + bonus sukcesu + `−α‖a‖²`. Trzymaj to w osobnej funkcji
`compute_reward(obs_dict, action)` — będzie strojone i musi być identyczne dla wszystkich metod.

**`info`:** `is_success`, `ik_failures`, `grasped` — potrzebne do ewaluacji i do HER.

## Krok 7 — poziomy trudności

Nie rób osobnych klas. Argument `level: Literal["L1","L2","L3"]` w konstruktorze,
sterujący wyłącznie randomizacją w `reset()` (L3 dodatkowo wstrzykuje perturbację w `step`).
Trzy klasy oznaczałyby trzy kopie reward i obserwacji — gwarantowany rozjazd.

---

## Weryfikacja

Kolejno, każdy krok musi przejść przed następnym:

1. **Unit, bez Gazebo:** `docker exec franka_sim bash -c "cd /ws && python3 -m pytest src/franka_rl/test -v"`
2. **Smoke IK na żywo:** bringup + skrypt, który wysyła 20 kroków `Δ = (0, 0, −0.01)`;
   `gz model -m fr3 -p` ma dalej pokazywać pozę bazy `0 0 0` (robot nie przewrócony),
   a EE ma zjechać o ~20 cm.
3. **Zamrożenie orientacji:** po 100 losowych krokach kąt między osią Z chwytaka a `(0,0,−1)`
   < 5°. To jest test, który wyłapie błąd znaku w `pose_error` — najbardziej prawdopodobną
   pomyłkę w całym zadaniu.
4. **Kontrakt Gym:** `gymnasium.utils.env_checker.check_env(env)` bez błędów.
5. **Losowa polityka:** 5 epizodów × 200 kroków bez wyjątku, bez wywrotki robota,
   `info["ik_failures"] / kroki` < 5%.
6. **Determinizm resetu:** `reset(seed=42)` dwa razy → identyczna obserwacja startowa
   (bit-w-bit przy fazie 2; przy fazie 1 free-run z tolerancją — i to jest właśnie
   argument, żeby przejść na `multi_step`).

## Czego świadomie NIE robimy teraz

- Strojenia tarcia chwytu — odłożone do momentu, gdy polityka realnie szarpie kostką
  (decyzja z thesis-context, pkt 5). Dopiero test 5 pokaże, czy jest problem.
- Skryptowanego eksperta i zbierania demonstracji — osobny etap, ale ma korzystać
  z **tego samego** `step()`, nie z własnej ścieżki do JTC.
- SAC/PPO — dopiero po zielonej weryfikacji 1–6.
