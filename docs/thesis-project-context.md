# Praca Inżynierska — Kontekst Projektu

## Informacje ogólne

- **Autor:** Wiktor, student ostatniego roku Automatyki i Robotyki, Politechnika Wrocławska
- **Rok akademicki:** 2025/2026
- **Rodzaj pracy:** Eksperymentalna (praca inżynierska)
- **Tytuł PL:** Porównanie uczenia ze wzmocnieniem i polityki dyfuzyjnej w generowaniu trajektorii manipulatora
- **Tytuł EN:** Comparison of Reinforcement Learning and Diffusion Policy in Manipulator Trajectory Generation

---

## Cel pracy

Systematyczne porównanie dwóch podejść do generowania trajektorii manipulatora:
- **Reinforcement Learning:** SAC (Soft Actor-Critic) i PPO (Proximal Policy Optimization)
- **Diffusion Policy:** trening per-task od zera na własnym zbiorze demonstracji (Chi et al., 2023)
  — **nie** finetuning pretrenowanego modelu, patrz „Korekta terminologii" w sekcji o DP

Zadanie: **Pick & Place** w trzech wariantach trudności:
1. **L1 — Stała scena:** pozycja obiektu i celu stała
2. **L2 — Randomizacja:** losowa pozycja obiektu w workspace
3. **L3 — Perturbacje:** L2 + dynamiczne zakłócenia w trakcie ruchu

Metryki porównawcze: skuteczność zadania (success rate), efektywność próbkowa, jakość trajektorii (smoothness), odporność na zakłócenia.

---

## Stack technologiczny

| Komponent | Technologia |
|---|---|
| Symulacja | Gazebo Harmonic (Gz Sim 8) — headless (`gz sim -s`) |
| Framework robotyczny | ROS 2 Jazzy (Ubuntu 24.04) |
| Manipulator | Franka FR3 (wcześniej planowana Panda/FER, zmieniono na FR3) |
| Sterowanie | ros2_control + gz_ros2_control |
| RL | stable-baselines3 (SAC, PPO) |
| Diffusion Policy | PyTorch + diffusers |
| Tracking | Weights & Biases |
| Konteneryzacja | Docker + Docker Compose |
| GPU | RTX 4060 (PC), zintegrowana grafika (laptop) |

---

## Harmonogram (10 tygodni)

| Tygodnie | Kamień milowy |
|---|---|
| 1–2 | Przegląd literatury + konfiguracja środowiska — **zamknięte** |
| 3–4 | Implementacja środowiska symulacyjnego (scena P&P, węzły ROS 2, 3 poziomy trudności) ← **tu jesteśmy** |
| 5–6 | Implementacja i trening SAC oraz PPO |
| 7–8 | Zbiór demonstracji + trening Diffusion Policy |
| 9–10 | Eksperymenty porównawcze + pisanie pracy |

---

## Ustalenia projektowe — obserwacje, akcje, sterowanie (research 07.2026)

### Przestrzeń obserwacji (wspólna dla SAC, PPO i DP)

Wariant **state-based** (oracle state z Gazebo), wektor ~20–30 wymiarów:

| Składnik | Wymiar | Uwagi |
|---|---|---|
| Pozycja końcówki (EE) | 3 | w bazie robota |
| Orientacja EE | 4 lub 6 | kwaternion albo reprezentacja 6D (Zhou 2019) — spójnie u obu metod |
| Rozwarcie chwytaka | 1 | |
| Pozycja obiektu (kostka) | 3 | rozważyć **względnie**: wektor EE→obiekt (jak panda-gym) |
| Pozycja celu | 3 | rozważyć względnie: wektor obiekt→cel |
| (opcjonalnie) kąty stawów | 7 | |
| (opcjonalnie) prędkości stawów | 7 | |

- Normalizacja wszystkiego do [-1, 1] per wymiar — jedna konwencja w obu pipeline'ach (DP: statystyki datasetu; SB3: `VecNormalize` lub ręcznie).
- DP przyjmuje historię obserwacji (T_o = 2), SAC/PPO pojedynczy stan — przy pełnym stanie markowskim nie psuje porównania, odnotować w pracy.
- Wariant vision-based jawnie odrzucony (koszt obliczeniowy, poza harmonogramem) → future work.

### Przestrzeń akcji (wspólna dla SAC, PPO, DP i skryptowanego eksperta)

**4D delta-EE, orientacja zamrożona (chwytak pionowo w dół):**

```
a = (Δx, Δy, Δz, g) ∈ [-1, 1]⁴
```

- Skalowanie: max **5 cm/krok** przy polityce **10–20 Hz** (≈ max 1 m/s EE) — limit bezpieczeństwa + identyczna dynamika dla wszystkich metod.
- Chwytak `g`: ciągłe wyjście sieci, interpretacja **binarna z progiem** (wzór: FurnitureBench), żeby polityka nie trzepotała chwytakiem.
- Uzasadnienie delta-EE: task space = akcje w przestrzeni zadania, wyższa efektywność próbkowa (Matas 2018, Martín-Martín 2019, Zhu 2020); position control > velocity control dla DP (Chi 2023).
- **Opcja zapasowa: delta joint position (7D)** — zero IK, brak problemu osobliwości; literatura pokazuje że bywa lepsza (Effective Tuning Strategies, arXiv:2410.01220 — delta-EE często narusza ograniczenia IK).

### Tor sterowania

```
Polityka (ΔEE @ 10–20 Hz)
  → clip akcji + clip do workspace box (x∈[-0.15,0.7], y∈[-0.6,0.6], z∈[0.42,0.65])
  → IK: damped least-squares na jakobianie (pinocchio), q̇ = Jᵀ(JJᵀ+λ²I)⁻¹·err
  → joint_trajectory_controller (jednopunktowa trajektoria, time_from_start ≈ dt)
  → gz_ros2_control (interfejs pozycyjny) → Gazebo
  → obserwacje wracają do polityki
```

- IK: własne DLS zamiast MoveIt Servo (deterministyczne, testowalne, bez węzłów MoveIt w pętli treningu).
- **Jakobian 6×7, nie 3×7 (decyzja skorygowana 08.2026).** Wcześniejszy zapis mówił „przy
  zamrożonej orientacji tylko wiersze pozycyjne jakobianu" — **odrzucone**. Przy 7 DOF i tylko
  3 więzach pozycyjnych nadmiarowe stopnie swobody pozwalają orientacji dryfować przez epizod:
  chwytak przestaje patrzeć pionowo w dół i chwyt się rozjeżdża. „Zamrożona orientacja" musi być
  **aktywnie regulowana**, nie tylko pominięta. Błąd 6D = `[p_des − p_cur ; log3(R_des · R_curᵀ)]`,
  człon obrotowy z osobną wagą `w_rot`. Frame IK: `fr3_hand_tcp` (punkt między palcami),
  jakobian w `pin.ReferenceFrame.LOCAL_WORLD_ALIGNED`.
- **Kinematyka: pinocchio**, nie własne wyprowadzenie FK ani KDL (`kdl_parser_py` bywa
  niedostępny w Jazzy). Praca opisuje metodę DLS, nie wyprowadzenie kinematyki FR3.
  Uwaga implementacyjna: model budowany z URDF **po** `strip_finger_mimic`, zawiera link `world`
  + fixed joint, więc `model.nq` = 9 (7 ramienia + 2 palce) — indeksować przez
  `model.getJointId(name)`, nigdy pozycyjnie.
- **Fakty o modelu pinocchio** zbudowanym z naszego URDF (po `strip_finger_mimic`) —
  zmierzone, nie założone; każdy wiersz zmienia implementację DLS:

  | Fakt | Wartość | Konsekwencja |
  |---|---|---|
  | `model.nq` | **9** (7 ramienia + 2 palce) | `world` + fixed joint nie wnoszą DOF; indeksuj przez `model.getJointId(name)`, nigdy pozycyjnie |
  | Frame IK | `fr3_hand_tcp` (id 29 z 54) | **nie** `fr3_link8` |
  | Jakobian z `computeFrameJacobian` | **6×9** | kolumny palców trzeba odciąć przed DLS, inaczej `JJᵀ` jest osobliwe |
  | `fr3_joint4` limity | `[-3.077, -0.117]` | `q4 = 0` jest **poza** zakresem — test osobliwości łokcia rób na `-0.117` |
  | FK w pozie ready | `p = [0.307, 0.0, 0.487]`, oś Z TCP = `[0,0,-1]` | poza ready już ma chwytak pionowo w dół → `R_des` bierz wprost z `FK(q_ready).rotation` |

  Pinocchio 4.0.0 wchodzi do obrazu jako zależność tranzytywna `ros-jazzy-moveit`, nie jest
  instalowane jawnie — sourcowanie `/opt/ros/jazzy/setup.bash` wystarcza, by go zaimportować.
- Porażka IK (osobliwość/limit stawu) = no-op + ewentualna mała kara; **logować częstość**
  (ciekawa statystyka porównawcza DP vs RL) — licznik `ik_failures` w `info` wrappera.
- JTC zamiast forward_position_controller — interpolacja między komendami = gładszy ruch, istotne przy metryce smoothness.
- **Zasada uczciwości porównania:** identyczny action space, identyczny kontroler i konfiguracja dla RL, DP **i eksperta zbierającego demonstracje** (ekspert nagrywa sekwencje (obs, ΔEE, g) wykonywane tym samym torem — NIE surowe plany MoveIt).

### Chwytak — ostrzeżenia praktyczne (decyzja podjęta)

1. Mimic joints w gz_ros2_control/DART **niewspierane** (potwierdzone) — `fr3_finger_joint2` nie podąża za `fr3_finger_joint1`.
2. **`DetachableJoint` przetestowany i odrzucony jako niekompatybilny z `gz_ros2_control`
   (gz-sim 8.11.0, ROS Jazzy vendor).** Mechanizm sam w sobie działa poprawnie
   (zweryfikowane na izolowanym minimalnym świecie: default-attach, `detach`,
   re-`attach` via topic — wszystko 1:1 zgodne z oczekiwaniami). Problem: gdy
   `parent_link` znajduje się na łańcuchu stawów aktuowanych przez
   `gz_ros2_control` (position command interface), rzeczywisty ruch ramienia NIE
   jest respektowany przez sztywne ograniczenie `DetachableJoint` — przyczepiony
   obiekt nie podąża za ruchem (zweryfikowane na `fr3_link1` i `fr3_link7`;
   działa tylko przy sztywnym teleportowaniu całego modelu, co nie ma zastosowania
   przy realnym sterowaniu). Wniosek: `gz_ros2_control`'s pozycyjne komendy
   najpewniej nie przechodzą przez pełny solver dynamiki zgodny z dodatkowymi
   (closed-loop) ograniczeniami.
3. **Ostateczna decyzja: Opcja A** — jawne sterowanie oboma palcami
   (`fr3_finger_joint2` jako pełnoprawny `command_interface`/`state_interface` w
   `ros2_control`, dopisany do `fr3_gripper_controller` w `controllers.yaml`).
   Wymagało dodatkowo usunięcia znacznika URDF `<mimic>` z `fr3_finger_joint2`
   (hardkodowany w `franka_hand.xacro`, brak parametru do wyłączenia) —
   `ros2_control` odmawia `command_interface` na mimic joint. Rozwiązane przez
   post-processing wygenerowanego URDF w `bringup.launch.py`
   (`strip_finger_mimic`, `xml.etree.ElementTree`) zamiast patchowania
   `franka_description`.
4. **Baza przytwierdzona do świata** (`<link name="world"/>` + fixed joint
   `world_to_base` w `fr3_gazebo.urdf.xacro`, xyz 0 0 0). Bez tego model jest
   w Gazebo free-floating i siły reakcji przy chwycie kostki przewracają całego
   robota (zaobserwowane przy pierwszych próbach chwytu z
   `rqt_joint_trajectory_controller`). Odpowiada realnemu FR3 przykręconemu do
   stanowiska; zero wpływu na action space i przestrzeń obserwacji.
5. **Fizyka chwytu — stan i pułapki (strojenie ODŁOŻONE).** Chwyt i podniesienie
   kostki zweryfikowane ręcznie (`rqt_joint_trajectory_controller`). Strojenie
   tarcia świadomie odłożone do momentu, gdy istnieje wrapper Gym: statyczne
   podniesienie nie testuje poślizgu, sensownym testem jest dopiero ruch z
   polityki (5 cm/krok, przyspieszenia, szarpnięcia). Przy okazji ustalone:
   - **DART ignoruje `<contact><ode>`** — `kp`, `kd`, `min_depth` w
     `fr3_world.sdf` to parametry ODE, silnik (`gz-physics-dartsim`) ich nie
     czyta. Z całej sekcji `<surface>` działa wyłącznie
     `<friction><ode><mu>/<mu2>`. Nie tracić czasu na kręcenie tymi gałkami.
   - **Tarcie liczy się z pary kontaktowej**, nie z jednego kształtu. Kostka ma
     `mu=1.0`, ale palce (`fr3_leftfinger` / `fr3_rightfinger` z
     `franka_description`) nie mają zdefiniowanego tarcia — biorą domyślne.
     Podbicie samej kostki daje ograniczony efekt. Docelowo bloki
     `<gazebo reference="fr3_leftfinger">` z `<mu1>`/`<mu2>` w naszym
     `fr3_gazebo.urdf.xacro` — **nie** patchujemy `franka_description` (ta sama
     zasada co przy `strip_finger_mimic`). Nazwy linków potwierdzić w
     wygenerowanym URDF przed implementacją.
   - **Drugi lever obok tarcia: siła normalna.** Siła tarcia = `mu` × siła
     normalna, a ta przy sterowaniu pozycyjnym bierze się z „przesterowania"
     komendy palców (kostka 5 cm → palec nominalnie 0.025 m; komenda niższa =
     docisk). Do rozstrzygnięcia razem z progiem binarnym `g` w action space —
     jaką konkretnie wartość zamknięcia wysyła polityka.
   - Przy finalnym strojeniu **odnotować w pracy**, czy użyte `mu` jest
     fizycznie uzasadnione (okładziny palców FR3 na plastiku), czy potraktowane
     jako czysty parametr symulacji.

### Reward i RL — ustalenia

- Sparse reward dla czystego SAC/PPO nierozwiązywalny w budżecie → **shaped reward** dla obu algorytmów: kara odległości EE–obiekt + bonus za chwyt + kara odległości obiekt–cel + bonus sukcesu + mała kara ‖a‖².
- Kara ‖a‖² w nagrodzie RL: tak (standard, cytat panda-gym/Fetch). **Żadnych filtrów dolnoprzepustowych na akcjach** u żadnej metody — smoothness raportowana z surowych trajektorii.
- SAC+HER możliwy jako eksperyment dodatkowy (tylko off-policy; SB3 `HerReplayBuffer`). PPO nie wspiera HER.
- Oczekiwanie: SAC 5–10× efektywniejszy próbkowo niż PPO; PPO może nie zdążyć na L2/L3 — to też jest wynik (metryka efektywności próbkowej).
- Curriculum: trening L2 startujący z wag L1 — element metodologii.

### Diffusion Policy — ustalenia

- Wariant **CNN (U-Net 1D + FiLM)**, nie Transformer (łatwiejszy tuning wg autorów).
- Trening: DDPM ~100 kroków; inferencja: **DDIM ~10 kroków** (inaczej nie zmieści się w częstotliwości sterowania). Latencja inferencji DP vs MLP = dodatkowa metryka.
- Chunking: T_o = 2, T_p = 16, T_a = 8 (wartości z papieru, do ablacji).
- **Korekta terminologii w pracy:** oryginalny DP trenowany per-task od zera na 50–200 demonstracjach — nie "finetuning pretrenowanego modelu" (to domena VLA typu Octo/π₀). U nas: trening od zera na własnym datasecie.
- Demonstracje: **scripted expert** (sekwencja waypointów przez IK / MoveIt), cel: 100–200 udanych demo na L2, zapis w formacie zgodnym z wrapperem Gym. Uwaga: scripted expert = demonstracje unimodalne → osłabia atut multimodalności DP, uczciwie przedyskutować (→ Mandlekar 2021).

### Ewaluacja i eksperymenty — ustalenia

- Protokół: ≥50 epizodów testowych × ≥3 seedy treningowe, średnia ± odchylenie, identyczne ziarna randomizacji sceny dla wszystkich metod.
- Smoothness: całka z jerku / suma kwadratów przyspieszeń stawów + długość ścieżki EE.
- Efektywność próbkowa — dwie osie: kroki środowiska (RL) vs koszt demonstracji (DP); raportować obie.
- **Hipoteza na L3 (najciekawszy potencjalny wynik):** chunking DP (otwarta pętla przez T_a kroków) vs reaktywność RL co krok — przewaga DP z L1/L2 może stopnieć/odwrócić się przy perturbacjach; ablacja T_a vs smoothness.
- Gazebo ↔ Gym: krokowanie symulacji przez serwis `/world/<name>/control` (`multi_step`) albo pauza+odpauzowanie — bez tego trening niepowtarzalny i wolny. Argument za Gazebo mimo to: integracja ROS 2 + realizm stacku sterowania (MuJoCo/Isaac = standard społeczności RL, odnotować).
- **Wdrożenie krokowania: etapowe (decyzja 08.2026).** Faza 1 — symulacja leci swobodnie,
  wrapper odmierza `dt` z `/clock` (`use_sim_time=True`); pozwala szybko domknąć end-to-end
  i testować IK oraz reward. Faza 2 — podmiana na `multi_step` przez `ros_gz_interfaces/srv/ControlWorld`.
  Warunek projektowy: cały kontakt z czasem schowany za **jedną** metodą `reserve_t(dt)` w warstwie
  ROS, więc faza 2 nie dotyka `gym_env`. Ryzyko fazy 2 do rozstrzygnięcia w implementacji: wyścig
  między publikacją trajektorii a krokiem świata (komenda musi dojść do JTC zanim ruszymy sim).
  Miarą, kiedy faza 2 jest konieczna, jest test determinizmu: `reset(seed=42)` dwa razy → ta sama
  obserwacja startowa.

---

## Struktura repozytorium

Repo: `~/Inżynierka/DiffRL-Panda/`

```
DiffRL-Panda/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── ros_entrypoint.sh
│   └── .env.example
│
├── src/                              # ROS 2 workspace (colcon)
│   ├── franka_sim/                   # Pakiet ament_cmake: scena Gazebo + launch files — ISTNIEJE
│   │   ├── launch/bringup.launch.py
│   │   ├── urdf/fr3_gazebo.urdf.xacro
│   │   ├── worlds/fr3_world.sdf      # ground plane + stół + kostka
│   │   └── config/controllers.yaml
│   │
│   ├── franka_task/                  # Pakiet ament_python: logika Pick & Place
│   │   ├── task_manager.py
│   │   ├── scene_randomizer.py
│   │   └── reward.py
│   │
│   ├── franka_rl/                    # Pakiet ament_python: środowisko Gym + SAC, PPO
│   │   ├── kinematics.py            # FK + jakobian (pinocchio), ZERO ROS
│   │   ├── ik.py                    # DLS + clip workspace/limity, ZERO ROS
│   │   ├── ros_bridge.py            # jedyne miejsce z rclpy; advance(dt)
│   │   ├── gym_env.py               # Wrapper Gymnasium ↔ ROS 2
│   │   ├── test/                    # pytest — uruchamialny bez Gazebo
│   │   ├── train_sac.py
│   │   ├── train_ppo.py
│   │   └── eval.py
│   │
│   └── franka_diffusion/             # Pakiet ament_python: Diffusion Policy
│       ├── data_collector.py
│       ├── dataset.py
│       ├── finetune.py
│       └── eval.py
│
├── evaluation/                       # Automatyczny protokół ewaluacji
├── data/                             # Demonstracje, checkpointy, wyniki (gitignored)
└── docs/
```

Podział na pakiety ROS 2:
- `franka_sim` → `ament_cmake` (launch files, konfiguracja Gazebo, URDF)
- `franka_task`, `franka_rl`, `franka_diffusion` → `ament_python`

---

## Stan implementacji — co działa

### Docker
- **Obraz bazowy:** `ros:jazzy-ros-base` + `ros-jazzy-desktop` (zainstalowane w całości zamiast pojedynczych paczek)
- **Dodatkowe paczki ROS 2:** `ros-jazzy-ros-gz`, `ros-jazzy-gz-ros2-control`, `ros-jazzy-moveit`
- **PyTorch z PyPI** (`pip install torch torchvision`, **bez** `--index-url download.pytorch.org` — CDN PyTorcha blokowany przez sieć, SSLV3_ALERT_HANDSHAKE_FAILURE). PyPI też daje build CUDA; jeden obraz, GPU na PC i CPU na laptopie automatycznie.
- **Pin `setuptools<81`** (warstwa po instalacjach pip). setuptools ≥ 82 nie zawiera już `pkg_resources`, a `/usr/bin/rosdep` go importuje → build wywalał się na `rosdep install` z `ModuleNotFoundError`. Instalacje pip podmieniają systemowy setuptools na najnowszy z PyPI, więc pin **musi** być po nich. W tej samej warstwie asercja `python3 -c "import pkg_resources"`, żeby regresja wywaliła się od razu, a nie 15 minut później.
- **Franka ze źródeł:** klonowane `franka_description` + `franka_ros2` (branch `jazzy`), budowane TYLKO `franka_description` i `franka_msgs` (`--packages-select`)
- **docker-compose:** jeden serwis `sim`, volumes montują `src/`, `evaluation/`, `data/`; `/dev/dri` + rezerwacja GPU nvidia (capabilities `gpu, graphics, display, compute, utility`) — renderowanie GUI przez GPU
- **Entrypoint + `.bashrc`:** sourcowanie ROS i `franka_ws` automatyczne, `docker exec` działa bez ręcznego `source`
- **`GZ_SIM_RESOURCE_PATH` jako `ENV`** w Dockerfile → meshe FR3 ładują się bez eksportu

### Franka FR3 w ROS 2 — środowisko sterowania DZIAŁA
- Własny wrapper `fr3_gazebo.urdf.xacro`: bazowy opis + `<ros2_control>` (position command, position+velocity state) + plugin `gz_ros2_control` + link `world` i fixed joint `world_to_base`
- `bringup.launch.py`: Gazebo headless (`-r -s fr3_world.sdf`) + most `/clock` + `robot_state_publisher` + spawn robota + spawnery kontrolerów sekwencjonowane przez `OnProcessExit`; dodatkowo `strip_finger_mimic` na wygenerowanym URDF
- Robot spawnuje się w pozie **ready**, trzy kontrolery `active`
- Chwyt i podniesienie kostki zweryfikowane ręcznie (`rqt_joint_trajectory_controller`)

### Pętla deweloperska `franka_rl` (pakiet `ament_python`)
- **`--symlink-install` NIE działa dla `ament_python`** (sprawdzone na `franka_rl`: zero
  symlinków w `install/`, edycja w `src/` niewidoczna bez rebuildu). Colcon degraduje się po
  cichu do zwykłej kopii — skutek zmiany trybu editable install w setuptools ≥ 64, bez
  ostrzeżenia. Nie wracać do tej flagi.
- **Rozwiązaniem dla TDD jest pusty `src/franka_rl/conftest.py`.** pytest w trybie importu
  „prepend" wstawia katalog zawierający `conftest.py` na początek `sys.path`, więc
  `import franka_rl` rozwiązuje się do `src/franka_rl/franka_rl/`, a nie do kopii w
  `/ws/install/`. Dzięki temu testy jednostkowe (`kinematics.py`, `ik.py` — zero ROS) lecą bez
  `colcon build` i bez Gazebo. **Plik jest pusty celowo — nie kasować.** `colcon build` jest
  potrzebny dopiero, gdy moduł ma działać jako zainstalowany pakiet (`ros2 run`, launch,
  skrypty treningowe).
- Pliki `__pycache__/`/`.pytest_cache/` tworzone przez `docker exec` są root-owe na hoście
  (ta sama przyczyna co przy colconie) — kasować z kontenera, nie `sudo` z hosta.

### Most serwisu `set_pose` — CLI, nie `config_file` (zmierzone 08.2026)

`ros_gz_bridge` 1.0.22 ma **dwie różne ścieżki kodu** do mostkowania serwisów i wybór między nimi
nie jest kosmetyczny:

- `parameters=[{'config_file': ...}]` → klasa `RosGzBridge`, która trzyma `heartbeat_timer_` (1 Hz)
  wołający `spin()`. Każdy tick ponownie woła `add_service_bridge` i **dopisuje** nowy obiekt do
  `std::vector<rclcpp::ServiceBase::SharedPtr> services_` — nic nie sprawdza, czy most już istnieje.
  Objaw widoczny: log `Creating ROS->GZ service bridge` co dokładnie 1.000 s.
  **Zmierzony skutek: RSS rośnie liniowo ~108 KB/s (1.06 MB / 10 s), czyli ~390 MB/h.** Przy
  wielogodzinnym treningu SAC to realny wyciek, nie hałas w logu.
- `arguments=['<svc>@<ros_srv_type>@<gz_req>@<gz_rep>']` → zwykły `main` w `parameter_bridge`, bez
  heartbeatu. Log leci raz, RSS płaski (zmierzone: 51036 KB stałe przez 50 s).

Używamy wariantu CLI. `ros2 run ros_gz_bridge parameter_bridge --help` potwierdza wsparcie dla
serwisów w CLI — mit, że serwisy wymagają YAML-a, jest fałszywy. Zweryfikowane funkcjonalnie:
`ros2 service call .../set_pose` → `success=True`, kostka faktycznie przeskakuje `(0.5,0)→(0.45,0.15)`
potwierdzone na `/model/cube/odometry`. `config/bridge.yaml` jest po tej zmianie martwy.

### Gazebo Harmonic
- Headless działa: `gz sim -s -r <world>`
- **GUI działa** — mit „Wayland to uniemożliwia" okazał się fałszywy, brakowało wyłącznie `xhost +local:docker` na hoście (XWayland). Klient `gz sim -g` dołącza do headless serwera z bringupa. GUI = podgląd/debug, trening zawsze headless.
- Scena `fr3_world.sdf`: ground plane + stół główny (0.6×0.8×0.4, static, pose `0.5 0 0.2`) +
  dwa stoły docelowe (0.25×0.25×0.4, static, pose `0 ±0.525 0.2`) + kostka 5 cm (masa 0.05 kg, `mu=1.0`)

### Cel zadania — dwa boczne stoły zamiast punktu w powietrzu (decyzja 08.2026)

Cel = środek blatu jednego z dwóch bocznych stołów: `(0, ±0.525, 0.425)`. Stoły stoją po lewej i
prawej stronie robota (90° w Z względem stołu głównego), robot jest w środku trójkąta.

- **Odrzucone: cel w powietrzu** (Fetch / panda-gym). Formalnie działa — sukces sprawdza się w
  chwili, gdy robot trzyma kostkę w tolerancji, epizod się kończy. Ale to zadanie „przynieś obiekt
  do pozy", nie „odłóż"; słabo broni się w opisie ewaluacji pracy o pick & place.
- **Odrzucone: podest wyższy od blatu.** Miał wymuszać podniesienie pionowym stopniem. Zbędny —
  szczelina daje to samo, a każdy centymetr wysokości to ryzyko kolizji dla DLS-IK, który nie
  unika przeszkód.
- **Wybrane: szczelina zamiast wysokości.** Boczne stoły są w `x∈[-0.125, 0.125]`, główny zaczyna
  się na `x=0.2` → przerwa ~7.5 cm nad podłogą przy kostce 5 cm. Popychanie nie ma rozwiązania:
  kostka spada, zamiast wjechać na stół docelowy. Degeneracja „pick & place → push" znika bez
  żadnych sztuczek w reward. Sukces jest sprawdzalny w stanie ustalonym (kostka *leży*, `‖v‖≈0`),
  co jest mocniejszą definicją niż migawka w powietrzu.
- **Dwa stoły, nie jeden** — wymusza ruch w obie strony (joint1 ±90°), daje symetryczne pokrycie
  workspace i bimodalny rozkład celów (atut dla DP, uczciwy test dla SAC). Jeden stół pozwalałby
  nauczyć się stałej trajektorii, co osłabiałoby wymowę L2.
- Rozkład celów wg poziomu: **L1** cel stały (lewy), **L2/L3** losowo 50/50 lewy/prawy.
  Rozkład celów jest dwupunktowy, ale wektor `cube→goal` w obserwacji pozostaje ciągły, bo
  losowana jest pozycja startowa kostki (`x∈[0.4,0.6]`, `y∈[-0.2,0.2]`).
- Jitter pozycji na blacie docelowym świadomie pominięty: stół 0.25 m przy kostce 5 cm daje mały
  zapas od krawędzi, a bimodalny cel już realizuje sens L2.

**Konsekwencje do zweryfikowania w symulacji (nie policzone, tylko oszacowane):**
- **Zamrożona orientacja przy 90° — ZWERYFIKOWANE, zapas duży (08.2026).** Test kinematyczny
  (DLS-IK, marsz po 5 cm z `q_init` z poprzedniego kroku — tak jak zrobi to polityka):
  oba cele `(0, ±0.525, 0.425)` osiągalne z błędem **0.50 mm**, cała strefa spawnu kostki
  (9 punktów) ≤ 0.61 mm, pełny transport (chwyt → lift → przeniesienie → odłożenie) w obu
  kierunkach bez potknięcia. **0 porażek IK na 154 waypointów.** Błąd orientacji ≤ **0.01°** —
  potwierdza sens decyzji o jakobianie 6×7 (orientacja realnie regulowana, nie dryfuje).
  joint7 wykorzystuje `-0.601 … +2.172` przy limitach `±3.051`; najmniejszy margines do limitu
  ze wszystkich 7 stawów to **0.720 rad (~41°)**. Wcześniejsze oszacowanie `π/4 ± π/2` było
  pesymistyczne. **Wariant awaryjny z 5D więzami (yaw swobodny) nie jest potrzebny.**
  Zastrzeżenie: to czysta kinematyka TCP — model kolizyjny nie był ładowany do pinocchio, więc
  test **nie** mówi nic o kolizjach ze stołami, autokolizjach ani o nadążaniu JTC.
- **Dystans transportu ~0.73 m** (kostka `(0.5,0)` → cel `(0,±0.525)`). Przy 5 cm/krok min. ~14
  kroków samego przenoszenia; `max_episode_steps=200` starczy. Zadanie wyraźnie trudniejsze niż
  typowe panda-gym (~0.15 m) → strojenie `mu` palców przestaje być odkładalne.
- **Kostka może spaść w szczelinę** (pod spodem jest podłoga `z=0`). W `step()` przerywać epizod
  przy `cube_z < 0.35` (`terminated=True`, bez bonusu), inaczej agent ciągnie 200 kroków stanu bez
  powrotu i zaśmieca replay buffer.
- **Narożniki stołu głównego przy `y=±0.4`** są blisko bocznych stołów; przedramię przechodzi nad
  tym rejonem przy joint1=±90°. Stół zwężono z 1.0 na 0.8 w `y` profilaktycznie.
  **ZAMKNIĘTE — kolizji nie ma (08.2026).** Przejazd 87 waypointów przez JTC w Gazebo (44 s,
  pełna sekwencja: chwyt w rogu strefy spawnu → lift → transport na przeciwległy stół →
  odłożenie → powrót, oba kierunki, joint1 przez ±90°): **max błąd nadążania 0.0062 rad (0.35°)**
  na 62997 próbkach `/fr3_arm_controller/controller_state`. Kolizja ze statycznym stołem
  objawiłaby się trwałym rozjazdem komenda ↔ stan, bo stół nie ustąpi. Test nie jest pusty —
  ramię faktycznie dojechało do ostatniego waypointu (`TCP = [0, -0.5246, 0.5748]`, błąd 0.47 mm).
  Metoda przydatna ponownie przy każdej zmianie geometrii sceny: mierz `error.positions`
  z `controller_state`, próg podejrzenia ~0.1 rad.

---

## Znane problemy i TODO

### Otwarte
1. **Strojenie fizyki chwytu** — świadomie odłożone do momentu, gdy istnieje wrapper Gym
   (szczegóły i pułapki: sekcja „Chwytak", pkt 5). Sensownym testem jest dopiero ruch z polityki.
2. **Zawieszony staw po długiej serii testów** — po dłuższej pracy w tym samym kontenerze
   zaobserwowano staw niereagujący na komendy mimo `claimed` w `list_hardware_interfaces`.
   Naprawia `docker compose restart sim`. Przyczyna nierozpoznana; jeśli wróci w pętli treningu,
   będzie wymagać diagnozy (podejrzenie: stan `gz_ros2_control` po wielu re-spawnach).
3. **Wyścig przy `multi_step`** (faza 2 krokowania) — nierozstrzygnięty do czasu implementacji.

### Rozwiązane
- `GZ_SIM_RESOURCE_PATH` → `ENV` w Dockerfile, meshe się ładują
- GUI na Waylandzie → `xhost +local:docker`
- `docker exec` bez sourcowania → sourcowanie dopisane do `/root/.bashrc`
- Mimic `fr3_finger_joint2` → Opcja A (jawne sterowanie oboma palcami, `strip_finger_mimic`)
- Robot przewracający się przy kontakcie → link `world` + fixed joint `world_to_base`
- Build Dockera padający na `rosdep` → pin `setuptools<81`

### Następne kroki
1. Naprawić `GZ_SIM_RESOURCE_PATH` w entrypoincie → potwierdzić że meshe FR3 ładują się w Gazebo — **Done**
2. Napisać minimalny launch file w pakiecie `franka_sim` (spawn FR3 + Gazebo + ros2_control) — **Done**
3. Zbudować scenę Pick & Place w Gazebo (stół + kostka SDF) — **Done**
4. Skonfigurować `controllers.yaml`: JTC dla ramienia + kontroler chwytaka; rozstrzygnąć mimic — **Done** (Opcja A)
5. Wrapper Gymnasium ↔ ROS 2 + DLS-IK — **WIP**, plan wykonawczy w `docs/plan-gym-wrapper-dls-ik.md`
6. Mosty Gazebo dla obserwacji i resetu: `dynamic_pose/info` (poza kostki), `SetEntityPose` (reset kostki) — **TODO**
7. Krokowanie Gazebo z pętli Gym, faza 2: `ControlWorld` / `multi_step` — **TODO**
8. Skryptowany ekspert zbierający demonstracje — **TODO**, ma iść przez ten sam `step()`, nie własną ścieżką do JTC

---

## Kluczowe decyzje podjęte

| Decyzja | Uzasadnienie |
|---|---|
| FR3 zamiast FER (Panda) | Nowszy model, lepsze wsparcie w repo |
| `ros-jazzy-desktop` zamiast pojedynczych paczek | Szybszy start, mniej debugowania brakujących zależności |
| PyTorch z PyPI (nie z `download.pytorch.org`) | CDN PyTorcha blokowany przez sieć (SSLV3_ALERT_HANDSHAKE_FAILURE); PyPI też daje build CUDA. Jeden obraz, na laptopie fallback na CPU |
| Tylko `franka_description` + `franka_msgs` ze źródeł | Reszta repo (gripper, hardware, gazebo_bringup) ciągnie `libfranka` — zbędne w symulacji |
| Integracja Gazebo pisana od zera w `franka_sim` | `franka_gazebo_bringup` z oficjalnego repo wymaga `franka_hardware` → `libfranka` |
| Gazebo headless | Wayland na hoście uniemożliwia GUI forwarding; trening RL i tak będzie headless |
| Obserwacje state-based (oracle) | Uczciwe porównanie, mieści się w harmonogramie i mocy GPU; vision → future work |
| Akcja: 4D delta-EE, orientacja zamrożona | Task space = efektywność próbkowa; position > velocity dla DP (Chi); prostota; plan B: delta joint position |
| Własne DLS-IK zamiast MoveIt Servo | Deterministyczne, testowalne, bez węzłów MoveIt w pętli treningu |
| Jakobian 6×7 z aktywnie regulowaną orientacją | 3×7 pozwala orientacji dryfować przez epizod przy 7 DOF — chwytak przestaje patrzeć w dół |
| Pinocchio zamiast własnej kinematyki / KDL | Gotowe i przetestowane FK+jakobian; praca opisuje DLS, nie kinematykę FR3; `kdl_parser_py` niepewny w Jazzy |
| Krokowanie sim wdrażane etapowo (free-run → `multi_step`) | Szybkie domknięcie end-to-end; cały czas schowany za `advance(dt)`, więc podmiana nie rusza `gym_env` |
| Poziomy L1/L2/L3 jako argument, nie trzy klasy | Trzy klasy = trzy kopie reward i obserwacji = gwarantowany rozjazd między wariantami |
| Baza przykręcona do świata (`world_to_base`) | Free-floating model przewraca się przy siłach reakcji chwytu; odpowiada realnemu FR3 na stanowisku |
| Pin `setuptools<81` w obrazie | setuptools ≥ 82 usunął `pkg_resources`, którego wymaga `rosdep` |
| JTC (position) jako kontroler | Interpolacja = gładkość; identyczny dla RL, DP i eksperta |
| Shaped reward + kara ‖a‖², bez filtrów akcji | Sparse nierozwiązywalny w budżecie; smoothness z surowych trajektorii |
| DP: wariant CNN, trening od zera, DDIM w inferencji | Łatwiejszy tuning; oryginalny DP jest per-task, nie pretrenowany |

---

## Aktualne pliki konfiguracyjne

### Dockerfile
```dockerfile
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-desktop \
    ros-jazzy-ros-gz \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-moveit \
    build-essential cmake git curl wget \
    python3-pip python3-venv python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir \
    torch torchvision

RUN pip3 install --break-system-packages --no-cache-dir \
    stable-baselines3 gymnasium diffusers wandb matplotlib pandas pyyaml scipy

# setuptools >= 82 nie zawiera pkg_resources, a /usr/bin/rosdep go importuje.
# Instalacje pip powyżej podmieniają systemowy setuptools — pin MUSI być po nich.
RUN pip3 install --break-system-packages --no-cache-dir "setuptools<81" \
    && python3 -c "import pkg_resources; print('pkg_resources OK')"

WORKDIR /opt/franka_ws/src
RUN git clone https://github.com/frankarobotics/franka_description.git
RUN git clone https://github.com/frankarobotics/franka_ros2.git -b jazzy

WORKDIR /opt/franka_ws
RUN source /opt/ros/jazzy/setup.bash && \
    apt-get update && rosdep update --rosdistro=jazzy && \
    rosdep install --from-paths src --ignore-src -r -y --skip-keys="Franka libfranka" && \
    rm -rf /var/lib/apt/lists/* && \
    colcon build --symlink-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select franka_description franka_msgs

ENV GZ_SIM_RESOURCE_PATH=/opt/franka_ws/install/franka_description/share
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "source /opt/franka_ws/install/setup.bash" >> /root/.bashrc && \
    echo "[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash" >> /root/.bashrc

WORKDIR /ws
# Sekcja kopiowania src/ + colcon build celowo zakomentowana: src/ jest bind-mountowane
# przez docker-compose, więc COPY zostałoby przykryte, a install/ w obrazie byłby nieaktualny.
# Odkomentować dopiero przy budowaniu samowystarczalnego obrazu — wtedy usunąć bind-mount.

COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
```

### docker-compose.yml
```yaml
services:
  sim:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: franka_sim
    environment:
      - DISPLAY=${DISPLAY}
      - QT_X11_NO_MITSHM=1
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
    volumes:
      - ../src:/ws/src
      - ../evaluation:/ws/evaluation
      - ../data:/ws/data
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
    devices:
      - /dev/dri:/dev/dri
    network_mode: host
    stdin_open: true
    tty: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu, graphics, display, compute, utility]
```

### ros_entrypoint.sh
```bash
#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash
source /opt/franka_ws/install/setup.bash

if [ -f /ws/install/setup.bash ]; then
    source /ws/install/setup.bash
fi

exec "$@"
```

### Launch
Tymczasowy launch z `/tmp` nieaktualny — zastąpiony przez
`src/franka_sim/launch/bringup.launch.py` (Gazebo + `/clock` bridge + RSP + spawn robota +
sekwencjonowane spawnery kontrolerów + `strip_finger_mimic`). Plik żyje w repo, nie kopiujemy
go tutaj, żeby nie utrzymywać dwóch wersji.

---

## Literatura

1. Chi et al. (2023) — Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (RSS 2023)
2. Haarnoja et al. (2018) — Soft Actor-Critic (ICML 2018)
3. Schulman et al. (2017) — Proximal Policy Optimization Algorithms
4. Ho et al. (2020) — Denoising Diffusion Probabilistic Models (NeurIPS 2020)
5. Mandlekar et al. (2021) — What Matters in Learning from Offline Human Demonstrations (CoRL 2021)
6. Gallouédec et al. (2021) — panda-gym: Open-source goal-conditioned environments for robotic learning — *najbliższy setup: Panda + P&P + SAC*
7. Andrychowicz et al. (2017) — Hindsight Experience Replay (NeurIPS 2017)
8. Zhou et al. (2019) — On the Continuity of Rotation Representations in Neural Networks (CVPR 2019) — *reprezentacja 6D orientacji*
9. Heo et al. (2023) — FurnitureBench (RSS 2023) — *wzorzec przestrzeni akcji: delta-EE + chwytak z progiem, OSC 10 Hz → 1 kHz*
10. Ren et al. (2024) — DPPO: Diffusion Policy Policy Optimization (ICLR 2025, arXiv:2409.00588) — *do related work: finetuning DP przez policy gradient, pomost RL↔DP*
11. Wang et al. (2022) — Diffusion Policies as an Expressive Policy Class for Offline RL (Diffusion-QL, ICLR 2023) — *related work*
12. arXiv:2410.01220 — Effective Tuning Strategies for Generalist Robot Manipulation Policies — *delta joint position vs delta-EE, argument za planem B*
13. arXiv:2602.23408 — Demystifying Action Space Design for Robotic Manipulation Policies — *systematyczne badanie wyboru przestrzeni akcji*

---

## Komendy robocze

Wszystkie komendy operacyjne żyją w `docs/cheatsheet.md`.

**Zasada podziału (obowiązująca przy każdej edycji obu plików):**
- `cheatsheet.md` = **jak**: komendy do skopiowania + najwyżej jednolinijkowa uwaga, bez której
  komenda nie zadziała lub zadziała źle. Żadnych akapitów tła, historii decyzji ani tabel faktów.
- `thesis-project-context.md` (ten plik) = **dlaczego**: decyzje projektowe i ich uzasadnienia,
  zmierzone fakty o modelu/silniku fizyki, odrzucone warianty, stan implementacji, TODO.

Jeśli treść tłumaczy *powód* — idzie tutaj, a cheatsheet co najwyżej odsyła.

## Dokumenty powiązane

- `docs/cheatsheet.md` — komendy operacyjne
- `docs/plan-gym-wrapper-dls-ik.md` — plan wykonawczy wrappera Gym + DLS-IK (etap bieżący)
- `CLAUDE.md` — reguły pracy w repo (walidacja przed uruchomieniem, rebuild po edycji `config/`/`launch/`/`urdf/`)
