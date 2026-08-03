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
- **Diffusion Policy:** finetuning wstępnie wytrenowanego modelu (Chi et al., 2023)

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
| 1–2 | Przegląd literatury + konfiguracja środowiska ← **tu jesteśmy** |
| 3–4 | Implementacja środowiska symulacyjnego (scena P&P, węzły ROS 2, 3 poziomy trudności) |
| 5–6 | Implementacja i trening SAC oraz PPO |
| 7–8 | Finetuning Diffusion Policy |
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
  → clip akcji + clip do workspace box (np. x∈[0.3,0.7], y∈[-0.3,0.3], z∈[0.02,0.5])
  → IK: damped least-squares na jakobianie (KDL/pinocchio), q̇ = Jᵀ(JJᵀ+λ²I)⁻¹·Δx
  → joint_trajectory_controller (jednopunktowa trajektoria, time_from_start ≈ dt)
  → gz_ros2_control (interfejs pozycyjny) → Gazebo
  → obserwacje wracają do polityki
```

- IK: własne DLS zamiast MoveIt Servo (deterministyczne, testowalne, bez węzłów MoveIt w pętli treningu). Przy zamrożonej orientacji tylko wiersze pozycyjne jakobianu.
- Porażka IK (osobliwość/limit stawu) = no-op + ewentualna mała kara; **logować częstość** (ciekawa statystyka porównawcza DP vs RL).
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
   `franka_description`. Fizyka chwytu (tarcie, kontakt) do dalszego tiuningu
   `mu`/`mu2`/`kp`/`kd`/`min_depth` na kostce w `fr3_world.sdf`.

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
│   ├── franka_sim/                   # Pakiet ament_cmake: scena Gazebo + launch files
│   │   ├── launch/
│   │   ├── worlds/
│   │   ├── models/
│   │   └── config/controllers.yaml
│   │
│   ├── franka_task/                  # Pakiet ament_python: logika Pick & Place
│   │   ├── task_manager.py
│   │   ├── scene_randomizer.py
│   │   └── reward.py
│   │
│   ├── franka_rl/                    # Pakiet ament_python: SAC, PPO
│   │   ├── gym_env.py               # Wrapper Gymnasium ↔ ROS 2
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
- **PyTorch z CUDA 12.4** (`cu124`) — jeden obraz, działa na GPU (PC) i CPU (laptop) automatycznie
- **Franka ze źródeł:** klonowane `franka_description` + `franka_ros2` (branch `jazzy`), budowane TYLKO `franka_description` i `franka_msgs` (`--packages-select`)
- **docker-compose:** jeden serwis `sim`, volumes montują `src/`, `evaluation/`, `data/`
- **Entrypoint:** automatycznie sourcuje `/opt/ros/jazzy/setup.bash` i `/opt/franka_ws/install/setup.bash`

### Franka FR3 w ROS 2
- URDF parsuje się poprawnie: `ros2 run xacro xacro .../fr3/fr3.urdf.xacro`
- `robot_state_publisher` działa, publikuje `/robot_description`
- Spawn do Gazebo działa (`ros2 run ros_gz_sim create -name fr3 -topic /robot_description`)

### Gazebo Harmonic
- Działa w trybie headless: `gz sim -s -r empty.sdf`
- GUI nie działa — host ma Waylanda, X11 forwarding do kontenera nie przechodzi (XWayland nie testowany)

---

## Znane problemy i TODO

### Do naprawienia
1. **Meshe FR3 w Gazebo** — Gazebo nie znajduje plików STL/DAE. Trzeba ustawić:
   ```bash
   export GZ_SIM_RESOURCE_PATH=/opt/franka_ws/install/franka_description/share:$GZ_SIM_RESOURCE_PATH
   ```
   To powinno być dodane do `ros_entrypoint.sh` na stałe.

2. **GUI Gazebo na Waylandzie** — host ma Wayland. Działa po `xhost +local:docker`. Na razie pracujemy headless (`gz sim -s`). Opcje:
   - Praca headless + RViz do wizualizacji

3. **`docker exec` nie sourcuje entrypointa** — przy `docker exec -it <kontener> bash` trzeba ręcznie: `source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash`. Rozwiązanie: dodać sourcowanie do `/root/.bashrc` w Dockerfile.

### Następne kroki
1. Naprawić `GZ_SIM_RESOURCE_PATH` w entrypoincie → potwierdzić że meshe FR3 ładują się w Gazebo - Done
2. Napisać minimalny launch file w pakiecie `franka_sim` (spawn FR3 + Gazebo + ros2_control) - Done
3. Zbudować scenę Pick & Place w Gazebo (stół + kostka SDF) - WIP
4. Napisać wrapper Gymnasium ↔ ROS 2 — observation/action space **wg sekcji „Ustalenia projektowe"** (state-based, 4D delta-EE) - TODO
5. Zaimplementować DLS-IK + clip do workspace (moduł testowalny jednostkowo) - TODO
6. Skonfigurować `controllers.yaml`: JTC dla ramienia + kontroler chwytaka; przetestować mimic joints - TODO
7. Rozwiązać krokowanie Gazebo z pętli Gym (serwis `/world/<name>/control`, `multi_step`) - TODO

---

## Kluczowe decyzje podjęte

| Decyzja | Uzasadnienie |
|---|---|
| FR3 zamiast FER (Panda) | Nowszy model, lepsze wsparcie w repo |
| `ros-jazzy-desktop` zamiast pojedynczych paczek | Szybszy start, mniej debugowania brakujących zależności |
| PyTorch CUDA (`cu124`) na obu maszynach | Jeden obraz; na laptopie automatyczny fallback na CPU |
| Tylko `franka_description` + `franka_msgs` ze źródeł | Reszta repo (gripper, hardware, gazebo_bringup) ciągnie `libfranka` — zbędne w symulacji |
| Integracja Gazebo pisana od zera w `franka_sim` | `franka_gazebo_bringup` z oficjalnego repo wymaga `franka_hardware` → `libfranka` |
| Gazebo headless | Wayland na hoście uniemożliwia GUI forwarding; trening RL i tak będzie headless |
| Obserwacje state-based (oracle) | Uczciwe porównanie, mieści się w harmonogramie i mocy GPU; vision → future work |
| Akcja: 4D delta-EE, orientacja zamrożona | Task space = efektywność próbkowa; position > velocity dla DP (Chi); prostota; plan B: delta joint position |
| Własne DLS-IK zamiast MoveIt Servo | Deterministyczne, testowalne, bez węzłów MoveIt w pętli treningu |
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
    torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --break-system-packages --no-cache-dir \
    stable-baselines3 gymnasium diffusers wandb matplotlib pandas pyyaml scipy

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

WORKDIR /ws

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
    network_mode: host
    stdin_open: true
    tty: true
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

### Minimalny launch file (tymczasowy, w /tmp)
```python
from launch import LaunchDescription
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    urdf = xacro.process_file(
        '/opt/franka_ws/src/franka_description/robots/fr3/fr3.urdf.xacro'
    ).toxml()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': urdf}],
        ),
    ])
```

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

## Komendy robocze (cheat sheet)

```bash
# Build obrazu
cd ~/Inżynierka/DiffRL-Panda/docker
docker compose build

# Uruchomienie kontenera
docker compose run sim

# Drugi terminal w kontenerze (ręczne sourcowanie!)
docker ps  # sprawdź nazwę
docker exec -it <nazwa_kontenera> bash
source /opt/ros/jazzy/setup.bash
source /opt/franka_ws/install/setup.bash

# Robot state publisher
ros2 launch /tmp/view_fr3.launch.py

# Gazebo headless
gz sim -s -r empty.sdf

# Spawn FR3 do Gazebo
export GZ_SIM_RESOURCE_PATH=/opt/franka_ws/install/franka_description/share:$GZ_SIM_RESOURCE_PATH
ros2 run ros_gz_sim create -name fr3 -topic /robot_description

# Sprawdzenie topików
ros2 topic list | grep robot
```
