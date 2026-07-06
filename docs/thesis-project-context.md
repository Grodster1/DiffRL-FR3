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

2. **GUI Gazebo na Waylandzie** — host ma Wayland, forwarding X11 do kontenera nie działa. Na razie pracujemy headless (`gz sim -s`). Opcje:
   - XWayland (nie testowany)
   - Gazebo nativo na hoście + transport do kontenera
   - Praca headless + RViz do wizualizacji

3. **`docker exec` nie sourcuje entrypointa** — przy `docker exec -it <kontener> bash` trzeba ręcznie: `source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash`. Rozwiązanie: dodać sourcowanie do `/root/.bashrc` w Dockerfile.

### Następne kroki
1. Naprawić `GZ_SIM_RESOURCE_PATH` w entrypoincie → potwierdzić że meshe FR3 ładują się w Gazebo
2. Napisać minimalny launch file w pakiecie `franka_sim` (spawn FR3 + Gazebo + ros2_control)
3. Zbudować scenę Pick & Place w Gazebo (stół + kostka SDF)
4. Napisać wrapper Gymnasium ↔ ROS 2 (observation/action space)

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
