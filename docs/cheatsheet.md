# Cheatsheet — DiffRL-Panda

> **Środowisko działa.** FR3 spawnuje się w Gazebo, trzy kontrolery `active`, robot stoi w pozie ready. GUI (RViz / Gazebo) działa przez XWayland. Poniżej aktualne komendy.

---

## Docker

```bash
cd ~/Inżynierka/DiffRL-Panda/docker

# Build obrazu
docker compose build
docker compose build --no-cache         

# Uruchomienie kontenera
docker compose up -d                     # ZALECANE: stabilna nazwa 'franka_sim'
docker compose run sim                   # UWAGA: losowa nazwa (docker-sim-run-xxxx), psuje docker exec

# Drugi terminal w działającym kontenerze
docker ps                                # sprawdź nazwę
docker exec -it franka_sim bash          # .bashrc sam sourcuje ROS + franka_ws 

# Zatrzymanie / sprzątanie
docker compose down                      # ubij serwis
docker system prune -f                   # usuń zbędne warstwy/kontenery
docker image prune -a -f                 # usuń nieużywane obrazy 
docker compose logs -f sim               # logi serwisu w tle
```

> **Uwaga o nazwie:** `docker exec franka_sim ...` działa tylko po `docker compose up -d` (nazwa z `container_name`). Po `docker compose run` nazwa jest losowa — sprawdź `docker ps`.

### Fallback — ręczne sourcowanie (gdyby .bashrc nie zadziałał)
```bash
source /opt/ros/jazzy/setup.bash
source /opt/franka_ws/install/setup.bash
[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash
```

---

## GUI na Waylandzie — DZIAŁA (przez XWayland)

Mit "GUI nie działa na Waylandzie" był fałszywy — brakowało tylko `xhost`.

```bash
# NA HOŚCIE 
xhost +local:docker

# Test że X11 z kontenera przechodzi
docker exec -it franka_sim bash -c "apt-get install -y x11-apps && xeyes"

# RViz 
docker exec -it franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash; rviz2"
#   W RViz: Fixed Frame = 'base', RobotModel → Description Topic = /robot_description

# Gazebo GUI (pełna scena z fizyką — klient dołącza do headless serwera z bringupa)
docker exec -it franka_sim bash -c "source /opt/ros/jazzy/setup.bash; gz sim -g"
```

> GUI = **podgląd/debug**, nie tryb treningu. Trening zawsze headless (renderowanie przy milionach kroków RL to strata). Jak GUI nagle przestaje działać → najpierw `xhost +local:docker` na hoście.

---

## Colcon (workspace ROS 2, w kontenerze)

```bash
cd /ws

# Build
colcon build --packages-select franka_sim
colcon build --packages-up-to franka_rl    
source install/setup.bash                 

# Czyszczenie 
rm -rf build install log

colcon list                                 # lista pakietów
```

> **Ważne:** `config/`, `launch/`, `urdf/`, `worlds/`, `models/` instalują się do `share/` przez `install(DIRECTORY ...)` w `CMakeLists.txt`. `ros2 launch` i `$(find franka_sim)` szukają w `share/`, NIE w `src/`. Po edycji tych plików → **rebuild**. Dodajesz nowy katalog → dopisz go do `install()`.

---

## Uruchomienie symulacji (główny flow)

```bash
# Pełny bringup: Gazebo headless + /clock bridge + RSP + spawn + kontrolery
ros2 launch franka_sim bringup.launch.py

# Szybki test bez rebuildu
ros2 launch /ws/src/franka_sim/launch/bringup.launch.py
```

Weryfikacja po starcie (drugi terminal):
```bash
ros2 control list_controllers            
ros2 topic echo /joint_states --once     

---

## Walidacja PRZED uruchomieniem (oszczędza core dumpy)

```bash
# YAML
python3 -c "import yaml; yaml.safe_load(open('src/franka_sim/config/controllers.yaml')); print('YAML OK')"

# xacro → URDF 
ros2 run xacro xacro src/franka_sim/urdf/fr3_gazebo.urdf.xacro > /tmp/test.urdf && echo OK
grep -A4 'joint name="fr3_joint1"' /tmp/test.urdf  
```

---

## URDF / xacro — nasz wrapper

```bash
# Mój plik
FR3=/ws/src/franka_sim/urdf/fr3_gazebo.urdf.xacro
ros2 run xacro xacro $FR3 > /tmp/fr3.urdf

# Bazowy opis Franki 
BASE=/opt/franka_ws/src/franka_description/robots/fr3/fr3.urdf.xacro
ros2 run xacro xacro $BASE > /tmp/fr3_base.urdf
```

> **NIE używać** flagi `ros2_control:=true` na bazowym xacro — generuje `<transmission>` w stylu ROS 1, niekompatybilne z `gz_ros2_control`. Dlatego mamy własny wrapper `fr3_gazebo.urdf.xacro` (position-only command, bug #343; `GazeboSimSystem` + `GazeboSimROS2ControlPlugin`).

---

## Gazebo Harmonic (Gz Sim 8)

```bash
gz sim -s -r empty.sdf                   # headless server (-s), run od razu (-r)
gz sim -g                                # klient GUI (dołącza do serwera)

# Introspekcja Gazebo 
gz topic -l                              # lista topików gz
gz model --list                          # modele w scenie
gz service -l                            # serwisy gz

# Resource path 
export GZ_SIM_RESOURCE_PATH=/opt/franka_ws/install/franka_description/share:$GZ_SIM_RESOURCE_PATH
```

---

## ros2_control 

Kontrolery: `joint_state_broadcaster`, `fr3_arm_controller` (JTC, 7 stawów), `fr3_gripper_controller` (JTC, `fr3_finger_joint1` + `fr3_finger_joint2`, sterowane jawnie).

```bash
ros2 control list_controllers            # stan
ros2 control list_hardware_interfaces    # dostępne command/state interfaces

# Test ruchu ramienia 
ros2 topic pub --once /fr3_arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: [fr3_joint1,fr3_joint2,fr3_joint3,fr3_joint4,fr3_joint5,fr3_joint6,fr3_joint7],
    points: [{positions: [0.0,-0.785,0.0,-2.356,0.0,1.571,0.785], time_from_start: {sec: 2}}]}"

# Test chwytaka (0.0 = zamknięty, 0.04 = otwarty) — oba palce jawnie w jednej komendzie
ros2 topic pub --once /fr3_gripper_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: [fr3_finger_joint1, fr3_finger_joint2], points: [{positions: [0.04, 0.04], time_from_start: {sec: 1}}]}"
```

> **Ryzyko #1 (mimic) — rozwiązane (Opcja A):** silnik DART w Harmonic NIE wspiera mimic constraints. `DetachableJoint` przetestowany i odrzucony — działa poprawnie w izolacji, ale nie respektuje ruchu stawów aktuowanych przez `gz_ros2_control` (position command). Finalnie: jawne sterowanie oboma palcami — `fr3_finger_joint2` dostał własny `command_interface` w `ros2_control` (usunięto `<mimic>` z URDF post-processingiem w `bringup.launch.py`, bo `franka_hand.xacro` nie ma parametru do jego wyłączenia). Szczegóły w `thesis-project-context.md`.
> **Uwaga:** po dłuższej serii testów w tym samym kontenerze zaobserwowano, że jeden ze stawów przestał reagować na komendy pozycji mimo poprawnej konfiguracji (`ros2 control list_hardware_interfaces` pokazywał `claimed`) — naprawił to `docker compose restart sim`. Jeśli komenda `joint_trajectory` nie daje efektu mimo aktywnego kontrolera, restart kontenera jest pierwszym krokiem diagnostyki.

---

## Trening RL (stable-baselines3)

```bash
cd /ws
python3 src/franka_rl/train_sac.py 2>&1 | tee data/logs/sac_$(date +%F_%H%M).log
python3 src/franka_rl/eval.py --checkpoint data/checkpoints/sac_best.zip --episodes 50
tensorboard --logdir data/logs --bind_all
```

### Weights & Biases
```bash
wandb login
export WANDB_MODE=offline                # trening offline, sync później
wandb sync data/wandb/offline-run-*
export WANDB_PROJECT=diffrl-panda
```

---

## Diffusion Policy

```bash
cd /ws
python3 src/franka_diffusion/data_collector.py --episodes 100 --out data/demos/
python3 src/franka_diffusion/finetune.py --config src/franka_diffusion/config.yaml
python3 src/franka_diffusion/eval.py --checkpoint data/checkpoints/dp_best.pt
```

---

## GPU / CUDA

```bash
nvidia-smi                               # stan GPU (host)
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

> **torch instaluj z PyPI** (`pip install torch torchvision`), NIE z `--index-url https://download.pytorch.org/whl/cu124` — sieć blokuje CDN PyTorcha (SSLV3_ALERT_HANDSHAKE_FAILURE). PyPI daje build CUDA.

---

## ROS 2 — introspekcja

```bash
ros2 node list
ros2 topic list | grep robot
ros2 topic echo /robot_description --once
ros2 topic hz /joint_states
ros2 topic info /joint_states -v
ros2 topic echo /tf_static --once        # sprawdzenie TF (dla RViz)
ros2 service list
```

---

## tmux (multi-terminal w jednym exec)

```bash
tmux
# Ctrl+b "  → podział poziomy   |   Ctrl+b %  → pionowy
# Ctrl+b o  → przełączanie      |   Ctrl+b d  → detach
tmux attach
```

---

## Git

```bash
cd ~/Inżynierka/DiffRL-Panda
git status
git add -p
git commit -m "..."
git check-ignore data/checkpoints/sac_best.zip   # czy data/ ignorowane
```

> **Uprawnienia:** colcon w kontenerze działa jako root → pliki `root`-owe na hoście. Jak edytor rzuca `EACCES`: `sudo chown -R $USER:$USER src/franka_sim/`. Twórz nowe pliki z hosta, żeby tego uniknąć.

---

## Szybki debug — typowe problemy

| Objaw | Sprawdź |
|---|---|
| Gazebo: brak meshy FR3 | `echo $GZ_SIM_RESOURCE_PATH` (jest jako ENV; fallback: export) |
| `command not found: ros2` w exec | `.bashrc` sourcuje ROS; fallback: ręczne `source` |
| `ros2 launch ... not found in share` | Katalog nie w `install(DIRECTORY)` w CMakeLists → dopisz + rebuild |
| controller_manager crash "value before ros__parameters" | YAML: taby zamiast spacji, albo złe wcięcia. Waliduj `yaml.safe_load` |
| launch: "too many values to unpack" | `launch_arguments={...}.items()` — brakuje `.items()` |
| Gazebo crash przy starcie kontrolerów | Rozjazd URDF↔YAML (interface not available) albo błąd w controllers.yaml |
| Chwytak: rusza się tylko jeden palec | Mimic niewspierany (DART) — to znane, plan B |
| torch: SSLV3_ALERT_HANDSHAKE_FAILURE | Instaluj z PyPI, nie z download.pytorch.org |
| GUI nie wyskakuje | `xhost +local:docker` na hoście (nie przeżywa restartu) |
| RViz: "frame [map] does not exist" | Fixed Frame → `base`; RobotModel Description Topic → `/robot_description` |
| Robot niewidoczny w RViz | RobotModel → Description Topic = `/robot_description` |
| `/clock` nie dochodzi do ROS | ros_gz_bridge dla `/clock` w launchu; sim z `-r` |
| Trening nie widzi GPU | `torch.cuda.is_available()`; `nvidia-smi` na hoście |
| colcon sypie się po refaktorze | `rm -rf build install log` i rebuild |