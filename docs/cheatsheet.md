# Cheatsheet — DiffRL-Panda

> Ten plik trzyma **jak** — komendy i krótkie uwagi operacyjne (jedna linia: co zrobić, żeby
> komenda zadziałała). **Dlaczego** — decyzje projektowe, fakty o modelu, uzasadnienia, stan
> implementacji — idzie do `docs/thesis-project-context.md`. Nie dopisuj tu akapitów tła.

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

# Restart (pierwszy krok, gdy staw przestał reagować mimo aktywnego kontrolera)
docker compose restart sim

# Zatrzymanie / sprzątanie
docker compose down                      # ubij serwis
docker system prune -f                   # usuń zbędne warstwy/kontenery
docker image prune -a -f                 # usuń nieużywane obrazy
docker compose logs -f sim               # logi serwisu w tle
```

> `docker exec franka_sim ...` działa tylko po `docker compose up -d`. Po `docker compose run`
> nazwa jest losowa — sprawdź `docker ps`.

---

## Sourcowanie

**Sesja interaktywna** — nic nie robisz, `.bashrc` sourcuje wszystkie trzy warstwy:
```bash
docker exec -it franka_sim bash
source /ws/install/setup.bash            # tylko po pierwszym buildzie / dodaniu nowego pakietu
```

**Pojedyncza komenda z hosta** — `docker exec franka_sim <cmd>` omija `.bashrc` (powłoka
nieinteraktywna), więc zawsze owijaj w `bash -c` z sourcowaniem:
```bash
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash && source /ws/install/setup.bash; <KOMENDA>"
```

Alias na hoście (`~/.bashrc`) — `-l -i` daje powłokę logowania i interaktywną, więc `.bashrc`
kontenera wykonuje się sam:
```bash
alias fx='docker exec -it franka_sim bash -lic'

fx 'ros2 control list_controllers'
fx 'python3 -c "import pinocchio; print(pinocchio.__version__)"'
fx 'cd /ws && colcon build --packages-select franka_rl'
```

| Source | Co odblokowuje |
|---|---|
| `/opt/ros/jazzy/setup.bash` | `ros2`, `colcon`, `rclpy`, **`pinocchio`**, `xacro`, mosty `ros_gz` |
| `/opt/franka_ws/install/setup.bash` | `franka_description`, `franka_msgs` (bazowy xacro FR3) |
| `/ws/install/setup.bash` | `franka_sim`, `franka_rl` — `ros2 launch`, `$(find franka_sim)` |

> `GZ_SIM_RESOURCE_PATH` jest `ENV` w Dockerfile — meshe działają bez sourcowania czegokolwiek.

Fallback, gdyby `.bashrc` nie zadziałał:
```bash
source /opt/ros/jazzy/setup.bash
source /opt/franka_ws/install/setup.bash
[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash
```

---

## GUI (przez XWayland)

```bash
# NA HOŚCIE — nie przeżywa restartu, powtórz po każdym
xhost +local:docker

# Test że X11 z kontenera przechodzi
docker exec -it franka_sim bash -c "apt-get install -y x11-apps && xeyes"

# RViz — Fixed Frame = 'base', RobotModel → Description Topic = /robot_description
docker exec -it franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash; rviz2"

# Gazebo GUI (klient dołącza do headless serwera z bringupa)
docker exec -it franka_sim bash -c "source /opt/ros/jazzy/setup.bash; gz sim -g"
```

---

## Colcon (workspace ROS 2, w kontenerze)

```bash
cd /ws

# Build
colcon build --packages-select franka_sim
colcon build --packages-select franka_rl
colcon build --packages-up-to franka_rl
source install/setup.bash

# Czyszczenie
rm -rf build install log

colcon list                                 # lista pakietów
```

> Po edycji `config/`, `launch/`, `urdf/`, `worlds/`, `models/` → **rebuild** (instalują się do
> `share/`). Nowy katalog → dopisz do `install(DIRECTORY ...)` w `CMakeLists.txt`.

> Nie używaj `--symlink-install` dla `ament_python` — nie działa (cicha degradacja do kopii).
> Testy `franka_rl` i tak lecą z `src/` dzięki `conftest.py`.

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
```

### Ręczne sterowanie stawami (suwaki rqt)

```bash
# Build
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash && cd /ws && colcon build && source install/setup.bash"

# Instalacja rqt (raz na kontener)
docker exec franka_sim bash -c "apt-get update && apt-get install -y ros-jazzy-rqt-joint-trajectory-controller"

# Terminal 1 — bringup
docker exec -it franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash; ros2 launch franka_sim bringup.launch.py"

# Terminal 2 — GUI
docker exec -it franka_sim bash -c "source /opt/ros/jazzy/setup.bash; gz sim -g"

# Terminal 3 — suwaki
docker exec -it franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash; ros2 run rqt_joint_trajectory_controller rqt_joint_trajectory_controller"
```

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

## URDF / xacro

```bash
# Nasz wrapper
FR3=/ws/src/franka_sim/urdf/fr3_gazebo.urdf.xacro
ros2 run xacro xacro $FR3 > /tmp/fr3.urdf

# Bazowy opis Franki
BASE=/opt/franka_ws/src/franka_description/robots/fr3/fr3.urdf.xacro
ros2 run xacro xacro $BASE > /tmp/fr3_base.urdf
```

> **NIE używać** flagi `ros2_control:=true` na bazowym xacro — dlatego mamy własny wrapper.

---

## Gazebo Harmonic (Gz Sim 8)

```bash
gz sim -s -r empty.sdf                   # headless server (-s), run od razu (-r)
gz sim -g                                # klient GUI (dołącza do serwera)

gz topic -l                              # lista topików gz
gz model --list                          # modele w scenie
gz model -m fr3 -p                        # poza modelu (musi zostać 0 0 0 — baza przykręcona)
gz service -l                            # serwisy gz

export GZ_SIM_RESOURCE_PATH=/opt/franka_ws/install/franka_description/share:$GZ_SIM_RESOURCE_PATH
```

---

## ros2_control

Kontrolery: `joint_state_broadcaster`, `fr3_arm_controller` (JTC, 7 stawów),
`fr3_gripper_controller` (JTC, `fr3_finger_joint1` + `fr3_finger_joint2`, sterowane jawnie).

```bash
ros2 control list_controllers            # stan
ros2 control list_hardware_interfaces    # dostępne command/state interfaces

# Test ruchu ramienia
ros2 topic pub --once /fr3_arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: [fr3_joint1,fr3_joint2,fr3_joint3,fr3_joint4,fr3_joint5,fr3_joint6,fr3_joint7],
    points: [{positions: [0.0,-0.785,0.0,-2.356,0.0,1.571,0.785], time_from_start: {sec: 2}}]}"

# Błąd nadążania JTC (wykrywanie kolizji: trwały rozjazd komenda↔stan, próg ~0.1 rad)
ros2 topic echo /fr3_arm_controller/controller_state --field error.positions

# Test chwytaka (0.0 = zamknięty, 0.04 = otwarty) — oba palce jawnie w jednej komendzie
ros2 topic pub --once /fr3_gripper_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: [fr3_finger_joint1, fr3_finger_joint2], points: [{positions: [0.04, 0.04], time_from_start: {sec: 1}}]}"
```

---

## `franka_rl` — Gym + DLS-IK

```bash
# Build
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash; cd /ws && colcon build --packages-select franka_rl"

# Sanity check, że ROS widzi pakiet
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash; ros2 pkg list | grep franka_rl"
```

### Testy jednostkowe — bez rebuildu i bez Gazebo

```bash
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash; cd /ws && python3 -m pytest src/franka_rl/test -v"

# pojedynczy test
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash; cd /ws && python3 -m pytest src/franka_rl/test/test_ik.py::{name} -v"

# sprzątanie śmieci po pytest (root-owe na hoście — kasuj z kontenera)
docker exec franka_sim bash -c "rm -rf /ws/src/franka_rl/.pytest_cache /ws/src/franka_rl/**/__pycache__"
```

> Wystarczy `source /opt/ros/jazzy/setup.bash` (stamtąd idzie `pinocchio`); `/ws/install` nie jest
> potrzebny. Działa dzięki pustemu `src/franka_rl/conftest.py` — **nie kasuj tego pliku**.

```bash
# Wersja pinocchio
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash; python3 -c 'import pinocchio; print(pinocchio.__version__)'"
```

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
pip install torch torchvision            # zawsze z PyPI, NIE z download.pytorch.org
```

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

sudo chown -R $USER:$USER src/franka_sim/        # gdy edytor rzuca EACCES (pliki root-owe z colcona)
```

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
| Staw nie reaguje mimo `claimed` w `list_hardware_interfaces` | `docker compose restart sim` |
| Kostka wyślizguje się z chwytaka | `mu`/`mu2` po **obu** stronach pary (kostka + `fr3_leftfinger`/`fr3_rightfinger`); `kp`/`kd`/`min_depth` DART ignoruje |
| Robot przewraca się / odjeżdża przy kontakcie | Brak `<link name="world"/>` + fixed jointa `world_to_base`. Sprawdź `gz model -m fr3 -p` |
| torch: SSLV3_ALERT_HANDSHAKE_FAILURE | Instaluj z PyPI, nie z download.pytorch.org |
| GUI nie wyskakuje | `xhost +local:docker` na hoście (nie przeżywa restartu) |
| RViz: "frame [map] does not exist" | Fixed Frame → `base`; RobotModel Description Topic → `/robot_description` |
| Robot niewidoczny w RViz | RobotModel → Description Topic = `/robot_description` |
| `/clock` nie dochodzi do ROS | ros_gz_bridge dla `/clock` w launchu; sim z `-r` |
| `parameter_bridge` loguje „Creating ROS->GZ service bridge" co sekundę | Most serwisu podany przez `parameters={'config_file':...}` — przełóż na `arguments=['<svc>@<ros_srv>@<gz_req>@<gz_rep>']` (wyciek ~390 MB/h) |
| Trening nie widzi GPU | `torch.cuda.is_available()`; `nvidia-smi` na hoście |
| colcon sypie się po refaktorze | `rm -rf build install log` i rebuild |
| pytest nie widzi zmian w `src/` | Skasowany `src/franka_rl/conftest.py` — przywróć (pusty plik) |
