# Cheatsheet — DiffRL-Panda
## Docker

```bash
cd ~/Inżynierka/DiffRL-Panda/docker

# Build obrazu
docker compose build
docker compose build --no-cache          # gdy cache gryzie (zmiany w apt/pip nie łapią)

# Uruchomienie kontenera (główny terminal, odpala entrypoint)
docker compose run sim                   # jednorazowo, usuwany po wyjściu
docker compose up -d                     # w tle, jako serwis

# Drugi terminal w działającym kontenerze
docker ps                                # sprawdź nazwę (franka_sim)
docker exec -it franka_sim bash          # .bashrc sam sourcuje ROS + franka_ws

# Zatrzymanie / sprzątanie
docker compose down                      # ubij serwis
docker stop franka_sim && docker rm franka_sim
docker system prune -f                   # usuń zbędne warstwy/kontenery
docker image prune -a -f                 # usuń nieużywane obrazy (ostrożnie)

# Podejrzenie logów serwisu w tle
docker compose logs -f sim
```

### Fallback — ręczne sourcowanie (gdyby .bashrc nie zadziałał)
```bash
source /opt/ros/jazzy/setup.bash
source /opt/franka_ws/install/setup.bash
[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash
```

---

## Colcon (workspace ROS 2)

```bash
cd /ws                                   # workspace w kontenerze

# Build
colcon build --symlink-install
colcon build --symlink-install --packages-select franka_sim
colcon build --symlink-install --packages-up-to franka_rl   # pakiet + zależności
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Po buildzie zawsze sourcuj
source install/setup.bash

# Czyszczenie (gdy build się sypie po zmianie struktury)
rm -rf build install log

# Sprawdzenie zależności pakietu
rosdep install --from-paths src --ignore-src -r -y \
  --skip-keys="Franka libfranka"

# Lista pakietów w workspace
colcon list
```

---

## ROS 2 — introspekcja

```bash
# Węzły
ros2 node list
ros2 node info /robot_state_publisher

# Topiki
ros2 topic list
ros2 topic list | grep robot
ros2 topic echo /robot_description --once
ros2 topic echo /joint_states
ros2 topic hz /joint_states              # częstotliwość publikacji
ros2 topic info /joint_states -v         # typy, QoS, liczba pub/sub

# Serwisy i akcje
ros2 service list
ros2 action list

# Parametry
ros2 param list
ros2 param get /robot_state_publisher robot_description

# Graf (wymaga rqt, GUI)
rqt_graph
```

---

## Franka FR3 / URDF / xacro

```bash
FR3_XACRO=/opt/franka_ws/src/franka_description/robots/fr3/fr3.urdf.xacro

# Parsowanie xacro → URDF (sprawdzenie czy się buduje)
ros2 run xacro xacro $FR3_XACRO

# Zapis do pliku i podgląd błędów
ros2 run xacro xacro $FR3_XACRO > /tmp/fr3.urdf
xmllint --noout /tmp/fr3.urdf && echo "URDF OK"

# Sprawdzenie z parametrami (np. gazebo:=true jeśli xacro to wspiera)
ros2 run xacro xacro $FR3_XACRO ros2_control:=true gazebo:=true

# Robot state publisher (tymczasowy launch)
ros2 launch /tmp/view_fr3.launch.py
```

---

## Gazebo Harmonic (Gz Sim 8)

```bash
# Headless (docelowe dla treningu)
gz sim -s -r empty.sdf                   # -s = server only, -r = run od razu

# Z konkretnym światem
gz sim -s -r /ws/src/franka_sim/worlds/pick_place.sdf

# Resource path (fallback, jeśli nie ma w ENV)
export GZ_SIM_RESOURCE_PATH=/opt/franka_ws/install/franka_description/share:$GZ_SIM_RESOURCE_PATH

# Spawn FR3 z /robot_description
ros2 run ros_gz_sim create -name fr3 -topic /robot_description
ros2 run ros_gz_sim create -name fr3 -topic /robot_description -z 0.0

# Introspekcja Gazebo (transport gz, nie ROS!)
gz topic -l                              # lista topików gz
gz topic -e -t /clock                    # echo topiku gz
gz model --list                          # modele w scenie
gz service -l                            # serwisy gz

# Sprawdzenie czy sim chodzi i czy publikuje /clock do ROS
ros2 topic echo /clock --once            # wymaga uruchomionego ros_gz_bridge
```

---

## ros2_control

```bash
# Menedżer kontrolerów — stan
ros2 control list_controllers
ros2 control list_hardware_interfaces    # dostępne command/state interfaces

# Ręczny spawn kontrolera (zwykle w launchu, ale przydatne do debugu)
ros2 run controller_manager spawner joint_state_broadcaster
ros2 run controller_manager spawner joint_trajectory_controller \
  --param-file /ws/src/franka_sim/config/controllers.yaml

# Załaduj / aktywuj / dezaktywuj
ros2 control load_controller joint_trajectory_controller
ros2 control set_controller_state joint_trajectory_controller active
ros2 control set_controller_state joint_trajectory_controller inactive

# Wysłanie testowej trajektorii (sanity check sterowania)
ros2 topic pub --once /joint_trajectory_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory "..."
```

---

## Trening RL (stable-baselines3)

```bash
cd /ws

# Trening (w tle + log)
python3 src/franka_rl/train_sac.py 2>&1 | tee data/logs/sac_$(date +%F_%H%M).log
python3 src/franka_rl/train_ppo.py

# Ewaluacja checkpointu
python3 src/franka_rl/eval.py --checkpoint data/checkpoints/sac_best.zip --episodes 50

# TensorBoard (jeśli SB3 loguje lokalnie)
tensorboard --logdir data/logs --bind_all
```

### Weights & Biases
```bash
wandb login                              # jednorazowo, wklej API key
export WANDB_MODE=offline                # trening offline, sync później
wandb sync data/wandb/offline-run-*      # dosynchronizowanie
export WANDB_PROJECT=diffrl-panda
```

---

## Diffusion Policy

```bash
cd /ws

# Zbieranie demonstracji
python3 src/franka_diffusion/data_collector.py --episodes 100 --out data/demos/

# Finetuning
python3 src/franka_diffusion/finetune.py --config src/franka_diffusion/config.yaml

# Ewaluacja
python3 src/franka_diffusion/eval.py --checkpoint data/checkpoints/dp_best.pt
```

---

## GPU / CUDA

```bash
nvidia-smi                               # stan GPU (na hoście)
watch -n1 nvidia-smi                     # monitoring na żywo

# Sprawdzenie czy PyTorch widzi CUDA (w kontenerze)
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Wersja CUDA w torchu
python3 -c "import torch; print(torch.version.cuda)"
```

---

## Multi-terminal headless (tmux)

Bez GUI wygodnie mieć kilka paneli w jednym `docker exec`:

```bash
tmux                                     # nowa sesja
# Ctrl+b "  → podział poziomy
# Ctrl+b %  → podział pionowy
# Ctrl+b o  → przełączanie paneli
# Ctrl+b d  → detach (sesja żyje dalej)
tmux attach                              # powrót do sesji
```

Typowy układ: panel 1 = `gz sim -s`, panel 2 = spawn + kontrolery, panel 3 = trening/introspekcja.

---

## Git

```bash
cd ~/Inżynierka/DiffRL-Panda
git status
git add -p                               # interaktywne stage'owanie
git commit -m "..."
git log --oneline --graph -15

# Sprawdzenie że data/ jest ignorowane
git check-ignore data/checkpoints/sac_best.zip
```

---

## Szybki debug — typowe problemy

| Objaw | Sprawdź |
|---|---|
| Gazebo: brak meshy FR3 (STL/DAE) | `echo $GZ_SIM_RESOURCE_PATH` — czy wskazuje na `franka_description/share` |
| `command not found: ros2` w exec | Czy `.bashrc` sourcuje ROS; fallback: ręczne `source` |
| Spawn się wiesza | Czy `/robot_description` publikowany: `ros2 topic echo /robot_description --once` |
| Kontroler nie startuje | `ros2 control list_controllers` — stan; `list_hardware_interfaces` — czy interfejsy istnieją |
| `/clock` nie dochodzi do ROS | Czy działa `ros_gz_bridge` dla `/clock`; sim musi chodzić z `-r` |
| Trening nie widzi GPU | `torch.cuda.is_available()` w kontenerze; `nvidia-smi` na hoście |
| colcon build sypie się po refaktorze | `rm -rf build install log` i rebuild |