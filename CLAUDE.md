# CLAUDE.md — DiffRL-Panda

Instrukcje pracy w tym repo dla Claude Code. Pełne tło decyzji projektowych (action space, reward, DP, ewaluacja) jest w `thesis-project-context.md` — czytaj go dla "dlaczego", ten plik jest dla "jak".

---

## Środowisko: WSZYSTKO działa w Dockerze

Ten projekt uruchamia się w kontenerze. **Komendy ROS 2 (`colcon`, `ros2`, `gz`, `rosdep`) NIE działają na hoście** — host nie ma zainstalowanego ROS-a. Muszą iść przez kontener.

### Uruchomienie kontenera (raz, na starcie sesji)
```bash
cd docker && docker compose up -d && cd ..
docker ps   # potwierdź, że kontener 'franka_sim' działa
```
Używaj `up -d` (nie `docker compose run`) — daje stabilną nazwę `franka_sim`. `run` generuje losową nazwę i psuje poniższe komendy.

### Uruchamianie komend ROS — przez `docker exec`
Owijaj każdą komendę ROS w:
```bash
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash && [ -f /ws/install/setup.bash ] && source /ws/install/setup.bash; cd /ws && <KOMENDA>"
```
Przykład (build):
```bash
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash; cd /ws && colcon build --packages-select franka_sim"
```

### Edycja plików — normalnie na hoście
Bind-mount (`../src:/ws/src` w compose) synchronizuje pliki host ↔ kontener w czasie rzeczywistym. Edytuj pliki bezpośrednio w `src/` na hoście — zmiany są natychmiast widoczne w `/ws/src` w kontenerze. **Nie** trzeba nic kopiować.

### Ścieżki: host vs kontener
| Host | Kontener |
|---|---|
| `~/Inżynierka/DiffRL-Panda/src` | `/ws/src` |
| `~/Inżynierka/DiffRL-Panda/data` | `/ws/data` |
| `~/Inżynierka/DiffRL-Panda/evaluation` | `/ws/evaluation` |

---

## Reguły, które MUSZĄ być przestrzegane

1. **Po edycji `controllers.yaml` / xacro / launch → walidacja PRZED uruchomieniem.**
   YAML (najczęstsze źródło crashy — wcięcia spacjami, nie tabami; `ros__parameters` z **podwójnym** podkreśleniem):
   ```bash
   python3 -c "import yaml; yaml.safe_load(open('src/franka_sim/config/controllers.yaml')); print('YAML OK')"
   ```
   xacro (render bez odpalania Gazebo):
   ```bash
   docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash; ros2 run xacro xacro /ws/src/franka_sim/urdf/fr3_gazebo.urdf.xacro > /tmp/test.urdf && echo OK"
   ```
   Nie odpalaj całego stosu Gazebo, żeby wyłapać literówkę.

2. **`config/`, `launch/`, `urdf/` instalują się do `share/` przez colcon.**
   `$(find franka_sim)/...` i `ros2 launch franka_sim ...` szukają w `share/`, NIE w `src/`. Po każdej edycji tych plików **rebuild**:
   ```bash
   docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash; cd /ws && colcon build --packages-select franka_sim && source install/setup.bash"
   ```
   Gdy dodajesz NOWY katalog do pakietu (np. `worlds/`, `models/`), dopisz go do `install(DIRECTORY ...)` w `CMakeLists.txt`, inaczej ROS go nie zobaczy.

3. **torch instaluj z PyPI, NIE z `download.pytorch.org`.**
   Sieć blokuje/dławi CDN PyTorcha (SSLV3_ALERT_HANDSHAKE_FAILURE). W Dockerfile: `pip install torch torchvision` **bez** `--index-url https://download.pytorch.org/whl/cu124`. PyPI działa i daje build CUDA.

4. **`launch_arguments` w `IncludeLaunchDescription` wymaga `.items()`** — słownik goły rzuca "too many values to unpack".

5. **NIE buduj `franka_hardware` / `franka_ros2`** (ciągną `libfranka`). Budujemy tylko `franka_description` + `franka_msgs` (w obrazie) i własne pakiety `franka_*` z `src/`.

6. **NIE używaj flagi `ros2_control:=true`** na bazowym `fr3.urdf.xacro` — generuje `<transmission>` w stylu ROS 1, niekompatybilne z `gz_ros2_control`. Mamy własny wrapper `fr3_gazebo.urdf.xacro`.

7. **Interfejs poleceń stawów: TYLKO `position`.** NIE dodawaj `velocity` jako *command* interface — bug gz_ros2_control #343 (position+velocity command naraz). `velocity` jest OK jako *state* interface.

8. **Pliki tworzone z kontenera są `root`-owe na hoście** (colcon działa jako root). Jeśli edycja z hosta rzuca `EACCES` → `sudo chown -R $USER:$USER src/franka_sim/`. Twórz nowe pliki/katalogi z hosta, żeby tego uniknąć.

---

## Stan implementacji (aktualny — środowisko sterowania DZIAŁA)

### Gotowe i przetestowane
- **Docker**: obraz się buduje, kontener `franka_sim` stoi. `GZ_SIM_RESOURCE_PATH` jako `ENV`, sourcowanie ROS w `.bashrc` (działa `docker exec` bez ręcznego source).
- **`franka_sim`** (ament_cmake) — zbudowany pakiet z `package.xml` + `CMakeLists.txt`. Zawiera:
  - `urdf/fr3_gazebo.urdf.xacro` — własny wrapper: include bazowego opisu + tag `<ros2_control>` (position-only command, position+velocity state, 7 stawów ramienia + `fr3_finger_joint1`) + plugin `gz_ros2_control`. Nazwy pluginów: `gz_ros2_control/GazeboSimSystem` (hardware), `gz_ros2_control-system` / `gz_ros2_control::GazeboSimROS2ControlPlugin` (menedżer).
  - `config/controllers.yaml` — `update_rate: 1000`; `joint_state_broadcaster` + `fr3_arm_controller` (JTC, 7 stawów) + `fr3_gripper_controller` (JTC, 1 palec). Wszystkie position control.
  - `launch/bringup.launch.py` — gazebo (headless `-r -s empty.sdf`) + most `/clock` (ros_gz_bridge) + robot_state_publisher (`use_sim_time: True`) + spawn robota + spawnery kontrolerów sekwencjonowane przez `RegisterEventHandler(OnProcessExit)`.
- **Uruchomienie**: `ros2 launch franka_sim bringup.launch.py` → Gazebo wstaje, robot spawnuje się w pozie **ready** (joint2=-π/4, joint4=-3π/4, joint6=π/2, joint7=π/4), **trzy kontrolery `active`**. Robot stoi stabilnie (world-anchoring NIE potrzebny — nie dodawać fixed joint do świata).

### Ryzyko #1 (mimic) — PRZETESTOWANE, wymaga decyzji
Silnik fizyki Gazebo Harmonic (DART) **nie wspiera mimic constraints**. `fr3_finger_joint2` (mimic w bazowym URDF) NIE podąża za `fr3_finger_joint1`. Obecnie kontroler chwytaka steruje tylko jednym palcem → chwytak zamyka się asymetrycznie.

**Decyzja otwarta** (do podjęcia przez użytkownika, nie rozstrzygać samodzielnie):
- **Opcja A**: jawne sterowanie oboma palcami (dodać `fr3_finger_joint2` do ros2_control + `controllers.yaml`, wysyłać tę samą komendę obu).
- **Opcja B**: `DetachableJoint` (plan B z dokumentu) — przyspawanie obiektu po wykryciu chwytu, omija fizykę chwytu. Prawdopodobnie lepsze jeśli i tak wystąpi ryzyko #2 (kostka wylatuje).
Zależy od tego, jak modelowany jest chwyt w action space (`g` = binarne z progiem).

### Następne kroki (etap 3–4)
1. Rozwiązać chwytak (Opcja A lub B — patrz wyżej).
2. Scena Pick & Place: stół + kostka SDF w `franka_sim/worlds/` + `models/`. Dopisać katalogi do `install()` w CMakeLists.
3. Wrapper Gymnasium ↔ ROS 2 (`franka_rl/gym_env.py`) — observation/action space wg `thesis-project-context.md` (state-based ~20-30D, 4D delta-EE).
4. DLS-IK + clip do workspace (moduł testowalny jednostkowo).
5. Krokowanie Gazebo z pętli Gym (serwis `/world/<n>/control`, `multi_step`).

---

## Kluczowe decyzje projektowe (skrót — pełne uzasadnienia w thesis-project-context.md)

- **Action space**: 4D delta-EE `(Δx, Δy, Δz, g)`, orientacja zamrożona (chwytak pionowo w dół), max 5 cm/krok @ 10–20 Hz. Plan B: delta joint position 7D.
- **Sterowanie**: position control (JTC), własne DLS-IK (nie MoveIt Servo w pętli), identyczny tor dla RL / DP / eksperta (zasada uczciwości porównania).
- **Obserwacje**: state-based (oracle z Gazebo), znormalizowane [-1,1]. Vision → future work.
- **RL**: SAC (preferowany, sample efficiency).
- **DP**: CNN (U-Net 1D + FiLM), trening od zera (NIE finetuning), DDPM trening / DDIM inferencja.

---

## Cheat sheet — najczęstsze komendy (wszystkie przez docker exec)

```bash
# Kontener
cd docker && docker compose up -d && cd ..
docker ps

# Build franka_sim
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash; cd /ws && colcon build --packages-select franka_sim && source install/setup.bash"

# Launch (główny terminal — użyj -it dla podglądu output)
docker exec -it franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash && source /ws/install/setup.bash; ros2 launch franka_sim bringup.launch.py"

# Introspekcja (drugi exec, gdy launch działa)
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash; ros2 control list_controllers"
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash; ros2 topic echo /joint_states --once"

# Walidacja YAML (na hoście, bez kontenera)
python3 -c "import yaml; yaml.safe_load(open('src/franka_sim/config/controllers.yaml')); print('YAML OK')"
```