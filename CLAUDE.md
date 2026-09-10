# CLAUDE.md - DiffRL-Panda

Instrukcje pracy w tym repo dla Claude Code. Pełne tło decyzji projektowych (action space, reward, DP, ewaluacja) jest w `docs/thesis-project-context.md` - czytaj go dla „dlaczego". Komendy operacyjne są w `docs/cheatsheet.md`. Ten plik trzyma reguły pracy w repo; podział treści między te trzy dokumenty opisuje reguła 8.

---

## Środowisko: WSZYSTKO działa w Dockerze

Ten projekt uruchamia się w kontenerze. **Komendy ROS 2 (`colcon`, `ros2`, `gz`, `rosdep`) NIE działają na hoście** - host nie ma zainstalowanego ROS-a. Muszą iść przez kontener.

### Uruchomienie kontenera (raz, na starcie sesji)
```bash
cd docker && docker compose up -d && cd ..
docker ps   # potwierdź, że kontener 'franka_sim' działa
```
Używaj `up -d` (nie `docker compose run`) - daje stabilną nazwę `franka_sim`. `run` generuje losową nazwę i psuje poniższe komendy.

### Uruchamianie komend ROS - przez `docker exec`
Owijaj każdą komendę ROS w:
```bash
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash && [ -f /ws/install/setup.bash ] && source /ws/install/setup.bash; cd /ws && <KOMENDA>"
```
Przykład (build):
```bash
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash; cd /ws && colcon build --packages-select franka_sim"
```

### Edycja plików - normalnie na hoście
Bind-mount (`../src:/ws/src` w compose) synchronizuje pliki host ↔ kontener w czasie rzeczywistym. Edytuj pliki bezpośrednio w `src/` na hoście - zmiany są natychmiast widoczne w `/ws/src` w kontenerze. **Nie** trzeba nic kopiować.

### Ścieżki: host vs kontener
| Host | Kontener |
|---|---|
| `~/Inżynierka/DiffRL-Panda/src` | `/ws/src` |
| `~/Inżynierka/DiffRL-Panda/data` | `/ws/data` |
| `~/Inżynierka/DiffRL-Panda/evaluation` | `/ws/evaluation` |

---

## Reguły, które MUSZĄ być przestrzegane

1. **Po edycji `controllers.yaml` / xacro / launch → walidacja PRZED uruchomieniem.**
   YAML (najczęstsze źródło crashy - wcięcia spacjami, nie tabami; `ros__parameters` z **podwójnym** podkreśleniem):
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

4. **`launch_arguments` w `IncludeLaunchDescription` wymaga `.items()`** - słownik goły rzuca "too many values to unpack".

5. **NIE buduj `franka_hardware` / `franka_ros2`** (ciągną `libfranka`). Budujemy tylko `franka_description` + `franka_msgs` (w obrazie) i własne pakiety `franka_*` z `src/`.

6. **NIE używaj flagi `ros2_control:=true`** na bazowym `fr3.urdf.xacro` - generuje `<transmission>` w stylu ROS 1, niekompatybilne z `gz_ros2_control`. Mamy własny wrapper `fr3_gazebo.urdf.xacro`.

7. **Interfejs poleceń stawów: TYLKO `position`.** NIE dodawaj `velocity` jako *command* interface - bug gz_ros2_control #343 (position+velocity command naraz). `velocity` jest OK jako *state* interface.

8. **Podział dokumentacji: `docs/cheatsheet.md` = „jak", `docs/thesis-project-context.md` = „dlaczego".**
   Przy każdej edycji dokumentów trzymaj się tego rozdziału:
   - **`docs/cheatsheet.md`** - wyłącznie komendy do skopiowania + najwyżej **jednolinijkowa**
     uwaga, bez której komenda nie zadziała lub zadziała źle. Żadnych akapitów tła, historii
     decyzji, odrzuconych wariantów ani tabel zmierzonych faktów.
   - **`docs/thesis-project-context.md`** - decyzje projektowe i ich uzasadnienia, zmierzone fakty
     o modelu/silniku fizyki, co odrzucono i dlaczego, stan implementacji, TODO.

   Test: jeśli treść tłumaczy **powód**, idzie do `thesis-project-context.md`, a cheatsheet co
   najwyżej odsyła. Nowej wiedzy nie dopisuj „gdziekolwiek pasuje" - najpierw zdecyduj, czy to
   „jak", czy „dlaczego".

9. **Pliki tworzone z kontenera są `root`-owe na hoście** (colcon działa jako root). Jeśli edycja z hosta rzuca `EACCES` → `sudo chown -R $USER:$USER src/franka_sim/`. Twórz nowe pliki/katalogi z hosta, żeby tego uniknąć.

---

## Stan implementacji (aktualny - środowisko sterowania DZIAŁA)

### Gotowe i przetestowane
- **Docker**: obraz się buduje, kontener `franka_sim` stoi. `GZ_SIM_RESOURCE_PATH` jako `ENV`, sourcowanie ROS w `.bashrc` (działa `docker exec` bez ręcznego source).
- **`franka_sim`** (ament_cmake) - zbudowany pakiet z `package.xml` + `CMakeLists.txt`. Zawiera:
  - `urdf/fr3_gazebo.urdf.xacro` - własny wrapper: include bazowego opisu + tag `<ros2_control>` (position-only command, position+velocity state, 7 stawów ramienia + **oba** palce: `fr3_finger_joint1` i `fr3_finger_joint2`) + link `world` i fixed joint `world_to_base` + plugin `gz_ros2_control`. Nazwy pluginów: `gz_ros2_control/GazeboSimSystem` (hardware), `gz_ros2_control-system` / `gz_ros2_control::GazeboSimROS2ControlPlugin` (menedżer).
  - `config/controllers.yaml` - `update_rate: 1000`; `joint_state_broadcaster` + `fr3_arm_controller` (JTC, 7 stawów) + `fr3_gripper_controller` (JTC, oba palce). Wszystkie position control.
  - `worlds/fr3_world.sdf` - ground plane + stół główny (0.6x0.8x0.4, static, `0.5 0 0.2`) + dwa stoły docelowe (0.25x0.25x0.4, static, `0 ±0.525 0.2`) + kostka 5 cm (0.05 kg, `mu=1.0`) z pluginem `PosePublisher`. Boczne stoły oddziela od głównego szczelina ~7.5 cm - kostki nie da się przepchnąć, musi zostać podniesiona (uzasadnienie: `docs/thesis-project-context.md`).
  - `launch/bringup.launch.py` - gazebo (headless `-r -s fr3_world.sdf`) + `strip_finger_mimic` na wygenerowanym URDF + mosty ros_gz (`/clock`, poza kostki `/model/cube/odometry`) + robot_state_publisher (`use_sim_time: True`) + spawn robota + spawnery kontrolerów sekwencjonowane przez `RegisterEventHandler(OnProcessExit)`.
- **Uruchomienie**: `ros2 launch franka_sim bringup.launch.py` → Gazebo wstaje, robot spawnuje się w pozie **ready** (joint2=-π/4, joint4=-3π/4, joint6=π/2, joint7=π/4), **trzy kontrolery `active`**. Baza jest **przytwierdzona do świata** - `fr3_gazebo.urdf.xacro` zawiera pusty `<link name="world"/>` + fixed joint `world_to_base` (`world` → `base`, xyz 0 0 0). Bez tego model jest free-floating i przewraca się przy pierwszym kontakcie chwytaka z kostką (wcześniejsza notatka „world-anchoring NIE potrzebny" była błędna - wynikała z testów bez kontaktu). Link `world` musi zostać bez `<inertial>` - sdformat traktuje tę nazwę specjalnie i wiąże model z ramką świata, **nie** robi z niego `<static>true</static>` (statyczny model zablokowałby stawy i zabił `gz_ros2_control`).

- **Chwytak - mimic ROZSTRZYGNIĘTY (Opcja A).** DART nie wspiera mimic constraints, a `DetachableJoint` został przetestowany i **odrzucony** (nie respektuje ruchu stawów aktuowanych przez `gz_ros2_control`). Finalnie: jawne sterowanie oboma palcami - `fr3_finger_joint2` ma własny `command_interface`, a `<mimic>` jest usuwany post-processingiem URDF (`strip_finger_mimic` w `bringup.launch.py`), bo `ros2_control` odmawia `command_interface` na mimic joint. **Nie patchujemy `franka_description`.** Uzasadnienia i wyniki testów: `docs/thesis-project-context.md`, sekcja „Chwytak".
- **Chwyt kostki** zweryfikowany ręcznie, a potem ruchem z polityki: skryptowany ekspert zebrał 100/100 udanych demo na L2 przy `drop_rate` 0.000. **Strojenie tarcia okazało się niepotrzebne** - nie kręć `mu`, dopóki `drop_rate` nie podniesie się przy RL albo DP.
- **`franka_rl`** (ament_python) - warstwa bez ROS (`kinematics.py`, `inverse_kinematics.py`, `observations.py`, `config.py`, `expert.py`) oddzielona od warstwy z `rclpy` (`ros_bridge.py`, `gym_env.py`, skrypty wykonawcze). Testy jednostkowe w `src/franka_rl/test` lecą bez Gazebo i bez rebuildu (`src/franka_rl/conftest.py` - **nie kasować**; poza efektem `sys.path` trzyma autodetekcję symulacji dla testów `@pytest.mark.sim`).
- **Demonstracje**: `collect_demos.py` (`ros2 run franka_rl collect_demos`) - `--save` domyślnie wyłączone, zapisuje tylko epizody udane. Pierwszy zbiór: `data/demos/demo_L2_seed0_20260907_1445/` (100 demo). Komendy: `docs/cheatsheet.md`.

### Następne kroki
1. Trening SAC i PPO (`train_sac.py`, `train_ppo.py`) - etap bieżący.
2. Trening Diffusion Policy na `data/demos/` (`franka_diffusion` czyta gotowe pliki, nie steruje robotem).
3. Krokowanie Gazebo z pętli Gym, faza 2: `ControlWorld` / `multi_step` (faza 1 = free-run + `dt` z `/clock`). Miara, że zrobione: `xfail` na `check_env` zaczyna przechodzić.

---

## Kluczowe decyzje projektowe (skrót - pełne uzasadnienia w thesis-project-context.md)

- **Action space**: 4D delta-EE `(Δx, Δy, Δz, g)`, orientacja zamrożona (chwytak pionowo w dół), max 5 cm/krok @ 10–20 Hz. Plan B: delta joint position 7D.
- **Sterowanie**: position control (JTC), własne DLS-IK (nie MoveIt Servo w pętli), identyczny tor dla RL / DP / eksperta (zasada uczciwości porównania).
- **Obserwacje**: state-based (oracle z Gazebo), znormalizowane [-1,1]. Vision → future work.
- **RL**: SAC (preferowany, sample efficiency).
- **DP**: CNN (U-Net 1D + FiLM), trening od zera (NIE finetuning), DDPM trening / DDIM inferencja.

---

## Cheat sheet - minimum startowe

Pełny zestaw komend: `docs/cheatsheet.md`. Tu tylko to, bez czego nie ruszysz sesji - nie
rozbudowuj tej sekcji, żeby nie utrzymywać dwóch kopii (patrz reguła 8).

```bash
# Kontener
cd docker && docker compose up -d && cd ..
docker ps

# Build franka_sim
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash; cd /ws && colcon build --packages-select franka_sim && source install/setup.bash"

# Launch (główny terminal - użyj -it dla podglądu output)
docker exec -it franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash && source /ws/install/setup.bash; ros2 launch franka_sim bringup.launch.py"

# Introspekcja (drugi exec, gdy launch działa)
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash; ros2 control list_controllers"
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash; ros2 topic echo /joint_states --once"

# Testy jednostkowe franka_rl (bez Gazebo, bez rebuildu)
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash; cd /ws && python3 -m pytest src/franka_rl/test -v"

# Walidacja YAML (na hoście, bez kontenera)
python3 -c "import yaml; yaml.safe_load(open('src/franka_sim/config/controllers.yaml')); print('YAML OK')"
```

---

## Dokumenty powiązane

- `docs/cheatsheet.md` - komendy operacyjne („jak")
- `docs/thesis-project-context.md` - decyzje projektowe i uzasadnienia („dlaczego")
- `docs/constants-derivations.md` - wyprowadzenia stałych liczbowych („skąd ta liczba"); dopisując tam stałą, zawsze oznacz status [W]/[Z]/[P]
- `docs/plan-gym-wrapper-dls-ik.md` - plan wykonawczy bieżącego etapu (Gym + DLS-IK)