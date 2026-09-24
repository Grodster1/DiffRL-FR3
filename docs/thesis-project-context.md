# Praca Inżynierska - Kontekst Projektu

## Informacje ogólne

- **Autor:** Wiktor, Automatyka i Robotyka, 7. semestr, Politechnika Wrocławska
- **Rok akademicki:** 2026/2027
- **Rodzaj pracy:** Eksperymentalna (praca inżynierska)
- **Tytuł PL:** Porównanie uczenia ze wzmocnieniem i polityki dyfuzyjnej w generowaniu trajektorii manipulatora
- **Tytuł EN:** Comparison of Reinforcement Learning and Diffusion Policy in Manipulator Trajectory Generation

---

## Cel pracy

Systematyczne porównanie dwóch podejść do generowania trajektorii manipulatora:
- **Reinforcement Learning:** SAC (Soft Actor-Critic) i PPO (Proximal Policy Optimization)
- **Diffusion Policy:** trening per-task od zera na własnym zbiorze demonstracji (Chi et al., 2023)

Zadanie: **Pick & Place** w trzech wariantach trudności:
1. **L1 - Stała scena:** pozycja obiektu i celu stała
2. **L2 - Randomizacja:** losowa pozycja obiektu w workspace
3. **L3 - Perturbacje:** L2 + dynamiczne zakłócenia w trakcie ruchu

Metryki porównawcze: skuteczność zadania (success rate), efektywność próbkowa, jakość trajektorii (smoothness), odporność na zakłócenia.

---

## Stack technologiczny

| Komponent | Technologia |
|---|---|
| Symulacja | Gazebo Harmonic (Gz Sim 8) - headless (`gz sim -s`) |
| Framework robotyczny | ROS 2 Jazzy (Ubuntu 24.04) |
| Manipulator | Franka FR3 (wcześniej planowana Panda/FER, zmieniono na FR3) |
| Sterowanie | ros2_control + gz_ros2_control |
| RL | stable-baselines3 (SAC, PPO) |
| Diffusion Policy | PyTorch + diffusers |
| Tracking | Weights & Biases |
| Konteneryzacja | Docker + Docker Compose |
| GPU | RTX 4060 (PC), zintegrowana grafika (laptop) |

---

## Harmonogram (10 tygodni + okres wakacyjny)

| Tygodnie | Kamień milowy |
|---|---|
| 1–2 | Przegląd literatury + konfiguracja środowiska - **zamknięte** |
| 3–4 | Implementacja środowiska symulacyjnego (scena P&P, węzły ROS 2, 3 poziomy trudności) - **zamknięte** (otwarta tylko faza 2 krokowania, `multi_step`) |
| 5–6 | Implementacja i trening SAC oraz PPO ← **in progress** |
| 7–8 | Zbiór demonstracji + trening Diffusion Policy - zbiór **zebrany z wyprzedzeniem** (100 demo L2, 07.09.2026); zostaje trening |
| 9–10 | Eksperymenty porównawcze + pisanie pracy |

---

## Ustalenia projektowe - obserwacje, akcje, sterowanie (research 07.2026)

### Przestrzeń obserwacji (wspólna dla SAC, PPO i DP)

Wariant **state-based** (oracle state z Gazebo), wektor ~20–30 wymiarów:

| Składnik | Wymiar | Uwagi |
|---|---|---|
| Pozycja końcówki (EE) | 3 | w bazie robota |
| Orientacja EE | 4 lub 6 | kwaternion albo reprezentacja 6D (Zhou 2019) - spójnie u obu metod |
| Rozwarcie chwytaka | 1 | |
| Pozycja obiektu (kostka) | 3 | rozważyć **względnie**: wektor EE→obiekt (jak panda-gym) |
| Pozycja celu | 3 | rozważyć względnie: wektor obiekt→cel |
| (opcjonalnie) kąty stawów | 7 | |
| (opcjonalnie) prędkości stawów | 7 | |

- Normalizacja wszystkiego do [-1, 1] per wymiar - jedna konwencja w obu pipeline'ach (DP: statystyki datasetu; SB3: `VecNormalize` lub ręcznie).
- DP przyjmuje historię obserwacji ($T_o = 2$), SAC/PPO pojedynczy stan - przy pełnym stanie markowskim nie psuje porównania, odnotować w pracy.
- Wariant vision-based jawnie odrzucony (koszt obliczeniowy, poza harmonogramem) → future work.

### Przestrzeń akcji (wspólna dla SAC, PPO, DP i skryptowanego eksperta)

**4D delta-EE, orientacja zamrożona (chwytak pionowo w dół):**

```
a = (Δx, Δy, Δz, g) ∈ [-1, 1]⁴
```

- Skalowanie: max **5 cm/krok** przy polityce **10–20 Hz** (≈ max 1 m/s EE) - limit bezpieczeństwa + identyczna dynamika dla wszystkich metod.
- Chwytak `g`: ciągłe wyjście sieci, interpretacja **binarna z progiem** (wzór: FurnitureBench), żeby polityka nie trzepotała chwytakiem.
- Uzasadnienie delta-EE: task space = akcje w przestrzeni zadania, wyższa efektywność próbkowa (Matas 2018, Martín-Martín 2019, Zhu 2020); position control > velocity control dla DP (Chi 2023).
- **Opcja zapasowa: delta joint position (7D)** - zero IK, brak problemu osobliwości; literatura pokazuje że bywa lepsza (Effective Tuning Strategies, arXiv:2410.01220 - delta-EE często narusza ograniczenia IK).

### Tor sterowania

```
Polityka (ΔEE @ 10–20 Hz)
  → clip akcji + clip do workspace box (x∈[-0.15,0.7], y∈[-0.6,0.6], z∈[0.42,0.65])
  → IK: damped least-squares na jakobianie (pinocchio), q̇ = Jᵀ(JJᵀ+λ²I)⁻¹·err
  → joint_trajectory_controller (jednopunktowa trajektoria, time_from_start ≈ dt)
  → gz_ros2_control (interfejs pozycyjny) → Gazebo
  → obserwacje wracają do polityki
```

- IK: własna implementacja DLS (deterministyczne, testowalne, bez węzłów MoveIt w pętli treningu).
- **Jakobian 6x7 (decyzja skorygowana 08.2026)**:
 Błąd 6D = `[p_des − p_cur ; log3(R_des · R_curᵀ)]`,
  człon obrotowy z osobną wagą `w_rot`. Frame IK: `fr3_hand_tcp` (punkt między palcami),
  jakobian w `pin.ReferenceFrame.LOCAL_WORLD_ALIGNED`.
- **Kinematyka: pinocchio** Praca opisuje metodę DLS, nie wyprowadzenie kinematyki FR3.
  Uwaga implementacyjna: model budowany z URDF **po** `strip_finger_mimic`, zawiera link `world` + fixed joint, więc `model.nq` = 9 (7 ramienia + 2 palce) - indeksowanie przez
  `model.getJointId(name)`.
- **Fakty o modelu pinocchio** zbudowanym z naszego URDF (po `strip_finger_mimic`)

  | Fakt | Wartość | Konsekwencja |
  |---|---|---|
  | `model.nq` | **9** (7 ramienia + 2 palce) | `world` + fixed joint nie wnoszą DOF; indeksuj przez `model.getJointId(name)`, nigdy pozycyjnie |
  | Frame IK | `fr3_hand_tcp` (id 29 z 54) | **nie** `fr3_link8` |
  | Jakobian z `computeFrameJacobian` | **6x9** | kolumny palców trzeba odciąć przed DLS - powód **wymiarowy**, nie numeryczny (`dq` wyszłoby 9-elementowe). Wcześniejszy zapis „inaczej `JJᵀ` jest osobliwe" **sprostowany**: kolumny palców mają normę `0.0`, więc `JJᵀ` się nie zmienia - wartości singularne `6x9` i `6x7` identyczne. Szczegóły: `docs/constants-derivations.md` §7 |
  | `fr3_joint4` limity | `[-3.077, -0.117]` | `q4 = 0` jest **poza** zakresem - test osobliwości łokcia rób na `-0.117` |
  | FK w pozie ready | `p = [0.307, 0.0, 0.487]`, oś Z TCP = `[0,0,-1]` | poza ready już ma chwytak pionowo w dół → `R_des` bierz wprost z `FK(q_ready).rotation` |

  Pinocchio 4.0.0 wchodzi do obrazu jako zależność tranzytywna `ros-jazzy-moveit`, nie jest
  instalowane jawnie - sourcowanie `/opt/ros/jazzy/setup.bash` wystarcza, by go zaimportować.
- Porażka IK (osobliwość/limit stawu) = no-op + ewentualna mała kara; **logować częstość**
  (ciekawa statystyka porównawcza DP vs RL) - licznik `ik_failures` w `info` wrappera.
- JTC zamiast forward_position_controller - interpolacja między komendami = gładszy ruch, istotne przy metryce smoothness.
- **Zasada uczciwości porównania:** identyczny action space, identyczny kontroler i konfiguracja dla RL, DP **i eksperta zbierającego demonstracje** (ekspert nagrywa sekwencje (obs, ΔEE, g) wykonywane tym samym torem - NIE surowe plany MoveIt).

### Chwytak - ostrzeżenia praktyczne (decyzja podjęta)

1. Mimic joints w gz_ros2_control/DART **niewspierane** (potwierdzone) - `fr3_finger_joint2` nie podąża za `fr3_finger_joint1`.
2. **`DetachableJoint` przetestowany i odrzucony jako niekompatybilny z `gz_ros2_control`
   (gz-sim 8.11.0, ROS Jazzy vendor).** Mechanizm sam w sobie działa poprawnie
   (zweryfikowane na izolowanym minimalnym świecie: default-attach, `detach`,
   re-`attach` via topic - wszystko 1:1 zgodne z oczekiwaniami). Problem: gdy
   `parent_link` znajduje się na łańcuchu stawów aktuowanych przez
   `gz_ros2_control` (position command interface), rzeczywisty ruch ramienia NIE
   jest respektowany przez sztywne ograniczenie `DetachableJoint` - przyczepiony
   obiekt nie podąża za ruchem (zweryfikowane na `fr3_link1` i `fr3_link7`;
   działa tylko przy sztywnym teleportowaniu całego modelu, co nie ma zastosowania
   przy realnym sterowaniu). Wniosek: `gz_ros2_control`'s pozycyjne komendy
   najpewniej nie przechodzą przez pełny solver dynamiki zgodny z dodatkowymi
   (closed-loop) ograniczeniami.
3. **Ostateczna decyzja: Opcja A** - jawne sterowanie oboma palcami
   (`fr3_finger_joint2` jako pełnoprawny `command_interface`/`state_interface` w
   `ros2_control`, dopisany do `fr3_gripper_controller` w `controllers.yaml`).
   Wymagało dodatkowo usunięcia znacznika URDF `<mimic>` z `fr3_finger_joint2`
   (hardkodowany w `franka_hand.xacro`, brak parametru do wyłączenia) -
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
5. **Fizyka chwytu - strojenie NIEPOTRZEBNE (rozstrzygnięte 07.09.2026).** Chwyt
   i podniesienie kostki zweryfikowane najpierw ręcznie
   (`rqt_joint_trajectory_controller`), a następnie ruchem z polityki: skryptowany
   ekspert zebrał 100/100 udanych demo na L2 przy `drop_rate` 0.000, na domyślnym
   tarciu palców i `mu=1.0` kostki. To był właśnie odkładany test poślizgu -
   statyczne podniesienie go nie zastępowało, ruch 5 cm/krok tak. Poniższe
   ustalenia zostają jako opis mechanizmu i punkt wyjścia, gdyby `drop_rate`
   podniósł się przy RL albo DP (polityka może szarpać inaczej niż automat faz):
   - **DART ignoruje `<contact><ode>`** - `kp`, `kd`, `min_depth` w
     `fr3_world.sdf` to parametry ODE, silnik (`gz-physics-dartsim`) ich nie
     czyta. Z całej sekcji `<surface>` działa wyłącznie
     `<friction><ode><mu>/<mu2>`. Nie tracić czasu na kręcenie tymi gałkami.
   - **Tarcie liczy się z pary kontaktowej**, nie z jednego kształtu. Kostka ma
     `mu=1.0`, ale palce (`fr3_leftfinger` / `fr3_rightfinger` z
     `franka_description`) nie mają zdefiniowanego tarcia - biorą domyślne.
     Podbicie samej kostki daje ograniczony efekt. Docelowo bloki
     `<gazebo reference="fr3_leftfinger">` z `<mu1>`/`<mu2>` w naszym
     `fr3_gazebo.urdf.xacro` - **nie** patchujemy `franka_description` (ta sama
     zasada co przy `strip_finger_mimic`). Nazwy linków potwierdzić w
     wygenerowanym URDF przed implementacją.
   - **Drugi lever obok tarcia: siła normalna.** Siła tarcia = `mu` x siła
     normalna, a ta przy sterowaniu pozycyjnym bierze się z „przesterowania"
     komendy palców (kostka 5 cm → palec nominalnie 0.025 m; komenda niższa =
     docisk). Do rozstrzygnięcia razem z progiem binarnym `g` w action space -
     jaką konkretnie wartość zamknięcia wysyła polityka.
   - Przy finalnym strojeniu **odnotować w pracy**, czy użyte `mu` jest
     fizycznie uzasadnione (okładziny palców FR3 na plastiku), czy potraktowane
     jako czysty parametr symulacji.

### Wznawianie treningu - `--resume` (09.09.2026)

`train_sac.py` co `--save-freq` kroków nadpisuje **jeden** punkt wznowienia w katalogu przebiegu:
`latest.zip` + `latest_replay_buffer.pkl`. Osobno od historycznych checkpointów, bo:

- **Sam model nie wznawia SAC-a.** Bez bufora uczeń rusza z pustym zbiorem off-policy, czyli
  formalnie „wznowiony" przebieg wyrzuca całą zebraną wiedzę o środowisku. Bufor musi lecieć razem.
- **Bufor waży ~150 MB** przy `buffer_size=1e6`, więc zapis jednej kopii per checkpoint kosztowałby
  gigabajty na przebieg. Stąd stała nazwa i nadpisywanie: to punkt przetrwania awarii, nie historia.
- **`--timesteps` jest CELEM, nie budżetem wywołania.** SB3 przy `reset_num_timesteps=False`
  **dodaje** `total_timesteps` do licznika, więc naiwne przekazanie celu przy wznowieniu przestrzeliwuje
  o wszystko, co już przetrenowane (zmierzone: wznowienie z 1200 na „1800" doszło do 3000).
  `remaining_timesteps()` odejmuje; wznowienie do już osiągniętej liczby kończy się komunikatem,
  nie kolejnym przebiegiem.
- **`monitor.csv` nie jest dopisywany, tylko nadpisywany** przez `Monitor`, więc każde wznowienie
  dostaje własny plik `resumeN.monitor.csv`. `load_results()` globuje `*monitor.csv` i widzi komplet.

### Reward i RL - ustalenia

- Sparse reward dla czystego SAC/PPO nierozwiązywalny w budżecie → **shaped reward** dla obu algorytmów: kara odległości EE–obiekt + bonus za chwyt + kara odległości obiekt–cel + bonus sukcesu + mała kara ‖a‖².
- Kara ‖a‖² w nagrodzie RL: tak (standard, cytat panda-gym/Fetch). **Żadnych filtrów dolnoprzepustowych na akcjach** u żadnej metody - smoothness raportowana z surowych trajektorii.
- SAC+HER możliwy jako eksperyment dodatkowy (tylko off-policy; SB3 `HerReplayBuffer`). PPO nie wspiera HER.
- Oczekiwanie: SAC 5–10x efektywniejszy próbkowo niż PPO; PPO może nie zdążyć na L2/L3 - to też jest wynik (metryka efektywności próbkowej).
- Curriculum: trening L2 startujący z wag L1 - element metodologii.

### Diffusion Policy - ustalenia

- Wariant **CNN (U-Net 1D + FiLM)**, nie Transformer (łatwiejszy tuning wg autorów).
- Trening: DDPM ~100 kroków; inferencja: **DDIM ~10 kroków** (inaczej nie zmieści się w częstotliwości sterowania). Latencja inferencji DP vs MLP = dodatkowa metryka.
- Chunking: T_o = 2, T_p = 16, T_a = 8 (wartości z papieru, do ablacji).
- **Korekta terminologii w pracy:** oryginalny DP trenowany per-task od zera na 50–200 demonstracjach - nie "finetuning pretrenowanego modelu" (to domena VLA typu Octo/π₀). U nas: trening od zera na własnym datasecie.
- Demonstracje: **scripted expert** - automat 8 faz (approach → descend → close → lift → transport
  → place → release → settle) emitujący akcje 4D w tej samej przestrzeni co SAC i DP, przepuszczane
  przez ten sam `env.step()`. **Nie** waypointy przez MoveIt (wcześniejszy zapis „sekwencja
  waypointów przez IK / MoveIt" sprostowany 09.2026) - demonstracja spoza przestrzeni akcji polityki
  jest niereprodukowalna, więc bezużyteczna jako cel imitacji. Cel: 100–200 udanych demo na L2 -
  **pierwsze 100 zebrane 07.09.2026** (`data/demos/demo_L2_seed0_20260907_1445/`, 100/100 prób,
  ~8 min). Dosypanie drugiej setki innym seedem jest tanie i daje materiał na ablację „rozmiar
  zbioru vs jakość DP".
  Uwaga: scripted expert = demonstracje unimodalne → osłabia atut multimodalności DP, uczciwie
  przedyskutować (→ Mandlekar 2021).
  Format zbioru: `ep_XXXX.npz` per epizod (`obs` T+1, `action` T, `reward` T, `goal_pos`) +
  `meta.json` z normalizacją i układem obserwacji - bez tego pliku zbiór jest za pół roku
  nieczytelny, bo `REL_SCALE`/`WORKSPACE_BOX` mogą się zmienić. Zapisywane są wyłącznie epizody
  udane; `episodes.jsonl` i `summary.json` trzymają wszystkie próby, żeby `success_rate` eksperta
  pozostał mierzalny.

### Ewaluacja i eksperymenty - ustalenia

- Protokół: ≥50 epizodów testowych x ≥3 seedy treningowe, średnia ± odchylenie, identyczne ziarna randomizacji sceny dla wszystkich metod.
- Smoothness: całka z jerku / suma kwadratów przyspieszeń stawów + długość ścieżki EE.
- Efektywność próbkowa - dwie osie: kroki środowiska (RL) vs koszt demonstracji (DP); raportować obie.
- **Hipoteza na L3 (najciekawszy potencjalny wynik):** chunking DP (otwarta pętla przez T_a kroków) vs reaktywność RL co krok - przewaga DP z L1/L2 może stopnieć/odwrócić się przy perturbacjach; ablacja T_a vs smoothness.
- Gazebo ↔ Gym: krokowanie symulacji przez serwis `/world/<name>/control` (`multi_step`) albo pauza+odpauzowanie - bez tego trening niepowtarzalny i wolny. Argument za Gazebo mimo to: integracja ROS 2 + realizm stacku sterowania (MuJoCo/Isaac = standard społeczności RL, odnotować).
- **Wdrożenie krokowania: etapowe (decyzja 08.2026).** Faza 1 - symulacja leci swobodnie,
  wrapper odmierza `dt` z `/clock` (`use_sim_time=True`); pozwala szybko domknąć end-to-end
  i testować IK oraz reward. Faza 2 - podmiana na `multi_step` przez `ros_gz_interfaces/srv/ControlWorld`.
  Warunek projektowy: cały kontakt z czasem schowany za **jedną** metodą `reserve_t(dt)` w warstwie
  ROS, więc faza 2 nie dotyka `gym_env`. Ryzyko fazy 2 do rozstrzygnięcia w implementacji: wyścig
  między publikacją trajektorii a krokiem świata (komenda musi dojść do JTC zanim ruszymy sim).
  Miarą, kiedy faza 2 jest konieczna, jest test determinizmu: `reset(seed=42)` dwa razy → ta sama
  obserwacja startowa.
- **`real_time_factor` 0 (bez dławika) - 3.1x szybciej, jakość zbioru bez zmian (09.09.2026).**
  W `fr3_world.sdf` `<real_time_factor>` zmieniony z `1.0` na `0` (silnik liczy tak szybko, jak
  pozwala CPU, `max_step_size` zostaje 1 ms). Ekspert na L2/seed 0: **32.4 FPS wobec 10.5**,
  100 udanych demo w ~2.7 min zamiast ~8. Metryki jakości identyczne (`success_rate`,
  `grasp_rate` 1.000, `drop_rate`, `ik_failure_rate` 0.000, zero porażek budżetu fazy).
  Zbiór: `data/demos/demo_L2_seed0_20260909_1213/`.
  **Uwaga (13.09.2026):** wartość `real_time_factor` nie ma związku z blokadą `fr3_joint2`.
  **Nie wywoływać `set_physics` w locie** - wyłącza grawitację. Szczegóły: „Znane problemy", pkt 1.
- **Cena zdjęcia dławika: `dt` przestaje być stałe (zmierzone 09.09.2026).** `reserve_t(dt)`
  jest **ograniczeniem dolnym** - czeka, aż zegar symulacji przesunie się o `dt`, ale niczego nie
  wstrzymuje, więc każda milisekunda liczona po stronie Pythona (IK, budowa obserwacji, u RL
  dodatkowo krok gradientu SAC) to sim-time, który upływa poza kontrolą pętli, a JTC trzyma w tym
  czasie ostatnią komendę. Przy `real_time_factor=1.0` symulacja była wolniejsza od Pythona
  i przeregulowanie się nie ujawniało; przy `0` ujawnia się jako **efektywny krok większy
  i zmienny**. Ślad w danych eksperta: mediana najlepszego zbliżenia EE→kostka **0.27 mm → 0.71 mm**,
  średnia długość epizodu 50.87 → 51.77 kroku. Dla eksperta bez znaczenia (chwyt dalej 100%),
  ale **narzut na krok rośnie z ciężarem polityki**, więc przy SAC/DP przeregulowanie będzie
  większe niż zmierzone tutaj. To najmocniejszy dotąd argument za fazą 2 (`multi_step`):
  dopiero ona daje `dt` z górnym ograniczeniem.
- **Faktyczny okres sterowania pod SAC - ZMIERZONY, przeregulowanie ~1% (09.09.2026).**
  `gym_env` liczy sim-time między kolejnymi publikacjami komendy (czyli pełny okres, razem
  z IK, obserwacjami i krokiem gradientu) i raportuje `sim_dt_mean` / `sim_dt_max` w `info`;
  `train_sac.py` przepuszcza je przez `Monitor(info_keywords=...)`, więc siedzą w `monitor.csv`.
  Przebieg L1, 2000 kroków, `buffer_size=20000`:

  | Faza | `sim_dt_mean` | `sim_dt_max` |
  |---|---|---|
  | przed `learning_starts` (epizody 0-4) | 50.3-50.4 ms | 51-52 ms |
  | epizod przecinający `learning_starts=1000` | **58.0 ms** | **1039 ms** |
  | po rozgrzaniu (epizody 6-9) | 50.4-50.6 ms | 52-94 ms |

  Wniosek: **obawa o rozjazd `dt` pod RL okazała się przesadzona.** W stanie ustalonym krok
  gradientu SAC kosztuje ~1% nominalnego `dt` (0.4-0.6 ms), sporadyczne szpilki sięgają ~2x`dt`
  i są rzadkie. Sekunda w epizodzie 5 to **jednorazowa** rozgrzewka CUDA przy pierwszym kroku
  uczenia, nie stan ustalony - psuje jeden epizod na przebieg. Faza 2 (`multi_step`) zostaje
  potrzebna dla **determinizmu** (`check_env`, powtarzalność ziaren), ale nie jest już warunkiem
  sensownego treningu SAC. Miarę zostawiamy włączoną: jeśli `sim_dt_mean` odjedzie od nominalnego
  `dt` przy DP (inferencja DDIM jest wyraźnie droższa od MLP), zobaczymy to w `monitor.csv`
  zamiast zgadywać.

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
│   ├── franka_sim/                   # Pakiet ament_cmake: scena Gazebo + launch files - ISTNIEJE
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
│   │   ├── config.py                # stałe (progi, wagi reward, geometria) - ZERO ROS
│   │   ├── kinematics.py            # FK + jakobian (pinocchio), ZERO ROS
│   │   ├── inverse_kinematics.py    # DLS + clip workspace/limity, ZERO ROS
│   │   ├── observations.py          # obs_dict (SI) + normalizacja, ZERO ROS
│   │   ├── urdf_utils.py            # strip_finger_mimic, ZERO ROS
│   │   ├── expert.py                # skryptowany ekspert (automat 8 faz), ZERO ROS
│   │   ├── ros_bridge.py            # jedyne miejsce z rclpy; reserve_t(dt)
│   │   ├── gym_env.py               # Wrapper Gymnasium ↔ ROS 2
│   │   ├── test/                    # pytest - uruchamialny bez Gazebo
│   │   ├── random_baseline.py       # baseline losowej polityki (diagnostyka środowiska)
│   │   ├── collect_demos.py         # pętla wykonawcza eksperta: diagnostyka + zapis demo
│   │   ├── train_sac.py
│   │   ├── train_ppo.py
│   │   └── eval.py
│   │
│   └── franka_diffusion/             # Pakiet ament_python: Diffusion Policy
│       ├── dataset.py               # czyta gotowe pliki demo z data/demos/
│       ├── train.py
│       └── eval.py
│
├── evaluation/                       # Automatyczny protokół ewaluacji
├── data/                             # Demonstracje, checkpointy, wyniki (gitignored)
└── docs/
```

Podział na pakiety ROS 2:
- `franka_sim` → `ament_cmake` (launch files, konfiguracja Gazebo, URDF)
- `franka_task`, `franka_rl`, `franka_diffusion` → `ament_python`

**Zbieranie demonstracji należy do `franka_rl`, nie do `franka_diffusion` (sprostowanie 09.2026).**
Wcześniejszy szkic struktury umieszczał `data_collector.py` w `franka_diffusion`. To jest złe
miejsce: kolektor musi importować `gym_env`, czyli ciągnie `rclpy`, `xacro` i całą warstwę ROS.
`franka_diffusion` ma **czytać gotowe pliki** z `data/demos/`, a nie sterować robotem - inaczej
trening DP nie da się uruchomić bez stojącego Gazebo. Ta sama granica, która oddziela
`kinematics.py` od `ros_bridge.py`, obowiązuje między pakietami.

Z tego samego powodu ekspert jest rozbity na dwa pliki: `expert.py` (czysty numpy, testowalny
bez Gazebo) i `collect_demos.py` (pętla wykonawcza, importuje `gym_env`). Gdyby siedziały razem,
`import franka_rl.expert` w teście jednostkowym zaciągnąłby cały stos ROS i testy eksperta
przestałyby chodzić poza kontenerem z żywą symulacją.

---

## Stan implementacji - co działa

### Docker
- **Obraz bazowy:** `ros:jazzy-ros-base` + `ros-jazzy-desktop` (zainstalowane w całości zamiast pojedynczych paczek)
- **Dodatkowe paczki ROS 2:** `ros-jazzy-ros-gz`, `ros-jazzy-gz-ros2-control`, `ros-jazzy-moveit`
- **PyTorch z PyPI** (`pip install torch torchvision`, **bez** `--index-url download.pytorch.org` - CDN PyTorcha blokowany przez sieć, SSLV3_ALERT_HANDSHAKE_FAILURE). PyPI też daje build CUDA; jeden obraz, GPU na PC i CPU na laptopie automatycznie.
- **Pin `setuptools<81`** (warstwa po instalacjach pip). setuptools ≥ 82 nie zawiera już `pkg_resources`, a `/usr/bin/rosdep` go importuje → build wywalał się na `rosdep install` z `ModuleNotFoundError`. Instalacje pip podmieniają systemowy setuptools na najnowszy z PyPI, więc pin **musi** być po nich. W tej samej warstwie asercja `python3 -c "import pkg_resources"`, żeby regresja wywaliła się od razu, a nie 15 minut później.
- **Franka ze źródeł:** klonowane `franka_description` + `franka_ros2` (branch `jazzy`), budowane TYLKO `franka_description` i `franka_msgs` (`--packages-select`)
- **docker-compose:** jeden serwis `sim`, volumes montują `src/`, `evaluation/`, `data/`; `/dev/dri` + rezerwacja GPU nvidia (capabilities `gpu, graphics, display, compute, utility`) - renderowanie GUI przez GPU
- **Entrypoint + `.bashrc`:** sourcowanie ROS i `franka_ws` automatyczne, `docker exec` działa bez ręcznego `source`
- **`GZ_SIM_RESOURCE_PATH` jako `ENV`** w Dockerfile → meshe FR3 ładują się bez eksportu

### Franka FR3 w ROS 2 - środowisko sterowania DZIAŁA
- Własny wrapper `fr3_gazebo.urdf.xacro`: bazowy opis + `<ros2_control>` (position command, position+velocity state) + plugin `gz_ros2_control` + link `world` i fixed joint `world_to_base`
- `bringup.launch.py`: Gazebo headless (`-r -s fr3_world.sdf`) + most `/clock` + `robot_state_publisher` + spawn robota + spawnery kontrolerów sekwencjonowane przez `OnProcessExit`; dodatkowo `strip_finger_mimic` na wygenerowanym URDF
- Robot spawnuje się w pozie **ready**, trzy kontrolery `active`
- Chwyt i podniesienie kostki zweryfikowane ręcznie (`rqt_joint_trajectory_controller`)

### Pętla deweloperska `franka_rl` (pakiet `ament_python`)
- **`--symlink-install` NIE działa dla `ament_python`** (sprawdzone na `franka_rl`: zero
  symlinków w `install/`, edycja w `src/` niewidoczna bez rebuildu). Colcon degraduje się po
  cichu do zwykłej kopii - skutek zmiany trybu editable install w setuptools ≥ 64, bez
  ostrzeżenia. Nie wracać do tej flagi.
- **Rozwiązaniem dla TDD jest `src/franka_rl/conftest.py`.** pytest w trybie importu
  „prepend" wstawia katalog zawierający `conftest.py` na początek `sys.path`, więc
  `import franka_rl` rozwiązuje się do `src/franka_rl/franka_rl/`, a nie do kopii w
  `/ws/install/`. Dzięki temu testy jednostkowe (`kinematics.py`, `ik.py` - zero ROS) lecą bez
  `colcon build` i bez Gazebo. **Nie kasować.** `colcon build` jest potrzebny dopiero, gdy
  moduł ma działać jako zainstalowany pakiet (`ros2 run`, launch, skrypty treningowe).

  Plik był początkowo pusty (efekt dawała sama jego obecność); od 09.2026 ma **drugie,
  niezależne zadanie** - autodetekcję żywej symulacji. To zmienia zasadę „nie kasować"
  z ezoterycznej w oczywistą, ale obie role trzeba trzymać w głowie osobno: usunięcie
  hooków nie przywraca stanu „pusty, ale potrzebny", tylko psuje jedną z dwóch rzeczy.
- **Testy dzielą się markerem `sim`, nie zmienną środowiskową (decyzja 09.2026).** Testy
  wymagające stojącego `bringup.launch.py` są oznaczone `@pytest.mark.sim`; `conftest.py`
  raz na sesję sprawdza, czy symulacja żyje, i pomija je z czytelnym powodem, gdy nie żyje.
  Jedna komenda (`pytest src/franka_rl/test`) działa w obu światach: bez Gazebo przechodzi
  część czysta, z Gazebo całość. Odrzucono wariant z `FRANKA_SIM_LIVE=1` - gate, o którym
  trzeba pamiętać, zamienia „testy nie przeszły" w „testy się nie uruchomiły" i nikt tego
  nie zauważa.
- **Detekcja żywej symulacji sprawdza WIADOMOŚĆ z `/clock`, nie wydawcę (pułapka zmierzona
  09.2026).** Pierwsza wersja pytała `count_publishers("/clock") > 0` i dawała fałszywy
  pozytyw: procesy `ros_gz_bridge` **przeżywają `gz sim`**, więc po zabiciu samego Gazebo
  na `/clock` dalej wisi wydawca, przez którego nic nie leci. Skutek byłby gorszy niż brak
  detekcji - testy nie zostałyby pominięte, tylko zawisłyby w `reserve_t()` czekając na tick,
  który nigdy nie przyjdzie. Warunek „przyszła wiadomość" jest dokładnie tym, czego wymaga
  `reserve_t`. Praktyczna konsekwencja przy sprzątaniu: `pkill -f 'gz sim'` **nie kończy
  sesji** - mosty trzeba ubić osobno, inaczej następny przebieg testów widzi zombie-graf ROS.
- **Kontrakt Gym: `check_env` nie przechodzi w fazie 1 i nie ma prawa przejść.** Gymnasium
  wymaga bit-w-bit identycznej obserwacji po `reset(seed)` przy tej samej akcji, a przy
  free-run zegar symulacji biegnie niezależnie od pętli Gym. W testach rozbite na dwa:
  statyczna część kontraktu (obserwacja mieści się w zadeklarowanej przestrzeni, typy
  zwrotek) przechodzi normalnie, pełny `check_env` jest `xfail` z uzasadnieniem.
  **Ten `xfail` jest operacyjną miarą, kiedy faza 2 (`multi_step`) jest zrobiona** - gdy
  zacznie przechodzić, przejście się dokonało. Zastępuje to opisany niżej „test determinizmu
  resetu" konkretnym, uruchamialnym warunkiem.
- Pliki `__pycache__/`/`.pytest_cache/` tworzone przez `docker exec` są root-owe na hoście
  (ta sama przyczyna co przy colconie) - kasować z kontenera, nie `sudo` z hosta.

### Most serwisu `set_pose` - CLI, nie `config_file` (zmierzone 08.2026)

`ros_gz_bridge` 1.0.22 ma **dwie różne ścieżki kodu** do mostkowania serwisów i wybór między nimi
nie jest kosmetyczny:

- `parameters=[{'config_file': ...}]` → klasa `RosGzBridge`, która trzyma `heartbeat_timer_` (1 Hz)
  wołający `spin()`. Każdy tick ponownie woła `add_service_bridge` i **dopisuje** nowy obiekt do
  `std::vector<rclcpp::ServiceBase::SharedPtr> services_` - nic nie sprawdza, czy most już istnieje.
  Objaw widoczny: log `Creating ROS->GZ service bridge` co dokładnie 1.000 s.
  **Zmierzony skutek: RSS rośnie liniowo ~108 KB/s (1.06 MB / 10 s), czyli ~390 MB/h.** Przy
  wielogodzinnym treningu SAC to realny wyciek, nie hałas w logu.
- `arguments=['<svc>@<ros_srv_type>@<gz_req>@<gz_rep>']` → zwykły `main` w `parameter_bridge`, bez
  heartbeatu. Log leci raz, RSS płaski (zmierzone: 51036 KB stałe przez 50 s).

Używamy wariantu CLI. `ros2 run ros_gz_bridge parameter_bridge --help` potwierdza wsparcie dla
serwisów w CLI - mit, że serwisy wymagają YAML-a, jest fałszywy. Zweryfikowane funkcjonalnie:
`ros2 service call .../set_pose` → `success=True`, kostka faktycznie przeskakuje `(0.5,0)→(0.45,0.15)`
potwierdzone na `/model/cube/odometry`. `config/bridge.yaml` jest po tej zmianie martwy.

### Gazebo Harmonic
- Headless działa: `gz sim -s -r <world>`
- **GUI działa** - mit „Wayland to uniemożliwia" okazał się fałszywy, brakowało wyłącznie `xhost +local:docker` na hoście (XWayland). Klient `gz sim -g` dołącza do headless serwera z bringupa. GUI = podgląd/debug, trening zawsze headless.
- Scena `fr3_world.sdf`: ground plane + stół główny (0.6x0.8x0.4, static, pose `0.5 0 0.2`) +
  dwa stoły docelowe (0.25x0.25x0.4, static, pose `0 ±0.525 0.2`) + kostka 5 cm (masa 0.05 kg, `mu=1.0`)

### Cel zadania - dwa boczne stoły zamiast punktu w powietrzu (decyzja 08.2026)

Cel = środek blatu jednego z dwóch bocznych stołów: `(0, ±0.525, 0.425)`. Stoły stoją po lewej i
prawej stronie robota (90° w Z względem stołu głównego), robot jest w środku trójkąta.

- **Odrzucone: cel w powietrzu** (Fetch / panda-gym). Formalnie działa - sukces sprawdza się w
  chwili, gdy robot trzyma kostkę w tolerancji, epizod się kończy. Ale to zadanie „przynieś obiekt
  do pozy", nie „odłóż"; słabo broni się w opisie ewaluacji pracy o pick & place.
- **Odrzucone: podest wyższy od blatu.** Miał wymuszać podniesienie pionowym stopniem. Zbędny -
  szczelina daje to samo, a każdy centymetr wysokości to ryzyko kolizji dla DLS-IK, który nie
  unika przeszkód.
- **Wybrane: szczelina zamiast wysokości.** Boczne stoły są w `x∈[-0.125, 0.125]`, główny zaczyna
  się na `x=0.2` → przerwa ~7.5 cm nad podłogą przy kostce 5 cm. Popychanie nie ma rozwiązania:
  kostka spada, zamiast wjechać na stół docelowy. Degeneracja „pick & place → push" znika bez
  żadnych sztuczek w reward. Sukces jest sprawdzalny w stanie ustalonym (kostka *leży*, `‖v‖≈0`),
  co jest mocniejszą definicją niż migawka w powietrzu.
- **Dwa stoły, nie jeden** - wymusza ruch w obie strony (joint1 ±90°), daje symetryczne pokrycie
  workspace i bimodalny rozkład celów (atut dla DP, uczciwy test dla SAC). Jeden stół pozwalałby
  nauczyć się stałej trajektorii, co osłabiałoby wymowę L2.
- Rozkład celów wg poziomu: **L1** cel stały (lewy), **L2/L3** losowo 50/50 lewy/prawy.
  Rozkład celów jest dwupunktowy, ale wektor `cube→goal` w obserwacji pozostaje ciągły, bo
  losowana jest pozycja startowa kostki (`x∈[0.4,0.6]`, `y∈[-0.2,0.2]`).
- Jitter pozycji na blacie docelowym świadomie pominięty: stół 0.25 m przy kostce 5 cm daje mały
  zapas od krawędzi, a bimodalny cel już realizuje sens L2.

**Konsekwencje do zweryfikowania w symulacji (nie policzone, tylko oszacowane):**
- **Zamrożona orientacja przy 90° - ZWERYFIKOWANE, zapas duży (08.2026).** Test kinematyczny
  (DLS-IK, marsz po 5 cm z `q_init` z poprzedniego kroku - tak jak zrobi to polityka):
  oba cele `(0, ±0.525, 0.425)` osiągalne z błędem **0.50 mm**, cała strefa spawnu kostki
  (9 punktów) ≤ 0.61 mm, pełny transport (chwyt → lift → przeniesienie → odłożenie) w obu
  kierunkach bez potknięcia. **0 porażek IK na 154 waypointów.** Błąd orientacji ≤ **0.01°** -
  potwierdza sens decyzji o jakobianie 6x7 (orientacja realnie regulowana, nie dryfuje).
  joint7 wykorzystuje `-0.601 … +2.172` przy limitach `±3.051`; najmniejszy margines do limitu
  ze wszystkich 7 stawów to **0.720 rad (~41°)**. Wcześniejsze oszacowanie `π/4 ± π/2` było
  pesymistyczne. **Wariant awaryjny z 5D więzami (yaw swobodny) nie jest potrzebny.**
  Zastrzeżenie: to czysta kinematyka TCP - model kolizyjny nie był ładowany do pinocchio, więc
  test **nie** mówi nic o kolizjach ze stołami, autokolizjach ani o nadążaniu JTC.
- **Dystans transportu ~0.73 m** (kostka `(0.5,0)` → cel `(0,±0.525)`). Przy 5 cm/krok min. ~14
  kroków samego przenoszenia; `max_episode_steps=200` starczy. Zadanie wyraźnie trudniejsze niż
  typowe panda-gym (~0.15 m) → strojenie `mu` palców przestaje być odkładalne.
- **Kostka może spaść w szczelinę** (pod spodem jest podłoga `z=0`). W `step()` przerywać epizod
  przy `cube_z < 0.35` (`terminated=True`, bez bonusu), inaczej agent ciągnie 200 kroków stanu bez
  powrotu i zaśmieca replay buffer.
- **Narożniki stołu głównego przy `y=±0.4`** są blisko bocznych stołów; przedramię przechodzi nad
  tym rejonem przy joint1=±90°. Stół zwężono z 1.0 na 0.8 w `y` profilaktycznie.
  **ZAMKNIĘTE - kolizji nie ma (08.2026).** Przejazd 87 waypointów przez JTC w Gazebo (44 s,
  pełna sekwencja: chwyt w rogu strefy spawnu → lift → transport na przeciwległy stół →
  odłożenie → powrót, oba kierunki, joint1 przez ±90°): **max błąd nadążania 0.0062 rad (0.35°)**
  na 62997 próbkach `/fr3_arm_controller/controller_state`. Kolizja ze statycznym stołem
  objawiłaby się trwałym rozjazdem komenda ↔ stan, bo stół nie ustąpi. Test nie jest pusty -
  ramię faktycznie dojechało do ostatniego waypointu (`TCP = [0, -0.5246, 0.5748]`, błąd 0.47 mm).
  Metoda przydatna ponownie przy każdej zmianie geometrii sceny: mierz `error.positions`
  z `controller_state`, próg podejrzenia ~0.1 rad.

---

## Znane problemy i TODO

### Otwarte
1. **Zawieszony staw po długiej serii testów** - po dłuższej pracy w tym samym kontenerze
   zaobserwowano staw niereagujący na komendy mimo `claimed` w `list_hardware_interfaces`.
   Naprawia `docker compose restart sim`. Przyczyna nierozpoznana; jeśli wróci w pętli treningu,
   będzie wymagać diagnozy (podejrzenie: stan `gz_ros2_control` po wielu re-spawnach).

   **Wrócił w pętli treningu - sygnatura zmierzona (12.09.2026).** Przebieg SAC L1 z PBRS
   (`sac_L1_seed0_20260912_1947`). Około epizodu 55-60 (~11-12k kroków) polityka wyprowadziła
   **`fr3_joint2` na dolny limit** i staw z niego nie zszedł: odczyt na żywo `q2 = -1.836` przy
   limicie URDF `lower="-1.8361"`, pozostałe sześć stawów dokładnie w `Q_READY`, kostka
   w pozycji domyślnej, palce otwarte. `‖q − Q_READY‖ = 1.051` przy `TOL = 0.05`.
   Nie jest potwierdzone, że to ten sam mechanizm co pierwsza obserwacja - wtedy nie
   sprawdzano, czy staw stał na limicie.

   - **Maskował się jako postęp.** Od epizodu ~60 `min_ee_to_cube` = **0.5425 m identycznie
     w każdym epizodzie** (median = best w każdym bloku), zero chwytów i upuszczeń, a mimo to
     `ep_rew_mean` rosło do +2.4. Stan się nie zmienia, więc każdy krok płaci stałe `(γ−1)·Φ`:
     przy tej pozie `Φ = −(0.5425 + 0.7245) = −1.267` → +0.0127/krok → ×200 ≈ +2.5
     (zmierzone +2.445). Równolegle `critic_loss` spadł do 8.6·10⁻⁵ (nieruchomy świat jest
     idealnie przewidywalny), a `ent_coef` do 2.3·10⁻⁴ (akcje nie mają wpływu, więc eksploracja
     jest bezwartościowa). Wszystkie trzy wyglądały jak zbieżność.
   - **Sygnał, który to zdradził:** `min_ee_to_cube` liczy także obserwację z `reset()`, a
     w pozie ready EE jest 0.2027 m od kostki. Wartość powyżej 0.2027 jest więc **niemożliwa przy
     działającym resecie** - to jest jednoznaczny detektor, niezależny od interpretacji nagrody.
   - **Punkt odniesienia dla `ep_rew_mean` pod PBRS:** `Monitor` loguje sumę **niezdyskontowaną**.
     Zdrowy 200-krokowy epizod bez chwytu, startujący z ready, daje ≈ `0.01 · 0.928 · 200` ≈
     **+1.86**. Wartość w tym rejonie to linia bazowa, nie postęp; alarmem jest dopiero stałe
     `min_ee_to_cube` między epizodami.
   - **Dlaczego przeszło niezauważone:** pętla `N_SETTLE` w `reset()` robi `break` po dojechaniu
     do `Q_READY`, ale gdy ramię nie dojedzie, **po cichu wychodzi po 60 iteracjach** i epizod
     startuje z ramieniem w złej pozie. ~24k z 36k kroków przebiegu to przejścia z zamrożonego
     ramienia - bufor jest zatruty, przebiegu **nie wznawiać** (`--resume` przeniosłoby je dalej).
   - **`docker compose restart sim` naprawia tę sygnaturę - potwierdzone (12.09.2026):** po
     restarcie `fr3_joint2` wrócił na `-0.785`. Restart usuwa objaw, nie przyczynę - wiadomo
     tylko, że stan zablokowania żyje w procesie symulacji, a nie w modelu czy konfiguracji.
   - **`reset()` rzuca wyjątkiem - DONE (12.09.2026).** `for ... else` na pętli `N_SETTLE`:
     gdy ramię nie osiągnie `Q_READY`, leci `RuntimeError` z nazwą i pozycją stawu, **zanim**
     `set_cube_pose` ruszy scenę. Pilnuje tego
     `test_reset_fails_loudly_when_the_arm_cannot_reach_ready` (sprawdzony mutacją: ciche
     `pass` zamiast `raise` wywala test). Zadziałało od razu w praktyce: kolejny start SAC padł
     **w automatycznym resecie po pierwszym epizodzie**, czyli jeszcze przed `learning_starts`,
     na samych losowych akcjach - awaria okazała się nie rzadka, tylko wcześniej niewidoczna.

   **Wyzwalacz POTWIERDZONY, mechanizm w silniku NIEWYJAŚNIONY (13.09.2026).**

   - **Jak polityka trafia na limit - zmierzone offline.** `WORKSPACE_BOX` ma `x ∈ [-0.15, 0.7]`,
     czyli obejmuje też rejon przy bazie i za nią, którego z zamrożoną orientacją (chwytak
     pionowo w dół) nie da się osiągnąć. Marsz DLS-IK z `Q_READY` po 5 cm w stronę
     `(-0.15, 0, 0.42)` dociska `fr3_joint2` **dokładnie** do `-1.8361` już **po 5 krokach**
     (EE przy `x ≈ 0.06`); w stronę `(-0.15, 0, 0.65)` na limit trafia `fr3_joint6`. Stare
     `clip_to_joint_limits` przycinało do `[lower, upper]`, więc cel równy limitowi szedł
     prosto do JTC. Losowej eksploracji wystarczy kilka kroków w miarę konsekwentnie w `-x`.
     Weryfikacja workspace'u z 08.2026 (sekcja „Cel zadania") sprawdzała boczne stoły i strefę
     spawnu kostki, **nie** środek pudełka przy bazie.
   - **Odtworzone na żywej symulacji, na samym `joint2`** (komenda JTC, reszta stawów w ready):

     | Próba | Cel `joint2` | Wynik |
     |---|---|---|
     | 1 | limit + 0.05 rad, wolno (3 s) | wrócił do ready |
     | 2 | limit + 0.05 rad, krokami po 50 ms jak `step()` | wrócił do ready |
     | 3 | **dokładnie limit** `-1.8361`, wolno (3 s) | **zablokowany** |

     Warunek jest deterministyczny i nie zależy od szybkości dojazdu. Po zablokowaniu JTC
     wysyła `reference = -0.785`, a `feedback` stoi na `-1.8361000000000027` z prędkością ~0.
   - **Hipotezy sprawdzone i odrzucone:**

     | Hipoteza | Test | Wynik |
     |---|---|---|
     | ogranicznik limitów `ros2_control` odrzuca komendy | log `controller_manager` | `Enforcing command limits is disabled` |
     | za duża komenda prędkości (`gz_ros2_control` robi z pozycji prędkość, `position_proportional_gain = 0.1`) | mały krok do `-1.80` | staw stoi |
     | samo dojechanie do limitu blokuje dowolny staw | `joint7` dokładnie na `3.0508` i z powrotem | wrócił |
     | uderzenie w limit trajektorią 50 ms | `joint7`, `joint5` wbite w limit | oba wróciły |
     | grawitacja przewyższa moment stawu | pinocchio `computeGeneralizedGravity` w zablokowanej pozie | **26.9 Nm = 31%** z 87 Nm |

     Zostaje różnica, której nie zweryfikowano: na `joint2` grawitacja **dociska** staw do
     ograniczenia (moment ~27 Nm w stronę limitu), a na nadgarstku nic go nie dociska. To
     najbardziej prawdopodobny wyróżnik, ale **nie** wyjaśnienie - statycznie staw ma trzykrotny
     zapas momentu, więc „grawitacja go przytrzymuje" w prostej postaci jest obalone. Mechanizm
     siedzi gdzieś w styku ograniczenia limitu DART z komendą prędkości `gz_ros2_control`.
   - **`CLIP_MARGIN = 0.05` rad [Z]** w `config.py`, `clip_to_joint_limits` przycina do
     `[lower + m, upper − m]`. Wartość wprost z próby 1-2 powyżej. Po zmianie ten sam marsz IK
     zatrzymuje `joint2` na `-1.7861`, najmniejszy zapas do limitu we wszystkich stawach =
     dokładnie 0.0500 rad. Odpowiada praktyce realnego FR3, który też nie dopuszcza komend przy
     twardych limitach. **Wpływ na zebrane demonstracje - wnioskowany, nie sprawdzony na plikach:**
     ścieżki eksperta leżą w rejonie, dla którego test z 08.2026 dał minimalny zapas 0.72 rad,
     więc nie powinny były nigdy wejść w pas przycinany przez margines.
     **Sprostowanie (13.09.2026): margines NIE rozwiązał problemu.** Kolejny przebieg SAC
     (`sac_L1_seed0_20260913_1652`, `ent_coef=0.1`) zablokował `joint2` po 86 epizodach (17.2k
     kroków), mimo że zainstalowany pakiet miał margines (sprawdzone `cmp` src ↔ install). Cele
     z IK nie schodziły poniżej `-1.7861`, a staw i tak dotknął `-1.8361` - przyczyna niżej.
     Zostaje jako zabezpieczenie, ale przy przestrzeleniu ~0.04 rad (pomiar niżej) zapas jest
     na styk. Próba 2 z tabeli powyżej sprawdzała tylko pozycję końcową, nie minimum w trakcie
     ruchu - „wrócił do ready" nie wyklucza, że staw po drodze dotknął limitu.

   **Właściwa przyczyna: ucieczka stawu przy szybkich komendach (13.09.2026). Mechanizm
   NIEZNANY, naprawy BRAK. „Obejście" przez `set_physics` WYCOFANE - wyłączało grawitację.**

   - **Offline: polityka zleca szybsze ruchy, niż staw wykona.** Losowa polityka przez prawdziwe
     IK (60 epizodów, 12k kroków, kinematycznie): `joint2` wchodzi w pas 0.15 rad od limitu
     **207 razy**, komendy w stronę limitu sięgają **0.21 rad/krok = 4.2 rad/s**, przy limicie
     prędkości stawu w URDF **2.62 rad/s**.
   - **Pomiar na żywo, świat z poprawną fizyką.** Skrypt wysyła `joint2` z ready do celu `-1.4`
     (0.44 rad przed limitem, więc pomiar nie blokuje stawu) komendami JTC co 50 ms czasu
     symulacji i zapisuje minimum `q2` z `/joint_states`. Zadana prędkość = krok / 50 ms. Na
     świeżym bringupie (`real_time_factor` 0 z pliku, dwa przebiegi; także 1000 z pliku, RTF
     zmierzony 3.7-3.8): **1.0 rad/s - przestrzelenie 0.000; ≥ 2 rad/s - 0.44 rad, ucieczka do
     limitu -1.8361** z prędkością wbitą w 2.62 rad/s. Staw nie hamuje przy celu, tylko jedzie do
     twardego ogranicznika, a jedno dotknięcie wystarcza do blokady. Tłumaczy to, czemu margines
     0.05 rad nie pomógł. Próg w okolicy 2 rad/s (raz 2.0 rad/s dało 0.026 rad, 2.6 już ucieczkę).
   - **Hipotezy sprawdzone i odrzucone:** brak przebudowy pakietu (`cmp` src ↔ install
     identyczne); druga aplikacja wysyłająca komendy (brak procesów i innych publisherów, log
     treningu nietknięty); wartość `real_time_factor` (0 i 1000 z pliku identycznie); efekt
     „pierwszego przebiegu po starcie" (drugi przebieg na tym samym bringupie też ucieka).
   - **Błędny trop: `set_physics` (wycofany, 13.09.2026).** Wywołanie serwisu
     `/world/fr3_world/set_physics` z parametrami identycznymi jak w SDF usuwało ucieczkę
     (przestrzelenie ≤ 0.039 rad przy RTF 1, 3.8 i po `set_pose`), więc trafiło do
     `bringup.launch.py` jako obejście. **Skutek uboczny: po wywołaniu grawitacja praktycznie
     znika.** Kostka postawiona na 1 m wisi 0.5 s (zmierzone przyspieszenie ~0.05 m/s² zamiast
     9.81). Wszystkie pomiary „po `set_physics`" szły więc bez grawitacji - pokazują tylko, że
     bez ciężaru ramienia staw nie ucieka, co wskazuje na **grawitację jako składnik
     mechanizmu** (przy odchyleniu do tyłu ciągnie `joint2` w stronę limitu), ale **nie** jest
     naprawą. Statycznie moment grawitacji to 27 z 87 Nm, więc to nie jest proste „ciężar
     przewyższa staw" - mechanizm dynamiczny pozostaje niewyjaśniony. Launch przywrócony z gita
     (`f571e2a`), `franka_sim` przebudowany.
     **Lekcja do metody:** poprawkę sprawdzano wyłącznie na objawie (przestrzelenie), bez kontroli
     skutków ubocznych w innej części fizyki. Test swobodnego spadku trwa sekundę i wykryłby to
     od razu.
   - **Przebieg `sac_L1_seed0_20260913_1748` (~53k kroków) - NIEWAŻNY**, trenowany bez grawitacji.
     Objawy w `monitor.csv`: `min_ee_to_cube` do **0.99 m** przez kilkanaście epizodów z rzędu
     (przy działającym resecie maksimum to 0.2027), kostka dryfująca po muśnięciu w linii prostej
     ze stałą prędkością (z: 0.90 → 3.54 m przy |v| = 0.27 m/s), 7 epizodów `dropped`, zero chwytów.
     Wcześniejsze przebiegi i zbiory demonstracji `set_physics` nie dotyczy.
   - **Wpływ na wcześniejsze dane - wnioskowany:** zbiory demonstracji (07.09 przy RTF 1.0,
     09.09 przy RTF 0) zbierano w świecie, w którym szybkie komendy uciekają. Ekspert ma 100/100
     sukcesów i `drop_rate` 0.000, co sugeruje, że jego ruchy nie przekraczały progu, ale prędkości
     stawów nie były logowane. Czy świat przy RTF = 1 bez `set_physics` też ucieka, **nie
     sprawdzono**.
   - **TODO:**
     1. **Ograniczyć zmianę stawów na krok** w `step()`. Jedyna zmierzona bezpieczna wartość to
        1.0 rad/s (przestrzelenie 0.000), czyli `|Δq| ≤ 0.05` rad przy `dt = 0.05`; limit
        prędkości z URDF (2.62 rad/s) jest **powyżej** progu ucieczki, więc sam nie wystarczy.
     2. **Test `@pytest.mark.sim` na grawitację** (swobodny spadek kostki) - każda zmiana fizyki
        świata ma go przechodzić.
     3. **Test `@pytest.mark.sim` na ucieczkę** (`joint2` do `-1.4` przy ~2.6 rad/s, przestrzelenie
        < 0.05 rad) - dziś oczekiwanie: pada, dopóki pkt 1 nie jest wdrożony.
   - **TODO: poprawić `WORKSPACE_BOX`.** Margines usuwa blokadę, ale nie powód, dla którego
     eksploracja wjeżdża w rejon przy bazie: tam dalej będą porażki IK / stawy przyklejone do
     marginesu i marnowane kroki. Wymaga osobnej analizy granicy osiągalności z zamrożoną
     orientacją (np. strefa wykluczenia wokół osi bazy zamiast prostopadłościanu).
2. **Wyścig przy `multi_step`** (faza 2 krokowania) - nierozstrzygnięty do czasu implementacji.

### Rozwiązane
- `GZ_SIM_RESOURCE_PATH` → `ENV` w Dockerfile, meshe się ładują
- GUI na Waylandzie → `xhost +local:docker`
- `docker exec` bez sourcowania → sourcowanie dopisane do `/root/.bashrc`
- Mimic `fr3_finger_joint2` → Opcja A (jawne sterowanie oboma palcami, `strip_finger_mimic`)
- Robot przewracający się przy kontakcie → link `world` + fixed joint `world_to_base`
- Build Dockera padający na `rosdep` → pin `setuptools<81`
- Strojenie fizyki chwytu → **okazało się niepotrzebne** (07.09.2026). Odłożone do czasu, aż
  będzie ruch z polityki; gdy ten ruch nastąpił, ekspert zebrał 100/100 demo na L2 bez ani
  jednego upuszczenia (`drop_rate` 0.000). Domyślne tarcie palców z `franka_description` przy
  `mu=1.0` kostki wystarcza dla 5 cm/krok. Gałki z sekcji „Chwytak" pkt 5 zostają opisane jako
  niewykorzystane; wracać do nich dopiero, gdy `drop_rate` podniesie się przy RL albo DP
  (polityka może generować szarpnięcia, których automat 8 faz nie produkuje).

### Następne kroki
1. Naprawić `GZ_SIM_RESOURCE_PATH` w entrypoincie → potwierdzić że meshe FR3 ładują się w Gazebo - **DONE**
2. Napisać minimalny launch file w pakiecie `franka_sim` (spawn FR3 + Gazebo + ros2_control) - **DONE**
3. Zbudować scenę Pick & Place w Gazebo (stół + kostka SDF) - **DONE**
4. Skonfigurować `controllers.yaml`: JTC dla ramienia + kontroler chwytaka; rozstrzygnąć mimic - **DONE** (Opcja A)
5. Wrapper Gymnasium ↔ ROS 2 + DLS-IK - **DONE**, plan wykonawczy w `docs/plan-gym-wrapper-dls-ik.md`
6. Mosty Gazebo dla obserwacji i resetu: `dynamic_pose/info` (poza kostki), `SetEntityPose` (reset kostki) - **DONE**
7. Krokowanie Gazebo z pętli Gym, faza 2: `ControlWorld` / `multi_step` - **TODO**
8. Skryptowany ekspert zbierający demonstracje - **DONE (07.09.2026)**. `expert.py` (automat 8 faz,
   zero ROS) + `collect_demos.py` (pętla wykonawcza, flagi `--verbose`/`--save`). Pierwszy przebieg
   na żywym Gazebo, **L2, seed 0: 100/100 udanych epizodów** - `success_rate` i `grasp_rate` 1.000,
   `drop_rate` 0.000, `ik_failure_rate` 0.000, zero porażek budżetu fazy. 5087 kroków w ~8 min
   (10.5 FPS), długości epizodów 44-61 (mediana 51) przy `max_episode_steps=200`. Cele rozłożone
   49/51 lewy/prawy - bimodalność z decyzji o dwóch stołach faktycznie jest w danych.
   Zbiór: `data/demos/demo_L2_seed0_20260907_1445/` (100 plików `.npz`, 668 KB), zweryfikowany
   strukturalnie (`obs` = `action`+1, wszystko w `[-1,1]`, zero NaN).
   **Powtórka bez dławika (09.09.2026):** ten sam przebieg przy `real_time_factor=0` -
   `data/demos/demo_L2_seed0_20260909_1213/`, 100/100, 5177 kroków, **32.4 FPS**, komplet metryk
   jakości bez zmian, ta sama walidacja strukturalna przeszła. Różnice: epizody dłuższe średnio
   o ~1 krok (51.77 vs 50.87), `return_mean` −8.62 vs −7.78 (to ta sama kara za krok, nie gorsze
   zachowanie), mediana `min_ee_to_cube` 0.71 vs 0.27 mm - przyczyna opisana wyżej przy
   `real_time_factor`. Który zbiór idzie do treningu DP, jeszcze nie rozstrzygnięte; oba są
   kompletne i zgodne formatem, więc nadają się też na ablację „rozmiar zbioru vs jakość".

---

## Kluczowe decyzje podjęte

| Decyzja | Uzasadnienie |
|---|---|
| FR3 zamiast FER (Panda) | Nowszy model, lepsze wsparcie w repo |
| `ros-jazzy-desktop` zamiast pojedynczych paczek | Szybszy start, mniej debugowania brakujących zależności |
| PyTorch z PyPI (nie z `download.pytorch.org`) | CDN PyTorcha blokowany przez sieć (SSLV3_ALERT_HANDSHAKE_FAILURE); PyPI też daje build CUDA. Jeden obraz, na laptopie fallback na CPU |
| Tylko `franka_description` + `franka_msgs` ze źródeł | Reszta repo (gripper, hardware, gazebo_bringup) ciągnie `libfranka` - zbędne w symulacji |
| Integracja Gazebo pisana od zera w `franka_sim` | `franka_gazebo_bringup` z oficjalnego repo wymaga `franka_hardware` → `libfranka` |
| Gazebo headless | Wayland na hoście uniemożliwia GUI forwarding; trening RL i tak będzie headless |
| Obserwacje state-based (oracle) | Uczciwe porównanie, mieści się w harmonogramie i mocy GPU; vision → future work |
| Akcja: 4D delta-EE, orientacja zamrożona | Task space = efektywność próbkowa; position > velocity dla DP (Chi); prostota; plan B: delta joint position |
| Własne DLS-IK zamiast MoveIt Servo | Deterministyczne, testowalne, bez węzłów MoveIt w pętli treningu |
| Jakobian 6x7 z aktywnie regulowaną orientacją | 3x7 pozwala orientacji dryfować przez epizod przy 7 DOF - chwytak przestaje patrzeć w dół |
| Pinocchio zamiast własnej kinematyki / KDL | Gotowe i przetestowane FK+jakobian; praca opisuje DLS, nie kinematykę FR3; `kdl_parser_py` niepewny w Jazzy |
| Krokowanie sim wdrażane etapowo (free-run → `multi_step`) | Szybkie domknięcie end-to-end; cały czas schowany za `advance(dt)`, więc podmiana nie rusza `gym_env` |
| Poziomy L1/L2/L3 jako argument, nie trzy klasy | Trzy klasy = trzy kopie reward i obserwacji = gwarantowany rozjazd między wariantami |
| Baza przykręcona do świata (`world_to_base`) | Free-floating model przewraca się przy siłach reakcji chwytu; odpowiada realnemu FR3 na stanowisku |
| Pin `setuptools<81` w obrazie | setuptools ≥ 82 usunął `pkg_resources`, którego wymaga `rosdep` |
| JTC (position) jako kontroler | Interpolacja = gładkość; identyczny dla RL, DP i eksperta |
| Shaped reward + kara ‖a‖², bez filtrów akcji | Sparse nierozwiązywalny w budżecie; smoothness z surowych trajektorii |
| DP: wariant CNN, trening od zera, DDIM w inferencji | Łatwiejszy tuning; oryginalny DP jest per-task, nie pretrenowany |

---

## Literatura

1. Chi et al. (2023) - Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (RSS 2023)
2. Haarnoja et al. (2018) - Soft Actor-Critic (ICML 2018)
3. Schulman et al. (2017) - Proximal Policy Optimization Algorithms
4. Ho et al. (2020) - Denoising Diffusion Probabilistic Models (NeurIPS 2020)
5. Mandlekar et al. (2021) - What Matters in Learning from Offline Human Demonstrations (CoRL 2021)
6. Gallouédec et al. (2021) - panda-gym: Open-source goal-conditioned environments for robotic learning - *najbliższy setup: Panda + P&P + SAC*
7. Andrychowicz et al. (2017) - Hindsight Experience Replay (NeurIPS 2017)
8. Zhou et al. (2019) - On the Continuity of Rotation Representations in Neural Networks (CVPR 2019) - *reprezentacja 6D orientacji*
9. Heo et al. (2023) - FurnitureBench (RSS 2023) - *wzorzec przestrzeni akcji: delta-EE + chwytak z progiem, OSC 10 Hz → 1 kHz*
10. Ren et al. (2024) - DPPO: Diffusion Policy Policy Optimization (ICLR 2025, arXiv:2409.00588) - *do related work: finetuning DP przez policy gradient, pomost RL↔DP*
11. Wang et al. (2022) - Diffusion Policies as an Expressive Policy Class for Offline RL (Diffusion-QL, ICLR 2023) - *related work*
12. arXiv:2410.01220 - Effective Tuning Strategies for Generalist Robot Manipulation Policies - *delta joint position vs delta-EE, argument za planem B*
13. arXiv:2602.23408 - Demystifying Action Space Design for Robotic Manipulation Policies - *systematyczne badanie wyboru przestrzeni akcji*

---

## Dokumenty powiązane

- `docs/cheatsheet.md` - komendy operacyjne
- `docs/constants-derivations.md` - wyprowadzenia stałych liczbowych (skąd `REL_SCALE`, `λ`, progi) - ściąga pod pisanie pracy
- `docs/plan-gym-wrapper-dls-ik.md` - plan wykonawczy wrappera Gym + DLS-IK (etap bieżący)
- `CLAUDE.md` - reguły pracy w repo (walidacja przed uruchomieniem, rebuild po edycji `config/`/`launch/`/`urdf/`)
