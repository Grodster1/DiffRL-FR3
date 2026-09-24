# Stałe liczbowe i ich wyprowadzenia

> Dokument-satelita `thesis-project-context.md`. Tam są **decyzje** (co wybrano i dlaczego),
> tutaj **liczby** (skąd wzięła się konkretna wartość i co się psuje, gdy jest zła).
> Powstał pod pisanie pracy: każda sekcja ma być gotowym akapitem uzasadnienia, a nie notatką.
>
> Reguła prowadzenia: dopisując tu stałą, zawsze podaj jej **status** - inaczej po miesiącu
> nie odróżnisz wyprowadzenia od zgadywanki, a w pracy nie wolno przedstawiać drugiego jako
> pierwszego.

**Status stałej:**

| Znacznik | Znaczenie |
|---|---|
| **[W]** | **Wyprowadzona** - wynika z geometrii sceny, URDF albo matematyki metody. W pracy podaje się wyprowadzenie. |
| **[Z]** | **Zmierzona** - wynik eksperymentu w symulacji. W pracy podaje się liczbę i metodę pomiaru. |
| **[P]** | **Przyjęta** - świadomy wybór projektowy z uzasadnieniem, ale bez twardego wyprowadzenia. W pracy trzeba to nazwać wprost; kandydat do ablacji. |

---

## 1. Geometria sceny

Wszystko poniżej wynika z `src/franka_sim/worlds/fr3_world.sdf` i jest wzajemnie zależne -
zmiana jednej liczby przesuwa resztę.

| Stała | Wartość | Status | Wyprowadzenie |
|---|---|---|---|
| Wysokość blatu | `0.40 m` | **[W]** | Stół główny: `<pose>0.5 0 0.2</pose>`, `<size>0.6 0.8 0.4</size>`. Środek na `z = 0.2`, połowa wysokości `0.2` → górna powierzchnia na `0.2 + 0.2 = 0.40`. |
| Bok kostki | `0.05 m` | **[P]** | Rozmiar typowy dla benchmarków pick & place (panda-gym, Fetch). Dobrany do rozwarcia chwytaka FR3 (patrz §5). |
| `z` kostki na blacie | `0.425 m` | **[W]** | `0.40` (blat) + `0.025` (połowa kostki). To jest jednocześnie docelowa wysokość TCP przy chwycie, bo `fr3_hand_tcp` leży między palcami. |
| Szczelina między stołami | `0.075 m` | **[W]** | Stoły boczne: `<pose>0 ±0.525 0.2</pose>`, `<size>0.25 0.25 0.4</size>` → zajmują `x ∈ [-0.125, 0.125]`. Stół główny zaczyna się na `x = 0.5 − 0.3 = 0.2`. Przerwa: `0.2 − 0.125 = 0.075 m`. |
| Cele | `(0, ±0.525, 0.425)` | **[W]** | Środek blatu stołu bocznego, wysokość jak dla kostki na blacie. |
| Strefa spawnu kostki | `x ∈ [0.4, 0.6]`, `y ∈ [-0.2, 0.2]` | **[P]** | Zawarta w stole głównym (`x ∈ [0.2, 0.8]`, `y ∈ [-0.4, 0.4]`) z marginesem 0.2 m od każdej krawędzi - kostka nigdy nie spawnuje się na skraju blatu. |
| Dystans transportu | `≈ 0.73 m` | **[W]** | `‖(0.5, 0) − (0, ±0.525)‖ = √(0.25 + 0.2756) ≈ 0.725 m`. Wartość referencyjna: panda-gym operuje na ~0.15 m, więc zadanie jest ~5x dłuższe w przestrzeni. |

**Smaczek do pracy - dlaczego 7.5 cm szczeliny to nie przypadek.** Szczelina jest większa niż
połowa boku kostki (2.5 cm), ale mniejsza niż jej pełny bok tylko o 2.5 cm. Kostka pchana po
blacie w kierunku stołu bocznego przekracza krawędź, traci podparcie i spada na podłogę,
zamiast „przeskoczyć". To eliminuje degenerację *pick & place → push* **geometrią sceny, a nie
karą w funkcji nagrody** - czyli bez wprowadzania hiperparametru, który trzeba by potem
uczciwie zaraportować i strojić. Alternatywa (wyższy podest) została odrzucona, bo DLS-IK nie
unika przeszkód, więc każdy centymetr pionowej przeszkody to ryzyko kolizji.

---

## 2. Normalizacja obserwacji (`observations.py`)

Wektor obserwacji ma **16 wymiarów**: `ee_pos (3) + ee_rot6d (6) + gripper (1) + ee_to_cube (3)
+ cube_to_goal (3)`.

> Rozbieżność z `thesis-project-context.md`, który zapowiadał „~20–30D": pozycje bezwzględne
> kostki i celu zastąpiono wektorami względnymi (konwencja panda-gym - polityka dostaje wprost
> to, co ma zredukować do zera), a opcjonalne `q (7)` i `dq (7)` pominięto. Przy pełnej
> obserwowalności stanu `q` jest redundantne względem `(p_ee, R_ee)` **poza** rozwiązaniem
> redundancji łokcia; gdyby polityka miała problem z osobliwościami, dopisanie `q` da 23D.

### 2.1 `WORKSPACE_BOX` - `x ∈ [-0.15, 0.70]`, `y ∈ [-0.6, 0.6]`, `z ∈ [0.42, 0.65]` - **[P]** z komponentem **[W]**

Pudełko pełni dwie role naraz: przycina cel IK (`clip_to_workspace`) i definiuje skalę
normalizacji `ee_pos`. Dolne granice są wyprowadzalne, górne to margines:

| Granica | Skąd |
|---|---|
| `z_min = 0.42` | **[W]** Blat `0.40` + `0.02`. TCP nie może zejść poniżej wysokości chwytu (`0.425`) o więcej niż 5 mm - próba schodzenia niżej to taranowanie blatu palcami. |
| `z_max = 0.65` | **[P]** `0.225 m` nad blatem. Wystarcza na przeniesienie ponad krawędziami stołów; wyżej nie ma po co, a wysoki sufit tylko rozrzedza normalizację. |
| `x ∈ [-0.15, 0.70]` | **[P]** Pokrywa cele (`x = 0`) i strefę spawnu (`x ≤ 0.6`) z marginesem 0.1–0.15 m. |
| `y ∈ [-0.6, 0.6]` | **[P]** Pokrywa cele (`y = ±0.525`) z marginesem 0.075 m. |

> **Znane ograniczenie (patrz §3.1a):** pudełko jest prostopadłościanem, a strefa robocza nim
> nie jest. Róg „daleko i wysoko" (`x = 0.6`, `z = 0.65`) leży już w obszarze osobliwym -
> IK tam nie dojeżdża (błąd 4.4 cm). Pudełko dopuszcza więc cele, których tor sterowania nie
> zrealizuje; to źródło systematycznych, przestrzennie skoncentrowanych `ik_failures`.

### 2.2 `REL_SCALE = [0.8, 0.8, 0.25]` - **[W]**

To jedyna stała w tym module, którą da się wyprowadzić do końca, i warto to w pracy pokazać,
bo ilustruje różnicę między normalizacją „na oko" a policzoną.

Zasada: dzielnik ma być **maksymalną możliwą wartością danej składowej**, żeby normalizacja
wykorzystywała pełny zakres `[-1, 1]`, a `clip` był bezpiecznikiem, nie stanem normalnym.
Zakresy wynikają z §1 i z `WORKSPACE_BOX`:

`ee_to_cube = p_cube − p_ee`, `cube_to_goal = p_goal − p_cube`

| Składowa | Zakres składników | Zakres różnicy | max \|·\| |
|---|---|---|---|
| `ee_to_cube.x` | `cube.x ∈ [0.4, 0.6]`, `ee.x ∈ [-0.15, 0.70]` | `[-0.30, 0.75]` | **0.75** |
| `ee_to_cube.y` | `cube.y ∈ [-0.2, 0.2]`, `ee.y ∈ [-0.6, 0.6]` | `[-0.80, 0.80]` | **0.80** |
| `ee_to_cube.z` | `cube.z ∈ [0.425, 0.65]`, `ee.z ∈ [0.42, 0.65]` | `[-0.23, 0.23]` | **0.23** |
| `cube_to_goal.x` | `goal.x = 0`, `cube.x ∈ [0.4, 0.6]` | `[-0.60, -0.40]` | **0.60** |
| `cube_to_goal.y` | `goal.y = ±0.525`, `cube.y ∈ [-0.2, 0.2]` | `[-0.725, 0.725]` | **0.73** |
| `cube_to_goal.z` | `goal.z = 0.425`, `cube.z ∈ [0.425, 0.65]` | `[-0.225, 0.0]` | **0.23** |

Stąd `[0.8, 0.8, 0.25]` - maksimum po obu wektorach na każdej osi, z zaokrągleniem w górę
(~10% zapasu na `z`).

**Odrzucony wariant: jeden skalar `0.6`** (pierwotna wartość, „mniej więcej rozpiętość
workspace'u w `y`"). Był zły w obie strony jednocześnie:
- **`x` i `y` obcinane w typowych sytuacjach** - `0.8 > 0.6`, więc `clip` do `±1` wchodził przy
  zwykłym sięganiu na przeciwległy stół. Polityka traciła informację o kierunku dokładnie tam,
  gdzie jest najbardziej potrzebna.
- **`z` zduszone** - przy realnym zakresie `0.23` obserwacja nigdy nie wychodziła poza `±0.38`,
  czyli 62% zakresu wejścia sieci było martwe na osi, która rozstrzyga o chwycie (wysokość
  podejścia). Tam potrzeba **najwięcej** rozdzielczości, nie najmniej.

**Uwaga o timingu (istotna metodologicznie).** `REL_SCALE` definiuje rozkład wejść polityki.
Zmiana po zebraniu demonstracji unieważnia dataset DP, a po treningu - checkpointy SAC.
Stała musi być zamrożona **przed** pierwszym zbieraniem danych i ta sama dla wszystkich
porównywanych metod (zasada uczciwości porównania).

### 2.3 `GRIPPER_RANGE = (0.0, 0.08)` - **[W]**

Zakres **jednego** palca to `<limit lower="0.0" upper="0.04">` na `fr3_finger_joint*` w URDF.
`GRIPPER_RANGE` opisuje **rozwarcie całkowite**, czyli sumę obu palców: `2 x 0.04 = 0.08 m`.
Pomiar (`ros_bridge.py`) sumuje oba stawy; komenda (`publish_gripper_command`) dzieli rozwarcie
na pół i wysyła każdemu palcowi jego połowę. Warstwa ROS jest jedynym miejscem, które wie,
że chwytak ma dwa stawy.

**Sprostowanie (09.2026) - to była przyczyna `grasp_rate = 0`.** Wcześniejszy zapis mówił
`(0.0, 0.04)` i uzasadniał odczyt jednego palca tym, że „oba dostają identyczną komendę, więc
pozycja drugiego nie wnosi informacji". Komendy istotnie są identyczne, ale **pozycje nie**:
przy kontakcie każdy palec staje tam, gdzie zablokuje go bryła kostki, a podział między nie
zależy od tego, gdzie bocznie kostka leży. Zmierzony chwyt (09.2026, `rqt`, kostka uniesiona
3.2 cm nad blat): `fr3_finger_joint1 = 0.0400`, `fr3_finger_joint2 = 0.0100`, suma `0.0500` -
dokładnie bok kostki. Odczyt samego `joint1` dawał `0.040`, czyli wartość nieodróżnialną od
chwytaka otwartego na oścież, więc warunek `holding` w `_update_grasped` nie mógł zapalić się
przy żadnym poprawnym chwycie. Trzy pozostałe warunki (odległość TCP-kostka `0.0259 < 0.05`,
uniesienie `0.032 > 0.02`, wysokość TCP `0.4625` wewnątrz `WORKSPACE_BOX`) były spełnione.

### 2.4 Dlaczego orientacja to 6 liczb, a nie 4 - **[P]**, uzasadnienie z literatury

Reprezentacja 6D (dwie pierwsze kolumny macierzy obrotu, Zhou i in. 2019) zamiast kwaternionu.
Powód nie jest kosmetyczny: kwaterniony i kąty Eulera są **nieciągłe jako cel regresji**
(`q` i `−q` to ten sam obrót; Euler ma gimbal lock), a sieć uczona regresji na nieciągłej
reprezentacji ma w tych miejscach nieograniczony błąd. Reprezentacja 6D jest ciągła i
surjektywna na SO(3). Dodatkowy zysk praktyczny: kolumny macierzy obrotu **z definicji leżą
w `[-1, 1]`**, więc przechodzą przez normalizację bez skalowania - jeden mniej hiperparametr.

Test kontrolny w `test_observations.py` sprawdza, że para kolumn jest ortonormalna i że
`col0 x col1` odtwarza trzecią kolumnę `R` - czyli że 6 liczb faktycznie koduje pełny obrót.

### 2.5 Dwa poziomy skal - obserwacja vs nagroda - **[W]**

`build_obs_dict` zwraca **metry**, `normalize` zwraca `[-1, 1]`. Nagroda liczona jest z dicta,
nie z wektora. Powód liczbowy: w wektorze każdy blok ma inny dzielnik (`ee_pos` przez
`WORKSPACE_BOX`, wektory względne przez `REL_SCALE`), więc `‖obs[10:13]‖` **nie jest**
odległością EE–kostka, tylko tą odległością przemnożoną przez ≈ `1/0.8`. Wagi w funkcji
nagrody przestałyby mieć interpretację fizyczną, a każda zmiana `REL_SCALE` wymuszałaby
przestrojenie nagrody. Progi sukcesu (`‖p_cube − p_goal‖ < 0.05 m`) też mają sens wyłącznie
w metrach.

---

## 3. DLS-IK (`inverse_kinematics.py`)

### 3.1 Współczynnik tłumienia `λ = 0.05` - **[P]** z pełnym wyprowadzeniem konsekwencji

To jest stała, przy której w pracy najbardziej opłaca się pokazać matematykę, bo tłumienie
w DLS ma zamkniętą interpretację przez SVD.

Dla `dq = Jᵀ(JJᵀ + λ²I)⁻¹ e` i rozkładu `J = UΣVᵀ` rozwiązanie rozkłada się na kierunki
singularne:

```
dq = Σᵢ  σᵢ / (σᵢ² + λ²) · (uᵢᵀe) · vᵢ
```

Czyli każdy kierunek jest wzmacniany przez `g(σ) = σ / (σ² + λ²)`. Z tego wynika wszystko, co
trzeba wiedzieć o `λ`:

| Reżim | Zachowanie | Wniosek |
|---|---|---|
| `σ ≫ λ` | `g(σ) ≈ 1/σ` | DLS degeneruje do pseudoodwrotności - z dala od osobliwości tłumienie **nie robi nic**. |
| `σ = λ` | `g = 1/(2λ)` - **maksimum** | Największe możliwe wzmocnienie. Dla `λ = 0.05` → **10 rad na metr błędu**. |
| `σ → 0` | `g → 0` | W osobliwości kierunek jest wygaszany, a nie wysadzany do nieskończoności. To cały sens metody (Nakamura & Hanafusa 1986, Wampler 1986). |

**Konsekwencja liczbowa dla naszej pętli.** Polityka żąda najwyżej `0.05 m` na krok, więc
w najgorszym przypadku (`σ = λ`) pojedyncza iteracja DLS daje `‖dq‖ ≤ 10 x 0.05 = 0.5 rad`
≈ 28.6° na krok. Czyli `λ = 0.05` jest **jednocześnie limitem bezpieczeństwa** - bez tłumienia
ta wartość jest nieograniczona.

### 3.1a Czy `λ` w ogóle działa - pomiar `σ_min` - **[Z]** (08.2026)

Tłumienie ma znaczenie tylko tam, gdzie `σ_min` schodzi do rzędu `λ`. Zmierzone: 33 punkty
(siatka strefy spawnu `x ∈ {0.4, 0.5, 0.6} x y ∈ {-0.2, 0, 0.2}` na trzech wysokościach
`z ∈ {0.425, 0.55, 0.65}` + oba cele na tych samych wysokościach), dojazd marszem po 5 cm
z `q_ready`, `σ_min = np.linalg.svd(kin.jacobian(q), compute_uv=False)[-1]` w pozie docelowej.

```
sigma_min:  min 0.0176   mediana 0.1378   max 0.2182
```

| Reżim | Gdzie | `σ_min` | Wniosek |
|---|---|---|---|
| Bezpieczny | Cała strefa chwytu (`z = 0.425`) | `0.138 – 0.218` | `σ_min ≈ 3–4 x λ` → tłumienie praktycznie nieaktywne, DLS zachowuje się jak pseudoodwrotność. |
| Graniczny | Cele na `z = 0.65`, `x = 0.5` na `z = 0.65` | `0.055 – 0.073` | `σ_min ≈ λ` → tłumienie wchodzi do gry, wzmocnienie bliskie maksimum `1/(2λ)`. |
| Osobliwy | **`(0.6, ±0.2, 0.65)`** | **`0.0176`** | `σ_min < λ`. IK **nie dojeżdża**: błąd końcowy **4.4 cm**, no-op na progu `dq_max`. |

**To jest realne znalezisko, nie ciekawostka.** Osobliwość siedzi w rogu „daleko **i** wysoko"
- ramię jest wtedy wyprostowane w łokciu, a `WORKSPACE_BOX` pozwala tam wejść (`x ≤ 0.70`,
`z ≤ 0.65`), więc polityka *będzie* tam trafiać podczas eksploracji. Konsekwencje:

- Statystyka `ik_failures` będzie **skoncentrowana przestrzennie**, nie rozłożona równomiernie.
  Raportując ją w pracy jako metrykę porównawczą DP vs RL, trzeba to zaznaczyć - inaczej wyższy
  wskaźnik u jednej metody czyta się jako wada metody, a nie jako informacja o tym, że częściej
  eksplorowała róg pudełka.
- `WORKSPACE_BOX` w obecnej formie jest **prostopadłościanem opisanym na strefie roboczej,
  która prostopadłościanem nie jest**. Uczciwe warianty naprawy: (a) obniżyć `z_max` przy dużym
  `x`, (b) zostawić i policzyć porażki. Wariant (b) jest metodologicznie w porządku pod
  warunkiem, że zostanie opisany.
- Zmierzone `0.0176 < λ = 0.05` **dowodzi, że tłumienie nie jest ozdobnikiem**: bez niego ten
  punkt dawałby wzmocnienie `1/σ ≈ 57 rad/m` zamiast ograniczonego `≤ 1/(2λ) = 10 rad/m`.
  To gotowy akapit uzasadniający wybór DLS zamiast czystej pseudoodwrotności.

Zastrzeżenie: to czysta kinematyka TCP, bez modelu kolizji i bez JTC - mówi o uwarunkowaniu
jakobianu, nie o wykonalności ruchu w Gazebo.

### 3.2 `dq_max = 1.0 rad` (próg no-op) - **[W]** z §3.1

Próg odrzucenia kroku IK jest ustawiony na `2 x` najgorszy przypadek z §3.1 (`0.5 rad`). Dzięki
temu wyzwala się **wyłącznie** wtedy, gdy uwarunkowanie jest realnie złe, a nie przy normalnym
sięganiu na skraj zasięgu. Przekroczenie → brak publikacji komendy + inkrementacja
`ik_failures` (statystyka porównawcza DP vs RL, `info`).

### 3.3 `tol = 1e-3` - **[P]**, spójne z pomiarem

Norma błędu 6D poniżej `1e-3`. Uwaga metodologiczna, którą warto w pracy nazwać wprost:
**`err6` miesza jednostki** - trzy pierwsze składowe są w metrach, trzy kolejne w radianach,
a norma liczona jest na złożeniu. `1e-3` odpowiada więc 1 mm **lub** 1 mrad (0.057°) i te dwa
progi są traktowane jak równoważne, co jest arbitralne. Zmierzony wynik `0.50 mm` na cel
i `≤ 0.01°` na orientację (154 waypointy, 0 porażek) mieści się w tym progu z zapasem.

### 3.4 Waga orientacji `w_rot` - **stan: nieobecna w kodzie, implicite `1.0`** - **[P]**

Plan przewidywał osobną wagę członu obrotowego. `pose_error` jej nie ma, więc efektywnie
`w_rot = 1.0` - 1 rad błędu orientacji waży tyle, co 1 m błędu pozycji. Przeliczając na
jednostki praktyczne: **błąd 1° = 0.0175 rad odpowiada 17.5 mm błędu pozycji**. Domyślna waga
jest więc silnie proorientacyjna, a nie neutralna, jak sugeruje wartość „1.0".

To jest zgodne z decyzją o jakobianie 6x7 (orientacja ma być aktywnie regulowana, nie
pomijana) i z pomiarem - orientacja trzyma się `≤ 0.01°`, czyli jest dociskana agresywnie.
Jeśli w treningu pojawią się porażki IK przy sięganiu na skraj, `w_rot < 1` jest pierwszą
gałką do pokręcenia: rozluźnia orientację, żeby kupić zasięg.

### 3.5 `max_iters = 50` - **[P]**

Iteracje wewnętrzne solvera na jeden krok polityki. Z §3.1: przy zbieżności liniowej i kroku
0.05 m do celu, praktycznie zawsze wystarcza kilka iteracji; 50 to sufit chroniący przed
zawieszeniem pętli treningu, a nie realny budżet.

---

## 4. Czas, częstotliwości, długość epizodu

| Stała | Wartość | Status | Wyprowadzenie |
|---|---|---|---|
| Krok fizyki | `1 ms` | **[P]** | `<max_step_size>0.001</max_step_size>` - wartość domyślna Gazebo, nie zmieniana. Wpływu większego kroku na jakość kontaktu palec–kostka **nie badaliśmy**; gdyby trening okazał się wolny, to pierwszy kandydat do ablacji (a zarazem pierwsze ryzyko przenikania kształtów). |
| `update_rate` kontrolerów | `1000 Hz` | **[W]** | Dopasowane 1:1 do kroku fizyki - kontroler liczy się raz na krok solvera, bez aliasowania. |
| `dt` polityki | `0.05 s` (20 Hz) | **[P]** | Górny koniec pasma 10–20 Hz z literatury (Chi 2023 dla DP). Wyższa częstotliwość = więcej kroków na epizod = dłuższy trening; niższa = szarpany ruch. |
| Kroków fizyki na krok polityki | `50` | **[W]** | `0.05 / 0.001`. |
| Skala akcji | `0.05 m/krok` | **[P]** | Przy 20 Hz daje `1 m/s` prędkości EE. Do podparcia w pracy kartą katalogową FR3 - jest to wartość wyraźnie poniżej limitu sprzętowego, więc pełni rolę limitu bezpieczeństwa i ujednolicenia dynamiki między metodami. |
| `max_episode_steps` | `200` (10 s) | **[W]** | Sam transport to `0.725 / 0.05 ≈ 15 kroków`. Doliczając sięgnięcie, chwyt (§5), podniesienie i odłożenie - rząd 40–60 kroków dla eksperta. `200` daje ~3–4x zapas na eksplorację RL, nie będąc na tyle długim, żeby zaśmiecać replay buffer stanem terminalnym. |
| `time_from_start` w JTC | `= dt` | **[P]** | Jednopunktowa trajektoria domykana dokładnie na następnym kroku polityki. **Nie schodzić poniżej** - JTC odrzuca trajektorię z `time_from_start = 0`. |

---

## 5. Chwytak

| Stała | Wartość | Status | Wyprowadzenie |
|---|---|---|---|
| Pozycja palca „dotyka kostki" | `0.025 m` **nominalnie** | **[W]** | Połowa boku kostki - wartość przy kostce idealnie wyśrodkowanej między palcami. **Realny podział bywa mocno niesymetryczny** (zmierzone: `0.040 / 0.010`), bo każdy palec staje tam, gdzie zablokuje go bryła. Niezmiennikiem jest **suma** = bok kostki, i tylko na niej wolno opierać detekcję chwytu (§2.3). |
| Komenda „zamknięte" | `0.020 m` | **[P]** | `5 mm` przesterowania poniżej styku. Przy sterowaniu **pozycyjnym** to jedyny sposób wygenerowania siły docisku - regulator próbuje domknąć do wartości niemożliwej i napiera. |
| Prędkość palca | `0.2 m/s` | **[W]** | Z URDF: `<limit velocity="0.2">`. |
| **Czas zamknięcia** | `≈ 0.1 s = 2 kroki polityki` | **[W]** | `(0.04 − 0.02) / 0.2 = 0.1 s`, przy `dt = 0.05` to **dwa** kroki. |
| `mu` kostki | `1.0` | **[P]** | Parametr symulacji, **nie** wartość fizyczna. Do zaraportowania w pracy jako taki. |

**Smaczek do pracy - chwytak nie jest binarny w czasie.** Choć akcja `g` jest interpretowana
binarnie (próg, wzór FurnitureBench), fizyczne zamknięcie zajmuje **2 kroki polityki**.
Polityka nie może więc zamknąć chwytaka i w tym samym kroku ruszyć w górę - musi „poczekać".
Dla RL to zwykły element dynamiki do nauczenia; dla **skryptowanego eksperta to jawny warunek
poprawności** - waypoint zamknięcia musi być utrzymany przez ≥ 2 kroki, inaczej demonstracje
będą zawierać podnoszenie z niedomkniętym chwytakiem i DP nauczy się tego błędu.

**Smaczek do pracy - dlaczego statyczny test chwytu niczego nie dowodzi.** Kostka waży
`0.05 kg` → `mg ≈ 0.49 N`. Utrzymanie jej tarciem dwóch palców wymaga
`2·μ·F_n ≥ mg`, czyli przy `μ = 1.0` zaledwie `F_n ≥ 0.245 N`. Ale w ruchu dochodzi siła
bezwładności: hamowanie z `1 m/s` w jednym kroku `0.05 s` to `a = 20 m/s² ≈ 2g`, co potraja
wymaganą siłę tarcia. Dlatego statyczne podniesienie (zweryfikowane ręcznie przez
`rqt_joint_trajectory_controller`) **nie jest** testem chwytu, a strojenie `μ` świadomie
odłożono do momentu, gdy kostką szarpie polityka.

---

## 6. Reset i warunki terminacji (`gym_env.py`)

| Stała | Wartość | Status | Wyprowadzenie |
|---|---|---|---|
| `TOL` (zbieżność do pozy home) | `0.05` | **[P]** | Norma `‖q − q_ready‖` po 7 stawach. Rząd 0.05 rad ≈ 2.9° - na tyle ciasno, że kolejne epizody startują z powtarzalnej pozy, na tyle luźno, żeby nie czekać na ogon regulacji JTC. |
| `N_settle` | `60` | **[W]** | Sufit czasu na dojazd do pozy home: `60 x 0.05 = 3 s`. Pętla wychodzi wcześniej po `TOL`; to zabezpieczenie, nie oczekiwany czas. |
| `N_cube_settle` | `15` | **[P]** | `0.75 s` na uspokojenie kostki po `SetEntityPose`. Kostka stawiana jest dokładnie na blacie (`z = 0.425`), więc nie spada - ten czas pochłania drgania kontaktu solvera, nie swobodny lot. |
| Próg upadku kostki | `cube_z < 0.35` | **[W]** | Blat jest na `0.40`, środek leżącej kostki na `0.425`. `0.35` leży **poniżej blatu**, więc warunek jest jednoznaczny: nie da się go spełnić kostką leżącą gdziekolwiek na stole. Bez tego warunku kostka wpadająca w szczelinę (§1) generuje 200 kroków martwego stanu w replay bufferze. |

### 6.1 Warunek sukcesu - trzy składniki, nie jedna norma

Sukces to **kostka leżąca na blacie docelowym w stanie ustalonym**, nie migawka pozycji
(uzasadnienie decyzji: `thesis-project-context.md`, „Cel zadania"). Sprawdzenie jedną normą 3D
`‖cube − goal‖ < tol` byłoby za słabe: kostka **trzymana w chwytaku 4 cm nad blatem docelowym**
przeszłaby taki test i agent inkasowałby `R_SUCCESS` bez odłożenia - dokładnie ta degeneracja,
którą odrzucono na poziomie projektu sceny.

| Stała | Wartość | Status | Wyprowadzenie |
|---|---|---|---|
| `SUCCESS_XY_TOL` | `0.05` | **[W]** | Stół docelowy 0.25 m, kostka 0.05 m. Środek kostki może odejść od środka blatu o `(0.25 − 0.05)/2 = 0.10 m`, zanim kostka zacznie wystawać - to kres. `0.05` zostawia ≥ 5 cm blatu za każdą krawędzią kostki, więc sukces oznacza „stoi pewnie", a nie „balansuje na skraju". Zbieżnie ze standardem panda-gym (`0.05`). |
| `SUCCESS_Z_TOL` | `0.01` | **[W]** | Kostka leżąca na blacie ma środek dokładnie na `0.425` (§1); dopuszczalne pasmo to `[0.415, 0.435]`. Oś Z **musi** być sprawdzana osobno od XY - to jest ten warunek, który odróżnia „odłożona" od „trzymana nad stołem", a wrzucona do wspólnej normy 3D rozmywa się. Górna granica `0.435` pokrywa się z progiem wyjścia latcha (`0.425 + LIFT_MARGIN_EXIT`, §6a), więc pasmo wysokości dopuszczające sukces jest dokładnie tym pasmem, w którym latch jest zwolniony - dwa warunki nie mogą sobie zaprzeczyć. |
| `SUCCESS_VEL_EPS` | `0.01` m/s | **[Z]** + **[P]** | Pomiar (niżej) obala założenie, na którym stała pierwotna notatka: DART **zeruje** prędkość leżącej kostki dokładnie. Próg nie filtruje więc szumu, tylko odróżnia „leży" od „jeszcze się rusza"; `0.01 m/s` = `0.5 mm` na krok `dt = 0.05 s`, czyli ruch o dwa rzędy poniżej `SUCCESS_XY_TOL`. |

#### Pomiar szumu prędkości spoczynkowej - **[Z]** (09.2026)

Metoda: subskrypcja `/model/cube/odometry` (50 Hz, `use_sim_time`), trzy scenariusze w
uruchomionym `bringup.launch.py`, bez ramienia w kontakcie z kostką.

| Scenariusz | Próbki | `max ‖v_lin‖` | `max ‖v_ang‖` |
|---|---|---|---|
| A - kostka stoi nietknięta na blacie głównym, 20 s | 1001 | **0.0** | **0.0** |
| B - `SetEntityPose` dokładnie na blat docelowy (`z = 0.425`), 10 s | 501 | **0.0** | **0.0** |
| C - `SetEntityPose` 2 cm nad blatem (spadek ≈ puszczenie z chwytaka), 10 s | 501 | `0.524` (przelot) | 0.0 |

Wynik jest **binarny, nie zaszumiony**: poza transjentem kontaktu odczyt to dokładne `0.0` -
2500 kolejnych próbek co do bitu. DART wygasza ciało spoczywające, więc dolnej granicy progu nie
wyznacza żadna liczba fizyczna. W scenariuszu C kostka trwale schodzi poniżej `0.001 m/s` po
**0.06 s** (~1 krok polityki), a `p95` całego okna wynosi `2.1e-4 m/s` - transjent lądowania jest
krótszy niż jeden krok i nie generuje fałszywych negatywów.

Górną granicę wyznacza sens zadania, nie pomiar: przy 5 cm/krok kostka niesiona w chwytaku ma
`‖v‖` rzędu `1 m/s`, a `0.01 m/s` odrzuca ją z zapasem dwóch rzędów. Stąd status mieszany -
zerowa podłoga jest **[Z]**, sam wybór `0.01` jako „ruch pomijalny" pozostaje **[P]**.

> Konsekwencja praktyczna: ten warunek jest w obecnej symulacji niemal zawsze spełniony lub
> niemal zawsze złamany, bez strefy szarej. Nie jest przez to zbędny - chroni przed zaliczeniem
> sukcesu kostce w locie nad blatem docelowym (`z` w tolerancji przez jedną klatkę) i kostce
> zsuwającej się z blatu. Gdyby po strojeniu tarcia palców pojawiły się w Gazebo drgania
> kontaktowe, pomiar trzeba powtórzyć - założenie „podłoga = 0" jest właściwością obecnej
> konfiguracji solvera, nie gwarancją.

Czwarty warunek, `not grasped`, nie ma stałej: kostka trzymana nie jest odłożona, niezależnie od
tego, jak dobrze trafia w tolerancje. Bez niego agent mógłby zawisnąć w tolerancji i zaliczyć
sukces, nigdy nie puszczając.

**Kolejność w `reset()` jest wymuszona fizyką, nie stylem:** najpierw odjazd ramienia do pozy
home, dopiero potem `SetEntityPose` na kostkę. Odwrotnie - kostka może zostać postawiona
wewnątrz geometrii palców, a solver kontaktu wystrzeli ją ze sceny.

---

## 6a. Wykrywanie chwytu - histereza (`_update_grasped`)

Predykat chwytu jest **automatem stanowym**, nie funkcją bieżącej klatki. Powód: warunek
wysokościowy siedzi na granicy przez wiele kroków transportu (agent nie musi podnosić kostki
wysoko), a przy 5 cm/krok i drganiach kontaktu DART przeskakiwałby w tę i z powrotem. Migotanie
predykatu to szum w nagrodzie o amplitudzie `R_GRASP`.

| Stała | Wartość | Status | Wyprowadzenie |
|---|---|---|---|
| `GRASP_OPENING_MIN / MAX` | `0.036 / 0.064` | **[Z]** | Rozwarcie **całkowite** (suma obu palców, zakres `[0, 0.08]` - patrz §2.3). Kostka 5 cm domyka chwytak do `0.050` (zmierzone 09.2026: `0.040 + 0.010`), więc próg typu `opening < 0.02` **nigdy nie odpali podczas realnego chwytu**. Okno zamiast progu odróżnia „zacisnięte na kostce" (≈0.050) od „zacisniętego w próżni" (≈0.0) i od otwartego na oścież (`0.080`). Szerokość `±28%` wokół boku kostki - ta sama tolerancja względna co w poprzedniej, jednopalcowej wersji. |
| `GRASP_DIST_THRESH` | `0.05` | **[P]** | `‖ee_to_cube‖`; `fr3_hand_tcp` leży między palcami, więc przy chwycie odległość jest bliska zeru. Próg rzędu rozmiaru kostki. |
| `LIFT_MARGIN_ENTER` | `0.02` | **[W]** | Wejście przy `cube_z > 0.445`. Referencją jest `0.425` (środek **leżącej** kostki), nie `0.40` (blat) - pomylenie tych dwóch daje predykat zawsze prawdziwy. |
| `LIFT_MARGIN_EXIT` | `0.01` | **[W]** | Wyjście przy `cube_z < 0.435`. Pasmo histerezy 1 cm. Kluczowe: `0.435 > 0.425`, więc kostka **leżąca lub odłożona zawsze zwalnia latch** - to jest gwarancja, że latch nie da się farmić. |
| `N_GRASP_CONFIRM` | `2` | **[P]** | Wejście wymaga 2 kolejnych kroków spełnienia warunków; wyjście jest natychmiastowe. Asymetria wynika z tego, że fałszywie dodatni (agent inkasuje bonus bez przenoszenia) jest kosztowniejszy niż fałszywie ujemny. |

Fałszywy chwyt „na stole" jest strukturalnie niemożliwy, bo wejście wymaga `cube_z > 0.445` -
kostka musi fizycznie być w powietrzu, czego agent nie osiągnie bez trzymania jej. Realne
fałszywie dodatnie to jednoklatkowe odbicie kostki przez palce, i to usuwa `N_GRASP_CONFIRM`.

**Koszt: flaga `is_grasped` musiała wejść do obserwacji** (`OBS_DIM` 16 → 17, mapowana na
`±1.0` dla spójności skali z resztą wektora). Bez latcha `grasped` dawało się wyliczyć z samej
obserwacji (`gripper`, `ee_to_cube`, a `cube_z = ee_pos.z + ee_to_cube.z`), więc nagroda była
markowowska. Latch to psuje - dwa identyczne wektory obserwacji w paśmie histerezy dostawałyby
różną nagrodę, a krytyk SAC nie miałby z czego wywnioskować różnicy. Dołożenie flagi do
obserwacji ten koszt zeruje.

---

## 6b. Wagi nagrody (`config.py`) - **[P]**, z rachunku na zwrotach

Wagi `1.0` przy członach gęstych dobrane tak, żeby nagroda była dosłownie „minus odległość
w metrach" - interpretowalna bez tabeli przeliczeniowej.

| Stała | Wartość | Wyprowadzenie |
|---|---|---|
| `W_REACH` | `1.0` | `d_reach` startowo ≈ 0.20 m (poza ready `[0.307, 0, 0.487]` → kostka `[0.5, 0, 0.425]`), max ≈ 0.35 m |
| `W_TRANSPORT` | `1.0` | `d_transport` startowo ≈ 0.73 m (§1), max ≈ 0.75 m |
| `R_GRASP` | `1.0` | **jednorazowo**, na zboczu `False → True` |
| `R_SUCCESS` | `20.0` | Musi przebić sumę kar gęstych udanego epizodu (≈ −11) z zapasem |
| `W_ENERGY` | `0.01` | `‖a‖² ≤ 4` (4 wymiary w `[-1,1]`), więc max `0.04`/krok - dwa rzędy mniej niż `R_GRASP`. Przy realnych akcjach (`‖a‖² ≈ 0.5`) przez cały epizod uzbiera się ≈ 1.0, więc kara gryzie dopiero przy szarpaniu na pełnej amplitudzie - o to chodzi w metryce smoothness. |

**Zwroty z epizodu przy tych wagach** (rachunek, nie pomiar):

| Strategia | Zwrot |
|---|---|
| Rozwiązanie zadania (~25 kroków) | **≈ +10** |
| Chwyt bez dowiezienia (200 kroków w powietrzu) | ≈ −145 |
| Nicnierobienie (200 kroków) | ≈ −185 |
| Zrzucenie kostki w szczelinę w 10. kroku | ≈ −9 |

Porządek jest właściwy: rozwiązanie zadania wygrywa wyraźnie, każda forma farmienia przegrywa.

### Dwie pułapki strukturalne wykryte przy tym rachunku

**1. `R_GRASP` nie może być płacone co krok** (dlatego jest na zboczu). Gdy agent dowiezie
kostkę nad cel, człon transportowy jest ≈ 0, więc każdy krok daje czyste `+R_GRASP`, a odłożenie
kończy epizod jednorazowym `R_SUCCESS`. Przy 170 pozostałych krokach i `R_GRASP = 0.5` wiszenie
jest warte 85, a odłożenie 10 - agent nauczyłby się trzymać kostkę nad celem i nie puszczać.
Bonus jednorazowy usuwa atraktor: po chwycie zostaje samo `−W_TRANSPORT·d`, więc opłaca się
jechać do celu, a przy celu opłaca się puścić. Markowowskości to nie łamie, bo `is_grasped`
jest w obserwacji, więc zbocze odczytuje się z pary `(s, s')`.

**2. Ucieczka przez zrzucenie kostki - ZNANE, świadomie zaakceptowane.** Nagroda gęsta jest
w całości ujemna, a `dropped` kończy epizod, więc skrócenie epizodu jest opłacalne (−9 zamiast
−185). Kara za upuszczenie tego nie naprawi - musiałaby wynosić ≈ −185, co rozjedzie krytyka.
To strukturalna właściwość „gęsta kara ujemna + wczesna terminacja", nie błąd doboru wag.

Ocena ryzyka: umiarkowane. Zepchnięcie kostki do szczeliny wymaga przepchnięcia jej 25–45 cm
w kierunku −x, czyli kilkunastu celowych kroków z kontaktem - losowa eksploracja tego nie
znajdzie, a gradient „zbliż się do kostki" jest natychmiastowy.

**Sygnatura w logach, jeśli jednak wystąpi:** rosnący odsetek epizodów kończonych przez
`dropped`, przy krótkiej średniej długości epizodu i braku wzrostu success rate. Właściwa
reakcja to przejście na shaping potencjałowy (Ng i in. 1999): `r = r_sparse + γΦ(s') − Φ(s)`
z `Φ = −(W_REACH·d_reach·(1−grasped) + W_TRANSPORT·d_transport)`. Człony gęste się wtedy
teleskopują, skracanie epizodu przestaje cokolwiek dawać, a twierdzenie o niezmienniczości
gwarantuje, że polityka optymalna się nie zmienia.

---

## 7. Pozostałe uwagi warte odnotowania w pracy

- **Obserwacja jest „oracle", nie zmierzona.** Pozycja kostki idzie z `PosePublisher` Gazebo,
  czyli ze stanu solvera - bez szumu, bez okluzji, bez opóźnienia estymatora. To świadome
  uproszczenie (wariant vision-based odrzucony jako poza harmonogramem) i w rozdziale
  o ograniczeniach pracy musi być nazwane wprost, bo zawyża wyniki **obu** porównywanych metod
  w podobny sposób.
- **Faza 1 krokowania nie jest deterministyczna.** Symulacja leci swobodnie, a `reserve_t(dt)`
  tylko odmierza `dt` z `/clock`, więc liczba kroków fizyki między obserwacjami waha się.
  Miarą, kiedy trzeba przejść na `multi_step`, jest test: `reset(seed=42)` dwukrotnie → ta sama
  obserwacja startowa. Dopóki wahanie jest poniżej progu tolerancji, faza 1 wystarcza.
- **`model.nq = 9`, nie 7.** Model pinocchio budowany jest z URDF zawierającego link `world`
  i oba palce, więc jakobian z `computeFrameJacobian` ma wymiar **6x9**, a kolumny palców trzeba
  odciąć przed DLS.
  **Sprostowanie wobec `thesis-project-context.md`** (tam napisano „inaczej `JJᵀ` jest
  osobliwe") - **[Z]**, sprawdzone: kolumny palców mają normę dokładnie `0.0` (TCP jest
  potomkiem `fr3_hand`, nie palców), a `JJᵀ = Σᵢ jᵢjᵢᵀ`, więc kolumny zerowe **nic nie wnoszą**.
  Wartości singularne `J_full (6x9)` i `J_arm (6x7)` są identyczne co do czwartego miejsca
  (`1.8075, 1.6755, 1.1492, 0.3417, 0.3049, 0.2211` w pozie ready). Prawdziwy powód odcięcia
  jest wymiarowy, nie numeryczny: `dq` wyszłoby 9-elementowe (z dwoma zerami na końcu)
  i nie dodałoby się do 7-elementowego `q_arm`. Błąd objawiłby się jako niezgodność kształtów
  albo - gorzej - jako ciche broadcastowanie.
- **`fr3_joint4` ma limity `[-3.077, -0.117]`** - czyli `q4 = 0` jest **poza** zakresem. Test
  osobliwości wyprostowanego łokcia trzeba robić na `-0.117`, nie na `0`. Klasyczna pułapka
  przy przenoszeniu testów z podręcznikowego robota na FR3.
- **Poza `ready` już ma chwytak pionowo w dół** (`FK(q_ready)`: `p = [0.307, 0, 0.487]`,
  oś Z TCP = `[0, 0, -1]`). Dlatego `R_frozen` bierze się wprost z `FK(Q_READY).rotation`,
  zamiast konstruować macierz obrotu ręcznie - o jedno miejsce na błąd znaku mniej.

---

## Dokumenty powiązane

- `docs/thesis-project-context.md` - decyzje projektowe i ich uzasadnienia
- `docs/cheatsheet.md` - komendy operacyjne
- `docs/plan-gym-wrapper-dls-ik.md` - plan wykonawczy etapu Gym + DLS-IK
