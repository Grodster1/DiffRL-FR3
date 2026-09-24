# Overview
Podczas treningu algorytmu SAC dla poziomu L1 (stały Goal Position) natrafiłem na kilka komplikacji. 
## Blokada stawu
Najważniejszą z nich była blokada ramienia (głównie joint2). Ramie trafiało na swoje ograniczenie i traciło możliwość dalszego ruchu. Komenda restart() nie działała, należało uruchomić ponownie bringup.

Aby usunąć ten błąd zastosowałem kilka poprawek, jednak żadna z nich nie tego wyeliminowała całkowicie.
- clip_to_joint_limits - dodanie w funkcji  `inverse_kinematics.py` marginesu CLIP_MARGIN do np.clip() 
- limit_joint_step - skaluje krok stawów zamiast obcinania per-staw. Zachowuje to kierunek obliczony przez ik. Po obliczeniu delty **q_target - q_current** sprawdzamy największą wartość w  macierzy. Jeżeli jest większa niż wartość **max_step** stosujemy skalowanie delty ($\Delta_q *= (\frac{max\_step}{largest}$) )
- MAX_JOINT_VEL - stała ustawiona na 1.0 wykorzystywana w limit_joint_step
## Potential Based Reward

Pierwszym podejściem do gęstej nagrody był **potential-based reward shaping** (Ng, Harada, Russell 1999). Kusił gwarancją teoretyczną: jeżeli człon kształtujący ma postać różnicy potencjałów
$$F(s, s') = \gamma\,\Phi(s') - \Phi(s)$$
to polityka optymalna zadania pozostaje niezmieniona. Można więc dosypać sygnału, nie zmieniając definicji zadania - a to dokładnie ten problem, który miałem: nagroda czysto rzadka (sukces/porażka) nie dawała SAC-owi żadnego gradientu przez pierwsze setki tysięcy kroków.

Przyjąłem potencjał:
$$\Phi(s) = -\big(W_{REACH}\cdot d_{reach}(s)\cdot[\lnot grasped] + W_{TRANSPORT}\cdot d_{transport}(s)\big)$$

gdzie $d_{reach}$ to odległość chwytaka od kostki, a $d_{transport}$ odległość kostki od celu. Wskaźnik $[\lnot grasped]$ wyłącza człon sięgania po chwyceniu kostki - po chwycie odległość chwytak–kostka przestaje być czymś, co warto minimalizować.

Podejście zawiodło i to z trzech niezależnych powodów.

**1. Dryf wynikający z $\gamma < 1$ - główna przyczyna.** Jeżeli stan się nie zmienia ($s' = s$), człon kształtujący nie zeruje się, tylko wynosi
$$F(s, s) = (\gamma - 1)\Phi(s) = -(1-\gamma)\Phi(s)$$
Ponieważ $\Phi \le 0$, jest to wartość **dodatnia**, proporcjonalna do $|\Phi(s)|$. Innymi słowy: samo stanie w miejscu przynosiło dochód, i to tym większy, im **dalej** agent był od celu. To wyjaśnia obserwację gdy agent traktował pozycję bliższą celu jako gorszą, mimo że dostawał nagrodę za zbliżanie się. Skala tego artefaktu przewyższała użyteczny sygnał. 

**2. Podatek za wczesne zakończenie.** Gwarancja PBRS wymaga $\Phi(\text{terminal}) = 0$. Upuszczenie kostki kończyło wtedy epizod, więc suma wypłacała się natychmiast - agentowi opłacało się porzucić zadanie, żeby zainkasować nagromadzone kształtowanie. Załatałem to stałą `R_DROP = -2.0`, dobraną tak, by przewyższała $|\Phi(s_0)|$

**3. Znikomy sygnał za chwyt.** Wydawało mi się, że wyłączenie członu $d_{reach}$ w momencie chwytu samo w sobie premiuje chwycenie kostki. W rzeczywistości w chwili chwytu $d_{reach} \approx 0{,}004$ m, więc skokowa nagroda za przejście przez tę bramkę wynosiła około **0,0007** - wartość tonąca w szumie. Nagroda za chwyt musiała i tak przyjść z osobnego członu `R_GRASP`.

Do tego dochodzi drobiazg implementacyjny: SB3 bootstrapuje przy obcięciu epizodu (`truncated`), więc teleskopowanie i tak nie zachodzi dokładnie przy limicie 200 kroków.

Wniosek: gwarancja Ng i in. jest prawdziwa, ale mówi o polityce optymalnej w granicy. Nie mówi nic o tym, czy sygnał przetrwa skończony budżet próbek - a tutaj użyteczna część nagrody była o rząd wielkości mniejsza niż artefakty jej własnej konstrukcji.

## Dense Reward

Zrezygnowałem z różnic potencjałów na rzecz **gęstego kosztu stanu** liczonego bezpośrednio ze stanu docelowego:
$$r = C(s') + R_{GRASP}\cdot[\text{pierwszy chwyt}] + R_{SUCCESS}\cdot[\text{sukces}] - W_{ENERGY}\lVert a \rVert^2$$

gdzie $C(s') \le 0$ jest karą za odległość od stanu docelowego. Kluczowa różnica: nagroda zależy od wartości bezwzględnej stanu, a nie od jego zmiany. Znika przez to cały problem 1 - stanie w miejscu nie daje dochodu, tylko kosztuje, i kosztuje tym więcej, im dalej agent jest od celu. Znika też problem 2, bo nie ma sumy teleskopowej do wypłacenia; upuszczenie kostki przestało kończyć epizod, więc porzucenie zadania nie daje już żadnej korzyści i stała `R_DROP` stała się zbędna.

Koszt stanu ma postać:
$$C(s) = -W_{TRANSPORT}\cdot d_{transport} - [\lnot grasped \wedge \lnot on\_goal]\big(W_{REACH}\cdot d_{reach} + W_G(1 - progress)\big) + W_{RELEASE}\cdot[\text{odłożona}]$$

Człon `progress` zastąpił nieskuteczną bramkę z punktu 3 - jest to łańcuch czterech etapów chwytu, z których każdy warunkuje następny:

- $s_1$ - zgranie w XY nad kostką,
- $s_2 = s_1 \cdot$ zgranie w Z,
- $s_3 = s_2 \cdot$ palce w oknie szerokości odpowiadającym trzymanej kostce,
- $s_4 = s_3 \cdot$ uniesienie kostki ponad blat,

a $progress = (s_1+s_2+s_3+s_4)/4$. Mnożenie zamiast sumowania wymusza kolejność: nie da się zainkasować nagrody za zacisk palców, stojąc metr od kostki.

Pierwsza wersja miała jednak poważną wadę, którą wychwyciłem dopiero po uruchomieniu treningu. Warunek wyłączający człon sięgania brzmiał po prostu $[\lnot grasped]$. W momencie, gdy agent zwalniał kostkę na celu, `is_grasped` przechodziło w `False`, człon $W_{REACH}\cdot d_{reach} + W_G(1-progress)$ wracał do gry i nagroda spadała skokowo - dokładnie w tym kroku, w którym agent robił rzecz pożądaną. Poprawką jest geometryczny warunek `on_goal`- jeżeli kostka leży w tolerancji celu, człon sięgania pozostaje wyłączony niezależnie od stanu chwytu.

**Premia za zwolnienie.** Dodatkowo stan „kostka na celu, chwytak rozwarty" dostaje premię `W_RELEASE`. Ponieważ jest to premia za stan, a nie za zdarzenie, agent mógłby ją farmić do końca epizodu - dlatego jej wartość musi spełniać
$$\frac{W_{RELEASE}}{1-\gamma} \ll R_{SUCCESS}$$
czyli suma premii możliwa do uzbierania przez nieskończony epizod ma być wyraźnie mniejsza niż jednorazowa nagroda za sukces. Przy `W_RELEASE = 0.05` i $\gamma = 0{,}99$ daje to 5 wobec `R_SUCCESS = 20`.

Efekt był jednoznaczny. W ostatnim przebiegu z nagrodą PBRS (`sac_L1_seed0_20260914_1419`, 891 epizodów, 178 tys. kroków) nie wystąpił ani jeden chwyt. Po przejściu na koszt stanu chwyt pojawił się, a następnie ustabilizował na poziomie bliskim 100% epizodów (`sac_L1_seed0_20260921_1331`, odcinek 700-900 tys. kroków: 912 chwytów na 1012 epizodów, przy czym w większości bloków po 50 epizodów było to 50/50 - braki pochodzą z jednego okresowego załamania polityki).

## Przeszacowanie krytyka 
