---

---

---
tags: [paper, reinforcement learning, stochastic, off-policy]
authors: Haarnoja, Zhou, Hartikainen, Abbeel, Levine
arxiv: 1812.05905 (improvement of 1801.01290)
status: przeczytane

---
## Overview
Artykuł przedstawia podejście Soft Actor-Critic czyli metoda off-policy model-free. Algorytm operuje na ciągłej przestrzeni stanów i akcji. W przeciwieństwie do klasycznych algorytmów, które skupiają się wyłącznie na maksymalizacji oczekiwanej nagrody, SAC wprowadza paradygmat **Maximum Entropy RL** - dąży do jednoczesnej maksymalizacji nagrody oraz entropii polityki. 

Dzięki temu osiągamy dane korzyści:
- **Głębsza eksploracja** - aktor jest zachęcany do testowania różnych trajektorii prowadzących do celu, co zapobiega przedwczesnemu utknięciu w minimach lokalnych
- **Efektywność i stabliność** - jako metoda off-policy, SAC wydajnie wykorzystuje zebrane wcześniej doświadczenia (replay buffer), zachowując przy tym stabilność uczenia
## Key concepts
 **Soft Policy Iteration** - teoretyczne ramy algorytmu skłądające sie z naprzemiennych faz:
 - Soft Policy Evaluation - szacowanie wartości (Q-value) obecnej polityki z uwzględnieniem bonusu za entropię
 - Soft Policy Improvement - aktualizacja wag aktora poprzez minimalizacji dywergencji $(D_{KL})$ między jego polityką a rozkłądem opartym na ocenach krytyka
 
 **Reparametrization trick** - matematyczny trik pozwalający na obliczanie gradientów w Backpropagation przez proces losowania akcji. Polega on na transformacji zewnętrznego szumu przy użyciu parametrów wypluwanych przez sieć aktora:
 
 1. Losowanie szumu: $$\epsilon \sim \mathcal{N}(0, 1)$$
2. Złożenie: $$ \alpha_{t} = \mu + \sigma \cdot \epsilon$$
3. Nałożenie ograniczeń $[-1, 1]$: $$\alpha_{t} = \tanh(\mu + \sigma \cdot \epsilon)$$

**Clipped Double Q-learning** - użycie dwóch niezależnych sieci Krytyka, które uczą się równolegle. Przy ocenie stanu algorytm zawsze wybiera niższą wartość predykcji spośród tych dwóch sieci. Skutecznie eliminuje to problem narastającego przeszacowywania wartości.

**Automatyczne strojenie temperatury ($\alpha$)** - Potraktowanie stałej określającej wagę entropii jako zmiennej optymalizacyjnej. Używając mnożników Lagrange'a, algorytm sam podnosi temperaturę (wymusza losowość) w nieznanych stanach i obniża ją, gdy zachowanie agenta staje się pewne, pilnując jedynie, by entropia nie spadła poniżej docelowego progu ($\overline{\mathcal{H}} = -dim(\mathcal{A})$).

**Soft Bellman Backup Operator** ($\mathcal{T}^\pi$) - wykonuje operację zwaną **backupem**. Polega ona na tym, że bierzemy nagrodę, którą dostaliśmy teraz oraz wartość, której spodziewamy się w następnym kroku, i przenosimy to wszystko w czasie do tyłu, aby zaaktualizować ocenę naszego obecnego stanu. Jest nam potrzebny ponieważ dzięki niemu krytyk się czegoś uczy. W oparciu o **Twierdzenie Banacha** o punkcie stałym wiemy, że jeśli wieloktronie będziemy aplikować $\mathcal{T}^\pi$ na jakiejś początkowej funkcji $\mathcal{Q}$ to błędy będą z każdym krokiem maleć aż funkcja zbiegnie do prawdziwej wartości:
$$\mathcal{T}^\pi Q(s_t, a_t) \triangleq r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1}}[V(s_{t+1})] $$

**Dywergencja Kullbacka-Leiblera $(D_{KL}(P \parallel Q)$** - miara różnicy rozkłądu P od Q. Używana jako funkcja kary, by Aktor dopasował swój rozkład do ocen wystawionych przez Krytyka.  Ważne jest to, że funkcja jest zawsze nieujmena, osiaga zero tylko wtedy gdy $Q=P$ oraz daje wyraźny gradient 
$$ D_{KL}(P \parallel Q) = \mathbb{E}_{x\sim P}[\log{\frac{P(x)}{Q(x)}}]$$

## Główne funkcje straty

**Loss Krytyka ($J_Q$)** - minimalizacja błędu średniokwadratowego (MSE) między predykcją a celem (Target):
$$J_Q(\theta) = \mathbb{E} \left[ \left( Q_\theta(s_t, a_t) - (r_t + \gamma (Q_{\text{target}}(s_{t+1}, a_{t+1}) - \alpha \log \pi(a_{t+1}\vert{}s_{t+1}))) \right)^2 \right]$$

**Loss Aktora ($J_\pi$)** - minimalizacja dywergencji KL (maksymalizacja oczekiwanej wartości minus oczekiwana entropia):
$$J_\pi(\phi) = \mathbb{E} \left[ \alpha \log \pi_\phi(a_t\vert{}s_t) - Q_\theta(s_t, a_t) \right]$$

**Loss Temperatury ($J(\alpha)$)** - karanie za odchylenie od docelowego poziomu losowości ($\overline{\mathcal{H}}$):

$$J(\alpha) = \mathbb{E} \left[ -\alpha (\log \pi(a_t\vert{}s_t) + \overline{\mathcal{H}}) \right]$$
