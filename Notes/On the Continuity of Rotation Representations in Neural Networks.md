
---
tags: [paper, rotation-representation, deep learning]
authors: Zhou, Barnes, Lu, Yang, Li
arxiv: 1812.07035
status: przeczytane

---
## Overview
Artykuł o **ciągłości reprezentacji** obrotów w sieciach neuronowych.

Reprezentacja to para odwzorowań $g: X \to R$ oraz $f: R \to X$ z warunkiem $f(g(x)) = x$ dla każdego $x \in X$, gdzie $X$ to przestrzeń oryginalna (np. SO(3)), a $R$ to przestrzeń reprezentacji. Sieć produkuje element $R$, a $f$ jest zaszytą w forward passie funkcją matematyczną (nie warstwą uczoną).

> [!important] Reprezentacja jest ciągła wtedy, gdy ciągłe jest $g$ - odwzorowanie „do środka" sieci.  Nie $f$. Jeśli $g$ jest nieciągłe, spójny zbiór obrotów mapuje się na **rozspójniony** zbiór celów w $R$ - czyli sam sygnał uczący ma skoki.

Fundamentalne twierdzenie: **SO(3) nie da się przekształcić homeomorficznie w euklidesowej przestrzeni o 4 lub mniej wymiarach.** Powszechnie używane reprezentacje (kąty Eulera, kwaterniony itd.) są więc z konieczności nieciągłe - istnieją punkty, w których bardzo podobne obroty odpowiadają odległym wartościom reprezentacji. Rozwiązanie: reprezentacje **6D** (Gram-Schmidt) i **5D** (rzut stereograficzny). W praktyce używa się 6D.

## Key Concepts
**Homeomorfizm** - ciągłe przekształcenie jednej przestrzeni w drugą, które jest odwracalne.
Dwie przestrzenie są **topologicznie równoważne**, gdy istnieje między nimi homeomorfizm.

 **Zanurzenie topologiczne (embedding)** - odwzorowanie $g: X \to R$ jest homeomorfizmem na swój obraz $g(X)$. Ciągła reprezentacja jest zanurzeniem $X$ w $R$.
> [!note] Dlaczego to jest sedno dowodu.
>  Jeśli $X$ nie jest homeomorficzne z żadnym podzbiorem $R$, to **żadna** ciągła reprezentacja nie istnieje - niezależnie od tego, jak sprytnie ją parametryzujemy.

**Reprezentacja 6D (Gram-Schmidt)** - $g_{GS}$ odrzuca ostatnią kolumnę macierzy obrotu. $f_{GS}$ ją odtwarza: $$ b_1 = N(a_1), \quad b_2 = N(a_2 - (b_1 \cdot a_2)b_1), \quad b_3 = b_1 \times b_2 $$ gdzie $N(\cdot)$ to normalizacja. Uogólnia się na SO(n): $n^2-n$ wymiarów, ostatnia kolumna z **uogólnionego iloczynu wektorowego**.

**Błąd geodezyjny** - minimalna różnica kątowa między dwoma obrotami - metryka niezależna od reprezentacji: $$ L_{angle} = \cos^{-1}\left(\frac{\mathrm{tr}(M M'^{-1}) - 1}{2}\right) $$



