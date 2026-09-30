"""'Öğren' sekmesinin içeriği: kısa Türkçe açıklamalar ve LaTeX formülleri."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from fourier_viz.app.common import (
    TAB_CONVERGENCE,
    TAB_EPICYCLES,
    TAB_SERIES,
    TAB_SPECTRUM,
)


@dataclass(frozen=True)
class LearnSection:
    """Bir kavram bölümü.

    Attributes:
        key: Benzersiz anahtar.
        title: Başlık.
        body: Markdown (``$…$`` ve ``$$…$$`` LaTeX içerebilir).
        target_tab: İlgili sekmenin etiketi.
        target_hint: Sekmede ne denenebileceğine dair kısa öneri.
    """

    key: str
    title: str
    body: str
    target_tab: str
    target_hint: str


SECTIONS: Final[tuple[LearnSection, ...]] = (
    LearnSection(
        key="series",
        title="1. Fourier serisi ve katsayı formülleri",
        body=r"""
Periyodu $T$ olan (parçalı düzgün) her sinyal, frekansları temel frekansın tam katları olan
kosinüs ve sinüslerin toplamı olarak yazılabilir. $\omega_0 = 2\pi/T$ olmak üzere

$$
f(t) \sim \frac{a_0}{2} + \sum_{n=1}^{\infty}\Big[a_n\cos(n\omega_0 t) + b_n\sin(n\omega_0 t)\Big]
$$

$$
a_n = \frac{2}{T}\int_{-T/2}^{T/2} f(t)\cos(n\omega_0 t)\,dt, \qquad
b_n = \frac{2}{T}\int_{-T/2}^{T/2} f(t)\sin(n\omega_0 t)\,dt .
$$

$a_0/2$ sinyalin ortalamasıdır. İlk $N$ harmonikle yetinirsek **kısmi toplamı** elde ederiz:

$$
S_N(t) = \frac{a_0}{2} + \sum_{n=1}^{N}\Big[a_n\cos(n\omega_0 t) + b_n\sin(n\omega_0 t)\Big].
$$

Dirichlet koşulları altında $S_N(t)$ süreklilik noktalarında $f(t)$'ye, bir sıçrama noktasında
ise sol ve sağ limitlerin **ortalamasına** yakınsar. Örneğin genliği 1 olan kare dalga için
$a_n = 0$ ve tek $n$'ler için $b_n = \dfrac{4}{n\pi}$ bulunur.
""",
        target_tab=TAB_SERIES,
        target_hint="N kaydırıcısını değiştirin ve harmonik katmanını açın.",
    ),
    LearnSection(
        key="orthogonality",
        title="2. Ortogonallik ve simetri",
        body=r"""
Katsayı formüllerinin sırrı **ortogonalliktir**: farklı harmoniklerin bir periyot üzerindeki
iç çarpımı sıfırdır.

$$
\begin{aligned}
\frac{2}{T}\int_T \cos(n\omega_0 t)\cos(m\omega_0 t)\,dt &= \delta_{nm},\\
\frac{2}{T}\int_T \sin(n\omega_0 t)\sin(m\omega_0 t)\,dt &= \delta_{nm},\\
\int_T \cos(n\omega_0 t)\sin(m\omega_0 t)\,dt &= 0 \qquad (n, m \ge 1).
\end{aligned}
$$

Seriyi $\cos(m\omega_0 t)$ ile çarpıp integral alırsak toplamdaki tüm terimler yok olur,
yalnızca $n = m$ terimi kalır; bu da doğrudan $a_m$ formülünü verir.

**Simetri kısayolları:** çift fonksiyonda ($f(-t) = f(t)$) tüm $b_n = 0$, tek fonksiyonda
($f(-t) = -f(t)$) tüm $a_n = 0$ (ortalama $a_0$ dahil). Aşağıdaki Gram matrisi, sayısal
integralle hesaplanan iç çarpımların birim matrise ne kadar yakın olduğunu gösterir.
""",
        target_tab=TAB_SPECTRUM,
        target_hint="Üçgen dalgada (çift) sinüs, testere dişinde (tek) kosinüs terimlerinin "
        "kaybolduğunu görün.",
    ),
    LearnSection(
        key="euler",
        title="3. Euler formülü ve karmaşık form",
        body=r"""
Euler formülü $e^{i\theta} = \cos\theta + i\sin\theta$ sayesinde

$$
\cos\theta = \frac{e^{i\theta} + e^{-i\theta}}{2}, \qquad
\sin\theta = \frac{e^{i\theta} - e^{-i\theta}}{2i}
$$

yazılabilir ve seri daha kısa bir biçim alır:

$$
f(t) \sim \sum_{n=-\infty}^{\infty} c_n\, e^{i n\omega_0 t}, \qquad
c_n = \frac{1}{T}\int_T f(t)\, e^{-i n\omega_0 t}\,dt .
$$

İki form arasındaki dönüşüm: $c_0 = \dfrac{a_0}{2}$, $\;c_{\pm n} = \dfrac{a_n \mp i\,b_n}{2}$,
tersine $a_n = c_n + c_{-n}$, $\;b_n = i\,(c_n - c_{-n})$. Gerçek bir sinyal için
$c_{-n} = \overline{c_n}$ olur. $|c_n|$ genliği, $\arg c_n$ fazı verir.

**Zaman kayması:** $f(t - \tau)$ sinyalinin katsayıları $c_n e^{-i n\omega_0\tau}$ olur —
genlikler değişmez, yalnızca fazlar döner.
""",
        target_tab=TAB_SPECTRUM,
        target_hint="Çift taraflı gösterimde |cₙ| değerlerinin n ↔ −n simetrisine bakın.",
    ),
    LearnSection(
        key="parseval",
        title="4. Parseval özdeşliği (enerjinin korunumu)",
        body=r"""
Sinyalin ortalama gücü, katsayılarının kareleri toplamına eşittir:

$$
\frac{1}{T}\int_T |f(t)|^2\,dt
= \Big(\frac{a_0}{2}\Big)^2 + \frac{1}{2}\sum_{n=1}^{\infty}\big(a_n^2 + b_n^2\big)
= \sum_{n=-\infty}^{\infty} |c_n|^2 .
$$

Sonuç olarak kısmi toplamın L2 hatası $\|f - S_N\|^2 = \|f\|^2 - E_N$ olur; burada $E_N$ ilk
$N$ harmoniğin enerjisidir. Bu yüzden **L2 hatası $N$ arttıkça hiç artmaz**.

Güzel bir yan ürün: $\pm 1$ kare dalga için $1 = \dfrac{8}{\pi^2}\sum_{k\ge1}\dfrac{1}{(2k-1)^2}$,
yani $\displaystyle\sum_{k\ge1}\frac{1}{(2k-1)^2} = \frac{\pi^2}{8}$.
""",
        target_tab=TAB_SPECTRUM,
        target_hint="Parseval kutusundaki yakalanan oranın N ile %100'e yaklaşmasını izleyin.",
    ),
    LearnSection(
        key="gibbs",
        title="5. Gibbs olayı ve yumuşatma",
        body=r"""
Bir sıçrama noktasında kısmi toplam, $N$ ne kadar büyük olursa olsun sıçramanın yaklaşık
**%8.95'i** kadar aşım yapar:

$$
\lim_{N\to\infty}\frac{\max S_N - f(t_d^{+})}{|J|}
= \frac{1}{\pi}\operatorname{Si}(\pi) - \frac{1}{2} \approx 0.0895, \qquad
\operatorname{Si}(x) = \int_0^x \frac{\sin u}{u}\,du .
$$

Aşımın genişliği $\sim T/N$ ile daralır ama yüksekliği azalmaz: yakınsama noktasaldır,
**düzgün değildir**. Katsayıları $\sigma_n$ ile ağırlıklandırmak bunu hafifletir:

* **Fejér (Cesàro ortalaması):**
  $\sigma_N f = \dfrac{1}{N+1}\sum_{k=0}^{N} S_k$, yani $\sigma_n = 1 - \dfrac{n}{N+1}$.
  Fejér çekirdeği pozitif olduğundan aşım **tamamen** kaybolur (bedeli: daha yumuşak geçiş).
* **Lanczos σ:** $\sigma_n = \operatorname{sinc}\!\Big(\dfrac{n}{N+1}\Big)
  = \dfrac{\sin(\pi n/(N+1))}{\pi n/(N+1)}$; aşım yaklaşık %1.2'ye iner.
""",
        target_tab=TAB_CONVERGENCE,
        target_hint="Kare dalga seçip ölçülen aşımı teorik %8.95 ile karşılaştırın.",
    ),
    LearnSection(
        key="rates",
        title="6. Yakınsama hızı ve düzgünlük",
        body=r"""
Katsayıların azalma hızı sinyalin düzgünlüğünü yansıtır:

* sıçrama içeren sinyaller (kare, testere): $|a_n|, |b_n| \sim 1/n$, L2 hatası $\sim N^{-1/2}$;
* sürekli ama türevi sıçrayan sinyaller (üçgen, parabol, doğrultulmuş sinüs):
  $\sim 1/n^2$, L2 hatası $\sim N^{-3/2}$;
* sonsuz türevlenebilir sinyallerde katsayılar üstel hızla küçülür.

Log-log grafikte hata eğrisinin eğimi bu üsleri doğrudan gösterir.
""",
        target_tab=TAB_CONVERGENCE,
        target_hint="Kare ve üçgen dalganın log-log eğimlerini karşılaştırın.",
    ),
    LearnSection(
        key="epicycles",
        title="7. Epicycle'lar: DFT ile şekil çizmek",
        body=r"""
Kapalı bir eğrinin noktalarını karmaşık sayılar olarak yazalım: $z_m = x_m + i\,y_m$,
$m = 0,\dots,M-1$. Ayrık Fourier dönüşümü

$$
X_k = \sum_{m=0}^{M-1} z_m\, e^{-2\pi i k m/M}, \qquad c_k = \frac{X_k}{M}
$$

ile eğri, dönen vektörlerin toplamı olur:

$$
z(s) = \sum_{k} c_k\, e^{2\pi i k s}, \qquad s \in [0, 1).
$$

Her terim yarıçapı $|c_k|$, başlangıç açısı $\arg c_k$ olan ve bir turda $k$ kez dönen bir
çemberdir. Çemberler **genliğe göre sıralanıp** uç uca eklendiğinde son ucun izi şekli çizer;
en büyük $K$ çember, Parseval gereği en küçük kare hatayı verir. Noktalar önce **eşit yay
uzunluğuna** göre yeniden örneklenir; aksi hâlde çizim hızı düzensiz olur ve gereksiz yüksek
frekanslar ortaya çıkar.
""",
        target_tab=TAB_EPICYCLES,
        target_hint="Çember sayısını 3, 10, 50 yaparak şeklin nasıl netleştiğini izleyin.",
    ),
)
"""Öğren sekmesindeki bölümler (sırasıyla)."""
