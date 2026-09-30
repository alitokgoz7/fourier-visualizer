# Matematiksel arka plan

Bu belge, uygulamadaki her hesaplamanın arkasındaki matematiği ve kullanılan sayısal
yöntemleri özetler. Tüm projede aynı sözleşmeler kullanılır.

## 1. Gerçek Fourier serisi

Periyodu $T$ olan bir $f$ için temel açısal frekans $\omega_0 = 2\pi/T$ olmak üzere

$$
f(t) \sim \frac{a_0}{2} + \sum_{n=1}^{\infty}\Big[a_n\cos(n\omega_0 t) + b_n\sin(n\omega_0 t)\Big],
$$

$$
a_n = \frac{2}{T}\int_{t_0}^{t_0+T} f(t)\cos(n\omega_0 t)\,dt, \qquad
b_n = \frac{2}{T}\int_{t_0}^{t_0+T} f(t)\sin(n\omega_0 t)\,dt .
$$

* $a_0/2$ sinyalin ortalamasıdır; kodda `RealCoefficients.a[0]` doğrudan $a_0$'ı tutar.
* $N$ terimli kısmi toplam $S_N(t) = \frac{a_0}{2} + \sum_{n=1}^{N}\big[a_n\cos(n\omega_0 t) + b_n\sin(n\omega_0 t)\big]$.
* **Genlik–faz biçimi:** $a_n\cos(n\omega_0 t) + b_n\sin(n\omega_0 t) = A_n\cos(n\omega_0 t + \varphi_n)$,
  $A_n = \sqrt{a_n^2 + b_n^2}$, $\varphi_n = \operatorname{atan2}(-b_n, a_n)$.
* **Dirichlet koşulları:** parçalı düzgün $f$ için $S_N(t) \to \tfrac12\big[f(t^-) + f(t^+)\big]$.
  Hazır sinyaller süreksizlik noktalarında bu ortalama değeri döndürür.

### Ortogonallik

$$
\frac{2}{T}\int_T \cos(n\omega_0 t)\cos(m\omega_0 t)\,dt = \delta_{nm}, \quad
\frac{2}{T}\int_T \sin(n\omega_0 t)\sin(m\omega_0 t)\,dt = \delta_{nm}, \quad
\int_T \cos(n\omega_0 t)\sin(m\omega_0 t)\,dt = 0
$$

($n, m \ge 1$). Seriyi $\cos(m\omega_0 t)$ ile çarpıp bir periyot boyunca integre edince
yalnızca $n = m$ terimi kalır ve katsayı formülü doğrudan elde edilir. Aynı nedenle kısmi toplam,
$f$'nin $\{1, \cos n\omega_0 t, \sin n\omega_0 t\}_{n \le N}$ uzayına **dik izdüşümüdür**:
bu uzaydaki tüm trigonometrik polinomlar arasında L2 hatasını en küçük yapan odur.

### Simetri

* Çift fonksiyon ($f(-t) = f(t)$): tüm $b_n = 0$ (yalnızca kosinüsler).
* Tek fonksiyon ($f(-t) = -f(t)$): tüm $a_n = 0$, ortalama $a_0$ dahil.

## 2. Hazır sinyaller ve analitik katsayılar

Faz $x = \omega_0 t \in [-\pi, \pi)$, genlik $A$:

| Sinyal | Tanım | Analitik katsayılar |
|---|---|---|
| Kare dalga | $A\operatorname{sgn}(x)$ | $b_n = \dfrac{4A}{n\pi}$ (tek $n$), diğerleri 0 |
| Testere dişi | $Ax/\pi$ | $b_n = \dfrac{2A(-1)^{n+1}}{n\pi}$ |
| Üçgen dalga | $A(1 - 2\lvert x\rvert/\pi)$ | $a_n = \dfrac{8A}{(n\pi)^2}$ (tek $n$) |
| Yarım dalga doğrultulmuş | $A\max(\sin x, 0)$ | $a_0 = \dfrac{2A}{\pi}$, $b_1 = \dfrac{A}{2}$, $a_n = -\dfrac{2A}{\pi(n^2-1)}$ (çift $n \ge 2$) |
| Tam dalga doğrultulmuş | $A\lvert\sin(x/2)\rvert$ | $a_n = -\dfrac{4A}{\pi(4n^2-1)}$ |
| $\lvert\sin\rvert$ | $A\lvert\sin x\rvert$ | $a_n = -\dfrac{4A}{\pi(n^2-1)}$ (çift $n$) |
| Darbe dizisi (doluluk $d$) | $A$ ($\lvert x\rvert < \pi d$), 0 | $a_0 = 2Ad$, $a_n = \dfrac{2A\sin(n\pi d)}{n\pi}$ |
| Parabolik dalga | $A(x/\pi)^2$ | $a_0 = \dfrac{2A}{3}$, $a_n = \dfrac{4A(-1)^n}{(n\pi)^2}$ |

"Tam dalga doğrultulmuş sinüs" ile "$\lvert\sin\rvert$" aynı dalga biçimidir; ikincisinde seçilen
periyot $T$, dalganın gerçek periyodunun iki katıdır. Bu yüzden $\lvert\sin(\omega_0 t)\rvert$
açılımında **yalnızca çift harmonikler** bulunur.

Örnek bir yan ürün: parabolik dalgayı $t = \pi$'de değerlendirmek
$\sum_{n\ge1} 1/n^2 = \pi^2/6$ (Basel problemi) sonucunu verir; bu, testlerde de doğrulanır.

## 3. Karmaşık form ve Euler formülü

$e^{i\theta} = \cos\theta + i\sin\theta$ ile

$$
f(t) \sim \sum_{n=-\infty}^{\infty} c_n e^{in\omega_0 t}, \qquad
c_n = \frac{1}{T}\int_T f(t)\,e^{-in\omega_0 t}\,dt .
$$

Dönüşümler: $c_0 = a_0/2$, $c_n = (a_n - ib_n)/2$, $c_{-n} = (a_n + ib_n)/2$; tersine
$a_n = c_n + c_{-n}$, $b_n = i(c_n - c_{-n})$. Gerçek sinyal için $c_{-n} = \overline{c_n}$
(Hermitsel simetri); bu koşul sağlanmıyorsa `complex_to_real` Türkçe bir hata verir.

**Zaman kayması:** $g(t) = f(t - \tau) \Rightarrow c_n(g) = c_n(f)\,e^{-in\omega_0\tau}$.
Genlikler $|c_n|$ değişmez, yalnızca fazlar döner. Gerçek formda
$\theta_n = n\omega_0\tau$ ile $a'_n = a_n\cos\theta_n - b_n\sin\theta_n$,
$b'_n = a_n\sin\theta_n + b_n\cos\theta_n$.

## 4. Parseval özdeşliği ve L2 hatası

$$
\frac{1}{T}\int_T |f(t)|^2\,dt = \Big(\frac{a_0}{2}\Big)^2 + \frac12\sum_{n\ge1}\big(a_n^2+b_n^2\big)
= \sum_{n} |c_n|^2 .
$$

L2 (RMS) hatası $\|f - S_N\| = \sqrt{\tfrac1T\int_T (f - S_N)^2\,dt}$ olarak tanımlanır.
Ortogonallikten $\|f - S_N\|^2 = \|f\|^2 - E_N$ (Bessel) çıkar; $E_N$ ilk $N$ harmoniğin
enerjisidir. Dolayısıyla **L2 hatası $N$ ile hiç artmaz**. Uygulama hem doğrudan sayısal
integrali hem de bu Parseval formülünü hesaplar ve karşılaştırır.

## 5. Yakınsama hızı

* Sıçramalı sinyaller (kare, testere, darbe): katsayılar $\sim 1/n$, L2 hatası $\sim N^{-1/2}$.
* Sürekli ama türevi sıçrayan sinyaller (üçgen, parabol, doğrultulmuş sinüsler):
  katsayılar $\sim 1/n^2$, L2 hatası $\sim N^{-3/2}$.

Log-log hata grafiğinin eğimi bu üsleri verir (`ConvergenceResult.l2_rate`). Süreksiz
sinyallerde **maksimum** hata $N$ ile azalmaz: $\sup|f - S_N| = |J|/2$ (sıçramanın yarısı),
çünkü $S_N$ sıçrama noktasında ortalamadan geçer — yakınsama noktasaldır, düzgün değildir.
Maksimum hata hesabı bu yüzden süreksizliklerin $\pm 10^{-9}T$ yakınına yoklama noktaları ekler.

## 6. Gibbs olayı ve yumuşatma

Kısmi toplam, Dirichlet çekirdeğiyle konvolüsyondur ($T = 2\pi$ için):

$$
S_N f(x) = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x-u)\,D_N(u)\,du, \qquad
D_N(u) = \frac{\sin\big((N+\tfrac12)u\big)}{\sin(u/2)} .
$$

$D_N$ negatif değerler de aldığından büyüklüğü $J$ olan bir sıçramanın hemen yanında

$$
\lim_{N\to\infty}\frac{\max S_N - f(t_d^+)}{|J|} = \frac{1}{\pi}\operatorname{Si}(\pi) - \frac12
\approx 0.0894899, \qquad \operatorname{Si}(x) = \int_0^x\frac{\sin u}{u}\,du
$$

kadar aşım oluşur (sıçramanın ≈ **%8.95**'i). $\pm1$ kare dalga için tepe
$\tfrac{2}{\pi}\operatorname{Si}(\pi) \approx 1.17898$ değerine yakınsar. Tepenin sıçramaya
uzaklığı $\approx T/(2N)$ ile daralır ama yüksekliği değişmez.

Ölçüm (`gibbs_overshoot`): sıçramanın yüksek tarafında, genişliği birkaç $T/N$ olan bir pencerede
sık ızgara + sınırlı skaler optimizasyonla $\max S_N$ bulunur; oran $(\max S_N - h)/|J|$'dir.

Katsayıları $\sigma_n$ ile ağırlıklandırmak aşımı azaltır:

* **Fejér (Cesàro):** $\sigma_N f = \frac{1}{N+1}\sum_{k=0}^{N}S_k f$, yani
  $\sigma_n = 1 - \frac{n}{N+1}$. Fejér çekirdeği $F_N \ge 0$ olduğundan
  $\min f \le \sigma_N f \le \max f$: aşım **tamamen** kaybolur (bedeli daha yavaş geçiştir).
* **Lanczos σ:** $\sigma_n = \operatorname{sinc}\!\big(\frac{n}{N+1}\big)
  = \frac{\sin(\pi n/(N+1))}{\pi n/(N+1)}$; aşım ≈ %1.19'a iner.

## 7. Sayısal integral (katsayı hesabı)

Katsayılar $\int f \approx \sum_k w_k f(t_k)$ biçimindeki kurallarla, tüm harmonikler için tek
matris çarpımıyla (bellek sınırlı parçalar hâlinde) hesaplanır:

| Yöntem | Açıklama | Doğruluk |
|---|---|---|
| `gauss` (varsayılan) | Periyot bilinen süreksizlik/kırılma noktalarından bölünür, her parça panellere ayrılır, panel başına 8 noktalı Gauss–Legendre | Parçalı düzgün sinyallerde makine hassasiyetine yakın |
| `trapezoid` | Düzgün periyodik trapez, $w_k = T/M$ | Düzgün periyodik fonksiyonlarda spektral; süreksizlikte $O(1/M)$; $\lvert n\rvert \ge M/2$ için örtüşme |
| `simpson` | Parçalı bileşik Simpson; parça uçları tek taraflı limitleri temsil eder | $O(h^4)$ parça içinde |
| `quad` | SciPy QUADPACK, salınımlı ağırlıklı QAWO, parça parça | Toleransla kontrol edilen adaptif referans (yavaş) |

Varsayılan düğüm sayısı $\max(4096, 16(N+1))$'dir; arayüzden "hassasiyet" olarak değiştirilebilir.

## 8. DFT ve epicycle'lar

Kapalı bir yolun $M$ noktası $z_m = x_m + iy_m$ ile

$$
X_k = \sum_{m=0}^{M-1} z_m e^{-2\pi ikm/M}, \qquad z_m = \frac1M\sum_{k=0}^{M-1}X_k e^{2\pi ikm/M}.
$$

`dft(..., method="direct")` tanımı uygular ($k\cdot m \bmod M$ ile önceden hesaplanmış birim
kökler kullanılarak yuvarlama hatası azaltılır) ve `numpy.fft` ile karşılaştırılarak test edilir.

Epicycle katsayıları $c_k = X_k/M$, işaretli frekans $k \in [-M/2, M/2)$:

$$
z(s) = c_0 + \sum_{k \ne 0} c_k e^{2\pi iks}, \qquad s \in [0, 1).
$$

Her terim yarıçapı $|c_k|$, başlangıç açısı $\arg c_k$ olan, periyot başına $k$ tur atan bir
çemberdir ($k < 0$: saat yönünde). Zincir $c_0$ (ağırlık merkezi) noktasından başlar; çemberler
$|c_k|$'ye göre azalan sırada eklenir. Ayrık Parseval özdeşliği gereği en büyük $K$ çemberi
tutmak ayrık L2 hatasını en aza indirir: hata$^2 = \sum_{\text{atılan}}|c_k|^2$
(özellik tabanlı testlerle doğrulanır). Tüm çemberlerle örnek noktalarından tam olarak geçilir.

### Eşit yay uzunluğuna göre yeniden örnekleme

DFT, noktaların eşit "zaman" aralıklarıyla alındığını varsayar. Poligon köşeleri gibi düzensiz
dağılmış noktalar çizim hızını değiştirir ve gereksiz yüksek frekanslar üretir. Bu yüzden
kümülatif yay uzunluğu $s_j$ hesaplanır ve yeni noktalar $s = kL/M'$ konumlarında doğrusal
interpolasyonla bulunur (kapalı yolda son noktadan ilk noktaya olan aralık da eşittir).

## 9. Güvenli ifade ayrıştırma

Kullanıcı ifadesi `ast.parse(mode="eval")` ile yalnızca **ayrıştırılır**; ağaç, beyaz listedeki
düğümlerden (sayı, `t`, `pi`/`e`/`tau`, izinli fonksiyon çağrıları, aritmetik, karşılaştırma,
`a if c else b`) oluşmuyorsa reddedilir ve NumPy işlemlerinden oluşan bir kapanışa derlenir.
`eval`/`exec` hiçbir yerde kullanılmaz. Tüm aritmetik `float64` ile yapıldığından
`9**9**9**9` gibi girdiler tam sayı patlamasına değil `inf` değerine yol açar ve NaN/sonsuz
denetimiyle Türkçe bir hata mesajına dönüşür.

### İfadelerde sıçrama ve tekillik tespiti

İfade sinyalleri için iç sıçramalar otomatik bulunur (`detect_jumps`): aralık $T/8192$
adımlarla taranır, olağan farkların çok üstündeki her aday aralık farkın büyük kaldığı yarıya
doğru ikiye bölünerek $\sim 10^{-13}T$ genişliğe daraltılır. Gerçek bir sıçramada fark daralan
aralıkta sabit kalır; dik ama sürekli bölgelerde (`tanh(50*t)`) veya sonsuz eğimli
kırılmalarda (`cbrt(t)`) küçülür — bu yüzden fark $10^{-10}T$ ve $10^{-13}T$ ölçeklerinde
neredeyse aynıysa sıçrama kabul edilir. Bulunan noktalar integralde parça sınırı olur (böylece
`where(t > 0, 1, -1)` kare dalgayla makine hassasiyetinde aynı katsayıları verir) ve Gibbs
analizinde kullanılır.

Her aday çevresinde ayrıca $|f|$'nin en büyük olduğu yere 10'ar kat yakınlaşılır: kutup tipi
tekilliklerde (`1/(t-1)`, `tan(t)`) değer büyümeye devam eder ve ifade Türkçe bir hata ile
reddedilir; dar ama sonlu tepelerde değer doyuma ulaşır ve kabul edilir.

