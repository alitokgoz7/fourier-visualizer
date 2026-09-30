# Fourier Serisi Görselleştirici — Geliştirme Planı

Bu belge, kod yazılmadan önce hazırlanan mimari ve iş planıdır. Proje ilerledikçe
"Durum" bölümü güncellenir.

## 1. Hedef

Eğitici, interaktif ve test edilmiş bir Fourier serisi görselleştiricisi:

* Matematik çekirdeği (NumPy/SciPy) arayüzden tamamen bağımsız, saf fonksiyonlardan oluşur.
* Görselleştirme katmanı (Plotly, Matplotlib) yalnızca çekirdeğin çıktısını çizer.
* Streamlit arayüzü ince bir katmandır: girdi toplar, önbellekli çekirdek çağrıları yapar,
  grafikleri gösterir.

## 2. Matematiksel kurallar (tüm projede tek sözleşme)

* Periyot `T`, temel açısal frekans `ω₀ = 2π/T`.
* Gerçek form (klasik `a₀/2` sözleşmesi):

  `f(t) ~ a₀/2 + Σ_{n≥1} [aₙ cos(nω₀t) + bₙ sin(nω₀t)]`,
  `aₙ = (2/T) ∫_T f(t) cos(nω₀t) dt`, `bₙ = (2/T) ∫_T f(t) sin(nω₀t) dt`.
  Diziler `a[0..N]`, `b[0..N]` olarak tutulur; `b[0] = 0`.
* Karmaşık form: `cₙ = (1/T) ∫_T f(t) e^{-inω₀t} dt`, `f(t) ~ Σ cₙ e^{inω₀t}`.
  Dönüşüm: `c₀ = a₀/2`, `cₙ = (aₙ − i bₙ)/2`, `c₋ₙ = (aₙ + i bₙ)/2`;
  ters yönde `aₙ = cₙ + c₋ₙ`, `bₙ = i(cₙ − c₋ₙ)`.
* Genlik/faz: `aₙ cos + bₙ sin = Aₙ cos(nω₀t + φₙ)`, `Aₙ = √(aₙ² + bₙ²)`, `φₙ = atan2(−bₙ, aₙ)`.
* Parseval: `(1/T)∫|f|² = (a₀/2)² + ½ Σ (aₙ² + bₙ²) = Σ |cₙ|²`.
* L2 hatası RMS olarak tanımlanır: `‖f − S_N‖ = √((1/T)∫|f − S_N|²)`.
* Süreksizlik noktalarında hazır sinyaller ortalama değeri (Dirichlet sözleşmesi) döndürür.
* Gibbs teorik oranı: `G = Si(π)/π − 1/2 ≈ 0.0894898722` (sıçramanın ≈ %8.95'i).
* Yumuşatma (σ-faktörleri, `n = 0..N`):
  Fejér `σₙ = 1 − n/(N+1)`, Lanczos `σₙ = sinc(n/(N+1)) = sin(πn/(N+1)) / (πn/(N+1))`.
* 2D yol DFT'si: `z_m = x_m + i y_m`, `X_k = Σ z_m e^{−2πi km/M}`, epicycle katsayısı
  `c_k = X_k / M`, işaretli frekans `k ∈ [−M/2, M/2)`, `z(s) = Σ c_k e^{2πi k s}`, `s ∈ [0, 1)`.

## 3. Mimari ve modüller

```
fourier_viz/
  __init__.py
  core/                     # Saf matematik; Streamlit/Plotly bağımlılığı YOK
    types.py                # Ortak tip takma adları (FloatArray, ComplexArray)
    integration.py          # Periyot üzerinde kareleme düğümleri/ağırlıkları
                            #   (trapez, Simpson, parçalı Gauss–Legendre, adaptif quad)
    series.py               # RealCoefficients, sayısal katsayılar, kısmi toplam S_N,
                            #   tek tek harmonikler, tüm N'ler için kümülatif toplamlar,
                            #   Fejér/Lanczos σ-faktörleri
    complex_dft.py          # ComplexCoefficients, gerçek↔karmaşık dönüşüm, sayısal cₙ,
                            #   doğrudan DFT/IDFT (numpy.fft ile doğrulanır), Epicycle/EpicycleSet
    expression.py           # AST beyaz listeli güvenli matematik ifadesi ayrıştırıcı (eval YOK)
    signals.py              # PeriodicSignal + hazır sinyaller + analitik katsayılar
                            #   + ifade tabanlı sinyal + kayma/toplama/ölçekleme
    analysis.py             # L2/maks hata, yakınsama çalışması, Gibbs ölçümü,
                            #   Parseval kontrolü, genlik/faz/güç spektrumu
    paths.py                # 2D şekiller, eşit yay uzunluğuna göre yeniden örnekleme,
                            #   CSV ayrıştırma, normalizasyon
    svg.py                  # SVG path 'd' ayrıştırıcı (M L H V C S Q T A Z, mutlak/göreli)
  viz/
    theme.py                # Ortak renk paleti (açık/koyu tema uyumlu)
    plots.py                # Plotly figürleri (sinyal, harmonikler, yakınsama animasyonu,
                            #   spektrum, hata, Gibbs yakın çekim, epicycle animasyonu)
    static.py               # Matplotlib statik figürler + PNG bayt dönüşümü (galeri/dışa aktarma)
    animation.py            # Matplotlib epicycle/yakınsama animasyonları → GIF/MP4 baytları
  app/
    streamlit_app.py        # Giriş noktası: sayfa düzeni, kenar çubuğu, sekmeler
    state.py                # Hashlenebilir SignalSpec/ShapeSpec + st.cache_data sarmalayıcıları
    tabs.py                 # Her sekmenin render fonksiyonu
    learn.py                # "Öğren" sekmesinin Türkçe/LaTeX içeriği
scripts/generate_gallery.py # docs/gallery altına Matplotlib görselleri üretir
tests/                      # core/, viz/, app/ ve özellik tabanlı testler
docs/                       # matematik.md, mimari.md, gallery/
```

### Tasarım kararları

* **Önerilen yapıdan sapmalar:** `expression.py`, `integration.py`, `svg.py`, `types.py`
  çekirdekte ayrı modüller oldu (tek sorumluluk, kolay test). Matplotlib statik grafikleri
  `viz/static.py`'de toplanır; hem galeri betiği hem PNG dışa aktarma bunu kullanır
  (Plotly PNG dışa aktarma Chrome gerektiren kaleido'ya bağlı olduğundan sunucu tarafında
  Matplotlib tercih edildi; Plotly grafiklerinin kendi araç çubuğundaki PNG indirmesi de açık).
* **Sayısal integral:** varsayılan yöntem, bilinen süreksizlik noktalarında bölünmüş
  parçalı Gauss–Legendre karelemesidir (parçalı düzgün sinyallerde çok hassas). Düzgün
  periyodik trapez kuralı (spektral doğruluk), Simpson ve SciPy `quad` (adaptif, toleranslı)
  seçilebilir. Hassasiyet `samples` (düğüm sayısı) veya `quad` toleranslarıyla ayarlanır.
  Tüm harmonikler tek matris çarpımıyla (parça parça, bellek sınırlı) hesaplanır.
* **Kısmi toplam:** `cos/sin(n ω₀ t)` matrisleri ile vektörleştirilmiş; tüm N değerleri için
  kümülatif toplam tek geçişte (`np.cumsum`) — yakınsama animasyonu için.
* **Güvenli ifade ayrıştırıcı:** `ast.parse(mode="eval")` → düğüm beyaz listesi
  (sayı sabitleri, `t`, `pi`, `e`, izinli fonksiyon çağrıları, + − * / ** % , karşılaştırmalar,
  `^` üs olarak). Öznitelik, indeks, lambda, string, anahtar kelime argümanı, `_` ile başlayan
  adlar reddedilir. Uzunluk ve düğüm sayısı sınırı, tüm hesap `float64` NumPy ile yapılır
  (dev tam sayı üsleri ile DoS mümkün değil; taşma `inf` olur ve yakalanır).
* **Epicycle:** DC terimi (k=0) zincirin başlangıç merkezi olur; diğer çemberler genliğe göre
  azalan sırada. `EpicycleSet.joints(s)` tüm zincir eklemlerini tek `cumsum` ile verir.
* **Çizim:** Üçüncü parti tuval bileşeni (streamlit-drawable-canvas) güncel Streamlit ile
  yüklenemediği için, Streamlit'in yerleşik `st.plotly_chart(on_select=..., selection_mode="lasso")`
  özelliği kullanılır: kullanıcı kement ile serbest şekil çizer, poligon noktaları yol olur.
* **Sekmeler:** `st.tabs(..., key=..., on_change="rerun")` ile yalnızca açık sekme hesaplanır;
  "Öğren" sekmesindeki bağlantı butonları `session_state` üzerinden sekme değiştirir.
* **Önbellek:** Hesaplamalar hashlenebilir `SignalSpec`/`ShapeSpec` değerleriyle
  `st.cache_data` içinde tutulur.
* **Hata mesajları:** Çekirdek istisnaları (`ExpressionError`, `SignalError`, `PathError`)
  Türkçe, kullanıcıya gösterilebilir mesajlar taşır; arayüz bunları `st.error` ile gösterir.
* **Docstring dili:** Arayüz ve belgeler Türkçe olduğu için docstring'ler Türkçe,
  tanımlayıcılar İngilizcedir.

## 4. Test stratejisi

* **Birim testleri** her modül için (`tests/core/...`): analitik katsayılar (kare, testere,
  üçgen, doğrultulmuş sinüsler, darbe, parabol), sayısal≈analitik, simetri, ortogonallik,
  Parseval, gerçek↔karmaşık dönüşüm, DFT↔numpy.fft, IDFT gidiş-dönüş, epicycle yeniden
  çizimi, yeniden örneklemede eşit aralık ve kapalılık, SVG/CSV ayrıştırma.
* **Özellik tabanlı testler** (hypothesis): rastgele trigonometrik polinomlarla doğrusallık,
  sonlu harmonikten tam yeniden oluşturma, zaman kaymasının genliği korunması, Parseval,
  DFT gidiş-dönüşü, ifade ayrıştırıcının rastgele güvenli/güvensiz girdileri.
* **Analitik davranış testleri:** sürekli sinyallerde L2 hatasının monoton azalması,
  Gibbs aşımının ≈ %8.95, Fejér ile aşımın ortadan kalkması, Lanczos ile belirgin azalma.
* **Güvenlik testleri:** `__import__`, `os.system`, öznitelik zinciri, `().__class__`,
  lambda, string, dev üsler vb. reddedilir/sonsuz döngüye girmez.
* **Uç durumlar:** N=0, N=1, sabit/sıfır fonksiyon, çok büyük N (ör. 5000), boş/tek noktalı/
  sıfır uzunluklu yol, NaN/sonsuz üreten ifade, geçersiz parametreler.
* **Performans:** N=500 kısmi toplam (2000 nokta) ve 2000 noktalı DFT < 1 sn.
* **Görselleştirme testleri:** figürler doğru iz/çerçeve sayısıyla üretilir; GIF/MP4/PNG
  baytları geçerli sihirli başlık baytlarıyla başlar.
* **UI duman testi:** `streamlit.testing.v1.AppTest` ile uygulama açılır, her sekme seçilip
  render edilir, kaydırıcı/seçim değişince istisna oluşmaz, hatalı ifade Türkçe hata verir.
* **Araçlar:** `pytest --cov` (çekirdek ≥ %90 hedefi, `fail_under` ile zorunlu), `ruff check`,
  `ruff format --check`, `mypy --strict` (fourier_viz paketi).

## 5. Adımlar (her biri ayrı commit; testler geçmeden sonraki adıma geçilmez)

1. PLAN.md
2. Proje iskeleti: pyproject.toml (bağımlılıklar, ruff/mypy/pytest/coverage ayarları),
   paket dizinleri, .gitignore
3. `core/integration.py` + testler
4. `core/series.py` (gerçek katsayılar, kısmi toplam, σ-faktörleri) + testler
5. `core/complex_dft.py` (karmaşık katsayılar, dönüşümler, DFT, epicycle) + testler
6. `core/expression.py` (güvenli ayrıştırıcı) + güvenlik testleri
7. `core/signals.py` (sinyal kütüphanesi, analitik katsayılar) + testler
8. `core/analysis.py` (hata, Gibbs, Parseval, spektrum) + testler
9. `core/paths.py` + `core/svg.py` + testler
10. Özellik tabanlı (hypothesis) testler ve performans testleri
11. `viz/plots.py`, `viz/static.py`, `viz/animation.py` + testler
12. Streamlit arayüzü (`app/`) + AppTest duman testleri
13. `scripts/generate_gallery.py`, görsellerin üretilip incelenmesi
14. README.md, docs/, GitHub Actions CI
15. Son kontrol: tüm testler + kapsama + ruff + mypy; düzeltmeler; PR

## 6. Bilinen riskler / sınırlamalar (öngörülen)

* Epicycle animasyonları Plotly çerçeveleri olarak istemciye gönderilir; çok sayıda çember ×
  çerçeve büyük JSON üretir → görünür çember sayısı ve çerçeve sayısı sınırlandırılır.
* SVG desteği yalnızca `path` öğelerinin `d` özniteliğiyle sınırlıdır (dönüşümler/`transform`,
  `circle`/`rect` gibi öğeler desteklenmez).
* Kement ile çizim tek bir kapalı poligon üretir (çoklu kontur yok).

## 7. Durum

- [x] 1. PLAN.md
- [x] 2. Proje iskeleti (pyproject, ruff/mypy/pytest/coverage ayarları)
- [x] 3. `core/integration.py` + testler
- [x] 4. `core/series.py` + testler
- [x] 5. `core/complex_dft.py` + testler
- [x] 6. `core/expression.py` + güvenlik testleri
- [x] 7. `core/signals.py` + testler (sonradan: ifadelerde otomatik sıçrama/tekillik tespiti)
- [x] 8. `core/analysis.py` + testler
- [x] 9. `core/paths.py` + `core/svg.py` + testler
- [x] 10. Özellik tabanlı ve performans testleri
- [x] 11. `viz/` (Plotly, Matplotlib, GIF/MP4) + `export.py` + testler
- [x] 12. Streamlit arayüzü + AppTest duman testleri (tarayıcıda Playwright ile de denendi)
- [x] 13. `scripts/generate_gallery.py`, görseller üretildi ve incelendi
- [x] 14. README.md, docs/, GitHub Actions CI
- [x] 15. Son kontrol ve PR

### Plandan sapmalar

* `export.py` (CSV/JSON) ayrı bir üst düzey modül oldu; `app/` modülü `common.py` ve
  `sidebar.py` ile bölündü.
* Epitrokoid varsayılanı, daha öğretici bir şekil için R=5, r=1, d=2 (beş ilmek) seçildi.
* Kullanıcı ifadeleri için otomatik sıçrama ve tekillik tespiti eklendi (başta "bilinen
  sınırlama" olarak düşünülmüştü).
