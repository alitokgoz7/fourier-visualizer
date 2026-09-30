# Mimari

Proje üç katmandan oluşur ve bağımlılıklar yalnızca yukarıdan aşağıya akar:

```
app (Streamlit)  ──►  viz (Plotly / Matplotlib)  ──►  core (NumPy / SciPy)
        └───────────────────────────────────────────────►┘
```

* **`fourier_viz/core`** — saf, durumsuz matematik. Streamlit veya çizim kütüphanesi içe aktarmaz;
  tüm fonksiyonlar tip ipuçlu ve formüllü docstring'lere sahiptir. Hatalar Türkçe mesaj taşıyan
  `ValueError` alt sınıflarıdır (`SeriesError`, `SignalError`, `ExpressionError`, `PathError`).
* **`fourier_viz/viz`** — hazır dizileri çizer; hesap yapmaz (yalnızca animasyon karelerinin
  geometrisi gibi çizime özgü küçük hesaplar).
* **`fourier_viz/app`** — ince bir arayüz katmanı: girdi toplar, önbellekli çekirdek çağrıları yapar,
  grafikleri gösterir.

## Modüller

| Modül | Sorumluluk |
|---|---|
| `core/types.py` | `FloatArray`, `ComplexArray`, `IntArray` tip takma adları |
| `core/integration.py` | Periyot üzerinde Gauss–Legendre / trapez / Simpson düğüm-ağırlıkları, SciPy `quad` (QAWO) |
| `core/series.py` | `RealCoefficients`, sayısal katsayılar, `partial_sum`, `harmonic_terms`, `partial_sums` (tüm N'ler tek matris çarpımı), σ-faktörleri |
| `core/complex_dft.py` | `ComplexCoefficients`, gerçek↔karmaşık dönüşüm, sayısal $c_n$, doğrudan DFT/IDFT, `Epicycle`/`EpicycleSet` |
| `core/expression.py` | AST beyaz listeli güvenli ifade derleyicisi |
| `core/signals.py` | `PeriodicSignal`, 8 hazır sinyal + analitik katsayılar, ifade/fonksiyon tabanlı sinyaller (otomatik sıçrama/tekillik tespiti), kayma/ölçek/toplam, kayıt defteri |
| `core/analysis.py` | L2/maks. hata, yakınsama çalışması ve eğim, Gibbs ölçümü, Parseval, spektrum, ortogonallik (Gram) matrisi |
| `core/paths.py` | Hazır 2D şekiller, eşit yay uzunluğuna göre yeniden örnekleme, normalizasyon, CSV ayrıştırma |
| `core/svg.py` | SVG `d` ayrıştırıcı (M L H V C S Q T A Z; XML ayrıştırıcı kullanmadan) |
| `viz/theme.py` | Açık/koyu, renk körlüğü açısından doğrulanmış palet |
| `viz/plots.py` | Plotly figürleri ve istemci tarafı animasyonlar (yakınsama, epicycle) |
| `viz/static.py` | Matplotlib statik figürler, galeri kartları, PNG baytları |
| `viz/animation.py` | GIF (Pillow) / MP4 (ffmpeg, `imageio-ffmpeg` yedeği) animasyonları |
| `export.py` | Katsayı ve epicycle CSV/JSON dışa aktarma (katsayılar için içe aktarma da) |
| `app/streamlit_app.py` | Giriş noktası: sayfa düzeni, sekmeler |
| `app/sidebar.py` | Kenar çubuğu ayarları → dondurulmuş `Settings` |
| `app/state.py` | Hashlenebilir ayar sınıfları ve `st.cache_data` / `st.cache_resource` sarmalayıcıları |
| `app/tabs.py` | Altı sekmenin çizim fonksiyonları |
| `app/learn.py` | "Öğren" sekmesinin Türkçe/LaTeX içeriği |

## Veri akışı

1. `render_sidebar()` widget değerlerinden dondurulmuş bir `Settings` (`SignalSpec`,
   `SeriesSettings`, `EpicycleSettings`) üretir.
2. Etkin sekmenin render fonksiyonu çağrılır; `st.tabs(..., on_change="rerun")` sayesinde gizli
   sekmeler için **hiç hesap yapılmaz**.
3. Hesaplar `app/state.py` içindeki önbellekli fonksiyonlardan geçer:
   * `load_signal` — `st.cache_resource`: sinyaller kapanış içerdiğinden pickle edilemez;
     değişmez oldukları için paylaşılmaları güvenlidir.
   * `compute_coefficients`, `compute_convergence`, `compute_gibbs`, `compute_epicycles`,
     `build_epicycle_figure`… — `st.cache_data`; argümanlar küçük, hashlenebilir değerlerdir.
   * Katsayılar bir kez $N = 500$'e kadar hesaplanır, kaydırıcı hareket ettikçe yalnızca kesilir
     (`truncate`); bu yüzden N kaydırıcısı anında tepki verir. (`quad` yöntemi yavaş olduğundan
     $N \le 100$ ile sınırlıdır ve yalnızca istenen N için hesaplanır.)
4. `viz` fonksiyonları hazır dizilerden figür üretir; Plotly figürleri `theme="streamlit"` ile
   gösterilir, böylece açık/koyu tema geçişine anında uyar.

## Tasarım kararları

* **Eval yok:** kullanıcı ifadeleri AST beyaz listesiyle derlenir (bkz. `docs/matematik.md` §9).
* **Serbest çizim:** Üçüncü parti tuval bileşeni (`streamlit-drawable-canvas`) güncel Streamlit ile
  yüklenemediği için Streamlit'in yerleşik Plotly **kement seçimi** kullanılır
  (`st.plotly_chart(on_select="rerun", selection_mode="lasso")`). Yakalanan poligon oturum
  durumuna yazılır ve tuval sıfırlanır.
* **Animasyonlar istemcide:** Plotly kareleri tarayıcıda oynatılır; sunucuya gidiş-dönüş yoktur.
  JSON boyutunu sınırlamak için en fazla 80 çember çizilir (tüm vektörler yine de hesaba katılır)
  ve diziler `float32` gönderilir.
* **Dışa aktarma:** PNG'ler Matplotlib ile sunucuda üretilir (Plotly'nin PNG'si için Chrome
  gerektiren kaleido'ya bağımlılık yok); Plotly araç çubuğundaki PNG düğmesi de açıktır.
  GIF/MP4 dosyaları indirme düğmesine **tıklanınca** üretilir (`st.download_button(data=callable)`).
* **Renkler:** kategorik yuvalar sabit sırayla kullanılır; özgün sinyal nötr referans rengiyle,
  yaklaşımlar kategorik renklerle, harmonik indisi gibi sıralı büyüklükler tek tonlu rampayla çizilir.
  İlk dört yuva her iki temada CVD (renk körlüğü) ayrışma eşiklerini geçer.

## Test stratejisi

| Katman | Dosyalar | Yaklaşım |
|---|---|---|
| Çekirdek | `tests/core/test_*.py` | Analitik değerler, simetri, ortogonallik, Parseval, uç durumlar, hata mesajları |
| Özellik tabanlı | `tests/core/test_properties.py`, ifade/SVG/yol testlerindeki `@given` | Doğrusallık, kayma, tam yeniden oluşturma, DFT gidiş-dönüşü, epicycle en iyiliği, bulanık (fuzz) girdiler |
| Performans | `tests/test_performance.py` (`perf` işareti) | N=500 kısmi toplam, 2000 noktalı DFT vb. < 1 sn |
| Görselleştirme | `tests/viz/` | Figür yapısı, renk rolleri, PNG/GIF/MP4 sihirli baytları |
| Arayüz | `tests/app/test_streamlit_app.py` | `streamlit.testing.v1.AppTest` ile her sekme, sinyal, şekil ve hata yolu |
| Galeri | `tests/test_gallery.py` | Betik tüm görselleri üretir ve matematik sağlamaları geçer |

`HYPOTHESIS_PROFILE=ci` daha fazla ve deterministik örnekle çalışır. Çekirdek kapsama eşiği CI'da
`coverage report --include="fourier_viz/core/*" --fail-under=90` ile zorunludur.
