"""Sekmelerin çizim fonksiyonları.

Her fonksiyon yalnızca kendi sekmesi açıkken çağrılır (``st.tabs(..., on_change="rerun")``),
bu yüzden gizli sekmeler için hiçbir hesap yapılmaz.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np
import pandas as pd
import streamlit as st

from fourier_viz.app.common import TAB_CONVERGENCE, TAB_SERIES, go_to_tab, show_error
from fourier_viz.app.learn import SECTIONS
from fourier_viz.app.sidebar import Settings
from fourier_viz.app.state import (
    MAX_N,
    QUAD_MAX_N,
    SMOOTHING_LABELS,
    EpicycleSettings,
    SignalSpec,
    build_epicycle_figure,
    coefficients_for,
    compute_animation_sums,
    compute_coefficients,
    compute_convergence,
    compute_curves,
    compute_epicycles,
    compute_errors,
    compute_gibbs,
    compute_gibbs_vs_n,
    compute_harmonics,
    compute_numeric_and_analytic,
    compute_signal_energy,
    library_shape,
    load_signal,
    parse_uploaded_shape,
    prepare_path,
)
from fourier_viz.core.analysis import (
    GIBBS_CONSTANT,
    orthogonality_matrix,
    spectrum,
    terms_for_energy,
)
from fourier_viz.core.complex_dft import EpicycleSet, points_to_complex, real_to_complex
from fourier_viz.core.expression import ExpressionError
from fourier_viz.core.paths import PathError
from fourier_viz.core.series import SeriesError
from fourier_viz.core.signals import PeriodicSignal, SignalError
from fourier_viz.core.types import FloatArray
from fourier_viz.export import (
    coefficients_to_csv,
    coefficients_to_json,
    epicycles_to_csv,
    epicycles_to_json,
)
from fourier_viz.viz import animation, plots, static
from fourier_viz.viz.theme import DARK, LIGHT, Palette

USER_ERRORS = (SignalError, SeriesError, ExpressionError, PathError)
"""Kullanıcı girdisinden kaynaklanan ve Türkçe mesajla gösterilen hatalar."""

PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}


def _chart(fig: object, key: str) -> None:
    st.plotly_chart(fig, theme="streamlit", key=key, config=PLOT_CONFIG)


def _load(settings: Settings) -> tuple[PeriodicSignal, SignalSpec] | None:
    """Sinyali yükler; hata varsa Türkçe mesaj gösterip ``None`` döndürür."""
    try:
        return load_signal(settings.signal), settings.signal
    except USER_ERRORS as exc:
        show_error(exc, "Sinyal oluşturulamadı")
        st.info("İpucu: Çarpma için `*`, üs için `**` veya `^` kullanın; örn. `t**2 * sin(3*t)`.")
        return None


def _fmt(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "—"
    if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.{digits - 1}e}"
    return f"{value:.{digits}g}"


def _slope(value: float) -> str:
    """Log-log eğimini okunaklı gösterir (≈ 0 için '≈ 0')."""
    if not np.isfinite(value):
        return "—"
    return "≈ 0" if abs(value) < 0.005 else f"{value:.2f}"


def _signal_formula(signal: PeriodicSignal) -> None:
    if signal.name == "expression":
        start = "0" if np.isclose(signal.start, 0.0) else "-T/2"
        st.markdown(f"**Özel ifade:** `f(t) = {signal.params.get('expression', '')}`")
        st.caption(
            f"Aralık: [{start}, {start}+T), T = {signal.period / np.pi:g}π; "
            "periyodik olarak genişletilir."
        )
    elif signal.latex:
        st.latex(signal.latex)


# ====================================================================== 1. Fourier serisi
def render_series_tab(settings: Settings, palette: Palette) -> None:
    """Sinyal, kısmi toplam, harmonik katmanı ve yakınsama animasyonu."""
    loaded = _load(settings)
    if loaded is None:
        return
    signal, spec = loaded
    series = settings.series
    try:
        curves = compute_curves(spec, series)
        errors = compute_errors(spec, series)
    except USER_ERRORS as exc:
        show_error(exc, "Hesaplama başarısız")
        return

    head, formula = st.columns([2, 3], vertical_alignment="center")
    with head:
        st.subheader(signal.label)
        st.caption(f"Periyot T = {signal.period / np.pi:g}π ≈ {signal.period:.4g}")
    with formula:
        _signal_formula(signal)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Terim sayısı N", series.n_terms, border=True)
    m2.metric("L2 hatası (RMS)", _fmt(errors.l2), border=True, help="√((1/T)∫(f − S_N)² dt)")
    m3.metric(
        "Maks. hata",
        _fmt(errors.max_abs),
        border=True,
        help="Süreksiz sinyallerde sıçramanın yarısına yakın kalır (düzgün olmayan yakınsama).",
    )
    m4.metric(
        "Yakalanan enerji",
        f"%{100 * errors.energy_fraction:.3f}",
        border=True,
        help="Parseval: ilk N harmoniğin enerjisi / sinyal enerjisi",
    )

    controls = st.columns([1, 2], vertical_alignment="center")
    show_layer = controls[0].toggle("Harmonikleri ayrı katmanda göster", key="show_harmonics")
    count = 6
    if show_layer:
        count = controls[1].slider(
            "Gösterilecek (sıfır olmayan) harmonik sayısı", 1, 12, 6, key="harmonic_count"
        )
    smoothing_note = (
        "" if series.smoothing == "none" else f" · {SMOOTHING_LABELS[series.smoothing]}"
    )
    label = f"S<sub>N</sub>(t), N = {series.n_terms}{smoothing_note}"
    harmonics = indices = None
    if show_layer:
        t_h, harmonics, indices = compute_harmonics(spec, series, count)
        del t_h
    fig = plots.signal_figure(
        curves["t"] if harmonics is None else np.linspace(curves["t"][0], curves["t"][-1], 1000),
        curves["f"]
        if harmonics is None
        else signal(np.linspace(curves["t"][0], curves["t"][-1], 1000)),
        curves["s"] if harmonics is None else None,
        series.n_terms,
        harmonics=harmonics,
        harmonic_indices=indices,
        approximation_label=label,
        palette=palette,
    )
    if harmonics is not None:
        fig.add_scatter(
            x=curves["t"],
            y=curves["s"],
            mode="lines",
            name=label,
            line={"color": palette.color(0), "width": 2},
            row=1,
            col=1,
        )
    _chart(fig, "series_chart")

    st.markdown("#### N arttıkça yakınsama")
    st.caption(
        "▶ ile oynatın veya kaydırıcıyla N'i seçin. Kısmi toplam, seçili N'e kadar "
        "adım adım sinyale yaklaşır."
    )
    t_a, f_a, sums, ns = compute_animation_sums(spec, series)
    _chart(
        plots.convergence_animation_figure(
            t_a, f_a, sums, ns, palette=palette, title=signal.label, frame_duration=300
        ),
        "convergence_animation",
    )

    with st.expander("Sayısal ve analitik katsayıların karşılaştırması", icon="🔎"):
        _coefficient_comparison(spec, settings)


def _coefficient_comparison(spec: SignalSpec, settings: Settings) -> None:
    n = min(settings.series.n_terms, 20)
    method = settings.series.method
    try:
        numeric, analytic = compute_numeric_and_analytic(spec, n, method, settings.series.samples)
    except USER_ERRORS as exc:
        show_error(exc)
        return
    data: dict[str, object] = {
        "n": np.arange(n + 1),
        "aₙ (sayısal)": numeric.a,
        "bₙ (sayısal)": numeric.b,
    }
    if analytic is None:
        st.info(
            "Bu sinyal için kapalı form (analitik) formül tanımlı değil; yalnızca sayısal "
            "katsayılar gösteriliyor."
        )
    else:
        diff = np.maximum(np.abs(numeric.a - analytic.a), np.abs(numeric.b - analytic.b))
        data.update(
            {"aₙ (analitik)": analytic.a, "bₙ (analitik)": analytic.b, "en büyük |fark|": diff}
        )
        worst = float(diff.max())
        st.success(
            f"İlk {n} harmonikte sayısal ve analitik katsayılar arasındaki en büyük fark: "
            f"{worst:.2e} (yöntem: {method}).",
            icon="✅",
        )
    st.dataframe(pd.DataFrame(data), hide_index=True, width="stretch")


# ====================================================================== 2. Epicycles
def _drawing_input(palette: Palette) -> FloatArray | None:
    st.markdown(
        "Grafiğin sağ üstündeki araç çubuğundan **kement (lasso)** aracı seçiliyken tuvale "
        "tek hamlede kapalı bir şekil çizin. Fareyi bıraktığınızda şekliniz epicycle'lara "
        "dönüştürülür."
    )
    version = st.session_state.setdefault("canvas_version", 0)
    drawn = st.session_state.get("drawn_points")
    event = st.plotly_chart(
        plots.drawing_canvas_figure(drawn, palette=palette),
        key=f"draw_canvas_{version}",
        on_select="rerun",
        selection_mode="lasso",
        theme="streamlit",
        config={"displaylogo": False, "scrollZoom": False},
    )
    lasso: list[dict[str, list[float]]] = []
    if event is not None and hasattr(event, "selection"):
        lasso = event.selection.get("lasso", []) or []
    if lasso:
        last = lasso[-1]
        pts = np.column_stack(
            [np.asarray(last.get("x", []), float), np.asarray(last.get("y", []), float)]
        )
        if len(pts) >= 3:
            st.session_state["drawn_points"] = pts
            # Tuvali sıfırla: Plotly'nin kesik çizgili seçim çerçevesi yerine yakalanan yol
            # temiz bir çizgi olarak gösterilsin (yeni anahtarlı grafikte seçim boştur).
            st.session_state["canvas_version"] = version + 1
            st.rerun()

    def clear() -> None:
        st.session_state.pop("drawn_points", None)
        st.session_state["canvas_version"] = version + 1

    st.button("🧹 Çizimi temizle", on_click=clear, key="clear_drawing")
    result = st.session_state.get("drawn_points")
    if result is None:
        st.info("Henüz bir çizim yok. Kement aracıyla bir şekil çizin.", icon="✍️")
    return result


def _upload_input() -> FloatArray | None:
    uploaded = st.file_uploader(
        "CSV (her satırda x, y) veya SVG dosyası yükleyin",
        type=["csv", "txt", "svg"],
        key="shape_file",
    )
    pasted = st.text_area(
        "…ya da SVG yol verisi (`d`) / CSV metni yapıştırın",
        key="shape_text",
        placeholder="M 0 0 C 40 -40, 80 40, 120 0 S 80 -80, 0 0 Z",
        height=90,
    )
    try:
        if uploaded is not None:
            return parse_uploaded_shape(uploaded.name, uploaded.getvalue())
        text = pasted.strip()
        if text:
            looks_svg = text[0] in "Mm" or "<svg" in text.lower() or "<path" in text.lower()
            return parse_uploaded_shape("pasted.svg" if looks_svg else "pasted.csv", text.encode())
    except PathError as exc:
        show_error(exc, "Şekil okunamadı")
        return None
    st.info("Bir dosya yükleyin veya yol verisi yapıştırın.", icon="📁")
    return None


def _shape_points(eps: EpicycleSettings, palette: Palette) -> FloatArray | None:
    if eps.source == "drawing":
        return _drawing_input(palette)
    if eps.source == "upload":
        return _upload_input()
    return library_shape(eps.shape_key, eps.shape_params)


def render_epicycles_tab(settings: Settings, palette: Palette) -> None:
    """Dönen çemberlerle şekil çizimi."""
    eps = settings.epicycles
    raw = _shape_points(eps, palette)
    if raw is None:
        return
    try:
        points = prepare_path(raw, eps.n_points)
        epi = compute_epicycles(points, eps.n_circles)
    except USER_ERRORS as exc:
        show_error(exc, "Şekil hazırlanamadı")
        return
    s = np.arange(eps.n_points) / eps.n_points
    rms = float(np.sqrt(np.mean(np.abs(epi.evaluate(s) - points_to_complex(points)) ** 2)))
    m1, m2, m3 = st.columns(3)
    m1.metric("Çember sayısı", f"{epi.n_circles} / {eps.n_points - 1}", border=True)
    m2.metric(
        "Yakalanan enerji",
        f"%{100 * epi.energy_fraction():.2f}",
        border=True,
        help="Tutulan çemberlerin |c_k|² toplamı / tüm çemberlerin toplamı (DC hariç)",
    )
    m3.metric(
        "RMS sapma",
        _fmt(rms),
        border=True,
        help="Örnek noktalarında çizim ile hedef şekil arasındaki kök ortalama kare uzaklık "
        "(şekil [-1, 1] kutusuna ölçeklidir).",
    )
    fig = build_epicycle_figure(points, eps, palette.mode)
    _chart(fig, "epicycle_chart")
    if epi.n_circles > plots.MAX_DRAWN_CIRCLES and eps.show_circles:
        st.caption(
            f"Performans için yalnızca en büyük {plots.MAX_DRAWN_CIRCLES} çember çiziliyor; "
            "tüm vektörler (kollar) hesaba katılıyor."
        )
    with st.expander("Çember listesi (genliğe göre)", icon="📋"):
        _epicycle_table(epi)


def _epicycle_table(epi: EpicycleSet, limit: int = 30) -> None:
    k = min(limit, epi.n_circles)
    table = pd.DataFrame(
        {
            "sıra": np.arange(1, k + 1),
            "frekans k (tur/periyot)": epi.frequencies[:k],
            "yarıçap |c_k|": epi.radii[:k],
            "başlangıç açısı (°)": np.degrees(epi.phases[:k]),
        }
    )
    st.caption(f"Merkez (DC terimi) c₀ = {epi.center.real:.4f} + {epi.center.imag:.4f}i")
    st.dataframe(table, hide_index=True, width="stretch")


# ====================================================================== 3. Spektrum
def render_spectrum_tab(settings: Settings, palette: Palette) -> None:
    """Genlik/faz spektrumu, Parseval kontrolü ve katsayı tablosu."""
    loaded = _load(settings)
    if loaded is None:
        return
    signal, spec = loaded
    try:
        coeffs = coefficients_for(spec, settings.series)
        energy = compute_signal_energy(spec)
    except USER_ERRORS as exc:
        show_error(exc, "Hesaplama başarısız")
        return

    c1, c2, c3 = st.columns([3, 2, 2], vertical_alignment="bottom")
    view = c1.segmented_control(
        "Gösterim",
        ["Tek taraflı (Aₙ, φₙ)", "Çift taraflı (|cₙ|, arg cₙ)"],
        default="Tek taraflı (Aₙ, φₙ)",
        key="spectrum_view",
        required=True,
    )
    style = c2.segmented_control(
        "Stil", ["Gövde", "Çubuk"], default="Gövde", key="spectrum_style", required=True
    )
    log_y = c3.toggle("Logaritmik ölçek", key="spectrum_log")
    style_key: plots.SpectrumStyle = "stem" if style == "Gövde" else "bar"

    if view and view.startswith("Tek"):
        spec_data = spectrum(coeffs)
        x = spec_data.harmonics
        amp, phase = spec_data.amplitude, spec_data.phase
        amp_label, phase_label, x_label = "Genlik Aₙ", "Faz φₙ (rad)", "Harmonik n"
    else:
        cplx = real_to_complex(coeffs)
        x = cplx.frequencies
        amp = cplx.magnitude()
        phase = np.where(amp > 1e-12 * max(1.0, float(amp.max())), cplx.phase(), 0.0)
        amp_label, phase_label, x_label = "|cₙ|", "arg cₙ (rad)", "n"
    left, right = st.columns(2)
    with left:
        _chart(
            plots.spectrum_figure(
                x,
                amp,
                style=style_key,
                log_y=log_y,
                x_label=x_label,
                y_label=amp_label,
                palette=palette,
                title="Genlik spektrumu",
            ),
            "amplitude_spectrum",
        )
    with right:
        _chart(
            plots.spectrum_figure(
                x,
                phase,
                style=style_key,
                x_label=x_label,
                y_label=phase_label,
                palette=palette,
                title="Faz spektrumu",
                phase=True,
            ),
            "phase_spectrum",
        )

    st.markdown("#### Parseval özdeşliği")
    st.latex(
        r"\frac{1}{T}\int_T |f|^2\,dt = \Big(\frac{a_0}{2}\Big)^2 + "
        r"\frac12\sum_{n\ge1}(a_n^2+b_n^2) = \sum_n |c_n|^2"
    )
    full = compute_coefficients(
        spec,
        settings.series.n_compute,
        settings.series.method,
        settings.series.samples,
        settings.series.use_analytic,
    )
    n99 = terms_for_energy(full, energy, 0.99)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Sinyal enerjisi ‖f‖²", _fmt(energy, 6), border=True)
    p2.metric(f"Katsayı enerjisi (N = {coeffs.n_terms})", _fmt(coeffs.energy(), 6), border=True)
    p3.metric(
        "Yakalanan oran",
        f"%{100 * coeffs.energy() / energy:.4f}" if energy > 0 else "—",
        border=True,
    )
    p4.metric(
        "%99 enerji için gereken N", str(n99) if n99 >= 0 else f"> {full.n_terms}", border=True
    )

    with st.expander("Katsayı tablosu", icon="🧾"):
        spec_data = spectrum(coeffs)
        table = pd.DataFrame(
            {
                "n": spec_data.harmonics,
                "aₙ": coeffs.a,
                "bₙ": coeffs.b,
                "Aₙ": spec_data.amplitude,
                "φₙ (rad)": spec_data.phase,
                "|cₙ|": np.abs(real_to_complex(coeffs).c[coeffs.n_terms :]),
            }
        )
        st.dataframe(table, hide_index=True, width="stretch", height=320)
    del signal


# ====================================================================== 4. Yakınsama ve hata
def render_convergence_tab(settings: Settings, palette: Palette) -> None:
    """Log-log hata eğrileri, Gibbs ölçümü ve yumuşatma karşılaştırması."""
    loaded = _load(settings)
    if loaded is None:
        return
    signal, spec = loaded
    series = settings.series
    n_max = QUAD_MAX_N if series.method == "quad" else MAX_N
    exclude = st.toggle(
        "Maksimum hatada süreksizlik çevresini hariç tut (±T/40)",
        key="exclude_jumps",
        disabled=signal.is_continuous,
        help="Süreksizliğin hemen yanında hata her zaman sıçramanın yarısı kadardır.",
    )
    try:
        study = compute_convergence(
            spec,
            series.method,
            series.samples,
            series.use_analytic,
            series.smoothing,
            signal.period / 40 if exclude else 0.0,
            n_max,
        )
    except USER_ERRORS as exc:
        show_error(exc, "Hata çalışması yapılamadı")
        return
    fig = plots.error_figure(
        study.n_values,
        {"L2 (RMS) hatası": study.l2, "Maksimum hata": study.max_abs},
        palette=palette,
        title=f"{signal.label}: hata – N (log-log)",
        highlight_n=series.n_terms,
        highlight_label=f"seçili N = {series.n_terms}",
    )
    _chart(fig, "error_chart")
    r1, r2 = st.columns(2)
    r1.metric(
        "L2 hata eğimi",
        _slope(study.l2_rate()),
        border=True,
        help="Log-log doğru eğimi p: hata ≈ C·N^p. Süreksiz ≈ −0.5, sürekli (kırılmalı) ≈ −1.5",
    )
    r2.metric(
        "Maks. hata eğimi",
        _slope(study.max_rate()),
        border=True,
        help="Süreksiz sinyallerde (hariç tutma kapalıyken) ≈ 0: maksimum hata azalmaz.",
    )
    st.divider()
    _gibbs_section(signal, spec, settings, palette)


def _gibbs_section(
    signal: PeriodicSignal, spec: SignalSpec, settings: Settings, palette: Palette
) -> None:
    st.markdown("#### Gibbs olayı")
    results = compute_gibbs(spec, settings.series)
    if results is None:
        st.info(
            "Bu sinyal sürekli olduğundan Gibbs aşımı oluşmaz. Gibbs olayını görmek için "
            "kare dalga, testere dişi veya darbe dizisi seçin.",
            icon="ℹ️",
        )
        return
    theory = 100 * GIBBS_CONSTANT
    plain = results["none"]
    g1, g2, g3, g4 = st.columns(4)
    g1.metric(
        "Ölçülen aşım (yumuşatmasız)",
        f"%{plain.percent:.3f}",
        delta=f"{plain.percent - theory:+.3f} puan (teoriye göre)",
        delta_color="off",
        border=True,
    )
    g2.metric("Teorik sınır", f"%{theory:.3f}", border=True, help="Si(π)/π − 1/2 ≈ 0.08949 (N → ∞)")
    g3.metric("Fejér ile aşım", f"%{results['fejer'].percent:.3f}", border=True)
    g4.metric("Lanczos ile aşım", f"%{results['lanczos'].percent:.3f}", border=True)

    jump = plain.jump
    width = min(
        signal.period / 4,
        max(6 * signal.period / (settings.series.n_terms + 1), 0.02 * signal.period),
    )
    t = np.linspace(jump.location - width, jump.location + width, 1200)
    coeffs = coefficients_for(spec, settings.series)
    curves = {
        SMOOTHING_LABELS[m]: coeffs.evaluate(t, smoothing=m)  # type: ignore[arg-type]
        for m in SMOOTHING_LABELS
    }
    high = max(jump.left, jump.right)
    fig = plots.comparison_figure(
        t,
        signal(t),
        curves,
        palette=palette,
        title=f"Süreksizlik çevresi (t = {jump.location:.3g}), N = {settings.series.n_terms}",
        reference_levels={f"teorik tepe (%{theory:.2f})": high + GIBBS_CONSTANT * abs(jump.size)},
        peaks={
            SMOOTHING_LABELS[m]: (r.peak_time, r.peak_value)
            for m, r in results.items()
            if r.overshoot > 0
        },
    )
    _chart(fig, "gibbs_chart")
    if settings.series.method == "quad":
        # N = 500'e kadar adaptif quad ile katsayı hesabı çok yavaş olurdu.
        st.caption(
            "SciPy quad yöntemi yavaş olduğundan aşımın N'e bağlı değişimi bu yöntemde "
            "gösterilmez; başka bir integral yöntemi seçin."
        )
        trend = None
    else:
        trend = compute_gibbs_vs_n(
            spec, settings.series.method, settings.series.samples, settings.series.use_analytic
        )
    if trend is not None:
        ns, ratios = trend
        _chart(
            plots.gibbs_trend_figure(
                ns,
                100 * np.asarray(ratios),
                theory,
                palette=palette,
                title="Aşım sıfıra gitmez: N → ∞ iken ≈ %8.95'e yaklaşır",
            ),
            "gibbs_trend",
        )
    table = pd.DataFrame(
        {
            "yöntem": [SMOOTHING_LABELS[m] for m in results],
            "tepe değeri": [r.peak_value for r in results.values()],
            "aşım (sıçramanın %'si)": [r.percent for r in results.values()],
            "tepe konumu t": [r.peak_time for r in results.values()],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")


# ====================================================================== 5. Öğren
def render_learn_tab(settings: Settings, palette: Palette) -> None:
    """Kavramların Türkçe açıklamaları ve ilgili sekmelere bağlantılar."""
    del settings
    st.markdown(
        "Aşağıdaki kısa notlar uygulamadaki her grafiğin arkasındaki matematiği özetler. "
        "Her bölümün sonundaki düğme sizi ilgili sekmeye götürür."
    )
    for section in SECTIONS:
        with st.container(border=True):
            st.subheader(section.title)
            st.markdown(section.body)
            if section.key == "orthogonality":
                n_max = st.slider("Harmonik sayısı", 1, 8, 4, key="gram_n")
                labels = [f"cos {k}ω₀t" for k in range(1, n_max + 1)] + [
                    f"sin {k}ω₀t" for k in range(1, n_max + 1)
                ]
                _chart(
                    plots.heatmap_figure(
                        orthogonality_matrix(n_max),
                        labels,
                        palette=palette,
                        title="Gram matrisi ⟨φᵢ, φⱼ⟩ (≈ birim matris)",
                        height=420,
                    ),
                    "gram_heatmap",
                )
            cols = st.columns([1, 3], vertical_alignment="center")
            cols[0].button(
                f"→ {section.target_tab}",
                key=f"learn_go_{section.key}",
                on_click=go_to_tab,
                args=(section.target_tab,),
            )
            cols[1].caption(section.target_hint)
    st.caption(
        f"Hızlı başlangıç: {TAB_SERIES} sekmesinde N'i artırın, sonra {TAB_CONVERGENCE} "
        "sekmesinde hatanın nasıl azaldığını inceleyin."
    )


# ====================================================================== 6. Dışa aktar
def _deferred(
    func: Callable[..., bytes | str], *args: object, **kwargs: object
) -> Callable[[], bytes | str]:
    """İndirme anında (tıklanınca) çalışacak argümansız fonksiyon."""
    return functools.partial(func, *args, **kwargs)


def _png(fig_factory: Callable[[], object]) -> bytes:
    return static.figure_to_png_bytes(fig_factory())  # type: ignore[arg-type]


def render_export_tab(settings: Settings, palette: Palette) -> None:
    """Grafik, animasyon ve katsayıların indirilmesi."""
    del palette
    mode = st.segmented_control(
        "Dışa aktarma teması", ["Açık", "Koyu"], default="Açık", key="export_theme", required=True
    )
    export_palette = DARK if mode == "Koyu" else LIGHT
    loaded = _load(settings)

    st.markdown("#### 🖼️ Grafikler (PNG)")
    if loaded is not None:
        signal, spec = loaded
        try:
            coeffs = coefficients_for(spec, settings.series)
            curves = compute_curves(spec, settings.series)
        except USER_ERRORS as exc:
            show_error(exc)
            return
        n = settings.series.n_terms
        name = signal.name
        spec_data = spectrum(coeffs)
        cols = st.columns(3)
        cols[0].download_button(
            "Sinyal + kısmi toplam",
            key="dl_signal_png",
            icon="📈",
            mime="image/png",
            file_name=f"{name}_N{n}.png",
            data=_deferred(
                _png,
                functools.partial(
                    static.signal_plot,
                    curves["t"],
                    curves["f"],
                    curves["s"],
                    n,
                    title=f"{signal.label} — N = {n}",
                    palette=export_palette,
                ),
            ),
        )
        cols[1].download_button(
            "Genlik spektrumu",
            key="dl_spectrum_png",
            icon="📊",
            mime="image/png",
            file_name=f"{name}_spektrum_N{n}.png",
            data=_deferred(
                _png,
                functools.partial(
                    static.spectrum_plot,
                    spec_data.harmonics,
                    spec_data.amplitude,
                    title=f"{signal.label} — genlik spektrumu",
                    palette=export_palette,
                ),
            ),
        )
        n_max = QUAD_MAX_N if settings.series.method == "quad" else MAX_N
        try:
            study = compute_convergence(
                spec,
                settings.series.method,
                settings.series.samples,
                settings.series.use_analytic,
                settings.series.smoothing,
                0.0,
                n_max,
            )
            cols[2].download_button(
                "Hata – N (log-log)",
                key="dl_error_png",
                icon="📉",
                mime="image/png",
                file_name=f"{name}_hata.png",
                data=_deferred(
                    _png,
                    functools.partial(
                        static.error_plot,
                        study.n_values,
                        {"L2": study.l2, "Maks.": study.max_abs},
                        title=f"{signal.label} — hata",
                        palette=export_palette,
                    ),
                ),
            )
        except USER_ERRORS as exc:
            show_error(exc)

        st.markdown("#### 🎞️ Yakınsama animasyonu")
        t_a, f_a, sums, ns = compute_animation_sums(spec, settings.series)
        a1, a2 = st.columns(2)
        a1.download_button(
            "GIF indir",
            key="dl_conv_gif",
            icon="🎞️",
            mime="image/gif",
            file_name=f"{name}_yakinsama.gif",
            data=_deferred(
                animation.convergence_animation,
                t_a,
                f_a,
                sums,
                ns,
                "gif",
                title=signal.label,
                palette=export_palette,
            ),
        )
        a2.download_button(
            "MP4 indir",
            key="dl_conv_mp4",
            icon="🎬",
            mime="video/mp4",
            file_name=f"{name}_yakinsama.mp4",
            disabled=not animation.ffmpeg_available(),
            data=_deferred(
                animation.convergence_animation,
                t_a,
                f_a,
                sums,
                ns,
                "mp4",
                title=signal.label,
                palette=export_palette,
            ),
        )

        st.markdown("#### 🔢 Katsayılar")
        meta = {
            "signal": signal.label,
            "params": dict(signal.params),
            "n_terms": n,
            "method": settings.series.method,
            "analytic": settings.series.use_analytic and signal.has_analytic,
        }
        k1, k2 = st.columns(2)
        k1.download_button(
            "CSV",
            data=coefficients_to_csv(coeffs),
            file_name=f"{name}_katsayilar.csv",
            mime="text/csv",
            key="dl_coeff_csv",
            icon="📄",
        )
        k2.download_button(
            "JSON",
            data=coefficients_to_json(coeffs, meta),
            file_name=f"{name}_katsayilar.json",
            mime="application/json",
            key="dl_coeff_json",
            icon="🧾",
        )

    st.markdown("#### 🌀 Epicycles")
    eps = settings.epicycles
    raw = (
        library_shape(eps.shape_key, eps.shape_params)
        if eps.source == "library"
        else (st.session_state.get("drawn_points") if eps.source == "drawing" else None)
    )
    if raw is None:
        st.info(
            "Epicycle dışa aktarımı için Epicycles sekmesinde bir şekil hazırlayın "
            "(yüklenen dosyalar orada işlenir).",
            icon="ℹ️",
        )
        return
    try:
        points = prepare_path(raw, eps.n_points)
        epi = compute_epicycles(points, eps.n_circles)
    except USER_ERRORS as exc:
        show_error(exc)
        return
    e = st.columns(5)
    e[0].download_button(
        "Anlık görüntü (PNG)",
        key="dl_epi_png",
        icon="🖼️",
        mime="image/png",
        file_name="epicycle.png",
        data=_deferred(
            _png,
            functools.partial(
                static.epicycle_snapshot,
                epi,
                target=points,
                show_circles=eps.show_circles,
                palette=export_palette,
            ),
        ),
    )
    anim_kwargs = {
        "n_frames": min(eps.n_frames, 120),
        "trail_fraction": eps.trail,
        "show_circles": eps.show_circles,
        "target": points,
        "palette": export_palette,
    }
    e[1].download_button(
        "GIF",
        key="dl_epi_gif",
        icon="🎞️",
        mime="image/gif",
        file_name="epicycle.gif",
        data=_deferred(animation.epicycle_animation, epi, "gif", **anim_kwargs),
    )
    e[2].download_button(
        "MP4",
        key="dl_epi_mp4",
        icon="🎬",
        mime="video/mp4",
        file_name="epicycle.mp4",
        disabled=not animation.ffmpeg_available(),
        data=_deferred(animation.epicycle_animation, epi, "mp4", **anim_kwargs),
    )
    e[3].download_button(
        "CSV",
        data=epicycles_to_csv(epi),
        file_name="epicycle.csv",
        mime="text/csv",
        key="dl_epi_csv",
        icon="📄",
    )
    e[4].download_button(
        "JSON",
        data=epicycles_to_json(epi),
        file_name="epicycle.json",
        mime="application/json",
        key="dl_epi_json",
        icon="🧾",
    )
    st.caption("GIF/MP4 dosyaları düğmeye tıklandığında üretilir; birkaç saniye sürebilir.")
    if not animation.ffmpeg_available():
        st.warning("MP4 için ffmpeg bulunamadı; `pip install imageio-ffmpeg` ile kurabilirsiniz.")


TAB_RENDERERS: tuple[Callable[[Settings, Palette], None], ...] = (
    render_series_tab,
    render_epicycles_tab,
    render_spectrum_tab,
    render_convergence_tab,
    render_learn_tab,
    render_export_tab,
)
"""Sekme sırasına göre çizim fonksiyonları (:data:`~fourier_viz.app.common.TABS` ile aynı sıra)."""
