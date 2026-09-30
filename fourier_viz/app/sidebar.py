"""Kenar çubuğu: tüm hesaplama ayarları."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import streamlit as st

from fourier_viz.app.state import (
    EXPRESSION_KEY,
    MAX_N,
    METHOD_LABELS,
    QUAD_MAX_N,
    SMOOTHING_LABELS,
    EpicycleSettings,
    SeriesSettings,
    ShapeSource,
    SignalSpec,
)
from fourier_viz.core.expression import ALLOWED_FUNCTIONS
from fourier_viz.core.integration import QuadratureMethod
from fourier_viz.core.paths import SHAPE_LIBRARY
from fourier_viz.core.series import Smoothing
from fourier_viz.core.signals import SIGNAL_LIBRARY

DEFAULT_EXPRESSION = "t**2 * sin(3*t)"

EXPRESSION_EXAMPLES = (
    "t**2 * sin(3*t)",
    "exp(-t**2) * cos(5*t)",
    "abs(t)",
    "where(t > 0, 1, 0)",
    "sign(sin(t)) + 0.5*sin(3*t)",
    "t",
)

SHAPE_SOURCES: dict[str, str] = {
    "library": "Hazır şekil",
    "drawing": "Kendin çiz (kement)",
    "upload": "Dosya yükle (CSV/SVG)",
}


@dataclass(frozen=True)
class Settings:
    """Kenar çubuğundan toplanan tüm ayarlar."""

    signal: SignalSpec
    series: SeriesSettings
    epicycles: EpicycleSettings


def _signal_label(key: str) -> str:
    if key == EXPRESSION_KEY:
        return "✏️ Kendi ifadem"
    return SIGNAL_LIBRARY[key].label


def _signal_section() -> SignalSpec:
    keys = [*SIGNAL_LIBRARY, EXPRESSION_KEY]
    key = st.selectbox(
        "Sinyal",
        keys,
        format_func=_signal_label,
        key="signal_key",
        help="Hazır sinyallerden birini seçin veya kendi matematiksel ifadenizi yazın.",
    )
    if key == EXPRESSION_KEY:
        example = st.selectbox(
            "Örnek ifadeler",
            EXPRESSION_EXAMPLES,
            key="expression_example",
            help="Bir örnek seçmek ifade kutusunu doldurur.",
            on_change=lambda: st.session_state.update(
                expression=st.session_state["expression_example"]
            ),
        )
        del example
        if "expression" not in st.session_state:
            st.session_state["expression"] = DEFAULT_EXPRESSION
        expression = st.text_input(
            "f(t) =",
            key="expression",
            help="Değişken: t · Sabitler: pi, e, tau · İşlemler: + − * / ** (^) % · "
            "Karşılaştırma: < > <= >= · Fonksiyonlar: " + ", ".join(sorted(ALLOWED_FUNCTIONS)),
        )
        period_pi = st.slider(
            "Periyot T (π'nin katı)",
            0.25,
            6.0,
            2.0,
            0.25,
            key="expr_period",
            help="İfade bir periyotta tanımlanır ve periyodik olarak genişletilir.",
        )
        interval = st.radio(
            "Tanım aralığı", ["[−T/2, T/2)", "[0, T)"], key="expr_interval", horizontal=True
        )
        return SignalSpec(
            key=EXPRESSION_KEY,
            expression=expression,
            period=float(period_pi * np.pi),
            start_at_zero=interval == "[0, T)",
        )
    info = SIGNAL_LIBRARY[key]
    params: list[tuple[str, float]] = []
    for spec in info.params:
        if spec.name == "period":
            period_pi = st.slider(
                "Periyot T (π'nin katı)", 0.25, 6.0, 2.0, 0.25, key=f"param_{key}_period"
            )
            params.append(("period", float(period_pi * np.pi)))
            continue
        value = st.slider(
            spec.label,
            float(spec.minimum),
            float(spec.maximum),
            float(spec.default),
            float(spec.step),
            key=f"param_{key}_{spec.name}",
        )
        params.append((spec.name, float(value)))
    st.caption(info.description)
    return SignalSpec(key=key, params=tuple(params))


def _series_section(has_analytic: bool) -> SeriesSettings:
    method = st.selectbox(
        "Sayısal integral yöntemi",
        list(METHOD_LABELS),
        format_func=METHOD_LABELS.__getitem__,
        key="method",
        help="Gauss–Legendre, bilinen süreksizlik noktalarında bölünerek uygulanır ve "
        "parçalı düzgün sinyallerde çok hassastır. 'quad' adaptiftir ama yavaştır.",
    )
    max_n = QUAD_MAX_N if method == "quad" else MAX_N
    if st.session_state.get("n_terms", 10) > max_n:
        st.session_state["n_terms"] = max_n
    n_terms = st.slider(
        "Terim sayısı N",
        1,
        max_n,
        10,
        key="n_terms",
        help="Kısmi toplamda kullanılan harmonik sayısı.",
    )
    if method == "quad":
        st.caption(f"SciPy quad yöntemi yavaş olduğundan N en fazla {QUAD_MAX_N} olabilir.")
    smoothing = st.radio(
        "Yumuşatma (Gibbs azaltma)",
        list(SMOOTHING_LABELS),
        format_func=SMOOTHING_LABELS.__getitem__,
        key="smoothing",
    )
    samples: int | None = None
    if method != "quad":
        auto = st.toggle("Otomatik hassasiyet", value=True, key="samples_auto")
        if not auto:
            samples = int(
                st.number_input(
                    "Düğüm sayısı (hassasiyet)",
                    min_value=64,
                    max_value=200_000,
                    value=8192,
                    step=1024,
                    key="samples",
                    help="Daha fazla düğüm daha hassas ama daha yavaş integral demektir. "
                    "Yüksek harmonikler için en az 2N düğüm gerekir.",
                )
            )
    use_analytic = False
    if has_analytic:
        use_analytic = st.toggle(
            "Analitik katsayıları kullan",
            value=False,
            key="use_analytic",
            help="Kapalı form formüller biliniyorsa sayısal integral yerine onları kullanır.",
        )
    return SeriesSettings(
        n_terms=int(n_terms),
        smoothing=cast(Smoothing, smoothing),
        method=cast(QuadratureMethod, method),
        samples=samples,
        use_analytic=use_analytic,
    )


def _epicycle_section() -> EpicycleSettings:
    source = cast(
        ShapeSource,
        st.radio(
            "Şekil kaynağı",
            list(SHAPE_SOURCES),
            format_func=SHAPE_SOURCES.__getitem__,
            key="shape_source",
        ),
    )
    shape_key = "heart"
    shape_params: list[tuple[str, float]] = []
    if source == "library":
        shape_key = st.selectbox(
            "Şekil",
            list(SHAPE_LIBRARY),
            format_func=lambda k: SHAPE_LIBRARY[k].label,
            key="shape_key",
        )
        for spec in SHAPE_LIBRARY[shape_key].params:
            if spec.integer:
                value: float = st.slider(
                    spec.label,
                    int(spec.minimum),
                    int(spec.maximum),
                    int(spec.default),
                    int(spec.step),
                    key=f"shape_{shape_key}_{spec.name}",
                )
            else:
                value = st.slider(
                    spec.label,
                    float(spec.minimum),
                    float(spec.maximum),
                    float(spec.default),
                    float(spec.step),
                    key=f"shape_{shape_key}_{spec.name}",
                )
            shape_params.append((spec.name, float(value)))
    n_points = st.select_slider(
        "Örnek nokta sayısı M",
        [64, 128, 256, 512, 1024],
        value=256,
        key="n_points",
        help="Yol eşit yay uzunluğuna göre M noktaya yeniden örneklenir.",
    )
    n_circles = st.slider(
        "Çember sayısı",
        1,
        int(n_points) - 1,
        min(60, int(n_points) - 1),
        key="n_circles",
        help="En büyük genlikli çemberler tutulur (DC merkez hariç).",
    )
    speed = st.slider("Hız", 0.25, 4.0, 1.0, 0.25, key="speed", format="%.2f×")
    trail = st.slider(
        "İz uzunluğu",
        5,
        100,
        100,
        5,
        key="trail",
        format="%d%%",
        help="%100: şekil baştan sona çizilir; daha küçük değerler kuyruklu yıldız etkisi verir.",
    )
    n_frames = st.select_slider(
        "Tur başına kare", [60, 90, 120, 180, 240], value=120, key="n_frames"
    )
    show_circles = st.toggle("Çemberleri göster", value=True, key="show_circles")
    return EpicycleSettings(
        source=source,
        shape_key=shape_key,
        shape_params=tuple(shape_params),
        n_points=int(n_points),
        n_circles=int(n_circles),
        speed=float(speed),
        trail=float(trail) / 100.0,
        show_circles=bool(show_circles),
        n_frames=int(n_frames),
    )


def render_sidebar() -> Settings:
    """Kenar çubuğunu çizer ve ayarları döndürür."""
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        with st.expander("Sinyal", expanded=True, icon="〰️"):
            signal = _signal_section()
        with st.expander("Seri ve hesaplama", expanded=True, icon="🧮"):
            has_analytic = (not signal.is_expression) and signal.key in SIGNAL_LIBRARY
            series = _series_section(has_analytic)
        with st.expander("Epicycle", expanded=False, icon="🌀"):
            epicycles = _epicycle_section()
        st.caption(
            "Açık/koyu tema: sağ üstteki ⋮ menüsü → *Settings* → *Theme*. "
            "Grafikler temaya otomatik uyum sağlar."
        )
    return Settings(signal=signal, series=series, epicycles=epicycles)
