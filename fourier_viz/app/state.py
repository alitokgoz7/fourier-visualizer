"""Arayüz durumu: hashlenebilir ayar sınıfları ve önbellekli hesaplamalar.

Ağır hesaplamalar ``st.cache_data`` ile önbelleğe alınır; bunların argümanları yalnızca
hashlenebilir, küçük değerlerdir (dondurulmuş dataclass'lar, sayılar, NumPy dizileri).
:class:`~fourier_viz.core.signals.PeriodicSignal` nesneleri kapanış (closure) içerdiğinden
pickle edilemez; bu yüzden onlar ``st.cache_resource`` ile (kopyalanmadan) tutulur — sinyaller
değişmez (immutable) olduğundan paylaşım güvenlidir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from fourier_viz.core.analysis import (
    ConvergenceResult,
    GibbsResult,
    convergence_study,
    gibbs_overshoot,
    l2_error,
    max_error,
    signal_energy,
)
from fourier_viz.core.complex_dft import EpicycleSet, epicycles_from_points
from fourier_viz.core.integration import QuadratureMethod
from fourier_viz.core.paths import (
    build_shape,
    normalize_path,
    parse_csv_points,
    resample_by_arclength,
)
from fourier_viz.core.series import (
    RealCoefficients,
    Smoothing,
    harmonic_terms,
    partial_sums,
)
from fourier_viz.core.signals import PeriodicSignal, build_signal, from_expression
from fourier_viz.core.svg import svg_to_points
from fourier_viz.core.types import FloatArray
from fourier_viz.viz.plots import epicycle_figure
from fourier_viz.viz.theme import get_palette

MAX_N: Final[int] = 500
"""Arayüzdeki en büyük terim sayısı."""

QUAD_MAX_N: Final[int] = 100
"""Yavaş ``quad`` yöntemi için izin verilen en büyük terim sayısı."""

EXPRESSION_KEY: Final[str] = "expression"

SMOOTHING_LABELS: Final[dict[str, str]] = {
    "none": "Yok",
    "fejer": "Fejér (Cesàro)",
    "lanczos": "Lanczos σ",
}

METHOD_LABELS: Final[dict[str, str]] = {
    "gauss": "Gauss–Legendre (önerilen)",
    "trapezoid": "Trapez (periyodik)",
    "simpson": "Simpson",
    "quad": "SciPy quad (adaptif, yavaş)",
}


# ====================================================================== ayar sınıfları
@dataclass(frozen=True)
class SignalSpec:
    """Bir sinyali tanımlayan hashlenebilir özellikler.

    Attributes:
        key: Kütüphane anahtarı veya ``"expression"``.
        params: Kütüphane sinyali parametreleri (ad, değer) çiftleri.
        expression: Kullanıcı ifadesi (yalnızca ``key == "expression"``).
        period: İfade sinyalinin periyodu.
        start_at_zero: İfade aralığı ``[0, T)`` mi (aksi hâlde ``[-T/2, T/2)``).
    """

    key: str
    params: tuple[tuple[str, float], ...] = ()
    expression: str = ""
    period: float = 2.0 * np.pi
    start_at_zero: bool = False

    @property
    def is_expression(self) -> bool:
        """Kullanıcı ifadesi mi."""
        return self.key == EXPRESSION_KEY

    def build(self) -> PeriodicSignal:
        """Sinyali oluşturur (hatalar Türkçe ``SignalError``)."""
        if self.is_expression:
            return from_expression(
                self.expression, self.period, 0.0 if self.start_at_zero else None
            )
        return build_signal(self.key, **dict(self.params))


@dataclass(frozen=True)
class SeriesSettings:
    """Seri hesaplama ayarları."""

    n_terms: int = 10
    smoothing: Smoothing = "none"
    method: QuadratureMethod = "gauss"
    samples: int | None = None
    use_analytic: bool = False

    @property
    def n_compute(self) -> int:
        """Hesaplanacak katsayı sayısı: ``quad`` için N, diğerleri için :data:`MAX_N`.

        Katsayılar bir kez :data:`MAX_N`'e kadar hesaplanıp kaydırıcı hareket ettikçe
        yalnızca kesilir (truncate); böylece kaydırıcı anında tepki verir.
        """
        return self.n_terms if self.method == "quad" else MAX_N


ShapeSource = Literal["library", "drawing", "upload"]


@dataclass(frozen=True)
class EpicycleSettings:
    """Epicycle sekmesi ayarları."""

    source: ShapeSource = "library"
    shape_key: str = "heart"
    shape_params: tuple[tuple[str, float], ...] = ()
    n_points: int = 256
    n_circles: int = 60
    speed: float = 1.0
    trail: float = 1.0
    show_circles: bool = True
    n_frames: int = 120

    @property
    def frame_duration(self) -> int:
        """Hıza göre kare süresi (ms)."""
        return max(10, round(45 / self.speed))


# ====================================================================== önbellekli hesaplar
@st.cache_resource(max_entries=64, show_spinner=False)
def load_signal(spec: SignalSpec) -> PeriodicSignal:
    """Sinyali oluşturur ve (kopyalamadan) önbellekte tutar."""
    return spec.build()


@st.cache_data(max_entries=64, show_spinner="Katsayılar hesaplanıyor…")
def compute_coefficients(
    spec: SignalSpec,
    n_compute: int,
    method: QuadratureMethod,
    samples: int | None,
    use_analytic: bool,
) -> RealCoefficients:
    """Sayısal (veya varsa ve istenirse analitik) katsayılar."""
    signal = load_signal(spec)
    if use_analytic and signal.has_analytic:
        return signal.analytic_coefficients(n_compute)
    return signal.coefficients(n_compute, method=method, samples=samples)


def coefficients_for(spec: SignalSpec, settings: SeriesSettings) -> RealCoefficients:
    """Ayarlara göre katsayıları döndürür (N'e kesilmiş)."""
    coeffs = compute_coefficients(
        spec, settings.n_compute, settings.method, settings.samples, settings.use_analytic
    )
    return coeffs.truncate(min(settings.n_terms, coeffs.n_terms))


@st.cache_data(max_entries=32, show_spinner=False)
def compute_numeric_and_analytic(
    spec: SignalSpec, n_terms: int, method: QuadratureMethod, samples: int | None
) -> tuple[RealCoefficients, RealCoefficients | None]:
    """Karşılaştırma tablosu için sayısal ve (varsa) analitik katsayılar."""
    signal = load_signal(spec)
    numeric = signal.coefficients(n_terms, method=method, samples=samples)
    analytic = signal.analytic_coefficients(n_terms) if signal.has_analytic else None
    return numeric, analytic


@st.cache_data(max_entries=32, show_spinner=False)
def compute_signal_energy(spec: SignalSpec) -> float:
    """Sinyalin ortalama gücü (Parseval karşılaştırması için)."""
    return signal_energy(load_signal(spec))


@dataclass(frozen=True)
class ErrorSummary:
    """Seçili N için özet hata ölçüleri."""

    l2: float
    max_abs: float
    energy_fraction: float


@st.cache_data(max_entries=64, show_spinner=False)
def compute_errors(spec: SignalSpec, settings: SeriesSettings) -> ErrorSummary:
    """Seçili N ve yumuşatma için L2/maks. hata ve yakalanan enerji oranı."""
    signal = load_signal(spec)
    coeffs = coefficients_for(spec, settings)
    energy = compute_signal_energy(spec)
    return ErrorSummary(
        l2=l2_error(signal, coeffs, smoothing=settings.smoothing),
        max_abs=max_error(signal, coeffs, smoothing=settings.smoothing),
        energy_fraction=min(1.0, coeffs.energy() / energy) if energy > 0 else 1.0,
    )


@st.cache_data(max_entries=32, show_spinner="Hata eğrileri hesaplanıyor…")
def compute_convergence(
    spec: SignalSpec,
    method: QuadratureMethod,
    samples: int | None,
    use_analytic: bool,
    smoothing: Smoothing,
    exclude: float,
    n_max: int,
) -> ConvergenceResult:
    """1..n_max aralığında logaritmik aralıklı N değerleri için hata çalışması."""
    signal = load_signal(spec)
    coeffs = compute_coefficients(spec, n_max, method, samples, use_analytic)
    n_values = np.unique(np.geomspace(1, n_max, 36).round().astype(int))
    return convergence_study(signal, coeffs, n_values.tolist(), smoothing, exclude=exclude)


@st.cache_data(max_entries=32, show_spinner=False)
def compute_gibbs(spec: SignalSpec, settings: SeriesSettings) -> dict[str, GibbsResult] | None:
    """Üç yumuşatma yöntemi için seçili N'de Gibbs ölçümü (süreksizlik yoksa ``None``)."""
    signal = load_signal(spec)
    if signal.is_continuous:
        return None
    coeffs = coefficients_for(spec, settings)
    return {
        method: gibbs_overshoot(signal, coeffs, smoothing=method)  # type: ignore[arg-type]
        for method in SMOOTHING_LABELS
    }


@st.cache_data(max_entries=16, show_spinner=False)
def compute_gibbs_vs_n(
    spec: SignalSpec, method: QuadratureMethod, samples: int | None, use_analytic: bool
) -> tuple[list[int], list[float]] | None:
    """Aşım oranının N'e bağlılığı (yumuşatmasız)."""
    signal = load_signal(spec)
    if signal.is_continuous:
        return None
    coeffs = compute_coefficients(spec, MAX_N, method, samples, use_analytic)
    ns = [5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 500]
    ratios = [gibbs_overshoot(signal, coeffs, n).ratio for n in ns]
    return ns, ratios


@st.cache_data(max_entries=32, show_spinner=False)
def compute_curves(
    spec: SignalSpec, settings: SeriesSettings, n_points: int = 1500
) -> dict[str, FloatArray]:
    """Grafikler için bir periyotluk ızgara, sinyal ve kısmi toplam."""
    signal = load_signal(spec)
    coeffs = coefficients_for(spec, settings)
    t = np.linspace(signal.start, signal.start + signal.period, n_points)
    return {
        "t": t,
        "f": signal(t),
        "s": coeffs.evaluate(t, smoothing=settings.smoothing),
    }


@st.cache_data(max_entries=16, show_spinner=False)
def compute_harmonics(
    spec: SignalSpec, settings: SeriesSettings, count: int, n_points: int = 1000
) -> tuple[FloatArray, FloatArray, list[int]]:
    """İlk ``count`` sıfır olmayan harmonik (DC dahil) ve indisleri."""
    signal = load_signal(spec)
    coeffs = coefficients_for(spec, settings)
    t = np.linspace(signal.start, signal.start + signal.period, n_points)
    terms = harmonic_terms(coeffs, t, smoothing=settings.smoothing)
    amplitude = coeffs.amplitude()
    tol = 1e-9 * max(1.0, float(amplitude.max()))
    indices = [int(n) for n in np.nonzero(amplitude > tol)[0][:count]]
    return t, terms[indices], indices


def animation_n_values(n_max: int, max_frames: int = 60) -> list[int]:
    """Yakınsama animasyonu için N değerleri (küçük N'lerde her biri, sonra seyrek)."""
    if n_max <= max_frames:
        return list(range(1, n_max + 1))
    values = np.unique(np.geomspace(1, n_max, max_frames).round().astype(int))
    return [int(v) for v in values]


@st.cache_data(max_entries=16, show_spinner="Animasyon hazırlanıyor…")
def compute_animation_sums(
    spec: SignalSpec, settings: SeriesSettings, n_points: int = 700
) -> tuple[FloatArray, FloatArray, FloatArray, list[int]]:
    """Yakınsama animasyonu için ``(t, f, S_N satırları, N değerleri)``."""
    signal = load_signal(spec)
    coeffs = coefficients_for(spec, settings)
    t = np.linspace(signal.start, signal.start + signal.period, n_points)
    ns = animation_n_values(max(1, settings.n_terms))
    sums = partial_sums(coeffs, t, ns, settings.smoothing)
    return t, signal(t), sums, ns


# ====================================================================== şekiller
@st.cache_data(max_entries=32, show_spinner=False)
def library_shape(key: str, params: tuple[tuple[str, float], ...]) -> FloatArray:
    """Kütüphane şeklinin ham noktaları."""
    return build_shape(key, **dict(params))


@st.cache_data(max_entries=16, show_spinner=False)
def parse_uploaded_shape(name: str, data: bytes) -> FloatArray:
    """Yüklenen CSV/SVG dosyasını noktalara çevirir (hatalar Türkçe ``PathError``)."""
    text = data.decode("utf-8-sig", errors="replace")
    if name.lower().endswith(".svg") or "<svg" in text[:2000].lower():
        return svg_to_points(text)
    return parse_csv_points(text)


@st.cache_data(max_entries=32, show_spinner=False)
def prepare_path(raw: FloatArray, n_points: int) -> FloatArray:
    """Eşit yay uzunluğuna göre yeniden örnekler ve ``[-1, 1]`` kutusuna normalleştirir."""
    return normalize_path(resample_by_arclength(raw, n_points))


@st.cache_data(max_entries=32, show_spinner=False)
def compute_epicycles(points: FloatArray, n_circles: int) -> EpicycleSet:
    """Hazırlanmış yol için epicycle zinciri."""
    return epicycles_from_points(points, n_circles)


@st.cache_data(max_entries=8, show_spinner="Epicycle animasyonu hazırlanıyor…")
def build_epicycle_figure(
    points: FloatArray, settings: EpicycleSettings, theme_mode: str | None
) -> go.Figure:
    """Epicycle animasyon figürü (önbellekli)."""
    epi = compute_epicycles(points, settings.n_circles)
    return epicycle_figure(
        epi,
        n_frames=settings.n_frames,
        trail_fraction=settings.trail,
        show_circles=settings.show_circles,
        target=points,
        frame_duration=settings.frame_duration,
        palette=get_palette(theme_mode),
    )
