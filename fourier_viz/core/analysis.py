r"""Hata metrikleri, yakınsama, Gibbs olayı, Parseval özdeşliği ve spektrum analizi.

**L2 (RMS) hatası.** :math:`\|f - S_N\| = \sqrt{\tfrac{1}{T}\int_T |f - S_N|^2\,dt}`.
Kısmi toplam, :math:`\{1, \cos n\omega_0 t, \sin n\omega_0 t\}` alt uzayına dik izdüşüm
olduğundan (Bessel eşitsizliği) bu hata :math:`N` ile **monoton azalır** ve

.. math::

    \|f - S_N\|^2 = \|f\|^2 - \Big[(a_0/2)^2 + \tfrac12\sum_{n=1}^{N}(a_n^2 + b_n^2)\Big].

**Parseval.** :math:`\dfrac{1}{T}\displaystyle\int_T |f|^2\,dt = \Big(\frac{a_0}{2}\Big)^2 +
\frac12\sum_{n\ge1}(a_n^2 + b_n^2) = \sum_n |c_n|^2`.

**Gibbs olayı.** Büyüklüğü :math:`J` olan bir sıçramada :math:`S_N` sıçramanın hemen
yanında :math:`N \to \infty` iken bile

.. math::

    \frac{\max S_N - f(t_d^{+})}{|J|} \;\longrightarrow\;
    \frac{1}{\pi}\operatorname{Si}(\pi) - \frac12 \approx 0.08949

kadar aşar (:math:`\operatorname{Si}(x) = \int_0^x \frac{\sin u}{u}du`). Aşım daralır ama
yüksekliği azalmaz. Fejér ortalaması (pozitif çekirdek) aşımı tamamen yok eder; Lanczos
σ-faktörleri ise büyük ölçüde azaltır.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, special

from fourier_viz.core.complex_dft import real_to_complex
from fourier_viz.core.integration import gauss_legendre
from fourier_viz.core.series import (
    RealCoefficients,
    Smoothing,
    default_samples,
    partial_sum,
    partial_sums,
)
from fourier_viz.core.signals import Jump, PeriodicSignal, SignalError
from fourier_viz.core.types import FloatArray, IntArray

GIBBS_CONSTANT: Final[float] = float(special.sici(np.pi)[0] / np.pi - 0.5)
r"""Gibbs aşım oranı :math:`\operatorname{Si}(\pi)/\pi - 1/2 \approx 0.0894899` (≈ %8.95)."""

WILBRAHAM_GIBBS_PEAK: Final[float] = float(2.0 * special.sici(np.pi)[0] / np.pi)
r"""Birim (:math:`\pm1`) kare dalga için :math:`\lim_{N\to\infty} \max S_N`.

Değeri :math:`\tfrac{2}{\pi}\operatorname{Si}(\pi) \approx 1.17898` (Wilbraham–Gibbs sabiti).
"""


# ====================================================================== sonuç türleri
@dataclass(frozen=True)
class ConvergenceResult:
    """Farklı :math:`N` değerleri için hata ölçümleri.

    Attributes:
        n_values: Terim sayıları.
        l2: RMS (L2) hataları.
        max_abs: Maksimum mutlak hatalar.
        smoothing: Kullanılan yumuşatma.
    """

    n_values: IntArray
    l2: FloatArray
    max_abs: FloatArray
    smoothing: Smoothing = "none"

    def l2_rate(self) -> float:
        """L2 hatasının log-log eğimi (ör. süreksiz sinyal ≈ −0.5, sürekli ≈ −1.5)."""
        return estimate_rate(self.n_values, self.l2)

    def max_rate(self) -> float:
        """Maksimum hatanın log-log eğimi."""
        return estimate_rate(self.n_values, self.max_abs)


@dataclass(frozen=True)
class GibbsResult:
    r"""Bir süreksizlikteki Gibbs aşımı ölçümü.

    Attributes:
        n_terms: Terim sayısı :math:`N`.
        jump: İncelenen süreksizlik.
        peak_time: :math:`S_N`'in tepe (veya çukur) yaptığı zaman.
        peak_value: Tepe değeri.
        overshoot: Mutlak aşım (sıçramanın yüksek tarafındaki değerin ne kadar üstüne çıkıldığı;
            negatifse aşım yoktur).
        ratio: Aşımın sıçrama büyüklüğüne oranı :math:`\text{aşım}/|J|`.
        theoretical_ratio: :data:`GIBBS_CONSTANT`.
        smoothing: Kullanılan yumuşatma.
    """

    n_terms: int
    jump: Jump
    peak_time: float
    peak_value: float
    overshoot: float
    ratio: float
    theoretical_ratio: float = GIBBS_CONSTANT
    smoothing: Smoothing = "none"

    @property
    def percent(self) -> float:
        """Aşım yüzdesi (sıçramaya göre, en az 0)."""
        return max(0.0, 100.0 * self.ratio)

    @property
    def relative_difference(self) -> float:
        """Teorik değere göre göreli fark :math:`(r - G)/G`."""
        return (self.ratio - self.theoretical_ratio) / self.theoretical_ratio


@dataclass(frozen=True)
class ParsevalResult:
    r"""Parseval özdeşliği kontrolü.

    Attributes:
        signal_energy: :math:`(1/T)\int |f|^2` (sayısal integral).
        coefficient_energy: :math:`(a_0/2)^2 + \tfrac12\sum (a_n^2+b_n^2)`.
        complex_energy: :math:`\sum |c_n|^2` (aynı değer olmalı).
    """

    signal_energy: float
    coefficient_energy: float
    complex_energy: float

    @property
    def residual(self) -> float:
        r"""Katsayılarda henüz yakalanmamış enerji (kuyruk): :math:`\|f\|^2 - E_N \ge 0`."""
        return self.signal_energy - self.coefficient_energy

    @property
    def captured_fraction(self) -> float:
        """Katsayıların yakaladığı enerji oranı (0–1)."""
        if self.signal_energy == 0.0:
            return 1.0
        return self.coefficient_energy / self.signal_energy


@dataclass(frozen=True)
class Spectrum:
    r"""Tek taraflı genlik/faz/güç spektrumu (:math:`n = 0..N`).

    Attributes:
        harmonics: Harmonik indisleri :math:`n`.
        frequencies: Frekanslar :math:`n/T` (Hz, :math:`T` saniye kabul edilirse).
        amplitude: :math:`A_n` (:math:`A_0 = |a_0|/2`).
        phase: :math:`\varphi_n = \operatorname{atan2}(-b_n, a_n)`.
        power: Ortalama güce katkı: :math:`A_0^2` ve :math:`A_n^2/2`.
    """

    harmonics: IntArray
    frequencies: FloatArray
    amplitude: FloatArray
    phase: FloatArray
    power: FloatArray

    def dominant(self, count: int = 5) -> IntArray:
        """En büyük genlikli ``count`` harmoniğin indisleri (azalan sırada)."""
        order = np.argsort(-self.amplitude, kind="stable")
        dominant: IntArray = self.harmonics[order[:count]]
        return dominant


# ====================================================================== temel metrikler
def signal_energy(signal: PeriodicSignal, samples: int = 16384) -> float:
    r"""Sinyalin ortalama gücü :math:`\|f\|^2 = \tfrac{1}{T}\int_T |f(t)|^2\,dt`."""
    quad = gauss_legendre(signal.start, signal.period, samples, signal.breakpoints)
    return quad.integrate(signal(quad.nodes) ** 2) / signal.period


def inner_product(
    f: Callable[[FloatArray], FloatArray],
    g: Callable[[FloatArray], FloatArray],
    period: float = 2.0 * np.pi,
    t_start: float | None = None,
    samples: int = 4096,
) -> float:
    r"""Normalize iç çarpım :math:`\langle f, g\rangle = \tfrac{2}{T}\int_T f(t) g(t)\,dt`.

    Bu normalizasyonla :math:`\langle\cos n\omega_0 t, \cos m\omega_0 t\rangle = \delta_{nm}`
    (:math:`n, m \ge 1`) olur; farklı harmonikler **ortogonaldir**.
    """
    t0 = -period / 2.0 if t_start is None else t_start
    quad = gauss_legendre(t0, period, samples)
    return 2.0 / period * quad.integrate(f(quad.nodes) * g(quad.nodes))


def orthogonality_matrix(
    n_max: int, period: float = 2.0 * np.pi, samples: int = 4096
) -> FloatArray:
    r"""Taban fonksiyonların Gram matrisi.

    Taban sırası: :math:`[\cos(1\omega_0 t), \dots, \cos(N\omega_0 t), \sin(1\omega_0 t), \dots,
    \sin(N\omega_0 t)]`. Ortogonallik nedeniyle sonuç birim matrise çok yakındır.

    Returns:
        ``(2N, 2N)`` matris :math:`G_{ij} = \langle\phi_i, \phi_j\rangle`.
    """
    if n_max < 1:
        raise ValueError("En az bir harmonik gereklidir (n_max ≥ 1).")
    quad = gauss_legendre(-period / 2.0, period, max(samples, 16 * (n_max + 1)))
    k = np.arange(1, n_max + 1)[:, None]
    wt = 2.0 * np.pi / period * quad.nodes[None, :]
    basis = np.vstack([np.cos(k * wt), np.sin(k * wt)])
    gram: FloatArray = 2.0 / period * (basis * quad.weights) @ basis.T
    return gram


def _error_quadrature(
    signal: PeriodicSignal, n_terms: int, samples: int | None
) -> tuple[FloatArray, FloatArray]:
    quad = gauss_legendre(
        signal.start, signal.period, samples or default_samples(n_terms), signal.breakpoints
    )
    return quad.nodes, quad.weights


def l2_error(
    signal: PeriodicSignal,
    coeffs: RealCoefficients,
    n_terms: int | None = None,
    smoothing: Smoothing = "none",
    samples: int | None = None,
) -> float:
    r"""RMS hatası :math:`\sqrt{\tfrac1T\int_T (f - S_N)^2\,dt}` (Gauss karelemesiyle)."""
    n = coeffs.n_terms if n_terms is None else n_terms
    nodes, weights = _error_quadrature(signal, n, samples)
    diff = signal(nodes) - partial_sum(coeffs, nodes, n, smoothing)
    return float(np.sqrt(max(0.0, np.dot(weights, diff**2) / signal.period)))


def l2_error_from_energy(
    energy: float, coeffs: RealCoefficients, n_terms: int | None = None
) -> float:
    r"""Parseval ile L2 hatası: :math:`\sqrt{\|f\|^2 - E_N}` (yalnızca yumuşatmasız :math:`S_N`)."""
    truncated = coeffs if n_terms is None else coeffs.truncate(n_terms)
    return float(np.sqrt(max(0.0, energy - truncated.energy())))


def _exclusion_mask(signal: PeriodicSignal, t: FloatArray, exclude: float) -> np.ndarray:
    mask = np.ones(t.shape, dtype=bool)
    if exclude <= 0.0:
        return mask
    for loc in signal.discontinuities:
        dist = np.abs(np.mod(t - loc + signal.period / 2.0, signal.period) - signal.period / 2.0)
        mask &= dist > exclude
    return mask


def max_error(
    signal: PeriodicSignal,
    coeffs: RealCoefficients,
    n_terms: int | None = None,
    smoothing: Smoothing = "none",
    n_points: int = 4001,
    exclude: float = 0.0,
) -> float:
    r"""Maksimum mutlak hata :math:`\sup_t |f(t) - S_N(t)|`.

    Eşit aralıklı ızgaraya ek olarak her süreksizliğin :math:`\pm 10^{-9}T` yakınına tek
    taraflı "yoklama" noktaları eklenir; böylece süreksiz sinyallerde supremum doğru olarak
    :math:`|J|/2`'ye yakın ölçülür ve :math:`N` ile sıfıra gitmez (düzgün olmayan yakınsama).
    ``exclude > 0`` ise süreksizliklere bu uzaklıktan yakın noktalar hesaba katılmaz.
    """
    t = _error_grid(signal, n_points, exclude)
    diff = np.abs(signal(t) - partial_sum(coeffs, t, n_terms, smoothing))
    return float(diff.max())


def _error_grid(signal: PeriodicSignal, n_points: int, exclude: float) -> FloatArray:
    grid: FloatArray = np.linspace(signal.start, signal.start + signal.period, n_points)
    if exclude <= 0.0 and signal.discontinuities:
        eps = 1e-9 * signal.period
        probes = [d + s * eps for d in signal.discontinuities for s in (-1.0, 1.0)]
        grid = np.concatenate([grid, probes])
    grid = grid[_exclusion_mask(signal, grid, exclude)]
    if grid.size == 0:
        raise SignalError("Dışlama bölgesi tüm aralığı kapsıyor; daha küçük bir değer seçin.")
    return grid


def convergence_study(
    signal: PeriodicSignal,
    coeffs: RealCoefficients,
    n_values: Sequence[int],
    smoothing: Smoothing = "none",
    samples: int | None = None,
    n_points: int = 4001,
    exclude: float = 0.0,
) -> ConvergenceResult:
    """Birden çok :math:`N` için L2 ve maksimum hataları tek geçişte hesaplar.

    Tüm kısmi toplamlar :func:`~fourier_viz.core.series.partial_sums` ile tek matris çarpımında
    elde edilir.
    """
    ns = np.asarray(sorted({int(n) for n in n_values}), dtype=np.int64)
    if ns.size == 0:
        raise ValueError("En az bir N değeri gereklidir.")
    nodes, weights = _error_quadrature(signal, int(ns.max()), samples)
    sums = partial_sums(coeffs, nodes, ns.tolist(), smoothing)
    sq = (signal(nodes)[None, :] - sums) ** 2
    l2 = np.sqrt(np.maximum(0.0, sq @ weights / signal.period))

    grid = _error_grid(signal, n_points, exclude)
    grid_sums = partial_sums(coeffs, grid, ns.tolist(), smoothing)
    max_abs = np.max(np.abs(signal(grid)[None, :] - grid_sums), axis=1)
    return ConvergenceResult(ns, l2, max_abs, smoothing)


def estimate_rate(n_values: ArrayLike, errors: ArrayLike) -> float:
    r"""Log-log en küçük kareler eğimi: :math:`\text{hata} \approx C N^{p}` için :math:`p`.

    Sıfır/negatif :math:`N` veya hata değerleri ve çok küçük (:math:`<10^{-14}`) hatalar
    dışlanır. Yeterli nokta yoksa ``nan`` döner.
    """
    n = np.asarray(n_values, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    mask = (n > 0) & (e > 1e-14) & np.isfinite(e)
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(n[mask]), np.log(e[mask]), 1)
    return float(slope)


# ====================================================================== Gibbs
def _gibbs_window(signal: PeriodicSignal, jump: Jump, n_terms: int) -> float:
    base = min(signal.period / 4.0, 4.0 * signal.period / (n_terms + 1))
    others = [d for d in signal.discontinuities if not np.isclose(d, jump.location)]
    for loc in others:
        dist = abs(
            (loc - jump.location + signal.period / 2.0) % signal.period - signal.period / 2.0
        )
        base = min(base, 0.5 * dist)
    return base


def gibbs_overshoot(
    signal: PeriodicSignal,
    coeffs: RealCoefficients,
    n_terms: int | None = None,
    smoothing: Smoothing = "none",
    jump: Jump | None = None,
    grid_points: int = 2001,
) -> GibbsResult:
    r"""Bir süreksizlikteki Gibbs aşımını ölçer.

    Sıçramanın **yüksek** tarafında (:math:`h = \max(f(t_d^-), f(t_d^+))`), süreksizliğe
    bitişik ve birkaç :math:`T/N` genişliğindeki pencerede :math:`\max S_N` bulunur (sık ızgara +
    sınırlı skaler optimizasyon). Aşım :math:`\max S_N - h`, oran ise aşım :math:`/ |J|`.

    Args:
        signal: Sinyal (en az bir bilinen süreksizlik içermeli).
        coeffs: Katsayılar.
        n_terms: Terim sayısı (varsayılan: tümü).
        smoothing: Yumuşatma yöntemi.
        jump: İncelenecek süreksizlik (varsayılan: en büyük sıçrama).
        grid_points: Pencere içi ızgara nokta sayısı.

    Raises:
        SignalError: Sinyalde süreksizlik yoksa.
    """
    n = coeffs.n_terms if n_terms is None else n_terms
    if jump is None:
        jumps = signal.jumps()
        if not jumps:
            raise SignalError(
                "Bu sinyal süreksizlik içermiyor; Gibbs olayı yalnızca sıçramalarda görülür."
            )
        center = signal.start + signal.period / 2.0
        # En büyük sıçrama; eşitlikte temel aralığın ortasına en yakın olan (grafikte görünür).
        jump = max(jumps, key=lambda j: (round(abs(j.size), 9), -abs(j.location - center)))
    width = _gibbs_window(signal, jump, max(n, 1))
    upward = jump.right >= jump.left
    high = jump.right if upward else jump.left
    lo, hi = (
        (jump.location, jump.location + width) if upward else (jump.location - width, jump.location)
    )
    grid = np.linspace(lo, hi, grid_points)
    values = partial_sum(coeffs, grid, n, smoothing)
    best = int(np.argmax(values))
    left = grid[max(best - 1, 0)]
    right = grid[min(best + 1, grid.size - 1)]
    peak_t, peak_v = float(grid[best]), float(values[best])
    if right > left:
        res = optimize.minimize_scalar(
            lambda x: -float(partial_sum(coeffs, np.array([x]), n, smoothing)[0]),
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-12 * max(1.0, signal.period)},
        )
        if -res.fun > peak_v:
            peak_t, peak_v = float(res.x), float(-res.fun)
    overshoot = peak_v - high
    size = abs(jump.size)
    ratio = overshoot / size if size > 0 else 0.0
    return GibbsResult(n, jump, peak_t, peak_v, overshoot, ratio, GIBBS_CONSTANT, smoothing)


# ====================================================================== Parseval & spektrum
def parseval_check(
    signal: PeriodicSignal, coeffs: RealCoefficients, samples: int = 16384
) -> ParsevalResult:
    """Sinyal enerjisini katsayı enerjisiyle (gerçek ve karmaşık form) karşılaştırır."""
    return ParsevalResult(
        signal_energy=signal_energy(signal, samples),
        coefficient_energy=coeffs.energy(),
        complex_energy=real_to_complex(coeffs).energy(),
    )


def cumulative_energy(coeffs: RealCoefficients) -> FloatArray:
    r""":math:`E_N = (a_0/2)^2 + \tfrac12\sum_{n=1}^{N}(a_n^2+b_n^2)`, :math:`N = 0..N_{max}`."""
    parts = np.empty(coeffs.n_terms + 1)
    parts[0] = (coeffs.a[0] / 2.0) ** 2
    parts[1:] = 0.5 * (coeffs.a[1:] ** 2 + coeffs.b[1:] ** 2)
    energy: FloatArray = np.cumsum(parts)
    return energy


def terms_for_energy(coeffs: RealCoefficients, total_energy: float, fraction: float = 0.99) -> int:
    """Toplam enerjinin en az ``fraction`` kadarını yakalamak için gereken en küçük :math:`N`.

    Mevcut katsayılar yetmiyorsa ``-1`` döner.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Oran 0 ile 1 arasında olmalıdır.")
    if total_energy <= 0.0:
        return 0
    reached = np.nonzero(cumulative_energy(coeffs) >= fraction * total_energy * (1 - 1e-12))[0]
    return int(reached[0]) if reached.size else -1


def spectrum(coeffs: RealCoefficients) -> Spectrum:
    """Genlik, faz ve güç spektrumunu döndürür."""
    amplitude = coeffs.amplitude()
    power = amplitude**2 / 2.0
    power[0] = amplitude[0] ** 2
    harmonics = np.arange(coeffs.n_terms + 1, dtype=np.int64)
    phase = coeffs.phase()
    phase[amplitude < 1e-12 * max(1.0, float(amplitude.max()))] = 0.0
    return Spectrum(
        harmonics=harmonics,
        frequencies=harmonics / coeffs.period,
        amplitude=amplitude,
        phase=phase,
        power=power,
    )
