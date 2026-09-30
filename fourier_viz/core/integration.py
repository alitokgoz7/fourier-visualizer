r"""Bir periyot üzerinde sayısal integral (kareleme) kuralları.

Fourier katsayıları periyot üzerindeki integrallerdir:

.. math::

    a_n = \frac{2}{T}\int_{t_0}^{t_0+T} f(t)\cos(n\omega_0 t)\,dt ,\qquad
    b_n = \frac{2}{T}\int_{t_0}^{t_0+T} f(t)\sin(n\omega_0 t)\,dt .

Bu modül, integrali :math:`\int f \approx \sum_k w_k f(t_k)` biçiminde yazan
**düğüm/ağırlık** çiftleri üretir. Böylece tüm harmonikler için katsayılar tek bir
matris çarpımıyla hesaplanabilir.

Desteklenen yöntemler:

* ``"gauss"`` — parçalı (composite) Gauss–Legendre. Periyot önce bilinen
  süreksizlik noktalarından (``breakpoints``) bölünür, her parça eşit panellere ayrılır ve her
  panelde ``order`` noktalı Gauss kuralı uygulanır. Parçalı düzgün sinyaller için
  varsayılan ve en hassas düğüm tabanlı yöntemdir.
* ``"trapezoid"`` — düzgün periyodik trapez kuralı (:math:`w_k = T/M`). Düzgün periyodik
  fonksiyonlarda üstel (spektral) yakınsar; süreksiz fonksiyonlarda :math:`O(1/M)`.
* ``"simpson"`` — parçalı bileşik Simpson kuralı (süreksizliklerde bölünür).
* ``"quad"`` — SciPy ``quad`` (QUADPACK, salınımlı ağırlıklı QAWO). Düğüm tabanlı değildir;
  :func:`quad_fourier_integrals` ile her harmonik ayrı ayrı adaptif hesaplanır.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Final, Literal, get_args

import numpy as np

from fourier_viz.core.types import FloatArray

QuadratureMethod = Literal["gauss", "trapezoid", "simpson", "quad"]
"""Desteklenen integral yöntemlerinin adları."""

QUADRATURE_METHODS: Final[tuple[str, ...]] = get_args(QuadratureMethod)

MIN_SAMPLES: Final[int] = 8
"""Düğüm tabanlı kurallar için izin verilen en küçük düğüm sayısı."""

MAX_SAMPLES: Final[int] = 2_000_000
"""Bellek güvenliği için izin verilen en büyük düğüm sayısı."""

_BREAKPOINT_RTOL: Final[float] = 1e-12


@dataclass(frozen=True)
class Quadrature:
    r"""Bir periyot için kareleme kuralı: :math:`\int_{t_0}^{t_0+T} f\,dt \approx w \cdot f(t)`.

    Attributes:
        nodes: Düğüm noktaları :math:`t_k` (``[t_0, t_0 + T]`` içinde).
        weights: Ağırlıklar :math:`w_k`; toplamları periyoda (:math:`T`) eşittir.
        period: Periyot :math:`T`.
        t_start: Periyodun başlangıcı :math:`t_0`.
    """

    nodes: FloatArray
    weights: FloatArray
    period: float
    t_start: float

    @property
    def size(self) -> int:
        """Düğüm sayısı."""
        return int(self.nodes.size)

    def integrate(self, values: FloatArray) -> float:
        r"""Düğümlerde verilen değerlerin integralini döndürür: :math:`\sum_k w_k v_k`."""
        return float(np.dot(self.weights, values))


def validate_period(period: float) -> float:
    """Periyodun sonlu ve pozitif olduğunu doğrular.

    Args:
        period: Periyot :math:`T`.

    Returns:
        ``float`` olarak periyot.

    Raises:
        ValueError: Periyot pozitif ve sonlu değilse.
    """
    value = float(period)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Periyot pozitif ve sonlu olmalıdır (verilen: {period!r}).")
    return value


def normalize_breakpoints(
    breakpoints: Iterable[float], t_start: float, period: float
) -> tuple[float, ...]:
    """Süreksizlik noktalarını ``(t_0, t_0 + T)`` aralığına katlar, sıralar ve tekrarları atar.

    Periyodun uç noktalarına (``t_0`` veya ``t_0 + T``) denk gelen noktalar zaten parça sınırı
    olduğundan çıkarılır.

    Args:
        breakpoints: Herhangi bir periyotta verilmiş noktalar.
        t_start: Periyodun başlangıcı :math:`t_0`.
        period: Periyot :math:`T`.

    Returns:
        ``(t_0, t_0 + T)`` içinde kesin artan sırada noktalar.
    """
    period = validate_period(period)
    tol = _BREAKPOINT_RTOL * max(1.0, period, abs(t_start))
    folded: list[float] = []
    for bp in breakpoints:
        value = float(bp)
        if not np.isfinite(value):
            continue
        offset = (value - t_start) % period
        if offset < tol or period - offset < tol:
            continue
        folded.append(t_start + offset)
    folded.sort()
    unique: list[float] = []
    for value in folded:
        if not unique or value - unique[-1] > tol:
            unique.append(value)
    return tuple(unique)


def _segments(t_start: float, period: float, breakpoints: Iterable[float]) -> FloatArray:
    inner = normalize_breakpoints(breakpoints, t_start, period)
    return np.array([t_start, *inner, t_start + period], dtype=np.float64)


def _validate_samples(samples: int) -> int:
    count = int(samples)
    if count < MIN_SAMPLES:
        raise ValueError(f"Düğüm sayısı en az {MIN_SAMPLES} olmalıdır (verilen: {samples}).")
    if count > MAX_SAMPLES:
        raise ValueError(f"Düğüm sayısı en fazla {MAX_SAMPLES} olabilir (verilen: {samples}).")
    return count


def _panel_counts(lengths: FloatArray, total_panels: int) -> list[int]:
    """Toplam paneli parçalara uzunluklarıyla orantılı dağıtır (her parçaya en az 1)."""
    share = lengths / lengths.sum() * total_panels
    return [max(1, int(np.ceil(s))) for s in share]


def gauss_legendre(
    t_start: float,
    period: float,
    samples: int = 4096,
    breakpoints: Iterable[float] = (),
    order: int = 8,
) -> Quadrature:
    r"""Parçalı Gauss–Legendre kuralı üretir.

    Her panel :math:`[\alpha, \beta]` üzerinde referans düğümler :math:`x_j \in (-1, 1)` ve
    ağırlıklar :math:`\omega_j` ile

    .. math::

        \int_\alpha^\beta f(t)\,dt \approx \frac{\beta-\alpha}{2}
        \sum_{j=1}^{p} \omega_j\,
        f\!\left(\tfrac{\beta-\alpha}{2}x_j + \tfrac{\alpha+\beta}{2}\right).

    Düğümler panel içinde kaldığından süreksizlik noktalarında hiçbir zaman değerlendirme yapılmaz.

    Args:
        t_start: Periyodun başlangıcı.
        period: Periyot.
        samples: Yaklaşık toplam düğüm sayısı (``order`` katına yuvarlanır).
        breakpoints: Süreksizlik/kırılma noktaları.
        order: Panel başına Gauss noktası sayısı (1–64).

    Returns:
        :class:`Quadrature` nesnesi.
    """
    period = validate_period(period)
    samples = _validate_samples(samples)
    if not 1 <= order <= 64:
        raise ValueError(f"Gauss mertebesi 1 ile 64 arasında olmalıdır (verilen: {order}).")
    edges = _segments(t_start, period, breakpoints)
    lengths = np.diff(edges)
    panels = _panel_counts(lengths, max(1, samples // order))
    ref_x, ref_w = np.polynomial.legendre.leggauss(order)
    nodes_parts: list[FloatArray] = []
    weight_parts: list[FloatArray] = []
    for left, right, count in zip(edges[:-1], edges[1:], panels, strict=True):
        panel_edges = np.linspace(left, right, count + 1)
        half = 0.5 * np.diff(panel_edges)
        mid = 0.5 * (panel_edges[:-1] + panel_edges[1:])
        nodes_parts.append((mid[:, None] + half[:, None] * ref_x[None, :]).ravel())
        weight_parts.append((half[:, None] * ref_w[None, :]).ravel())
    return Quadrature(
        nodes=np.concatenate(nodes_parts),
        weights=np.concatenate(weight_parts),
        period=period,
        t_start=float(t_start),
    )


def periodic_trapezoid(t_start: float, period: float, samples: int = 4096) -> Quadrature:
    r"""Düzgün periyodik trapez kuralı: :math:`t_k = t_0 + kT/M`, :math:`w_k = T/M`.

    Periyodik bir fonksiyon için bu kural, :math:`|n| < M/2` olan harmonikleri tam olarak
    integre eder; daha yüksek harmonikler örtüşme (aliasing) yoluyla katlanır.

    Args:
        t_start: Periyodun başlangıcı.
        period: Periyot.
        samples: Düğüm sayısı :math:`M`.

    Returns:
        :class:`Quadrature` nesnesi.
    """
    period = validate_period(period)
    samples = _validate_samples(samples)
    nodes = t_start + period * np.arange(samples, dtype=np.float64) / samples
    weights = np.full(samples, period / samples)
    return Quadrature(nodes=nodes, weights=weights, period=period, t_start=float(t_start))


def composite_simpson(
    t_start: float,
    period: float,
    samples: int = 4096,
    breakpoints: Iterable[float] = (),
) -> Quadrature:
    r"""Parçalı bileşik Simpson kuralı.

    Her parça çift sayıda alt aralığa bölünür ve
    :math:`\tfrac{h}{3}[f_0 + 4f_1 + 2f_2 + \dots + 4f_{m-1} + f_m]` uygulanır. Parça uçlarındaki
    düğümler, tek taraflı limitleri temsil etmeleri için parçanın içine çok az kaydırılır
    (süreksizlikte ortalama değer yerine doğru taraftaki değer kullanılır).

    Args:
        t_start: Periyodun başlangıcı.
        period: Periyot.
        samples: Yaklaşık toplam alt aralık sayısı.
        breakpoints: Süreksizlik noktaları.

    Returns:
        :class:`Quadrature` nesnesi.
    """
    period = validate_period(period)
    samples = _validate_samples(samples)
    edges = _segments(t_start, period, breakpoints)
    lengths = np.diff(edges)
    intervals = _panel_counts(lengths, samples)
    nodes_parts: list[FloatArray] = []
    weight_parts: list[FloatArray] = []
    for left, right, count in zip(edges[:-1], edges[1:], intervals, strict=True):
        m = max(2, count + (count % 2))  # çift sayıda alt aralık
        nodes = np.linspace(left, right, m + 1)
        nudge = 1e-9 * (right - left)
        nodes[0] += nudge
        nodes[-1] -= nudge
        h = (right - left) / m
        weights = np.full(m + 1, 2.0)
        weights[1::2] = 4.0
        weights[0] = weights[-1] = 1.0
        nodes_parts.append(nodes)
        weight_parts.append(weights * h / 3.0)
    return Quadrature(
        nodes=np.concatenate(nodes_parts),
        weights=np.concatenate(weight_parts),
        period=period,
        t_start=float(t_start),
    )


def make_quadrature(
    method: QuadratureMethod,
    t_start: float,
    period: float,
    samples: int = 4096,
    breakpoints: Iterable[float] = (),
) -> Quadrature:
    """Ada göre düğüm tabanlı bir kareleme kuralı oluşturur.

    Args:
        method: ``"gauss"``, ``"trapezoid"`` veya ``"simpson"``.
        t_start: Periyodun başlangıcı.
        period: Periyot.
        samples: Düğüm sayısı (hassasiyet).
        breakpoints: Süreksizlik noktaları (trapez kuralı bunları kullanmaz).

    Returns:
        :class:`Quadrature` nesnesi.

    Raises:
        ValueError: Yöntem bilinmiyorsa veya ``"quad"`` ise (düğüm tabanlı değildir).
    """
    if method == "gauss":
        return gauss_legendre(t_start, period, samples, breakpoints)
    if method == "trapezoid":
        return periodic_trapezoid(t_start, period, samples)
    if method == "simpson":
        return composite_simpson(t_start, period, samples, breakpoints)
    if method == "quad":
        raise ValueError("'quad' yöntemi düğüm tabanlı değildir; quad_fourier_integrals kullanın.")
    raise ValueError(
        f"Bilinmeyen integral yöntemi: {method!r}. Geçerli yöntemler: {QUADRATURE_METHODS}."
    )


def quad_fourier_integrals(
    func: Callable[[FloatArray], FloatArray],
    n_terms: int,
    t_start: float,
    period: float,
    breakpoints: Iterable[float] = (),
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    limit: int = 200,
) -> tuple[FloatArray, FloatArray]:
    r"""SciPy ``quad`` ile :math:`\int f\cos(n\omega_0 t)` ve :math:`\int f\sin(n\omega_0 t)`.

    Her süreksizlik parçası için QUADPACK'in salınımlı ağırlık rutini (QAWO,
    ``weight="cos"/"sin"``) kullanılır. Yavaş fakat toleransı ayarlanabilir, referans
    niteliğinde bir yöntemdir.

    Args:
        func: Vektörleştirilmiş fonksiyon :math:`f(t)`.
        n_terms: En yüksek harmonik :math:`N`.
        t_start: Periyodun başlangıcı.
        period: Periyot.
        breakpoints: Süreksizlik noktaları.
        epsabs: Mutlak tolerans.
        epsrel: Göreli tolerans.
        limit: Alt aralık sınırı.

    Returns:
        ``(C, S)`` dizileri; :math:`C_n = \int f\cos`, :math:`S_n = \int f\sin`, ``n = 0..N``.
    """
    from scipy import integrate

    period = validate_period(period)
    omega0 = 2.0 * np.pi / period
    edges = _segments(t_start, period, breakpoints)

    def scalar(x: float) -> float:
        return float(np.asarray(func(np.array([x], dtype=np.float64)), dtype=np.float64)[0])

    cos_part = np.zeros(n_terms + 1)
    sin_part = np.zeros(n_terms + 1)
    for left, right in pairwise(edges):
        for n in range(n_terms + 1):
            if n == 0:
                value, _ = integrate.quad(
                    scalar, left, right, epsabs=epsabs, epsrel=epsrel, limit=limit
                )
                cos_part[0] += value
                continue
            wvar = n * omega0
            c_val, _ = integrate.quad(
                scalar, left, right, weight="cos", wvar=wvar,
                epsabs=epsabs, epsrel=epsrel, limit=limit,
            )  # fmt: skip
            s_val, _ = integrate.quad(
                scalar, left, right, weight="sin", wvar=wvar,
                epsabs=epsabs, epsrel=epsrel, limit=limit,
            )  # fmt: skip
            cos_part[n] += c_val
            sin_part[n] += s_val
    return cos_part, sin_part
