r"""Gerçek (trigonometrik) Fourier serisi.

Periyodu :math:`T` ve temel açısal frekansı :math:`\omega_0 = 2\pi/T` olan bir :math:`f` için

.. math::

    f(t) \sim \frac{a_0}{2} + \sum_{n=1}^{\infty}
        \left[a_n \cos(n\omega_0 t) + b_n \sin(n\omega_0 t)\right],

.. math::

    a_n = \frac{2}{T}\int_T f(t)\cos(n\omega_0 t)\,dt, \qquad
    b_n = \frac{2}{T}\int_T f(t)\sin(n\omega_0 t)\,dt .

:math:`N` terimli (ve isteğe bağlı :math:`\sigma_n` ağırlıklı) kısmi toplam

.. math::

    S_N(t) = \frac{a_0}{2} + \sum_{n=1}^{N} \sigma_n
        \left[a_n \cos(n\omega_0 t) + b_n \sin(n\omega_0 t)\right].

Gibbs etkisini azaltmak için iki :math:`\sigma` yumuşatması sunulur:

* Fejér (Cesàro ortalaması): :math:`\sigma_n = 1 - \dfrac{n}{N+1}`
* Lanczos sigma: :math:`\sigma_n = \operatorname{sinc}\!\left(\dfrac{n}{N+1}\right)
  = \dfrac{\sin(\pi n/(N+1))}{\pi n/(N+1)}`
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, get_args

import numpy as np
from numpy.typing import ArrayLike

from fourier_viz.core.integration import (
    QuadratureMethod,
    make_quadrature,
    quad_fourier_integrals,
    validate_period,
)
from fourier_viz.core.types import FloatArray

if TYPE_CHECKING:
    from fourier_viz.core.complex_dft import ComplexCoefficients

Smoothing = Literal["none", "fejer", "lanczos"]
"""Kısmi toplam yumuşatma yöntemleri."""

SMOOTHING_METHODS: Final[tuple[str, ...]] = get_args(Smoothing)

TWO_PI: Final[float] = 2.0 * np.pi

MAX_TERMS: Final[int] = 100_000
"""Güvenlik için izin verilen en büyük harmonik sayısı."""

_CHUNK_ELEMENTS: Final[int] = 2_000_000
"""Vektörleştirilmiş trig matrislerinde tek seferde tutulacak en fazla eleman sayısı."""


class SeriesError(ValueError):
    """Fourier serisi hesaplamalarındaki kullanıcıya gösterilebilir hatalar."""


def _as_float_array(values: ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def _validate_terms(n_terms: int, *, maximum: int = MAX_TERMS) -> int:
    if isinstance(n_terms, bool) or int(n_terms) != n_terms:
        raise SeriesError(f"Terim sayısı tam sayı olmalıdır (verilen: {n_terms!r}).")
    value = int(n_terms)
    if value < 0:
        raise SeriesError(f"Terim sayısı negatif olamaz (verilen: {value}).")
    if value > maximum:
        raise SeriesError(f"Terim sayısı en fazla {maximum} olabilir (verilen: {value}).")
    return value


def _row_chunks(n_rows: int, n_cols: int) -> Iterable[slice]:
    step = max(1, _CHUNK_ELEMENTS // max(1, n_cols))
    for start in range(0, n_rows, step):
        yield slice(start, min(n_rows, start + step))


@dataclass(frozen=True, eq=False)
class RealCoefficients:
    r"""Gerçek Fourier katsayıları :math:`a_0..a_N` ve :math:`b_0..b_N` (:math:`b_0 = 0`).

    Attributes:
        a: Kosinüs katsayıları, ``a[0]`` = :math:`a_0` (ortalama :math:`a_0/2`'dir).
        b: Sinüs katsayıları, ``b[0]`` her zaman 0'dır.
        period: Periyot :math:`T`.
    """

    a: FloatArray
    b: FloatArray
    period: float = field(default=TWO_PI)

    def __post_init__(self) -> None:
        a = np.array(self.a, dtype=np.float64, copy=True).ravel()
        b = np.array(self.b, dtype=np.float64, copy=True).ravel()
        if a.size == 0:
            raise SeriesError("Katsayı dizisi boş olamaz (en az a₀ gereklidir).")
        if a.shape != b.shape:
            raise SeriesError(f"a ve b dizileri aynı uzunlukta olmalıdır ({a.size} ≠ {b.size}).")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise SeriesError("Katsayılar sonlu olmalıdır (NaN/sonsuz değer bulundu).")
        b[0] = 0.0
        a.setflags(write=False)
        b.setflags(write=False)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "period", validate_period(self.period))

    # ------------------------------------------------------------------ özellikler
    @property
    def n_terms(self) -> int:
        """Saklanan en yüksek harmonik :math:`N`."""
        return int(self.a.size - 1)

    @property
    def omega0(self) -> float:
        r"""Temel açısal frekans :math:`\omega_0 = 2\pi/T`."""
        return TWO_PI / self.period

    @property
    def harmonics(self) -> FloatArray:
        """Harmonik indisleri ``0..N`` (``float`` dizi)."""
        return np.arange(self.n_terms + 1, dtype=np.float64)

    @property
    def mean(self) -> float:
        r"""Sinyalin ortalama değeri :math:`a_0/2`."""
        return float(self.a[0] / 2.0)

    def amplitude(self) -> FloatArray:
        r"""Harmonik genlikleri :math:`A_n = \sqrt{a_n^2 + b_n^2}` (:math:`A_0 = |a_0|/2`).

        :math:`a_n\cos + b_n\sin = A_n\cos(n\omega_0 t + \varphi_n)` gösterimindeki genliktir.
        """
        amp = np.hypot(self.a, self.b)
        amp[0] = abs(self.a[0]) / 2.0
        return amp

    def phase(self) -> FloatArray:
        r"""Harmonik fazları :math:`\varphi_n = \operatorname{atan2}(-b_n, a_n)` (radyan)."""
        return np.arctan2(-self.b, self.a)

    def energy(self) -> float:
        r"""Parseval enerjisi :math:`(a_0/2)^2 + \tfrac12\sum_{n\ge1}(a_n^2 + b_n^2)`."""
        return float((self.a[0] / 2.0) ** 2 + 0.5 * np.sum(self.a[1:] ** 2 + self.b[1:] ** 2))

    # ------------------------------------------------------------------ işlemler
    def truncate(self, n_terms: int) -> RealCoefficients:
        """İlk ``n_terms`` harmoniği tutan yeni katsayı nesnesi döndürür."""
        n = _validate_terms(n_terms, maximum=self.n_terms)
        return RealCoefficients(self.a[: n + 1], self.b[: n + 1], self.period)

    def padded(self, n_terms: int) -> RealCoefficients:
        """Eksik yüksek harmonikleri sıfırla doldurarak ``n_terms`` uzunluğa genişletir."""
        n = _validate_terms(n_terms)
        if n <= self.n_terms:
            return self.truncate(n)
        a = np.zeros(n + 1)
        b = np.zeros(n + 1)
        a[: self.a.size] = self.a
        b[: self.b.size] = self.b
        return RealCoefficients(a, b, self.period)

    def _check_compatible(self, other: RealCoefficients) -> None:
        if not np.isclose(self.period, other.period, rtol=1e-12, atol=0.0):
            raise SeriesError("Farklı periyotlu katsayılar birleştirilemez.")

    def __add__(self, other: RealCoefficients) -> RealCoefficients:
        if not isinstance(other, RealCoefficients):
            return NotImplemented
        self._check_compatible(other)
        n = max(self.n_terms, other.n_terms)
        lhs, rhs = self.padded(n), other.padded(n)
        return RealCoefficients(lhs.a + rhs.a, lhs.b + rhs.b, self.period)

    def __neg__(self) -> RealCoefficients:
        return RealCoefficients(-self.a, -self.b, self.period)

    def __sub__(self, other: RealCoefficients) -> RealCoefficients:
        if not isinstance(other, RealCoefficients):
            return NotImplemented
        return self + (-other)

    def __mul__(self, scalar: float) -> RealCoefficients:
        if not isinstance(scalar, int | float | np.floating | np.integer):
            return NotImplemented
        return RealCoefficients(self.a * float(scalar), self.b * float(scalar), self.period)

    __rmul__ = __mul__

    def allclose(self, other: RealCoefficients, *, atol: float = 1e-8, rtol: float = 1e-6) -> bool:
        """İki katsayı kümesinin (eksikler sıfır sayılarak) yaklaşık eşit olup olmadığı."""
        self._check_compatible(other)
        n = max(self.n_terms, other.n_terms)
        lhs, rhs = self.padded(n), other.padded(n)
        return bool(
            np.allclose(lhs.a, rhs.a, atol=atol, rtol=rtol)
            and np.allclose(lhs.b, rhs.b, atol=atol, rtol=rtol)
        )

    def to_complex(self) -> ComplexCoefficients:
        """Karmaşık forma dönüştürür (bkz. :func:`fourier_viz.core.complex_dft.real_to_complex`)."""
        from fourier_viz.core.complex_dft import real_to_complex

        return real_to_complex(self)

    def evaluate(
        self,
        t: ArrayLike,
        n_terms: int | None = None,
        smoothing: Smoothing = "none",
    ) -> FloatArray:
        """Kısmi toplamı hesaplar; :func:`partial_sum` kısayoludur."""
        return partial_sum(self, t, n_terms, smoothing)


# ---------------------------------------------------------------------- σ-faktörleri
def sigma_factors(n_terms: int, method: Smoothing = "none") -> FloatArray:
    r"""Kısmi toplam için yumuşatma ağırlıkları :math:`\sigma_0..\sigma_N`.

    * ``"none"``: :math:`\sigma_n = 1`
    * ``"fejer"``: :math:`\sigma_n = 1 - n/(N+1)` — Fejér çekirdeği pozitif olduğundan
      :math:`\min f \le \sigma_N f \le \max f`; Gibbs aşımı tamamen kaybolur.
    * ``"lanczos"``: :math:`\sigma_n = \operatorname{sinc}(n/(N+1))` — aşımı yaklaşık on kat
      azaltır, Fejér'e göre daha keskin geçiş sağlar.

    Args:
        n_terms: Harmonik sayısı :math:`N`.
        method: Yumuşatma yöntemi.

    Returns:
        Uzunluğu :math:`N+1` olan ağırlık dizisi (:math:`\sigma_0 = 1`).
    """
    n = _validate_terms(n_terms)
    k = np.arange(n + 1, dtype=np.float64)
    if method == "none":
        return np.ones(n + 1)
    if method == "fejer":
        return 1.0 - k / (n + 1)
    if method == "lanczos":
        return np.sinc(k / (n + 1))
    raise SeriesError(
        f"Bilinmeyen yumuşatma yöntemi: {method!r}. Geçerli yöntemler: {SMOOTHING_METHODS}."
    )


# ---------------------------------------------------------------------- katsayılar
def _domain_of(
    func: Callable[[FloatArray], FloatArray],
    period: float | None,
    t_start: float | None,
    breakpoints: Iterable[float] | None,
) -> tuple[float, float, tuple[float, ...]]:
    """Fonksiyonun (varsa) ``period``/``t_start``/``breakpoints`` özniteliklerini kullanır."""
    resolved_period = validate_period(
        period if period is not None else getattr(func, "period", TWO_PI)
    )
    resolved_start = float(
        t_start if t_start is not None else getattr(func, "t_start", -resolved_period / 2.0)
    )
    resolved_bps = tuple(
        breakpoints if breakpoints is not None else getattr(func, "breakpoints", ())
    )
    return resolved_period, resolved_start, resolved_bps


def default_samples(n_terms: int) -> int:
    """``n_terms`` harmonik için güvenli varsayılan düğüm sayısı: ``max(4096, 16 (N+1))``."""
    return max(4096, 16 * (int(n_terms) + 1))


def evaluate_finite(func: Callable[[FloatArray], FloatArray], t: FloatArray) -> FloatArray:
    """Fonksiyonu değerlendirir; sonucu ``t`` biçimine yayar ve sonlu olduğunu doğrular.

    Raises:
        SeriesError: Fonksiyon NaN/sonsuz veya karmaşık değer üretirse.
    """
    with np.errstate(all="ignore"):
        raw = np.asarray(func(t))
    if np.iscomplexobj(raw):
        if np.any(np.abs(raw.imag) > 1e-12):
            raise SeriesError("Fonksiyon karmaşık değerler üretiyor; gerçek değer bekleniyordu.")
        raw = raw.real
    values = np.broadcast_to(raw.astype(np.float64, copy=False), t.shape).copy()
    if not np.all(np.isfinite(values)):
        bad = t[~np.isfinite(values)]
        raise SeriesError(
            "Fonksiyon bazı noktalarda tanımsız (NaN veya sonsuz) değer üretiyor; "
            f"ilk sorunlu nokta t ≈ {bad[0]:.4g}."
        )
    return values


def real_coefficients(
    func: Callable[[FloatArray], FloatArray],
    n_terms: int,
    *,
    period: float | None = None,
    t_start: float | None = None,
    breakpoints: Iterable[float] | None = None,
    method: QuadratureMethod = "gauss",
    samples: int | None = None,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
) -> RealCoefficients:
    r"""Periyodik bir fonksiyonun gerçek Fourier katsayılarını sayısal integralle hesaplar.

    .. math::

        a_n \approx \frac{2}{T}\sum_k w_k f(t_k)\cos(n\omega_0 t_k), \qquad
        b_n \approx \frac{2}{T}\sum_k w_k f(t_k)\sin(n\omega_0 t_k).

    Tüm harmonikler, bellek sınırlı parçalar hâlinde tek matris çarpımıyla hesaplanır.
    ``func`` üzerinde ``period``, ``t_start`` veya ``breakpoints`` öznitelikleri varsa
    (ör. :class:`~fourier_viz.core.signals.PeriodicSignal`) varsayılan olarak onlar kullanılır.

    Args:
        func: Vektörleştirilmiş, gerçek değerli, :math:`T`-periyodik fonksiyon.
        n_terms: En yüksek harmonik :math:`N` (``0`` ise yalnızca :math:`a_0`).
        period: Periyot :math:`T` (varsayılan :math:`2\pi`).
        t_start: İntegral aralığının başı (varsayılan :math:`-T/2`).
        breakpoints: Süreksizlik noktaları (Gauss/Simpson parçalamasında kullanılır).
        method: ``"gauss"``, ``"trapezoid"``, ``"simpson"`` veya ``"quad"``.
        samples: Düğüm sayısı (hassasiyet); ``None`` ise :func:`default_samples`.
        epsabs: ``quad`` için mutlak tolerans.
        epsrel: ``quad`` için göreli tolerans.

    Returns:
        :class:`RealCoefficients`.

    Raises:
        SeriesError: Geçersiz terim sayısı veya fonksiyon sonlu olmayan değer üretirse.
    """
    n = _validate_terms(n_terms)
    T, t0, bps = _domain_of(func, period, t_start, breakpoints)
    omega0 = TWO_PI / T

    if method == "quad":

        def checked(x: FloatArray) -> FloatArray:
            return evaluate_finite(func, x)

        cos_part, sin_part = quad_fourier_integrals(
            checked, n, t0, T, bps, epsabs=epsabs, epsrel=epsrel
        )
        return RealCoefficients(2.0 / T * cos_part, 2.0 / T * sin_part, T)

    quad = make_quadrature(method, t0, T, samples or default_samples(n), bps)
    weighted = quad.weights * evaluate_finite(func, quad.nodes)
    a = np.empty(n + 1)
    b = np.empty(n + 1)
    harmonics = np.arange(n + 1, dtype=np.float64)
    for rows in _row_chunks(n + 1, quad.size):
        phase = np.outer(harmonics[rows], omega0 * quad.nodes)
        a[rows] = np.cos(phase) @ weighted
        b[rows] = np.sin(phase) @ weighted
    return RealCoefficients(2.0 / T * a, 2.0 / T * b, T)


def trig_polynomial(
    a: Sequence[float] | FloatArray,
    b: Sequence[float] | FloatArray,
    period: float = TWO_PI,
) -> Callable[[FloatArray], FloatArray]:
    r"""Verilen katsayılardan sonlu trigonometrik polinom fonksiyonu üretir.

    :math:`p(t) = a_0/2 + \sum_{n=1}^N [a_n\cos(n\omega_0 t) + b_n\sin(n\omega_0 t)]`.
    """
    coeffs = RealCoefficients(_as_float_array(a), _as_float_array(b), period)

    def poly(t: FloatArray) -> FloatArray:
        return partial_sum(coeffs, t)

    return poly


# ---------------------------------------------------------------------- kısmi toplamlar
def _resolve_terms(coeffs: RealCoefficients, n_terms: int | None) -> int:
    if n_terms is None:
        return coeffs.n_terms
    n = _validate_terms(n_terms)
    if n > coeffs.n_terms:
        raise SeriesError(
            f"İstenen terim sayısı ({n}) mevcut katsayı sayısını ({coeffs.n_terms}) aşıyor."
        )
    return n


def partial_sum(
    coeffs: RealCoefficients,
    t: ArrayLike,
    n_terms: int | None = None,
    smoothing: Smoothing = "none",
) -> FloatArray:
    r"""Kısmi toplam :math:`S_N(t)` (isteğe bağlı :math:`\sigma` yumuşatmalı).

    .. math::

        S_N(t) = \frac{a_0}{2} + \sum_{n=1}^{N}\sigma_n
            \left[a_n\cos(n\omega_0 t) + b_n\sin(n\omega_0 t)\right]

    Hesap vektörleştirilmiştir: :math:`\cos(n\omega_0 t)` matrisleri bellek sınırlı parçalar
    hâlinde oluşturulup katsayı vektörüyle çarpılır.

    Args:
        coeffs: Katsayılar.
        t: Zaman noktaları (herhangi bir biçimde dizi).
        n_terms: Kullanılacak harmonik sayısı :math:`N` (varsayılan: tümü).
        smoothing: ``"none"``, ``"fejer"`` veya ``"lanczos"``.

    Returns:
        ``t`` ile aynı biçimde :math:`S_N(t)` değerleri.
    """
    n = _resolve_terms(coeffs, n_terms)
    t_arr = _as_float_array(t)
    flat = t_arr.ravel()
    result = np.full(flat.shape, coeffs.a[0] / 2.0)
    if n > 0:
        sigma = sigma_factors(n, smoothing)[1:]
        a = coeffs.a[1 : n + 1] * sigma
        b = coeffs.b[1 : n + 1] * sigma
        harmonics = np.arange(1, n + 1, dtype=np.float64)
        wt = coeffs.omega0 * flat
        for rows in _row_chunks(n, flat.size):
            phase = np.outer(harmonics[rows], wt)
            result += a[rows] @ np.cos(phase) + b[rows] @ np.sin(phase)
    return result.reshape(t_arr.shape)


def harmonic_terms(
    coeffs: RealCoefficients,
    t: ArrayLike,
    n_terms: int | None = None,
    smoothing: Smoothing = "none",
) -> FloatArray:
    r"""Her harmoniği ayrı ayrı döndürür.

    Satır 0 sabit terim :math:`a_0/2`, satır :math:`n \ge 1` ise
    :math:`\sigma_n[a_n\cos(n\omega_0 t) + b_n\sin(n\omega_0 t)]` olur. Satırların toplamı
    :func:`partial_sum` sonucuna eşittir.

    Returns:
        ``(N + 1, len(t))`` biçiminde dizi.
    """
    n = _resolve_terms(coeffs, n_terms)
    flat = _as_float_array(t).ravel()
    sigma = sigma_factors(n, smoothing)
    harmonics = np.arange(n + 1, dtype=np.float64)[:, None]
    phase = harmonics * (coeffs.omega0 * flat)[None, :]
    terms = (coeffs.a[: n + 1] * sigma)[:, None] * np.cos(phase) + (coeffs.b[: n + 1] * sigma)[
        :, None
    ] * np.sin(phase)
    terms[0] = coeffs.a[0] / 2.0
    return terms


def partial_sums(
    coeffs: RealCoefficients,
    t: ArrayLike,
    n_values: Sequence[int],
    smoothing: Smoothing = "none",
) -> FloatArray:
    r"""Birden çok :math:`N` için kısmi toplamları tek seferde hesaplar.

    Harmonik matrisi :math:`H_{n}(t)` (:func:`harmonic_terms`) bir kez hesaplanır; her
    :math:`N_i` için ağırlık satırı :math:`W_{i,n} = \sigma_n(N_i)` (:math:`n \le N_i`, aksi
    hâlde 0) kurulur ve tüm kısmi toplamlar tek bir matris çarpımıyla elde edilir:
    :math:`S = W H`. Yakınsama animasyonları ve hata çalışmaları için idealdir.

    Returns:
        ``(len(n_values), len(t))`` biçiminde dizi.
    """
    flat = _as_float_array(t).ravel()
    ns = [_resolve_terms(coeffs, n) for n in n_values]
    if not ns:
        return np.empty((0, flat.size))
    n_max = max(ns)
    weights = np.zeros((len(ns), n_max + 1))
    for row, n in enumerate(ns):
        weights[row, : n + 1] = sigma_factors(n, smoothing)
    result: FloatArray = weights @ harmonic_terms(coeffs, flat, n_max)
    return result
