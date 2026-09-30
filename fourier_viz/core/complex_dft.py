r"""Karmaşık Fourier katsayıları, ayrık Fourier dönüşümü (DFT) ve epicycle'lar.

**Karmaşık form.** Euler formülü :math:`e^{i\theta} = \cos\theta + i\sin\theta` ile

.. math::

    f(t) \sim \sum_{n=-\infty}^{\infty} c_n e^{i n\omega_0 t}, \qquad
    c_n = \frac{1}{T}\int_T f(t)\, e^{-i n\omega_0 t}\,dt .

Gerçek formla ilişki: :math:`c_0 = a_0/2`, :math:`c_n = (a_n - i b_n)/2`,
:math:`c_{-n} = (a_n + i b_n)/2` ve tersine :math:`a_n = c_n + c_{-n}`,
:math:`b_n = i(c_n - c_{-n})`. Gerçek bir :math:`f` için :math:`c_{-n} = \overline{c_n}`
(Hermitsel simetri).

**DFT.** :math:`M` örnekli :math:`z_m` dizisi için

.. math::

    X_k = \sum_{m=0}^{M-1} z_m\, e^{-2\pi i k m / M}, \qquad
    z_m = \frac{1}{M}\sum_{k=0}^{M-1} X_k\, e^{2\pi i k m / M}.

**Epicycle'lar.** Kapalı bir yolun noktaları :math:`z_m = x_m + i y_m`,
:math:`s_m = m/M` "zamanlarında" örneklenmiş kabul edilir. :math:`c_k = X_k/M` ve işaretli
frekans :math:`k \in [-M/2, M/2)` ile

.. math::

    z(s) = \sum_k c_k\, e^{2\pi i k s}, \qquad s \in [0, 1),

her terim yarıçapı :math:`|c_k|`, başlangıç açısı :math:`\arg c_k` olan ve periyot başına
:math:`k` tur atan bir çemberdir. Çemberler uç uca eklendiğinde son ucun izi şekli çizer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike

from fourier_viz.core.integration import (
    QuadratureMethod,
    make_quadrature,
    validate_period,
)
from fourier_viz.core.series import (
    TWO_PI,
    RealCoefficients,
    SeriesError,
    _domain_of,
    _row_chunks,
    _validate_terms,
    default_samples,
    real_coefficients,
)
from fourier_viz.core.types import ComplexArray, FloatArray, IntArray

DFTMethod = Literal["fft", "direct"]
"""``"fft"``: :func:`numpy.fft.fft`; ``"direct"``: :math:`O(M^2)` tanım tabanlı uygulama."""

MAX_DIRECT_DFT: Final[int] = 20_000
"""Doğrudan (O(M²)) DFT için izin verilen en büyük örnek sayısı."""


# ====================================================================== katsayılar
@dataclass(frozen=True, eq=False)
class ComplexCoefficients:
    r"""Karmaşık Fourier katsayıları :math:`c_{-N}, \dots, c_0, \dots, c_N`.

    Attributes:
        c: Uzunluğu :math:`2N+1` olan dizi; ``c[j]`` :math:`c_{j-N}`'dir.
        period: Periyot :math:`T`.
    """

    c: ComplexArray
    period: float = field(default=TWO_PI)

    def __post_init__(self) -> None:
        c = np.array(self.c, dtype=np.complex128, copy=True).ravel()
        if c.size == 0 or c.size % 2 == 0:
            raise SeriesError(
                f"Karmaşık katsayı dizisinin uzunluğu tek olmalıdır (2N+1), verilen: {c.size}."
            )
        if not np.all(np.isfinite(c)):
            raise SeriesError("Katsayılar sonlu olmalıdır (NaN/sonsuz değer bulundu).")
        c.setflags(write=False)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "period", validate_period(self.period))

    @property
    def n_terms(self) -> int:
        """En yüksek harmonik :math:`N`."""
        return int((self.c.size - 1) // 2)

    @property
    def omega0(self) -> float:
        r"""Temel açısal frekans :math:`\omega_0 = 2\pi/T`."""
        return TWO_PI / self.period

    @property
    def frequencies(self) -> IntArray:
        """Harmonik indisleri ``-N..N``."""
        return np.arange(-self.n_terms, self.n_terms + 1, dtype=np.int64)

    def __getitem__(self, n: int) -> complex:
        """:math:`c_n` değerini döndürür (``|n| > N`` ise 0)."""
        if abs(n) > self.n_terms:
            return 0j
        return complex(self.c[n + self.n_terms])

    def magnitude(self) -> FloatArray:
        r""":math:`|c_n|` değerleri."""
        return np.abs(self.c)

    def phase(self) -> FloatArray:
        r""":math:`\arg c_n` değerleri (radyan)."""
        return np.angle(self.c)

    def energy(self) -> float:
        r"""Parseval enerjisi :math:`\sum_n |c_n|^2`."""
        return float(np.sum(np.abs(self.c) ** 2))

    def is_hermitian(self, atol: float = 1e-9) -> bool:
        r""":math:`c_{-n} = \overline{c_n}` (gerçek değerli sinyal) olup olmadığı."""
        return bool(np.allclose(self.c, np.conj(self.c[::-1]), atol=atol, rtol=1e-9))

    def shifted(self, tau: float) -> ComplexCoefficients:
        r"""Zaman kaymasının katsayılara etkisi: :math:`f(t-\tau) \to c_n e^{-in\omega_0\tau}`.

        Yalnızca fazlar değişir; genlikler :math:`|c_n|` korunur.
        """
        factor = np.exp(-1j * self.frequencies * self.omega0 * float(tau))
        return ComplexCoefficients(self.c * factor, self.period)

    def to_real(self, atol: float = 1e-9) -> RealCoefficients:
        """Gerçek forma dönüştürür (bkz. :func:`complex_to_real`)."""
        return complex_to_real(self, atol=atol)

    def evaluate(self, t: ArrayLike, n_terms: int | None = None) -> ComplexArray:
        r""":math:`\sum_{|n|\le N} c_n e^{in\omega_0 t}` toplamını hesaplar (karmaşık değerli)."""
        n = self.n_terms if n_terms is None else _validate_terms(n_terms, maximum=self.n_terms)
        t_arr = np.asarray(t, dtype=np.float64)
        flat = t_arr.ravel()
        coeffs = self.c[self.n_terms - n : self.n_terms + n + 1]
        freqs = np.arange(-n, n + 1, dtype=np.float64)
        result = np.zeros(flat.shape, dtype=np.complex128)
        for rows in _row_chunks(freqs.size, flat.size):
            result += coeffs[rows] @ np.exp(1j * np.outer(freqs[rows], self.omega0 * flat))
        return result.reshape(t_arr.shape)


def real_to_complex(coeffs: RealCoefficients) -> ComplexCoefficients:
    r"""Gerçek katsayıları karmaşık forma çevirir.

    .. math::

        c_0 = \frac{a_0}{2}, \qquad c_n = \frac{a_n - i b_n}{2}, \qquad
        c_{-n} = \frac{a_n + i b_n}{2} = \overline{c_n}.
    """
    positive = 0.5 * (coeffs.a - 1j * coeffs.b)
    positive[0] = coeffs.a[0] / 2.0
    full = np.concatenate([np.conj(positive[:0:-1]), positive])
    return ComplexCoefficients(full, coeffs.period)


def complex_to_real(coeffs: ComplexCoefficients, atol: float = 1e-9) -> RealCoefficients:
    r"""Karmaşık katsayıları gerçek forma çevirir.

    :math:`a_n = c_n + c_{-n}` ve :math:`b_n = i(c_n - c_{-n})`.

    Args:
        coeffs: Karmaşık katsayılar.
        atol: Hermitsel simetri için mutlak tolerans (gerçek değerli sinyal koşulu).

    Raises:
        SeriesError: Katsayılar gerçek bir sinyale ait değilse (Hermitsel değilse).
    """
    scale = max(1.0, float(np.max(np.abs(coeffs.c))))
    if not coeffs.is_hermitian(atol=atol * scale):
        raise SeriesError(
            "Katsayılar gerçek değerli bir sinyale ait değil (c₋ₙ ≠ conj(cₙ)); "
            "gerçek forma dönüştürülemez."
        )
    n = coeffs.n_terms
    pos = coeffs.c[n:]
    neg = coeffs.c[n::-1]
    a = (pos + neg).real
    b = (1j * (pos - neg)).real
    return RealCoefficients(a, b, coeffs.period)


def complex_coefficients(
    func: Callable[[FloatArray], ArrayLike],
    n_terms: int,
    *,
    period: float | None = None,
    t_start: float | None = None,
    breakpoints: Iterable[float] | None = None,
    method: QuadratureMethod = "gauss",
    samples: int | None = None,
) -> ComplexCoefficients:
    r"""(Gerçek ya da karmaşık değerli) bir fonksiyonun :math:`c_n` katsayılarını hesaplar.

    .. math::

        c_n \approx \frac{1}{T}\sum_k w_k f(t_k) e^{-in\omega_0 t_k}, \qquad |n| \le N.

    ``method="quad"`` için :math:`c_n(f) = c_n(\operatorname{Re} f) + i\,c_n(\operatorname{Im} f)`
    ayrışımı ve gerçek katsayılar üzerinden adaptif integral kullanılır.

    Args:
        func: Vektörleştirilmiş fonksiyon (karmaşık değer dönebilir).
        n_terms: En yüksek harmonik :math:`N`.
        period: Periyot (varsayılan: ``func.period`` ya da :math:`2\pi`).
        t_start: İntegral başlangıcı (varsayılan :math:`-T/2`).
        breakpoints: Süreksizlik noktaları.
        method: İntegral yöntemi.
        samples: Düğüm sayısı.

    Returns:
        :class:`ComplexCoefficients`.
    """
    n = _validate_terms(n_terms)
    T, t0, bps = _domain_of(func, period, t_start, breakpoints)  # type: ignore[arg-type]

    def values_at(x: FloatArray) -> ComplexArray:
        with np.errstate(all="ignore"):
            vals = np.broadcast_to(np.asarray(func(x), dtype=np.complex128), x.shape)
        if not np.all(np.isfinite(vals)):
            raise SeriesError(
                "Fonksiyon bazı noktalarda tanımsız (NaN veya sonsuz) değer üretiyor."
            )
        return vals

    if method == "quad":
        parts = [
            real_coefficients(part, n, period=T, t_start=t0, breakpoints=bps, method="quad")
            for part in (
                lambda x: values_at(x).real.astype(np.float64),
                lambda x: values_at(x).imag.astype(np.float64),
            )
        ]
        combined = real_to_complex(parts[0]).c + 1j * real_to_complex(parts[1]).c
        return ComplexCoefficients(combined, T)

    quad = make_quadrature(method, t0, T, samples or default_samples(n), bps)
    weighted = quad.weights * values_at(quad.nodes)
    freqs = np.arange(-n, n + 1, dtype=np.float64)
    c = np.empty(freqs.size, dtype=np.complex128)
    omega0 = TWO_PI / T
    for rows in _row_chunks(freqs.size, quad.size):
        c[rows] = np.exp(-1j * np.outer(freqs[rows], omega0 * quad.nodes)) @ weighted
    return ComplexCoefficients(c / T, T)


# ====================================================================== DFT
def _as_complex_vector(values: ArrayLike, name: str = "Girdi") -> ComplexArray:
    arr = np.asarray(values, dtype=np.complex128)
    if arr.ndim != 1:
        raise ValueError(f"{name} tek boyutlu olmalıdır (biçim: {arr.shape}).")
    if arr.size == 0:
        raise ValueError(f"{name} boş olamaz.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} NaN veya sonsuz değer içeriyor.")
    return arr


def _direct_transform(values: ComplexArray, sign: float) -> ComplexArray:
    """Tanımdan DFT: ``Σ_m v_m exp(sign·2πi·k·m/M)``.

    Yuvarlama hatasını azaltmak için üsteki ``k·m`` çarpımı ``mod M`` alınarak önceden
    hesaplanmış :math:`M` adet birim kökten (twiddle tablosu) seçilir.
    """
    size = values.size
    if size > MAX_DIRECT_DFT:
        raise ValueError(
            f"Doğrudan DFT en fazla {MAX_DIRECT_DFT} örnek için desteklenir; 'fft' kullanın."
        )
    twiddle = np.exp(sign * 2j * np.pi * np.arange(size) / size)
    m = np.arange(size, dtype=np.int64)
    out = np.empty(size, dtype=np.complex128)
    for rows in _row_chunks(size, size):
        k = m[rows]
        out[rows] = twiddle[np.outer(k, m) % size] @ values
    return out


def dft(z: ArrayLike, method: DFTMethod = "fft") -> ComplexArray:
    r"""Ayrık Fourier dönüşümü :math:`X_k = \sum_m z_m e^{-2\pi i k m/M}` (NumPy sözleşmesi).

    Args:
        z: Tek boyutlu (karmaşık) örnekler.
        method: ``"fft"`` (hızlı, :math:`O(M\log M)`) veya ``"direct"`` (eğitici, :math:`O(M^2)`).

    Returns:
        :math:`X_0, \dots, X_{M-1}`.
    """
    values = _as_complex_vector(z)
    if method == "fft":
        return np.fft.fft(values)
    if method == "direct":
        return _direct_transform(values, -1.0)
    raise ValueError(f"Bilinmeyen DFT yöntemi: {method!r} ('fft' veya 'direct').")


def idft(spectrum: ArrayLike, method: DFTMethod = "fft") -> ComplexArray:
    r"""Ters DFT :math:`z_m = \tfrac{1}{M}\sum_k X_k e^{2\pi i k m/M}`."""
    values = _as_complex_vector(spectrum, "Spektrum")
    if method == "fft":
        return np.fft.ifft(values)
    if method == "direct":
        return _direct_transform(values, 1.0) / values.size
    raise ValueError(f"Bilinmeyen DFT yöntemi: {method!r} ('fft' veya 'direct').")


def dft_frequencies(size: int) -> IntArray:
    """DFT çıktısının işaretli frekansları (``numpy.fft.fftfreq(M) * M`` ile aynı sıra)."""
    if size <= 0:
        raise ValueError("Örnek sayısı pozitif olmalıdır.")
    freqs: IntArray = np.rint(np.fft.fftfreq(size) * size).astype(np.int64)
    return freqs


# ====================================================================== epicycle'lar
def points_to_complex(points: ArrayLike) -> ComplexArray:
    """``(M, 2)`` biçimli ``(x, y)`` dizisini veya karmaşık diziyi :math:`x + iy` dizisine çevirir.

    Raises:
        ValueError: Biçim uygun değilse, dizi boşsa veya NaN/sonsuz değer varsa.
    """
    arr = np.asarray(points)
    if np.iscomplexobj(arr):
        return _as_complex_vector(arr, "Nokta dizisi")
    arr = arr.astype(np.float64)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return _as_complex_vector(arr[:, 0] + 1j * arr[:, 1], "Nokta dizisi")
    if arr.ndim == 1:
        return _as_complex_vector(arr, "Nokta dizisi")
    raise ValueError(f"Noktalar (M, 2) biçiminde olmalıdır (verilen: {arr.shape}).")


@dataclass(frozen=True)
class Epicycle:
    r"""Tek bir dönen çember (vektör): :math:`c_k e^{2\pi i k s}`.

    Attributes:
        frequency: Periyot başına tur sayısı :math:`k` (negatif: saat yönü).
        coefficient: Karmaşık katsayı :math:`c_k`.
    """

    frequency: int
    coefficient: complex

    @property
    def radius(self) -> float:
        r"""Yarıçap :math:`|c_k|`."""
        return abs(self.coefficient)

    @property
    def phase(self) -> float:
        r"""Başlangıç açısı :math:`\arg c_k` (radyan)."""
        return float(np.angle(self.coefficient))

    def position(self, s: float) -> complex:
        r"""Normalize zamanda vektörün ucu: :math:`c_k e^{2\pi i k s}`."""
        return complex(self.coefficient * np.exp(2j * np.pi * self.frequency * s))


@dataclass(frozen=True, eq=False)
class EpicycleSet:
    r"""Genliğe göre sıralanmış dönen çemberler zinciri.

    Zincir :math:`c_0` (DC terimi, şeklin ağırlık merkezi) noktasından başlar; ardından
    çemberler :math:`|c_k|` değerine göre azalan sırada uç uca eklenir.

    Attributes:
        center: DC terimi :math:`c_0`.
        frequencies: Sıralı frekanslar :math:`k` (0 hariç).
        coefficients: Sıralı katsayılar :math:`c_k`.
        n_samples: Katsayıların hesaplandığı örnek sayısı :math:`M`.
        total_energy: Kırpılmadan önceki tüm çemberlerin enerjisi :math:`\sum_{k\ne0}|c_k|^2`
            (verilmezse mevcut çemberlerden hesaplanır).
    """

    center: complex
    frequencies: IntArray
    coefficients: ComplexArray
    n_samples: int
    total_energy: float | None = None

    def __post_init__(self) -> None:
        freqs = np.array(self.frequencies, dtype=np.int64, copy=True).ravel()
        coeffs = np.array(self.coefficients, dtype=np.complex128, copy=True).ravel()
        if freqs.shape != coeffs.shape:
            raise ValueError("Frekans ve katsayı dizileri aynı uzunlukta olmalıdır.")
        freqs.setflags(write=False)
        coeffs.setflags(write=False)
        object.__setattr__(self, "frequencies", freqs)
        object.__setattr__(self, "coefficients", coeffs)
        object.__setattr__(self, "center", complex(self.center))
        if self.total_energy is None:
            object.__setattr__(self, "total_energy", float(np.sum(np.abs(coeffs) ** 2)))

    @property
    def n_circles(self) -> int:
        """Çember sayısı (DC terimi hariç)."""
        return int(self.frequencies.size)

    @property
    def radii(self) -> FloatArray:
        r"""Çember yarıçapları :math:`|c_k|` (azalan sırada)."""
        return np.abs(self.coefficients)

    @property
    def phases(self) -> FloatArray:
        r"""Başlangıç açıları :math:`\arg c_k`."""
        return np.angle(self.coefficients)

    def circles(self) -> list[Epicycle]:
        """Çemberleri :class:`Epicycle` listesi olarak döndürür."""
        return [
            Epicycle(int(k), complex(c))
            for k, c in zip(self.frequencies, self.coefficients, strict=True)
        ]

    def limit(self, n_circles: int | None) -> EpicycleSet:
        """Yalnızca en büyük ``n_circles`` çemberi tutan yeni küme (``None``: hepsi)."""
        if n_circles is None:
            return self
        if n_circles < 0:
            raise ValueError("Çember sayısı negatif olamaz.")
        k = min(int(n_circles), self.n_circles)
        return EpicycleSet(
            self.center,
            self.frequencies[:k],
            self.coefficients[:k],
            self.n_samples,
            self.total_energy,
        )

    def _rotations(self, s: ArrayLike) -> ComplexArray:
        s_arr = np.asarray(s, dtype=np.float64).ravel()
        rotations: ComplexArray = self.coefficients[None, :] * np.exp(
            2j * np.pi * np.outer(s_arr, self.frequencies.astype(np.float64))
        )
        return rotations

    def joints(self, s: ArrayLike) -> ComplexArray:
        r"""Zincirin tüm eklem noktaları.

        Args:
            s: Normalize zaman(lar) :math:`s \in [0, 1)`.

        Returns:
            ``(len(s), K + 1)`` biçiminde dizi; sütun 0 merkez :math:`c_0`, sütun :math:`j`
            ise ilk :math:`j` vektörün toplamıdır. Son sütun izi çizen uçtur.
        """
        vectors = self._rotations(s)
        joints = np.empty((vectors.shape[0], vectors.shape[1] + 1), dtype=np.complex128)
        joints[:, 0] = self.center
        joints[:, 1:] = self.center + np.cumsum(vectors, axis=1)
        return joints

    def evaluate(self, s: ArrayLike) -> ComplexArray:
        r"""Zincirin ucu :math:`z(s) = c_0 + \sum_j c_{k_j} e^{2\pi i k_j s}`."""
        s_arr = np.asarray(s, dtype=np.float64)
        flat = s_arr.ravel()
        result = np.full(flat.shape, self.center, dtype=np.complex128)
        freqs = self.frequencies.astype(np.float64)
        for cols in _row_chunks(freqs.size, flat.size):
            result += np.exp(2j * np.pi * np.outer(flat, freqs[cols])) @ self.coefficients[cols]
        return result.reshape(s_arr.shape)

    def energy_fraction(self) -> float:
        r"""Tutulan çemberlerin enerji oranı :math:`\sum_{tutulan}|c_k|^2 / \sum_{k\ne0}|c_k|^2`."""
        kept = float(np.sum(np.abs(self.coefficients) ** 2))
        total = float(self.total_energy or 0.0)
        if total == 0.0:
            return 1.0
        return min(1.0, kept / total)


def epicycles_from_points(
    points: ArrayLike,
    n_circles: int | None = None,
    method: DFTMethod = "fft",
) -> EpicycleSet:
    r"""Kapalı bir yolun noktalarından epicycle zinciri üretir.

    :math:`c_k = X_k / M` hesaplanır, :math:`k` işaretli frekanslara (:math:`[-M/2, M/2)`)
    eşlenir, DC terimi merkez olarak ayrılır ve kalan terimler :math:`|c_k|`'ye göre azalan
    sırada dizilir (eşitlikte küçük :math:`|k|` önce).

    Args:
        points: ``(M, 2)`` gerçek dizi ya da uzunluğu :math:`M` olan karmaşık dizi.
        n_circles: Tutulacak çember sayısı (``None``: :math:`M-1` çemberin tümü).
        method: DFT yöntemi.

    Returns:
        :class:`EpicycleSet`; tüm çemberlerle :math:`z(m/M) = z_m` olur.
    """
    z = points_to_complex(points)
    size = z.size
    coeffs = dft(z, method) / size
    freqs = dft_frequencies(size)
    mask = freqs != 0
    freqs_nz, coeffs_nz = freqs[mask], coeffs[mask]
    order = np.lexsort((np.abs(freqs_nz), -np.abs(coeffs_nz)))
    full = EpicycleSet(complex(coeffs[0]), freqs_nz[order], coeffs_nz[order], size)
    return full.limit(n_circles)
