r"""Periyodik sinyal kütüphanesi ve analitik Fourier katsayıları.

Tüm hazır sinyaller genlik :math:`A` ve periyot :math:`T` ile parametrelenir; temel aralık
:math:`t \in [-T/2, T/2)` ve bu aralıktaki faz :math:`x = \omega_0 t \in [-\pi, \pi)` ile
tanımlanır, ardından periyodik olarak genişletilir. Süreksizlik noktalarında sinyal, Fourier
serisinin de yakınsadığı **ortalama değeri** (Dirichlet koşulu) döndürür.

Sinyaller (:math:`x \in [-\pi, \pi)`) ve analitik katsayıları:

* **Kare dalga** :math:`A\,\operatorname{sgn}(x)` —
  :math:`b_n = 4A/(n\pi)` (tek :math:`n`), diğerleri 0.
* **Testere dişi** :math:`Ax/\pi` — :math:`b_n = 2A(-1)^{n+1}/(n\pi)`.
* **Üçgen dalga** :math:`A(1 - 2|x|/\pi)` — :math:`a_n = 8A/(n\pi)^2` (tek :math:`n`).
* **Yarım dalga doğrultulmuş sinüs** :math:`A\max(\sin x, 0)` — :math:`a_0 = 2A/\pi`,
  :math:`b_1 = A/2`, :math:`a_n = -2A/(\pi(n^2-1))` (çift :math:`n \ge 2`).
* **Tam dalga doğrultulmuş sinüs** :math:`A|\sin(x/2)|` — :math:`a_n = -4A/(\pi(4n^2-1))`.
* :math:`|\sin|` :math:`A|\sin x|` — :math:`a_n = -4A/(\pi(n^2-1))` (çift :math:`n`).
* **Darbe dizisi** (doluluk :math:`d`) :math:`A` (:math:`|x| < \pi d`), aksi hâlde 0 —
  :math:`a_0 = 2Ad`, :math:`a_n = 2A\sin(n\pi d)/(n\pi)`.
* **Parabolik dalga** :math:`A(x/\pi)^2` — :math:`a_0 = 2A/3`, :math:`a_n = 4A(-1)^n/(n\pi)^2`.

"Tam dalga doğrultulmuş sinüs" ile ":math:`|\sin|`" aynı dalga biçimidir; farkı seçilen
periyottur: ilkinde :math:`T` dalganın gerçek (temel) periyodu, ikincisinde ise dalga
:math:`T/2` ile tekrar eder. Bu yüzden :math:`|\sin(\omega_0 t)|` açılımında yalnızca **çift**
harmonikler görünür — periyot seçiminin spektruma etkisini gösteren eğitici bir örnek.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike

from fourier_viz.core.expression import ExpressionError, parse_expression
from fourier_viz.core.integration import QuadratureMethod, validate_period
from fourier_viz.core.series import TWO_PI, RealCoefficients, real_coefficients
from fourier_viz.core.types import FloatArray

AnalyticCoefficients = Callable[[int], RealCoefficients]
"""``N`` alıp ``a_0..a_N``, ``b_0..b_N`` döndüren analitik katsayı fonksiyonu."""

_EDGE_RTOL: Final[float] = 1e-12
_VALIDATION_POINTS: Final[int] = 4097


class SignalError(ValueError):
    """Sinyal tanımı/parametre hataları (Türkçe, kullanıcıya gösterilebilir)."""


# ====================================================================== yardımcılar
def wrap_to_period(t: ArrayLike, period: float, t_start: float) -> FloatArray:
    r"""Zamanı temel aralığa katlar: :math:`t_0 + ((t - t_0) \bmod T) \in [t_0, t_0 + T)`."""
    t_arr = np.asarray(t, dtype=np.float64)
    wrapped: FloatArray = t_start + np.mod(t_arr - t_start, period)
    return wrapped


def _phase(t: ArrayLike, period: float) -> FloatArray:
    r"""Temel faz :math:`x = \omega_0 t` değerini :math:`[-\pi, \pi)` aralığına katlar."""
    return TWO_PI * wrap_to_period(t, period, -period / 2.0) / period


def _near(x: FloatArray, value: float) -> np.ndarray:
    mask: np.ndarray = np.abs(x - value) <= _EDGE_RTOL * max(1.0, abs(value)) * 8
    return mask


def _harmonics(n_terms: int) -> FloatArray:
    return np.arange(n_terms + 1, dtype=np.float64)


def _check_amplitude(amplitude: float) -> float:
    value = float(amplitude)
    if not np.isfinite(value):
        raise SignalError("Genlik sonlu bir sayı olmalıdır.")
    return value


# ====================================================================== sinyal sınıfı
@dataclass(frozen=True, eq=False)
class Jump:
    """Bir süreksizlik noktası.

    Attributes:
        location: Süreksizliğin konumu :math:`t_d`.
        left: Soldan limit :math:`f(t_d^-)`.
        right: Sağdan limit :math:`f(t_d^+)`.
    """

    location: float
    left: float
    right: float

    @property
    def size(self) -> float:
        """İşaretli sıçrama :math:`f(t_d^+) - f(t_d^-)`."""
        return self.right - self.left

    @property
    def midpoint(self) -> float:
        """Fourier serisinin süreksizlikte yakınsadığı ortalama değer."""
        return 0.5 * (self.left + self.right)


@dataclass(frozen=True, eq=False)
class PeriodicSignal:
    r"""Periyodik, gerçek değerli bir sinyal.

    Attributes:
        func: Vektörleştirilmiş fonksiyon; tüm :math:`t` için tanımlı (zaten periyodik).
        period: Periyot :math:`T`.
        name: Makine adı (ör. ``"square"``).
        label: Türkçe görünen ad.
        t_start: Temel aralığın başlangıcı (varsayılan :math:`-T/2`).
        discontinuities: Temel aralıktaki sıçrama noktaları.
        kinks: Türevin süreksiz olduğu (kırılma) noktalar; integrali hassaslaştırır.
        analytic: Varsa analitik katsayı fonksiyonu.
        latex: Tanımın LaTeX gösterimi.
        params: Sinyali üreten parametreler (gösterim/önbellek için).
    """

    func: Callable[[FloatArray], FloatArray]
    period: float = TWO_PI
    name: str = "custom"
    label: str = "Özel sinyal"
    t_start: float | None = None
    discontinuities: tuple[float, ...] = ()
    kinks: tuple[float, ...] = ()
    analytic: AnalyticCoefficients | None = field(default=None, repr=False)
    latex: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        period = validate_period(self.period)
        object.__setattr__(self, "period", period)
        if self.t_start is None:
            object.__setattr__(self, "t_start", -period / 2.0)
        object.__setattr__(self, "discontinuities", tuple(float(d) for d in self.discontinuities))
        object.__setattr__(self, "kinks", tuple(float(k) for k in self.kinks))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    # ------------------------------------------------------------------ temel
    @property
    def start(self) -> float:
        """Temel aralığın başlangıcı :math:`t_0` (``float`` olarak)."""
        return float(self.t_start if self.t_start is not None else -self.period / 2.0)

    @property
    def omega0(self) -> float:
        r"""Temel açısal frekans :math:`\omega_0 = 2\pi/T`."""
        return TWO_PI / self.period

    @property
    def breakpoints(self) -> tuple[float, ...]:
        """İntegralde parça sınırı olarak kullanılan noktalar (süreksizlik + kırılma)."""
        return tuple(sorted({*self.discontinuities, *self.kinks}))

    @property
    def is_continuous(self) -> bool:
        """Bilinen bir sıçrama noktası yoksa ``True``."""
        return not self.discontinuities

    @property
    def has_analytic(self) -> bool:
        """Analitik katsayılar biliniyor mu."""
        return self.analytic is not None

    def __call__(self, t: ArrayLike) -> FloatArray:
        """Sinyali ``t`` noktalarında değerlendirir (``t`` ile aynı biçimde)."""
        t_arr = np.asarray(t, dtype=np.float64)
        with np.errstate(all="ignore"):
            values = np.asarray(self.func(t_arr), dtype=np.float64)
        return np.broadcast_to(values, t_arr.shape).copy()

    def sample(self, n_points: int = 1000, periods: float = 1.0) -> tuple[FloatArray, FloatArray]:
        """Sinyali ``periods`` periyot boyunca eşit aralıklı örnekler.

        Returns:
            ``(t, f(t))`` çifti; ``t``, :math:`t_0`'dan başlayıp :math:`t_0 + kT`'de biter.
        """
        if n_points < 2:
            raise SignalError("En az 2 örnek noktası gereklidir.")
        t = np.linspace(self.start, self.start + periods * self.period, int(n_points))
        return t, self(t)

    # ------------------------------------------------------------------ katsayılar
    def coefficients(
        self,
        n_terms: int,
        method: QuadratureMethod = "gauss",
        samples: int | None = None,
    ) -> RealCoefficients:
        """Katsayıları sayısal integralle hesaplar.

        Bkz. :func:`~fourier_viz.core.series.real_coefficients`; sinyalin periyodu, temel
        aralığı ve kırılma noktaları otomatik kullanılır.
        """
        return real_coefficients(
            self,
            n_terms,
            period=self.period,
            t_start=self.start,
            breakpoints=self.breakpoints,
            method=method,
            samples=samples,
        )

    def analytic_coefficients(self, n_terms: int) -> RealCoefficients:
        """Kapalı formdaki katsayılar.

        Raises:
            SignalError: Bu sinyal için analitik formül yoksa.
        """
        if self.analytic is None:
            raise SignalError(f"'{self.label}' için analitik katsayı formülü tanımlı değil.")
        if n_terms < 0:
            raise SignalError("Terim sayısı negatif olamaz.")
        return self.analytic(int(n_terms))

    def jumps(self, eps: float | None = None) -> list[Jump]:
        """Bilinen süreksizliklerdeki tek taraflı limitleri sayısal olarak hesaplar."""
        delta = eps if eps is not None else 1e-9 * self.period
        result: list[Jump] = []
        for loc in self.discontinuities:
            left, right = self(np.array([loc - delta, loc + delta]))
            result.append(Jump(loc, float(left), float(right)))
        return result

    # ------------------------------------------------------------------ dönüşümler
    def shifted(self, tau: float) -> PeriodicSignal:
        r"""Zamanda kaydırılmış sinyal :math:`g(t) = f(t - \tau)`.

        Katsayılar :math:`\theta_n = n\omega_0\tau` ile
        :math:`a'_n = a_n\cos\theta_n - b_n\sin\theta_n`,
        :math:`b'_n = a_n\sin\theta_n + b_n\cos\theta_n` olur; genlikler değişmez.
        """
        tau = float(tau)
        base = self
        analytic: AnalyticCoefficients | None = None
        if base.analytic is not None:
            base_analytic = base.analytic

            def analytic(n_terms: int) -> RealCoefficients:
                c = base_analytic(n_terms)
                theta = _harmonics(n_terms) * base.omega0 * tau
                a = c.a * np.cos(theta) - c.b * np.sin(theta)
                b = c.a * np.sin(theta) + c.b * np.cos(theta)
                return RealCoefficients(a, b, base.period)

        def func(t: FloatArray) -> FloatArray:
            return base(t - tau)

        return replace(
            self,
            func=func,
            name=f"{self.name}_shifted",
            label=f"{self.label} (τ={tau:g} kaydırılmış)",
            discontinuities=tuple(d + tau for d in self.discontinuities),
            kinks=tuple(k + tau for k in self.kinks),
            analytic=analytic,
            params={**self.params, "tau": tau},
        )

    def scaled(self, factor: float) -> PeriodicSignal:
        """Genliği ``factor`` ile çarpılmış sinyal."""
        k = float(factor)
        base = self
        analytic: AnalyticCoefficients | None = None
        if base.analytic is not None:
            base_analytic = base.analytic

            def analytic(n_terms: int) -> RealCoefficients:
                return base_analytic(n_terms) * k

        def func(t: FloatArray) -> FloatArray:
            return k * base(t)

        return replace(self, func=func, label=f"{k:g}·{self.label}", analytic=analytic)

    def __add__(self, other: PeriodicSignal) -> PeriodicSignal:
        if not isinstance(other, PeriodicSignal):
            return NotImplemented
        if not np.isclose(self.period, other.period, rtol=1e-12, atol=0.0):
            raise SignalError("Farklı periyotlu sinyaller toplanamaz.")
        lhs, rhs = self, other
        analytic: AnalyticCoefficients | None = None
        if lhs.analytic is not None and rhs.analytic is not None:
            la, ra = lhs.analytic, rhs.analytic

            def analytic(n_terms: int) -> RealCoefficients:
                return la(n_terms) + ra(n_terms)

        def func(t: FloatArray) -> FloatArray:
            return lhs(t) + rhs(t)

        return PeriodicSignal(
            func=func,
            period=self.period,
            name=f"{self.name}+{other.name}",
            label=f"{self.label} + {other.label}",
            t_start=self.start,
            discontinuities=(*self.discontinuities, *other.discontinuities),
            kinks=(*self.kinks, *other.kinks),
            analytic=analytic,
        )


# ====================================================================== hazır sinyaller
def square_wave(amplitude: float = 1.0, period: float = TWO_PI) -> PeriodicSignal:
    r"""Kare dalga :math:`f = A\operatorname{sgn}(x)`; :math:`b_n = 4A/(n\pi)` (tek :math:`n`)."""
    A = _check_amplitude(amplitude)
    T = validate_period(period)

    def func(t: FloatArray) -> FloatArray:
        x = _phase(t, T)
        return np.where(_near(x, -np.pi), 0.0, A * np.sign(x))

    def analytic(n_terms: int) -> RealCoefficients:
        n = _harmonics(n_terms)
        b = np.zeros_like(n)
        odd = n % 2 == 1
        b[odd] = 4.0 * A / (np.pi * n[odd])
        return RealCoefficients(np.zeros_like(n), b, T)

    return PeriodicSignal(
        func, T, "square", "Kare dalga",
        discontinuities=(-T / 2.0, 0.0),
        analytic=analytic,
        latex=r"f(t) = A\,\operatorname{sgn}(\sin \omega_0 t)",
        params={"amplitude": A, "period": T},
    )  # fmt: skip


def sawtooth_wave(amplitude: float = 1.0, period: float = TWO_PI) -> PeriodicSignal:
    r"""Testere dişi :math:`f = Ax/\pi`; :math:`b_n = 2A(-1)^{n+1}/(n\pi)`."""
    A = _check_amplitude(amplitude)
    T = validate_period(period)

    def func(t: FloatArray) -> FloatArray:
        x = _phase(t, T)
        return np.where(_near(x, -np.pi), 0.0, A * x / np.pi)

    def analytic(n_terms: int) -> RealCoefficients:
        n = _harmonics(n_terms)
        b = np.zeros_like(n)
        b[1:] = 2.0 * A * (-1.0) ** (n[1:] + 1) / (np.pi * n[1:])
        return RealCoefficients(np.zeros_like(n), b, T)

    return PeriodicSignal(
        func, T, "sawtooth", "Testere dişi",
        discontinuities=(-T / 2.0,),
        analytic=analytic,
        latex=r"f(t) = \frac{A}{\pi}\,\omega_0 t,\quad -\tfrac{T}{2} < t < \tfrac{T}{2}",
        params={"amplitude": A, "period": T},
    )  # fmt: skip


def triangle_wave(amplitude: float = 1.0, period: float = TWO_PI) -> PeriodicSignal:
    r"""Üçgen dalga :math:`f = A(1 - 2|x|/\pi)`; :math:`a_n = 8A/(n\pi)^2` (tek :math:`n`)."""
    A = _check_amplitude(amplitude)
    T = validate_period(period)

    def func(t: FloatArray) -> FloatArray:
        x = _phase(t, T)
        return A * (1.0 - 2.0 * np.abs(x) / np.pi)

    def analytic(n_terms: int) -> RealCoefficients:
        n = _harmonics(n_terms)
        a = np.zeros_like(n)
        odd = n % 2 == 1
        a[odd] = 8.0 * A / (np.pi * n[odd]) ** 2
        return RealCoefficients(a, np.zeros_like(n), T)

    return PeriodicSignal(
        func, T, "triangle", "Üçgen dalga",
        kinks=(-T / 2.0, 0.0),
        analytic=analytic,
        latex=r"f(t) = A\left(1 - \frac{2|\omega_0 t|}{\pi}\right),\quad |t| \le \tfrac{T}{2}",
        params={"amplitude": A, "period": T},
    )  # fmt: skip


def half_wave_rectified_sine(amplitude: float = 1.0, period: float = TWO_PI) -> PeriodicSignal:
    r"""Yarım dalga doğrultulmuş sinüs :math:`f = A\max(\sin x, 0)`.

    :math:`a_0 = 2A/\pi`, :math:`b_1 = A/2`, çift :math:`n \ge 2` için
    :math:`a_n = -2A/(\pi(n^2-1))`, diğerleri 0.
    """
    A = _check_amplitude(amplitude)
    T = validate_period(period)

    def func(t: FloatArray) -> FloatArray:
        return A * np.maximum(np.sin(_phase(t, T)), 0.0)

    def analytic(n_terms: int) -> RealCoefficients:
        n = _harmonics(n_terms)
        a = np.zeros_like(n)
        b = np.zeros_like(n)
        a[0] = 2.0 * A / np.pi
        even = (n % 2 == 0) & (n >= 2)
        a[even] = -2.0 * A / (np.pi * (n[even] ** 2 - 1.0))
        if n_terms >= 1:
            b[1] = A / 2.0
        return RealCoefficients(a, b, T)

    return PeriodicSignal(
        func, T, "half_rectified", "Yarım dalga doğrultulmuş sinüs",
        kinks=(-T / 2.0, 0.0),
        analytic=analytic,
        latex=r"f(t) = A\,\max(\sin \omega_0 t,\ 0)",
        params={"amplitude": A, "period": T},
    )  # fmt: skip


def full_wave_rectified_sine(amplitude: float = 1.0, period: float = TWO_PI) -> PeriodicSignal:
    r"""Tam dalga doğrultulmuş sinüs :math:`f = A|\sin(\pi t/T)|` (temel periyodu :math:`T`).

    :math:`a_n = -4A/(\pi(4n^2 - 1))` (böylece :math:`a_0 = 4A/\pi`), :math:`b_n = 0`.
    """
    A = _check_amplitude(amplitude)
    T = validate_period(period)

    def func(t: FloatArray) -> FloatArray:
        return A * np.abs(np.sin(_phase(t, T) / 2.0))

    def analytic(n_terms: int) -> RealCoefficients:
        n = _harmonics(n_terms)
        a = -4.0 * A / (np.pi * (4.0 * n**2 - 1.0))
        return RealCoefficients(a, np.zeros_like(n), T)

    return PeriodicSignal(
        func, T, "full_rectified", "Tam dalga doğrultulmuş sinüs",
        kinks=(-T / 2.0, 0.0),
        analytic=analytic,
        latex=r"f(t) = A\,\left|\sin\left(\frac{\pi t}{T}\right)\right|",
        params={"amplitude": A, "period": T},
    )  # fmt: skip


def abs_sine(amplitude: float = 1.0, period: float = TWO_PI) -> PeriodicSignal:
    r"""Mutlak sinüs :math:`f = A|\sin(\omega_0 t)|`.

    Yalnızca çift harmonikler: :math:`a_n = -4A/(\pi(n^2-1))` (çift :math:`n`).
    """
    A = _check_amplitude(amplitude)
    T = validate_period(period)

    def func(t: FloatArray) -> FloatArray:
        return A * np.abs(np.sin(_phase(t, T)))

    def analytic(n_terms: int) -> RealCoefficients:
        n = _harmonics(n_terms)
        a = np.zeros_like(n)
        even = n % 2 == 0
        a[even] = -4.0 * A / (np.pi * (n[even] ** 2 - 1.0))
        return RealCoefficients(a, np.zeros_like(n), T)

    return PeriodicSignal(
        func, T, "abs_sine", "|sin| (mutlak sinüs)",
        kinks=(-T / 2.0, 0.0),
        analytic=analytic,
        latex=r"f(t) = A\,|\sin \omega_0 t|",
        params={"amplitude": A, "period": T},
    )  # fmt: skip


def pulse_train(
    amplitude: float = 1.0, period: float = TWO_PI, duty: float = 0.25
) -> PeriodicSignal:
    r"""Darbe dizisi: :math:`|t| < dT/2` için :math:`A`, aksi hâlde 0 (doluluk oranı :math:`d`).

    :math:`a_0 = 2Ad`, :math:`a_n = \dfrac{2A}{n\pi}\sin(n\pi d)`, :math:`b_n = 0`.
    """
    A = _check_amplitude(amplitude)
    T = validate_period(period)
    d = float(duty)
    if not 0.0 < d < 1.0:
        raise SignalError(
            f"Doluluk oranı 0 ile 1 arasında (uçlar hariç) olmalıdır (verilen: {duty})."
        )
    edge = np.pi * d

    def func(t: FloatArray) -> FloatArray:
        x = np.abs(_phase(t, T))
        values = np.where(x < edge, A, 0.0)
        return np.where(_near(x, edge), A / 2.0, values)

    def analytic(n_terms: int) -> RealCoefficients:
        n = _harmonics(n_terms)
        a = np.empty_like(n)
        a[0] = 2.0 * A * d
        a[1:] = 2.0 * A * np.sin(n[1:] * np.pi * d) / (n[1:] * np.pi)
        return RealCoefficients(a, np.zeros_like(n), T)

    half_width = d * T / 2.0
    return PeriodicSignal(
        func, T, "pulse", "Darbe dizisi",
        discontinuities=(-half_width, half_width),
        analytic=analytic,
        latex=r"f(t) = \begin{cases} A, & |t| < dT/2 \\ 0, & dT/2 < |t| \le T/2 \end{cases}",
        params={"amplitude": A, "period": T, "duty": d},
    )  # fmt: skip


def parabolic_wave(amplitude: float = 1.0, period: float = TWO_PI) -> PeriodicSignal:
    r"""Parabolik dalga :math:`f = A(x/\pi)^2`.

    :math:`a_0 = 2A/3`, :math:`a_n = 4A(-1)^n/(n\pi)^2`, :math:`b_n = 0`.
    """
    A = _check_amplitude(amplitude)
    T = validate_period(period)

    def func(t: FloatArray) -> FloatArray:
        return A * (_phase(t, T) / np.pi) ** 2

    def analytic(n_terms: int) -> RealCoefficients:
        n = _harmonics(n_terms)
        a = np.empty_like(n)
        a[0] = 2.0 * A / 3.0
        a[1:] = 4.0 * A * (-1.0) ** n[1:] / (np.pi * n[1:]) ** 2
        return RealCoefficients(a, np.zeros_like(n), T)

    return PeriodicSignal(
        func, T, "parabolic", "Parabolik dalga",
        kinks=(-T / 2.0,),
        analytic=analytic,
        latex=r"f(t) = A\left(\frac{2t}{T}\right)^2,\quad |t| \le \tfrac{T}{2}",
        params={"amplitude": A, "period": T},
    )  # fmt: skip


# ====================================================================== özel sinyaller
MAX_DETECTED_JUMPS: Final[int] = 64
""":func:`detect_jumps` ile aranacak en fazla aday sıçrama sayısı."""


def _scalar(func: Callable[[FloatArray], FloatArray], x: float) -> float:
    with np.errstate(all="ignore"):
        return float(np.asarray(func(np.array([x], dtype=np.float64)), dtype=np.float64).ravel()[0])


def _check_bounded(
    func: Callable[[FloatArray], FloatArray],
    lo: float,
    hi: float,
    magnitude: float,
    zooms: int = 12,
    points: int = 21,
) -> None:
    """``[lo, hi]`` içinde ``|f|``'nin en büyük olduğu yere 10'ar kat yakınlaşarak tekillik arar.

    Kutup tipi tekilliklerde (``1/x``, ``1/x²``, ``tan``) değer her yakınlaşmada katlanarak büyür;
    integre edilebilir logaritmik tekilliklerde (``log|x|``) ise yalnızca yavaşça artar.

    Raises:
        SignalError: Değer ızgara büyüklüğünün :math:`10^6` katını aşarsa veya tanımsızsa.
    """
    center, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    peaks: list[float] = []
    for _ in range(zooms):
        xs = np.linspace(center - half, center + half, points)
        with np.errstate(all="ignore"):
            fx = np.abs(np.broadcast_to(np.asarray(func(xs), dtype=np.float64), xs.shape))
        if not np.all(np.isfinite(fx)):
            raise SignalError(
                f"Fonksiyon t ≈ {xs[~np.isfinite(fx)][0]:.4g} noktasında tanımsız (NaN/sonsuz) "
                "değer üretiyor (tekillik)."
            )
        j = int(np.argmax(fx))
        center = float(xs[j])
        peaks.append(float(fx[j]))
        half /= 10.0
    # Kutupta değer büyümeye devam eder; sonlu (ama dar ve yüksek) tepelerde doyuma ulaşır.
    still_growing = peaks[-1] > 5.0 * peaks[max(0, len(peaks) - 4)]
    if peaks[-1] > 1e6 * magnitude and still_growing:
        raise SignalError(
            f"Fonksiyon t ≈ {center:.4g} civarında sınırsız büyüyor (tekillik); bu sinyalin "
            "Fourier katsayıları tanımlı değil. Tanım aralığını veya ifadeyi değiştirin."
        )


def detect_jumps(
    func: Callable[[FloatArray], FloatArray],
    t_start: float,
    period: float,
    samples: int = 8192,
) -> tuple[float, ...]:
    r"""Temel aralığın **içindeki** sıçrama (süreksizlik) noktalarını bulur.

    Yöntem: aralık ``samples`` alt aralığa bölünüp ardışık değer farkları taranır; olağan
    farkların çok üstündeki her aday aralık, farkın büyük kaldığı yarıya doğru ikiye bölünerek
    :math:`\sim 10^{-13}T` genişliğe daraltılır. Gerçek bir sıçramada fark daralan aralıkta
    **sabit kalır**; dik ama sürekli bölgelerde (ör. ``tanh(50*t)``) veya sonsuz eğimli
    kırılmalarda (ör. ``cbrt(t)``) küçülür. Bu yüzden fark, iki farklı ölçekte
    (:math:`10^{-10}T` ve :math:`10^{-13}T`) neredeyse aynıysa sıçrama kabul edilir.

    Args:
        func: Vektörleştirilmiş fonksiyon (temel aralıkta).
        t_start: Temel aralığın başlangıcı :math:`t_0`.
        period: Periyot :math:`T`.
        samples: Tarama alt aralığı sayısı.

    Returns:
        :math:`(t_0, t_0 + T)` içinde artan sırada sıçrama konumları (en fazla
        :data:`MAX_DETECTED_JUMPS`; uç noktalardaki periyodik sıçrama dahil değildir).

    Raises:
        SignalError: Fonksiyon bir noktada sınırsız büyüyorsa (integre edilemeyen tekillik,
            ör. ``1/(t-1)``) veya tanımsız değer üretiyorsa.
    """
    T = validate_period(period)
    t = np.linspace(t_start, t_start + T, int(samples) + 1)
    with np.errstate(all="ignore"):
        v = np.broadcast_to(np.asarray(func(t), dtype=np.float64), t.shape)
    if not np.all(np.isfinite(v)):
        bad = t[~np.isfinite(v)][0]
        raise SignalError(
            f"Fonksiyon t ≈ {bad:.4g} noktasında tanımsız (NaN/sonsuz) değer üretiyor."
        )
    diffs = np.abs(np.diff(v))
    # Sağlam büyüklük ölçüsü: ızgara bir kutba tam denk gelse bile şişmeyen yüzdelik değer.
    magnitude = max(1.0, float(np.percentile(np.abs(v), 99.5)))
    scale = max(1.0, float(np.ptp(v)))
    threshold = max(10.0 * float(np.median(diffs)), 1e-9 * scale)
    candidates = np.nonzero(diffs > threshold)[0]
    if candidates.size > MAX_DETECTED_JUMPS:
        strongest = np.argsort(diffs[candidates])[::-1][:MAX_DETECTED_JUMPS]
        candidates = candidates[strongest]
    coarse_width = 1e-10 * max(1.0, T)
    fine_width = 1e-13 * max(1.0, abs(t_start), abs(t_start + T), T)
    found: list[float] = []
    for i in sorted(int(c) for c in candidates):
        _check_bounded(func, float(t[max(i - 1, 0)]), float(t[min(i + 2, t.size - 1)]), magnitude)
        lo, hi, f_lo, f_hi = float(t[i]), float(t[i + 1]), float(v[i]), float(v[i + 1])
        coarse_jump: float | None = None
        while hi - lo > fine_width:
            mid = 0.5 * (lo + hi)
            if not lo < mid < hi:
                break
            f_mid = _scalar(func, mid)
            if not np.isfinite(f_mid):
                raise SignalError(
                    f"Fonksiyon t ≈ {mid:.4g} noktasında tanımsız (NaN/sonsuz) değer üretiyor."
                )
            if abs(f_mid - f_lo) >= abs(f_hi - f_mid):
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
            if coarse_jump is None and hi - lo <= coarse_width:
                coarse_jump = abs(f_hi - f_lo)
        location = 0.5 * (lo + hi)
        jump = abs(f_hi - f_lo)
        is_jump = jump > 1e-6 * scale and (coarse_jump is None or jump >= 0.9 * coarse_jump)
        near_edge = min(location - t_start, t_start + T - location) <= 1e-9 * T
        if is_jump and not near_edge and (not found or location - found[-1] > 1e-9 * T):
            found.append(location)
    return tuple(found)


def from_function(
    func: Callable[[FloatArray], FloatArray],
    period: float = TWO_PI,
    t_start: float | None = None,
    *,
    name: str = "custom",
    label: str = "Özel fonksiyon",
    discontinuities: Iterable[float] = (),
    kinks: Iterable[float] = (),
) -> PeriodicSignal:
    """Bir periyotta tanımlı fonksiyonu periyodik olarak genişletir.

    ``func`` yalnızca temel aralık :math:`[t_0, t_0 + T)` üzerinde çağrılır. Uçlardaki değerler
    farklıysa (periyodik genişlemede sıçrama) :math:`t_0` noktası süreksizlik olarak eklenir.
    Sınırdaki ve verilen iç süreksizlik noktalarında (Dirichlet sözleşmesi gereği) sol ve sağ
    limitlerin ortalaması döndürülür.
    """
    T = validate_period(period)
    t0 = float(-T / 2.0 if t_start is None else t_start)
    with np.errstate(all="ignore"):
        ends = np.asarray(func(np.array([t0, t0 + T])), dtype=np.float64)
    left_end, right_end = float(ends[0]), float(ends[1])
    boundary_jump = bool(np.isfinite(ends).all() and not np.isclose(left_end, right_end))
    mid = 0.5 * (left_end + right_end)
    eps = 1e-9 * T
    interior_midpoints = [
        (float(d), 0.5 * (_scalar(func, float(d) - eps) + _scalar(func, float(d) + eps)))
        for d in discontinuities
        if t0 < float(d) < t0 + T
    ]

    def periodic(t: FloatArray) -> FloatArray:
        wrapped = wrap_to_period(t, T, t0)
        values = np.asarray(func(wrapped), dtype=np.float64)
        values = np.broadcast_to(values, wrapped.shape)
        if boundary_jump:
            values = np.where(_near(wrapped, t0), mid, values)
        for location, average in interior_midpoints:
            values = np.where(_near(wrapped, location), average, values)
        return values

    disc = tuple(discontinuities)
    if boundary_jump and not any(np.isclose(d, t0) for d in disc):
        disc = (t0, *disc)
    return PeriodicSignal(
        periodic, T, name, label, t0, disc, tuple(kinks), None, "", {"period": T, "t_start": t0}
    )


def from_expression(
    text: str,
    period: float = TWO_PI,
    t_start: float | None = None,
    *,
    validation_points: int = _VALIDATION_POINTS,
) -> PeriodicSignal:
    """Kullanıcı ifadesinden (güvenli ayrıştırıcıyla) periyodik sinyal üretir.

    İfade temel aralık :math:`[t_0, t_0+T)` üzerinde tanımlıdır ve periyodik olarak genişletilir.
    Oluşturma sırasında aralık, ``validation_points`` noktalı (uçlar ve 0 dahil) bir ızgarada
    değerlendirilerek NaN/sonsuz değer olmadığı doğrulanır; ardından :func:`detect_jumps` ile iç
    sıçramalar bulunur (ör. ``where(t > 0, 1, -1)``, ``floor(t)``). Bulunan sıçramalar integralde
    parça sınırı olur (hassasiyet) ve Gibbs analizi için kullanılır.

    Raises:
        SignalError: İfade geçersizse, tanımsız değer üretiyorsa veya integre edilemeyen bir
            tekilliği varsa (Türkçe açıklama ile).
    """
    try:
        expr = parse_expression(text)
    except ExpressionError as exc:
        raise SignalError(str(exc)) from exc
    T = validate_period(period)
    t0 = float(-T / 2.0 if t_start is None else t_start)
    grid = np.linspace(t0, t0 + T, max(3, int(validation_points)))
    grid = np.union1d(grid, [0.0]) if t0 <= 0.0 <= t0 + T else grid
    try:
        expr.evaluate(grid)
    except ExpressionError as exc:
        raise SignalError(str(exc)) from exc
    jumps = detect_jumps(expr, t0, T)
    signal = from_function(
        expr, T, t0, name="expression", label=f"f(t) = {expr.source}", discontinuities=jumps
    )
    return replace(
        signal,
        latex=f"f(t) = {expr.source}",
        params={"expression": expr.source, "period": T, "t_start": t0},
    )


# ====================================================================== kayıt defteri
@dataclass(frozen=True)
class ParamSpec:
    """Arayüzde gösterilecek bir sinyal parametresinin tanımı."""

    name: str
    label: str
    default: float
    minimum: float
    maximum: float
    step: float


@dataclass(frozen=True)
class SignalInfo:
    """Kütüphanedeki bir sinyalin tanımı."""

    key: str
    label: str
    factory: Callable[..., PeriodicSignal]
    params: tuple[ParamSpec, ...]
    description: str


_AMPLITUDE = ParamSpec("amplitude", "Genlik A", 1.0, 0.1, 5.0, 0.1)
_PERIOD = ParamSpec("period", "Periyot T", float(TWO_PI), 0.5, 20.0, 0.1)
_DUTY = ParamSpec("duty", "Doluluk oranı d", 0.25, 0.05, 0.95, 0.05)

SIGNAL_LIBRARY: Final[Mapping[str, SignalInfo]] = MappingProxyType(
    {
        info.key: info
        for info in (
            SignalInfo("square", "Kare dalga", square_wave, (_AMPLITUDE, _PERIOD),
                       "Tek harmonikler, 1/n ile azalan genlik; Gibbs etkisinin klasik örneği."),
            SignalInfo("sawtooth", "Testere dişi", sawtooth_wave, (_AMPLITUDE, _PERIOD),
                       "Tüm harmonikler, 1/n ile azalan ve işaret değiştiren sinüs katsayıları."),
            SignalInfo("triangle", "Üçgen dalga", triangle_wave, (_AMPLITUDE, _PERIOD),
                       "Sürekli sinyal: katsayılar 1/n² ile hızla azalır, Gibbs etkisi yoktur."),
            SignalInfo("half_rectified", "Yarım dalga doğrultulmuş sinüs",
                       half_wave_rectified_sine, (_AMPLITUDE, _PERIOD),
                       "DC bileşen, temel sinüs ve çift kosinüs harmonikleri."),
            SignalInfo("full_rectified", "Tam dalga doğrultulmuş sinüs",
                       full_wave_rectified_sine, (_AMPLITUDE, _PERIOD),
                       "Doğrultucu çıkışı: yalnızca kosinüs terimleri, 1/n² azalma."),
            SignalInfo("abs_sine", "|sin| (mutlak sinüs)", abs_sine, (_AMPLITUDE, _PERIOD),
                       "|sin(ω₀t)| aslında T/2 periyotludur; bu yüzden yalnızca çift harmonikler."),
            SignalInfo("pulse", "Darbe dizisi", pulse_train, (_AMPLITUDE, _PERIOD, _DUTY),
                       "Doluluk oranı spektrumun sinc zarfını belirler."),
            SignalInfo("parabolic", "Parabolik dalga", parabolic_wave, (_AMPLITUDE, _PERIOD),
                       "t² ifadesinin periyodik genişlemesi; ζ(2) = π²/6 bu seriden çıkar."),
        )
    }
)  # fmt: skip
"""Hazır sinyallerin kayıt defteri (anahtar → :class:`SignalInfo`)."""


def build_signal(key: str, **params: float) -> PeriodicSignal:
    """Kütüphaneden anahtar ve parametrelerle sinyal oluşturur.

    Raises:
        SignalError: Anahtar bilinmiyorsa veya parametre geçersizse.
    """
    info = SIGNAL_LIBRARY.get(key)
    if info is None:
        raise SignalError(f"Bilinmeyen sinyal: {key!r}. Geçerli: {', '.join(SIGNAL_LIBRARY)}.")
    allowed = {p.name for p in info.params}
    unknown = set(params) - allowed
    if unknown:
        raise SignalError(f"'{info.label}' için bilinmeyen parametre(ler): {sorted(unknown)}.")
    try:
        return info.factory(**params)
    except SignalError:
        raise
    except ValueError as exc:
        raise SignalError(str(exc)) from exc
