from __future__ import annotations

import numpy as np
import pytest

from fourier_viz.core.series import partial_sum
from fourier_viz.core.signals import (
    SIGNAL_LIBRARY,
    Jump,
    PeriodicSignal,
    SignalError,
    abs_sine,
    build_signal,
    from_expression,
    from_function,
    full_wave_rectified_sine,
    half_wave_rectified_sine,
    parabolic_wave,
    pulse_train,
    sawtooth_wave,
    square_wave,
    triangle_wave,
    wrap_to_period,
)

TWO_PI = 2.0 * np.pi
N = 40
n = np.arange(N + 1, dtype=float)
odd = n % 2 == 1
even = n % 2 == 0

ALL_KEYS = sorted(SIGNAL_LIBRARY)
EVEN_SIGNALS = ["triangle", "full_rectified", "abs_sine", "pulse", "parabolic"]
ODD_SIGNALS = ["square", "sawtooth"]


# ---------------------------------------------------------------- analytic coefficients
class TestAnalyticCoefficients:
    def test_square(self) -> None:
        c = square_wave().analytic_coefficients(N)
        assert np.allclose(c.a, 0.0)
        assert np.allclose(c.b[odd], 4.0 / (n[odd] * np.pi))
        assert np.allclose(c.b[even], 0.0)
        assert c.b[1] == pytest.approx(4 / np.pi)
        assert c.b[3] == pytest.approx(4 / (3 * np.pi))

    def test_sawtooth(self) -> None:
        c = sawtooth_wave().analytic_coefficients(N)
        assert np.allclose(c.a, 0.0)
        assert np.allclose(c.b[1:], 2.0 * (-1.0) ** (n[1:] + 1) / (n[1:] * np.pi))
        assert c.b[1] == pytest.approx(2 / np.pi)
        assert c.b[2] == pytest.approx(-1 / np.pi)

    def test_triangle(self) -> None:
        c = triangle_wave().analytic_coefficients(N)
        assert np.allclose(c.b, 0.0)
        assert np.allclose(c.a[odd], 8.0 / (n[odd] * np.pi) ** 2)
        assert np.allclose(c.a[even], 0.0)

    def test_half_rectified(self) -> None:
        c = half_wave_rectified_sine(2.0).analytic_coefficients(6)
        assert c.a[0] == pytest.approx(4 / np.pi)
        assert c.b[1] == pytest.approx(1.0)
        assert c.a[2] == pytest.approx(-4 / (3 * np.pi))
        assert c.a[3] == 0.0
        assert c.a[4] == pytest.approx(-4 / (15 * np.pi))
        assert half_wave_rectified_sine().analytic_coefficients(0).n_terms == 0

    def test_full_rectified_and_abs_sine(self) -> None:
        full = full_wave_rectified_sine().analytic_coefficients(3)
        assert np.allclose(
            full.a, [4 / np.pi, -4 / (3 * np.pi), -4 / (15 * np.pi), -4 / (35 * np.pi)]
        )
        absn = abs_sine().analytic_coefficients(4)
        assert np.allclose(absn.a, [4 / np.pi, 0.0, -4 / (3 * np.pi), 0.0, -4 / (15 * np.pi)])

    def test_pulse(self) -> None:
        c = pulse_train(1.0, TWO_PI, 0.5).analytic_coefficients(4)
        # d = 1/2 → kare dalganın 0..1 aralığına kaydırılmış (ve çift) hâli
        assert c.a[0] == pytest.approx(1.0)
        assert np.allclose(c.a[1:], [2 / np.pi, 0.0, -2 / (3 * np.pi), 0.0])

    def test_parabolic(self) -> None:
        c = parabolic_wave(np.pi**2).analytic_coefficients(3)
        # t² üzerinde [-π, π): a0 = 2π²/3, an = 4(-1)^n / n²
        assert c.a[0] == pytest.approx(2 * np.pi**2 / 3)
        assert np.allclose(c.a[1:], [-4.0, 1.0, -4 / 9])

    def test_basel_problem(self) -> None:
        # t = π'de parabolik seri ζ(2) = π²/6'yı verir.
        c = parabolic_wave(np.pi**2).analytic_coefficients(20000)
        value = partial_sum(c, np.array([np.pi]))[0]
        assert value == pytest.approx(np.pi**2, rel=1e-4)

    def test_missing_analytic(self) -> None:
        sig = from_function(np.sin)
        assert not sig.has_analytic
        with pytest.raises(SignalError, match="analitik"):
            sig.analytic_coefficients(3)
        with pytest.raises(SignalError, match="negatif"):
            square_wave().analytic_coefficients(-1)


# ---------------------------------------------------------------- numeric vs analytic
@pytest.mark.parametrize("key", ALL_KEYS)
@pytest.mark.parametrize("period", [TWO_PI, 1.0, 7.5])
def test_numeric_matches_analytic(key: str, period: float) -> None:
    sig = build_signal(key, period=period, amplitude=1.7)
    numeric = sig.coefficients(N)
    analytic = sig.analytic_coefficients(N)
    assert numeric.allclose(analytic, atol=1e-9, rtol=0)


@pytest.mark.parametrize("key", ALL_KEYS)
@pytest.mark.parametrize("method", ["trapezoid", "simpson"])
def test_other_methods_close_to_analytic(key: str, method: str) -> None:
    sig = build_signal(key)
    numeric = sig.coefficients(20, method=method, samples=20000)  # type: ignore[arg-type]
    assert numeric.allclose(sig.analytic_coefficients(20), atol=5e-4, rtol=0)


def test_quad_method_high_precision() -> None:
    sig = square_wave()
    assert sig.coefficients(10, method="quad").allclose(sig.analytic_coefficients(10), atol=1e-9)


# ---------------------------------------------------------------- symmetry
@pytest.mark.parametrize("key", EVEN_SIGNALS)
def test_even_signals_have_no_sine_terms(key: str) -> None:
    sig = build_signal(key)
    t = np.linspace(-3, 3, 101)
    assert np.allclose(sig(t), sig(-t))
    assert np.allclose(sig.coefficients(N).b, 0.0, atol=1e-12)


@pytest.mark.parametrize("key", ODD_SIGNALS)
def test_odd_signals_have_no_cosine_terms(key: str) -> None:
    sig = build_signal(key)
    t = np.linspace(-3, 3, 101)
    assert np.allclose(sig(t), -sig(-t))
    c = sig.coefficients(N)
    assert np.allclose(c.a, 0.0, atol=1e-12)  # a₀ dahil
    assert abs(c.a[0]) < 1e-12


# ---------------------------------------------------------------- signal values
@pytest.mark.parametrize("key", ALL_KEYS)
def test_periodicity(key: str) -> None:
    sig = build_signal(key, period=3.0)
    t = np.linspace(-5, 5, 257)
    assert np.allclose(sig(t), sig(t + 3.0), atol=1e-12)
    assert np.allclose(sig(t), sig(t - 6.0), atol=1e-12)


def test_values_and_midpoints() -> None:
    sq = square_wave(2.0)
    assert np.allclose(sq(np.array([0.0, np.pi, -np.pi, 1.0, -1.0])), [0.0, 0.0, 0.0, 2.0, -2.0])
    saw = sawtooth_wave()
    assert np.allclose(saw(np.array([np.pi, -np.pi, np.pi / 2, 0.0])), [0.0, 0.0, 0.5, 0.0])
    tri = triangle_wave()
    assert np.allclose(tri(np.array([0.0, np.pi, np.pi / 2])), [1.0, -1.0, 0.0])
    pulse = pulse_train(3.0, TWO_PI, 0.5)
    assert np.allclose(pulse(np.array([0.0, np.pi / 2, 2.0, 3.0])), [3.0, 1.5, 0.0, 0.0])
    assert np.allclose(parabolic_wave()(np.array([0.0, np.pi / 2, -np.pi])), [0.0, 0.25, 1.0])
    assert np.allclose(half_wave_rectified_sine()(np.array([np.pi / 2, -np.pi / 2])), [1.0, 0.0])
    assert np.allclose(full_wave_rectified_sine()(np.array([np.pi, -np.pi, 0.0])), [1.0, 1.0, 0.0])
    assert np.allclose(abs_sine()(np.array([np.pi / 2, -np.pi / 2])), [1.0, 1.0])


def test_signal_shape_and_scalars() -> None:
    sig = square_wave()
    assert sig(0.5).shape == ()
    assert sig(np.zeros((2, 3))).shape == (2, 3)


def test_sample() -> None:
    t, f = triangle_wave(period=2.0).sample(11, periods=2)
    assert t[0] == -1.0
    assert t[-1] == pytest.approx(3.0)
    assert f.shape == (11,)
    with pytest.raises(SignalError):
        triangle_wave().sample(1)


def test_properties() -> None:
    sig = square_wave(period=4.0)
    assert sig.start == -2.0
    assert sig.omega0 == pytest.approx(np.pi / 2)
    assert not sig.is_continuous
    assert triangle_wave().is_continuous
    assert sig.breakpoints == (-2.0, 0.0)
    assert dict(sig.params) == {"amplitude": 1.0, "period": 4.0}


def test_jumps() -> None:
    jumps = square_wave(1.5).jumps()
    assert len(jumps) == 2
    by_loc = {round(j.location, 6): j for j in jumps}
    assert by_loc[0.0].size == pytest.approx(3.0)
    assert by_loc[0.0].midpoint == pytest.approx(0.0)
    assert by_loc[round(-np.pi, 6)].size == pytest.approx(-3.0)
    pulse_jumps = pulse_train(2.0, 10.0, 0.4).jumps()
    assert [j.location for j in pulse_jumps] == pytest.approx([-2.0, 2.0])
    assert [j.size for j in pulse_jumps] == pytest.approx([2.0, -2.0])
    assert Jump(0.0, 1.0, 3.0).midpoint == 2.0


# ---------------------------------------------------------------- transformations
@pytest.mark.parametrize("key", ALL_KEYS)
def test_time_shift_changes_only_phase(key: str) -> None:
    sig = build_signal(key)
    shifted = sig.shifted(0.9)
    t = np.linspace(-3, 3, 50)
    assert np.allclose(shifted(t), sig(t - 0.9))
    c0, c1 = sig.coefficients(N), shifted.coefficients(N)
    assert np.allclose(c1.amplitude(), c0.amplitude(), atol=1e-9)
    assert c1.allclose(shifted.analytic_coefficients(N), atol=1e-9)
    assert "kaydırılmış" in shifted.label
    assert shifted.params["tau"] == 0.9


def test_scale_and_add() -> None:
    sq = square_wave()
    tri = triangle_wave()
    combo = sq.scaled(2.0) + tri
    t = np.linspace(-3, 3, 31)
    assert np.allclose(combo(t), 2 * sq(t) + tri(t))
    expected = 2 * sq.analytic_coefficients(N) + tri.analytic_coefficients(N)
    assert combo.analytic_coefficients(N).allclose(expected)
    assert combo.coefficients(N).allclose(expected, atol=1e-9)
    assert combo.discontinuities and combo.kinks
    no_analytic = sq + from_function(np.cos)
    assert not no_analytic.has_analytic
    assert not from_function(np.cos).scaled(2).has_analytic
    with pytest.raises(SignalError, match="periyot"):
        _ = sq + square_wave(period=1.0)
    assert sq.__add__(5) is NotImplemented  # type: ignore[operator]


# ---------------------------------------------------------------- custom signals
class TestCustom:
    def test_wrap_to_period(self) -> None:
        assert np.allclose(
            wrap_to_period([0.0, 7.0, -7.0], TWO_PI, -np.pi), [0.0, 7 - TWO_PI, -7 + TWO_PI]
        )

    def test_from_function_boundary_jump(self) -> None:
        sig = from_function(lambda t: t)
        assert sig.discontinuities == (-np.pi,)
        assert sig(np.array([np.pi]))[0] == pytest.approx(0.0)  # ortalama değer
        assert sig(np.array([1.0 + TWO_PI]))[0] == pytest.approx(1.0)
        c = sig.coefficients(10)
        assert c.allclose(sawtooth_wave(np.pi).analytic_coefficients(10), atol=1e-9)

    def test_from_function_continuous(self) -> None:
        sig = from_function(np.cos, period=TWO_PI, t_start=0.0)
        assert sig.is_continuous
        assert sig.coefficients(3).a[1] == pytest.approx(1.0)

    def test_from_expression_parabola(self) -> None:
        sig = from_expression("t**2")
        assert sig.name == "expression"
        assert sig.params["expression"] == "t**2"
        assert sig.is_continuous
        c = sig.coefficients(N)
        assert c.allclose(parabolic_wave(np.pi**2).analytic_coefficients(N), atol=1e-8)

    def test_from_expression_user_example(self) -> None:
        sig = from_expression("t**2 * sin(3*t)")
        c = sig.coefficients(10)
        assert np.allclose(c.a, 0.0, atol=1e-10)  # tek fonksiyon
        t = np.linspace(-3, 3, 11)
        assert np.allclose(sig(t), t**2 * np.sin(3 * t))

    def test_from_expression_custom_interval(self) -> None:
        sig = from_expression("t", period=2.0, t_start=0.0)
        assert sig.start == 0.0
        assert sig(np.array([2.5]))[0] == pytest.approx(0.5)
        assert sig(np.array([0.0]))[0] == pytest.approx(1.0)  # (0 + 2)/2

    @pytest.mark.parametrize(
        ("text", "match"),
        [("1/t", "sonsuz"), ("log(t)", "NaN"), ("sqrt(t - 1)", "NaN"), ("os.system", "öznitelik"),
         ("3t", "Sözdizimi"), ("", "boş")],
    )  # fmt: skip
    def test_from_expression_errors(self, text: str, match: str) -> None:
        with pytest.raises(SignalError, match=match):
            from_expression(text)

    def test_constant_and_zero_expressions(self) -> None:
        const = from_expression("2.5")
        c = const.coefficients(5)
        assert c.a[0] == pytest.approx(5.0)
        assert np.allclose(c.a[1:], 0.0, atol=1e-12)
        zero = from_expression("0")
        cz = zero.coefficients(5)
        assert np.allclose(cz.a, 0.0)
        assert np.allclose(cz.b, 0.0)


# ---------------------------------------------------------------- registry
class TestRegistry:
    def test_library_contents(self) -> None:
        assert set(ALL_KEYS) == {
            "square", "sawtooth", "triangle", "half_rectified", "full_rectified",
            "abs_sine", "pulse", "parabolic",
        }  # fmt: skip
        for key, info in SIGNAL_LIBRARY.items():
            sig = build_signal(key)
            assert isinstance(sig, PeriodicSignal)
            assert sig.name == key
            assert info.label and info.description
            assert sig.latex.startswith("f(t)")
            for p in info.params:
                assert p.minimum <= p.default <= p.maximum

    def test_errors(self) -> None:
        with pytest.raises(SignalError, match="Bilinmeyen sinyal"):
            build_signal("sine")
        with pytest.raises(SignalError, match="bilinmeyen parametre"):
            build_signal("square", duty=0.3)
        with pytest.raises(SignalError, match="Doluluk"):
            build_signal("pulse", duty=1.0)
        with pytest.raises(SignalError, match="Periyot"):
            build_signal("square", period=-1.0)
        with pytest.raises(SignalError, match="Genlik"):
            build_signal("square", amplitude=np.inf)
