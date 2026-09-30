from __future__ import annotations

import numpy as np
import pytest

from fourier_viz.core.series import (
    RealCoefficients,
    SeriesError,
    default_samples,
    harmonic_terms,
    partial_sum,
    partial_sums,
    real_coefficients,
    sigma_factors,
    trig_polynomial,
)

TWO_PI = 2.0 * np.pi


def square(t: np.ndarray) -> np.ndarray:
    return np.sign(np.sin(t))


def square_expected(n_terms: int) -> np.ndarray:
    n = np.arange(n_terms + 1)
    b = np.zeros(n_terms + 1)
    odd = n % 2 == 1
    b[odd] = 4.0 / (np.pi * n[odd])
    return b


# ---------------------------------------------------------------- RealCoefficients
class TestRealCoefficients:
    def test_basic_properties(self) -> None:
        c = RealCoefficients(np.array([2.0, 3.0, 0.0]), np.array([9.0, 0.0, 4.0]), period=4.0)
        assert c.n_terms == 2
        assert c.b[0] == 0.0  # b₀ zorla sıfırlanır
        assert c.mean == 1.0
        assert c.omega0 == pytest.approx(np.pi / 2)
        assert np.allclose(c.amplitude(), [1.0, 3.0, 4.0])
        assert np.allclose(c.phase(), [0.0, 0.0, -np.pi / 2])
        assert c.energy() == pytest.approx(1.0 + 0.5 * (9 + 16))
        assert np.array_equal(c.harmonics, [0.0, 1.0, 2.0])

    def test_immutable(self) -> None:
        c = RealCoefficients(np.array([1.0, 2.0]), np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="read-only"):
            c.a[0] = 5.0

    def test_input_is_copied(self) -> None:
        a = np.array([1.0, 2.0])
        c = RealCoefficients(a, np.zeros(2))
        a[0] = 100.0
        assert c.a[0] == 1.0

    @pytest.mark.parametrize(
        ("a", "b", "match"),
        [
            ([], [], "boş"),
            ([1.0, 2.0], [0.0], "aynı uzunlukta"),
            ([np.nan], [0.0], "sonlu"),
        ],
    )
    def test_invalid(self, a: list[float], b: list[float], match: str) -> None:
        with pytest.raises(SeriesError, match=match):
            RealCoefficients(np.array(a), np.array(b))

    def test_arithmetic(self) -> None:
        c1 = RealCoefficients(np.array([1.0, 2.0]), np.array([0.0, 3.0]))
        c2 = RealCoefficients(np.array([1.0, 1.0, 1.0]), np.array([0.0, 1.0, 1.0]))
        s = c1 + c2
        assert np.allclose(s.a, [2.0, 3.0, 1.0])
        assert np.allclose(s.b, [0.0, 4.0, 1.0])
        d = c2 - c1
        assert np.allclose(d.a, [0.0, -1.0, 1.0])
        assert np.allclose((2 * c1).a, [2.0, 4.0])
        assert np.allclose((c1 * 0.5).b, [0.0, 1.5])
        assert c1.allclose(c1.padded(5))
        assert not c1.allclose(c2)
        assert c1.__add__(3) is NotImplemented  # type: ignore[operator]
        assert c1.__sub__(3) is NotImplemented  # type: ignore[operator]
        assert c1.__mul__("x") is NotImplemented  # type: ignore[operator]

    def test_period_mismatch(self) -> None:
        c1 = RealCoefficients(np.array([1.0]), np.array([0.0]), period=1.0)
        c2 = RealCoefficients(np.array([1.0]), np.array([0.0]), period=2.0)
        with pytest.raises(SeriesError, match="periyot"):
            _ = c1 + c2

    def test_truncate_and_pad(self) -> None:
        c = RealCoefficients(np.arange(5.0), np.arange(5.0))
        assert c.truncate(2).n_terms == 2
        assert c.padded(2).n_terms == 2
        assert c.padded(8).n_terms == 8
        assert c.padded(8).a[-1] == 0.0
        with pytest.raises(SeriesError):
            c.truncate(10)


# ---------------------------------------------------------------- sigma factors
class TestSigmaFactors:
    def test_none(self) -> None:
        assert np.array_equal(sigma_factors(4, "none"), np.ones(5))

    def test_fejer(self) -> None:
        assert np.allclose(sigma_factors(3, "fejer"), [1.0, 0.75, 0.5, 0.25])

    def test_lanczos(self) -> None:
        sigma = sigma_factors(3, "lanczos")
        expected = [1.0] + [np.sin(np.pi * k / 4) / (np.pi * k / 4) for k in (1, 2, 3)]
        assert np.allclose(sigma, expected)

    def test_zero_terms(self) -> None:
        for method in ("none", "fejer", "lanczos"):
            assert np.array_equal(sigma_factors(0, method), [1.0])  # type: ignore[arg-type]

    def test_unknown(self) -> None:
        with pytest.raises(SeriesError, match="yumuşatma"):
            sigma_factors(3, "hann")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [-1, 1.5, True])
    def test_invalid_terms(self, bad: int) -> None:
        with pytest.raises(SeriesError):
            sigma_factors(bad)


# ---------------------------------------------------------------- numeric coefficients
class TestRealCoefficientsNumeric:
    @pytest.mark.parametrize("method", ["gauss", "simpson", "quad"])
    def test_square_wave(self, method: str) -> None:
        c = real_coefficients(
            square,
            15,
            period=TWO_PI,
            t_start=-np.pi,
            breakpoints=[0.0],
            method=method,  # type: ignore[arg-type]
        )
        assert np.allclose(c.a, 0.0, atol=1e-7)
        assert np.allclose(c.b, square_expected(15), atol=1e-7)

    def test_trapezoid_square_wave_converges(self) -> None:
        # Trapez kuralı süreksizlikte O(1/M); yine de iyi bir yaklaşım verir.
        c = real_coefficients(square, 9, method="trapezoid", samples=20000)
        assert np.allclose(c.b, square_expected(9), atol=1e-3)

    def test_trig_polynomial_exact_all_methods(self) -> None:
        a = [1.0, 0.5, 0.0, -2.0]
        b = [0.0, 1.5, 3.0, 0.25]
        f = trig_polynomial(a, b, period=3.0)
        for method in ("gauss", "trapezoid", "simpson", "quad"):
            c = real_coefficients(f, 6, period=3.0, method=method)  # type: ignore[arg-type]
            assert c.allclose(RealCoefficients(np.array(a), np.array(b), 3.0), atol=1e-8)

    def test_uses_function_attributes(self) -> None:
        def f(t: np.ndarray) -> np.ndarray:
            return np.cos(2 * np.pi * t / 5.0)

        f.period = 5.0  # type: ignore[attr-defined]
        f.t_start = 0.0  # type: ignore[attr-defined]
        c = real_coefficients(f, 3)
        assert c.period == 5.0
        assert c.a[1] == pytest.approx(1.0)

    def test_constant_and_zero_function(self) -> None:
        c = real_coefficients(lambda t: np.full_like(t, 3.0), 5)
        assert c.a[0] == pytest.approx(6.0)
        assert np.allclose(c.a[1:], 0.0, atol=1e-12)
        assert np.allclose(c.b, 0.0, atol=1e-12)
        z = real_coefficients(lambda t: np.zeros_like(t), 5)
        assert np.allclose(z.a, 0.0)
        assert np.allclose(z.b, 0.0)

    def test_scalar_returning_function_is_broadcast(self) -> None:
        c = real_coefficients(lambda t: 2.0, 2)  # type: ignore[arg-type,return-value]
        assert c.a[0] == pytest.approx(4.0)

    def test_n_terms_zero(self) -> None:
        c = real_coefficients(square, 0)
        assert c.n_terms == 0
        assert c.a[0] == pytest.approx(0.0, abs=1e-12)

    def test_nonfinite_function_rejected(self) -> None:
        with pytest.raises(SeriesError, match="NaN veya sonsuz"):
            real_coefficients(lambda t: 1.0 / (t - t[3]), 3, method="trapezoid", samples=64)
        with pytest.raises(SeriesError, match="NaN veya sonsuz"):
            real_coefficients(lambda t: np.log(t), 3)  # negatif t → NaN
        with pytest.raises(SeriesError, match="NaN veya sonsuz"):
            real_coefficients(lambda t: np.log(t), 3, method="quad")

    def test_complex_function(self) -> None:
        with pytest.raises(SeriesError, match="karmaşık"):
            real_coefficients(lambda t: np.exp(1j * t), 2)  # type: ignore[arg-type,return-value]
        # İhmal edilebilir sanal kısım kabul edilir.
        c = real_coefficients(lambda t: np.cos(t) + 0j, 2)  # type: ignore[arg-type,return-value]
        assert c.a[1] == pytest.approx(1.0)

    def test_default_samples(self) -> None:
        assert default_samples(10) == 4096
        assert default_samples(1000) == 16 * 1001


# ---------------------------------------------------------------- partial sums
class TestPartialSum:
    def test_matches_manual_formula(self) -> None:
        c = RealCoefficients(np.array([2.0, 1.0, 0.5]), np.array([0.0, -1.0, 2.0]), period=2.0)
        t = np.linspace(-1, 1, 7)
        w = np.pi
        manual = (
            1.0 + np.cos(w * t) - np.sin(w * t) + 0.5 * np.cos(2 * w * t) + 2 * np.sin(2 * w * t)
        )
        assert np.allclose(partial_sum(c, t), manual)
        assert np.allclose(c.evaluate(t, 1), 1.0 + np.cos(w * t) - np.sin(w * t))

    def test_n_zero_is_mean(self) -> None:
        c = RealCoefficients(np.array([4.0, 1.0]), np.array([0.0, 1.0]))
        assert np.allclose(partial_sum(c, [0.0, 1.0, 2.0], 0), 2.0)

    def test_shape_preserved(self) -> None:
        c = RealCoefficients(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        t = np.zeros((3, 4))
        assert partial_sum(c, t).shape == (3, 4)
        assert partial_sum(c, 0.0).shape == ()

    def test_too_many_terms(self) -> None:
        c = RealCoefficients(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        with pytest.raises(SeriesError, match="aşıyor"):
            partial_sum(c, [0.0], 5)

    def test_smoothing_scales_terms(self) -> None:
        c = RealCoefficients(np.array([0.0, 1.0, 1.0]), np.zeros(3))
        t = np.array([0.0])
        assert partial_sum(c, t, smoothing="fejer")[0] == pytest.approx(2 / 3 + 1 / 3)

    def test_harmonic_terms_sum_to_partial_sum(self) -> None:
        rng = np.random.default_rng(0)
        c = RealCoefficients(rng.normal(size=12), rng.normal(size=12), period=3.0)
        t = np.linspace(0, 3, 50)
        for smoothing in ("none", "fejer", "lanczos"):
            terms = harmonic_terms(c, t, smoothing=smoothing)  # type: ignore[arg-type]
            assert terms.shape == (12, 50)
            assert np.allclose(terms.sum(axis=0), partial_sum(c, t, smoothing=smoothing))  # type: ignore[arg-type]

    def test_partial_sums_many_n(self) -> None:
        rng = np.random.default_rng(1)
        c = RealCoefficients(rng.normal(size=20), rng.normal(size=20))
        t = np.linspace(-np.pi, np.pi, 40)
        ns = [0, 1, 5, 19]
        for smoothing in ("none", "lanczos"):
            stacked = partial_sums(c, t, ns, smoothing)  # type: ignore[arg-type]
            for row, n in zip(stacked, ns, strict=True):
                assert np.allclose(row, partial_sum(c, t, n, smoothing))  # type: ignore[arg-type]
        assert partial_sums(c, t, []).shape == (0, 40)

    def test_large_n_chunking(self) -> None:
        # Çok büyük N: parçalı hesap tek parça ile aynı sonucu verir.
        n = 5000
        a = np.zeros(n + 1)
        a[n] = 1.0
        c = RealCoefficients(a, np.zeros(n + 1))
        t = np.linspace(0, 1, 997)
        assert np.allclose(partial_sum(c, t), np.cos(n * t), atol=1e-9)
