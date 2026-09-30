from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from fourier_viz.core.complex_dft import (
    MAX_DIRECT_DFT,
    ComplexCoefficients,
    Epicycle,
    EpicycleSet,
    complex_coefficients,
    complex_to_real,
    dft,
    dft_frequencies,
    epicycles_from_points,
    idft,
    points_to_complex,
    real_to_complex,
)
from fourier_viz.core.series import RealCoefficients, SeriesError, partial_sum

TWO_PI = 2.0 * np.pi
RNG = np.random.default_rng(42)


def random_real(n: int, period: float = TWO_PI) -> RealCoefficients:
    return RealCoefficients(RNG.normal(size=n + 1), RNG.normal(size=n + 1), period)


# ---------------------------------------------------------------- conversions
class TestConversions:
    def test_known_values(self) -> None:
        real = RealCoefficients(np.array([4.0, 2.0]), np.array([0.0, 6.0]))
        cc = real_to_complex(real)
        assert cc.n_terms == 1
        assert np.array_equal(cc.frequencies, [-1, 0, 1])
        assert cc[0] == pytest.approx(2.0)
        assert cc[1] == pytest.approx(1.0 - 3.0j)
        assert cc[-1] == pytest.approx(1.0 + 3.0j)
        assert cc[5] == 0j
        assert cc.is_hermitian()

    def test_roundtrip(self) -> None:
        real = random_real(10, period=3.0)
        back = complex_to_real(real_to_complex(real))
        assert back.allclose(real, atol=1e-13)
        assert back.period == 3.0
        assert real.to_complex().to_real().allclose(real)

    def test_energy_consistent(self) -> None:
        real = random_real(8)
        assert real.to_complex().energy() == pytest.approx(real.energy())

    def test_evaluate_matches_real_partial_sum(self) -> None:
        real = random_real(6, period=2.5)
        t = np.linspace(-2, 2, 33)
        values = real.to_complex().evaluate(t)
        assert np.allclose(values.imag, 0.0, atol=1e-12)
        assert np.allclose(values.real, partial_sum(real, t))
        assert np.allclose(real.to_complex().evaluate(t, 2).real, partial_sum(real, t, 2))

    def test_non_hermitian_rejected(self) -> None:
        cc = ComplexCoefficients(np.array([0.0, 0.0, 1.0 + 0j]))
        assert not cc.is_hermitian()
        with pytest.raises(SeriesError, match="gerçek değerli"):
            cc.to_real()

    @pytest.mark.parametrize("values", [[], [1.0, 2.0], [np.nan, 0.0, 0.0]])
    def test_invalid_complex_coefficients(self, values: list[float]) -> None:
        with pytest.raises(SeriesError):
            ComplexCoefficients(np.array(values, dtype=complex))

    def test_shift_changes_only_phase(self) -> None:
        cc = random_real(7).to_complex()
        shifted = cc.shifted(0.7)
        assert np.allclose(shifted.magnitude(), cc.magnitude())
        assert not np.allclose(shifted.phase(), cc.phase())
        t = np.linspace(0, 6, 25)
        assert np.allclose(shifted.evaluate(t), cc.evaluate(t - 0.7))


# ---------------------------------------------------------------- numeric c_n
class TestComplexCoefficients:
    @pytest.mark.parametrize("method", ["gauss", "trapezoid", "simpson", "quad"])
    def test_complex_exponential(self, method: str) -> None:
        cc = complex_coefficients(
            lambda t: 2 * np.exp(3j * t) - 1j * np.exp(-1j * t),
            4,
            method=method,  # type: ignore[arg-type]
        )
        expected = np.zeros(9, dtype=complex)
        expected[4 + 3] = 2.0
        expected[4 - 1] = -1j
        assert np.allclose(cc.c, expected, atol=1e-9)

    def test_real_signal_matches_real_coefficients(self) -> None:
        real = random_real(5)
        func = real.evaluate
        cc = complex_coefficients(func, 5)
        assert cc.to_real().allclose(real, atol=1e-10)

    def test_nonfinite_rejected(self) -> None:
        with pytest.raises(SeriesError, match="NaN veya sonsuz"):
            complex_coefficients(lambda t: 1 / (t - t), 2)


# ---------------------------------------------------------------- DFT
class TestDFT:
    @pytest.mark.parametrize("size", [1, 2, 7, 64, 101])
    def test_direct_matches_numpy(self, size: int) -> None:
        z = RNG.normal(size=size) + 1j * RNG.normal(size=size)
        assert np.allclose(dft(z, "direct"), np.fft.fft(z), atol=1e-9)
        assert np.allclose(dft(z), np.fft.fft(z))
        assert np.allclose(idft(np.fft.fft(z), "direct"), z, atol=1e-12)

    def test_roundtrip(self) -> None:
        z = RNG.normal(size=500) + 1j * RNG.normal(size=500)
        for method in ("fft", "direct"):
            assert np.allclose(idft(dft(z, method), method), z, atol=1e-10)  # type: ignore[arg-type]

    def test_real_input(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.allclose(dft(x, "direct"), [10, -2 + 2j, -2, -2 - 2j])

    @pytest.mark.parametrize("bad", [[], [[1, 2], [3, 4]], [1.0, np.inf]])
    def test_invalid_input(self, bad: list[float]) -> None:
        with pytest.raises(ValueError):
            dft(bad)
        with pytest.raises(ValueError):
            idft(bad)

    def test_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="DFT yöntemi"):
            dft([1.0], "slow")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="DFT yöntemi"):
            idft([1.0], "slow")  # type: ignore[arg-type]

    def test_direct_size_limit(self) -> None:
        with pytest.raises(ValueError, match="fft"):
            dft(np.zeros(MAX_DIRECT_DFT + 1), "direct")

    def test_frequencies(self) -> None:
        assert np.array_equal(dft_frequencies(4), [0, 1, -2, -1])
        assert np.array_equal(dft_frequencies(5), [0, 1, 2, -2, -1])
        with pytest.raises(ValueError):
            dft_frequencies(0)


# ---------------------------------------------------------------- epicycles
def ellipse_points(m: int) -> np.ndarray:
    s = np.arange(m) / m
    return np.column_stack([3 * np.cos(TWO_PI * s) + 1.0, np.sin(TWO_PI * s) - 2.0])


class TestEpicycles:
    def test_points_to_complex(self) -> None:
        pts = np.array([[1.0, 2.0], [3.0, -1.0]])
        assert np.allclose(points_to_complex(pts), [1 + 2j, 3 - 1j])
        assert np.allclose(points_to_complex(np.array([1 + 1j])), [1 + 1j])
        assert np.allclose(points_to_complex([1.0, 2.0]), [1.0, 2.0])
        with pytest.raises(ValueError):
            points_to_complex(np.zeros((3, 3)))
        with pytest.raises(ValueError, match="boş"):
            points_to_complex(np.zeros((0, 2)))

    def test_single_circle(self) -> None:
        s = np.arange(16) / 16
        z = 2.0 * np.exp(2j * np.pi * 3 * s + 0.5j) + (1 - 1j)
        epi = epicycles_from_points(z, n_circles=1)
        assert epi.center == pytest.approx(1 - 1j)
        assert epi.n_circles == 1
        assert int(epi.frequencies[0]) == 3
        assert epi.radii[0] == pytest.approx(2.0)
        assert epi.phases[0] == pytest.approx(0.5)
        circle = epi.circles()[0]
        assert isinstance(circle, Epicycle)
        assert circle.radius == pytest.approx(2.0)
        assert circle.phase == pytest.approx(0.5)
        assert circle.position(0.0) == pytest.approx(2.0 * np.exp(0.5j))
        assert epi.energy_fraction() == pytest.approx(1.0)

    def test_sorted_by_amplitude(self) -> None:
        epi = epicycles_from_points(RNG.normal(size=(64, 2)))
        assert np.all(np.diff(epi.radii) <= 1e-15)
        assert 0 not in epi.frequencies
        assert epi.n_circles == 63

    def test_full_reconstruction(self) -> None:
        pts = RNG.normal(size=(101, 2))
        epi = epicycles_from_points(pts, method="direct")
        s = np.arange(101) / 101
        assert np.allclose(epi.evaluate(s), pts[:, 0] + 1j * pts[:, 1], atol=1e-10)
        joints = epi.joints(s)
        assert joints.shape == (101, 101)
        assert np.allclose(joints[:, 0], epi.center)
        assert np.allclose(joints[:, -1], epi.evaluate(s))

    def test_even_count_reconstruction(self) -> None:
        pts = ellipse_points(64) + RNG.normal(scale=0.01, size=(64, 2))
        epi = epicycles_from_points(pts)
        s = np.arange(64) / 64
        assert np.allclose(epi.evaluate(s), points_to_complex(pts), atol=1e-10)

    def test_truncation_error_decreases(self) -> None:
        pts = ellipse_points(128) + RNG.normal(scale=0.05, size=(128, 2))
        target = points_to_complex(pts)
        s = np.arange(128) / 128
        errors = [
            np.sqrt(np.mean(np.abs(epicycles_from_points(pts, k).evaluate(s) - target) ** 2))
            for k in (1, 2, 5, 20, 127)
        ]
        assert all(b <= a + 1e-12 for a, b in pairwise(errors))
        assert errors[-1] < 1e-10
        # Elips yalnızca iki baskın çemberle (k=+1, k=-1) iyi yaklaşık
        assert errors[1] < 0.1
        epi2 = epicycles_from_points(pts, 2)
        assert set(epi2.frequencies.tolist()) == {1, -1}
        assert 0.95 < epi2.energy_fraction() <= 1.0

    def test_limit(self) -> None:
        epi = epicycles_from_points(ellipse_points(32))
        assert epi.limit(None) is epi
        assert epi.limit(3).n_circles == 3
        assert epi.limit(1000).n_circles == 31
        assert epi.limit(0).n_circles == 0
        assert np.allclose(epi.limit(0).evaluate([0.1, 0.2]), epi.center)
        with pytest.raises(ValueError):
            epi.limit(-1)

    def test_single_point_path(self) -> None:
        epi = epicycles_from_points(np.array([[2.0, 3.0]]))
        assert epi.n_circles == 0
        assert epi.center == pytest.approx(2 + 3j)
        assert epi.energy_fraction() == 1.0

    def test_mismatched_arrays(self) -> None:
        with pytest.raises(ValueError, match="aynı uzunlukta"):
            EpicycleSet(0j, np.array([1, 2]), np.array([1j]), 3)

    def test_evaluate_shape(self) -> None:
        epi = epicycles_from_points(ellipse_points(16))
        assert epi.evaluate(np.zeros((2, 3))).shape == (2, 3)
