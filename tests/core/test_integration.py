from __future__ import annotations

import numpy as np
import pytest

from fourier_viz.core.integration import (
    MAX_SAMPLES,
    composite_simpson,
    gauss_legendre,
    make_quadrature,
    normalize_breakpoints,
    periodic_trapezoid,
    quad_fourier_integrals,
    validate_period,
)

TWO_PI = 2.0 * np.pi


@pytest.mark.parametrize("method", ["gauss", "trapezoid", "simpson"])
def test_weights_sum_to_period(method: str) -> None:
    quad = make_quadrature(method, -1.5, 3.0, 512, breakpoints=(0.2, 0.7))  # type: ignore[arg-type]
    assert quad.weights.sum() == pytest.approx(3.0, rel=1e-12)
    assert quad.nodes.min() >= -1.5
    assert quad.nodes.max() <= 1.5
    assert quad.size == quad.nodes.size == quad.weights.size


@pytest.mark.parametrize("method", ["gauss", "trapezoid", "simpson"])
def test_integrates_trig_polynomial_exactly(method: str) -> None:
    quad = make_quadrature(method, 0.0, TWO_PI, 1024)  # type: ignore[arg-type]
    values = 2.0 + np.cos(3 * quad.nodes) ** 2 + np.sin(5 * quad.nodes)
    # ∫ (2 + cos²3t + sin5t) dt = 4π + π
    assert quad.integrate(values) == pytest.approx(5.0 * np.pi, rel=1e-10)


def test_gauss_handles_discontinuity_exactly() -> None:
    # Basamak fonksiyonu: süreksizlik noktası verilince Gauss tam sonuç verir.
    quad = gauss_legendre(0.0, 1.0, 64, breakpoints=(0.3,))
    values = np.where(quad.nodes < 0.3, 1.0, 5.0)
    assert quad.integrate(values) == pytest.approx(0.3 + 5.0 * 0.7, abs=1e-13)
    assert not np.any(np.isclose(quad.nodes, 0.3))


def test_simpson_uses_one_sided_limits_at_breakpoints() -> None:
    quad = composite_simpson(0.0, 1.0, 64, breakpoints=(0.5,))
    values = np.where(quad.nodes < 0.5, 0.0, 1.0)
    assert quad.integrate(values) == pytest.approx(0.5, abs=1e-8)


def test_trapezoid_is_uniform() -> None:
    quad = periodic_trapezoid(1.0, 2.0, 16)
    assert np.allclose(np.diff(quad.nodes), 2.0 / 16)
    assert np.allclose(quad.weights, 2.0 / 16)
    assert quad.nodes[0] == 1.0


def test_normalize_breakpoints_folds_sorts_and_dedupes() -> None:
    result = normalize_breakpoints([7.0, 1.0, 1.0 + 1e-15, -np.pi, np.pi, np.inf], -np.pi, TWO_PI)
    # ±π periyot uçlarıdır ve çıkarılır; 7 → 7 - 2π ≈ 0.7168
    assert result == pytest.approx((7.0 - TWO_PI, 1.0))


def test_quad_matches_known_integrals() -> None:
    def square(t: np.ndarray) -> np.ndarray:
        return np.sign(np.sin(t))

    cos_part, sin_part = quad_fourier_integrals(square, 5, -np.pi, TWO_PI, breakpoints=(0.0,))
    expected_sin = np.array([0.0, 4.0, 0.0, 4.0 / 3.0, 0.0, 4.0 / 5.0])
    assert np.allclose(sin_part, expected_sin, atol=1e-9)
    assert np.allclose(cos_part, 0.0, atol=1e-9)


@pytest.mark.parametrize("period", [0.0, -1.0, np.inf, np.nan])
def test_invalid_period_rejected(period: float) -> None:
    with pytest.raises(ValueError, match="Periyot"):
        validate_period(period)


@pytest.mark.parametrize("samples", [0, 3, MAX_SAMPLES + 1])
def test_invalid_samples_rejected(samples: int) -> None:
    with pytest.raises(ValueError, match="Düğüm sayısı"):
        periodic_trapezoid(0.0, 1.0, samples)


def test_invalid_method_and_order() -> None:
    with pytest.raises(ValueError, match="Bilinmeyen"):
        make_quadrature("midpoint", 0.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="quad"):
        make_quadrature("quad", 0.0, 1.0)
    with pytest.raises(ValueError, match="mertebesi"):
        gauss_legendre(0.0, 1.0, 64, order=0)
