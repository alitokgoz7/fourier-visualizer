"""Hypothesis ile özellik tabanlı testler: doğrusallık, kayma, yeniden oluşturma, Parseval."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from fourier_viz.core.analysis import parseval_check
from fourier_viz.core.complex_dft import (
    complex_to_real,
    dft,
    epicycles_from_points,
    idft,
    points_to_complex,
    real_to_complex,
)
from fourier_viz.core.series import RealCoefficients, partial_sum, real_coefficients
from fourier_viz.core.signals import SIGNAL_LIBRARY, build_signal, from_function

TWO_PI = 2.0 * np.pi
finite = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)
periods = st.floats(0.5, 20.0, allow_nan=False)


@st.composite
def trig_coefficients(draw: st.DrawFn, max_terms: int = 8) -> RealCoefficients:
    n = draw(st.integers(0, max_terms))
    a = draw(arrays(np.float64, n + 1, elements=finite))
    b = draw(arrays(np.float64, n + 1, elements=finite))
    period = draw(periods)
    return RealCoefficients(a, b, period)


# ---------------------------------------------------------------- linearity
@given(trig_coefficients(), trig_coefficients(), finite, finite)
def test_linearity_of_coefficients(
    cf: RealCoefficients, cg: RealCoefficients, alpha: float, beta: float
) -> None:
    cg = RealCoefficients(cg.a, cg.b, cf.period)  # aynı periyot
    f, g = cf.evaluate, cg.evaluate
    n = max(cf.n_terms, cg.n_terms) + 2

    def combo(t: np.ndarray) -> np.ndarray:
        return alpha * f(t) + beta * g(t)

    kwargs = {"period": cf.period, "samples": 512}
    lhs = real_coefficients(combo, n, **kwargs)  # type: ignore[arg-type]
    rhs = alpha * real_coefficients(f, n, **kwargs) + beta * real_coefficients(g, n, **kwargs)  # type: ignore[arg-type]
    assert lhs.allclose(rhs, atol=1e-9, rtol=1e-9)
    assert lhs.allclose(alpha * cf + beta * cg, atol=1e-8, rtol=1e-8)


@given(
    st.sampled_from(sorted(SIGNAL_LIBRARY)), st.sampled_from(sorted(SIGNAL_LIBRARY)), finite, finite
)
def test_linearity_on_library_signals(key_f: str, key_g: str, alpha: float, beta: float) -> None:
    f, g = build_signal(key_f), build_signal(key_g)
    combo = f.scaled(alpha) + g.scaled(beta)
    n = 25
    lhs = combo.coefficients(n)
    rhs = alpha * f.coefficients(n) + beta * g.coefficients(n)
    assert lhs.allclose(rhs, atol=1e-9, rtol=1e-9)
    assert lhs.allclose(combo.analytic_coefficients(n), atol=1e-8, rtol=1e-8)


# ---------------------------------------------------------------- reconstruction
@given(trig_coefficients())
def test_finite_harmonics_reconstructed_exactly(coeffs: RealCoefficients) -> None:
    """K harmonikli bir sinyal, K terimle (neredeyse) tam yeniden oluşturulur."""
    k = coeffs.n_terms
    recovered = real_coefficients(coeffs.evaluate, k, period=coeffs.period, samples=256)
    assert recovered.allclose(coeffs, atol=1e-9, rtol=1e-9)
    t = np.linspace(-coeffs.period, coeffs.period, 301)
    assert np.allclose(partial_sum(recovered, t), coeffs.evaluate(t), atol=1e-9)
    # Fazladan istenen harmonikler sıfırdır
    extra = real_coefficients(coeffs.evaluate, k + 5, period=coeffs.period, samples=256)
    assert np.allclose(extra.a[k + 1 :], 0.0, atol=1e-9)
    assert np.allclose(extra.b[k + 1 :], 0.0, atol=1e-9)


# ---------------------------------------------------------------- time shift
@given(st.sampled_from(sorted(SIGNAL_LIBRARY)), st.floats(-10, 10, allow_nan=False))
def test_time_shift_preserves_amplitudes(key: str, tau: float) -> None:
    sig = build_signal(key, period=3.0)
    base = sig.coefficients(30)
    shifted = sig.shifted(tau).coefficients(30)
    assert np.allclose(shifted.amplitude(), base.amplitude(), atol=1e-9)
    # c_n → c_n e^{-i n ω₀ τ}: yalnızca faz değişir
    expected = real_to_complex(base).shifted(tau)
    assert np.allclose(real_to_complex(shifted).c, expected.c, atol=1e-9)


@given(trig_coefficients(), st.floats(-5, 5, allow_nan=False))
def test_time_shift_on_trig_polynomial(coeffs: RealCoefficients, tau: float) -> None:
    shifted_fn = from_function(coeffs.evaluate, coeffs.period).shifted(tau)
    shifted = shifted_fn.coefficients(coeffs.n_terms, samples=256)
    assert np.allclose(shifted.amplitude(), coeffs.amplitude(), atol=1e-8)


# ---------------------------------------------------------------- symmetry
@given(arrays(np.float64, st.integers(1, 8), elements=finite), periods)
def test_even_polynomial_has_no_sine_terms(a: np.ndarray, period: float) -> None:
    coeffs = RealCoefficients(a, np.zeros_like(a), period)
    c = real_coefficients(coeffs.evaluate, a.size + 2, period=period, samples=256)
    assert np.allclose(c.b, 0.0, atol=1e-9)


@given(arrays(np.float64, st.integers(1, 8), elements=finite), periods)
def test_odd_polynomial_has_no_cosine_terms(b: np.ndarray, period: float) -> None:
    coeffs = RealCoefficients(np.zeros_like(b), b, period)
    c = real_coefficients(coeffs.evaluate, b.size + 2, period=period, samples=256)
    assert np.allclose(c.a, 0.0, atol=1e-9)  # a₀ dahil


# ---------------------------------------------------------------- Parseval & conversions
@given(trig_coefficients())
def test_parseval_for_trig_polynomials(coeffs: RealCoefficients) -> None:
    sig = from_function(coeffs.evaluate, coeffs.period)
    result = parseval_check(sig, coeffs, samples=1024)
    scale = max(1.0, result.signal_energy)
    assert abs(result.residual) < 1e-9 * scale
    assert result.complex_energy == pytest.approx(result.coefficient_energy, rel=1e-12, abs=1e-12)


@given(trig_coefficients())
def test_real_complex_roundtrip(coeffs: RealCoefficients) -> None:
    cplx = real_to_complex(coeffs)
    assert cplx.is_hermitian()
    assert complex_to_real(cplx).allclose(coeffs, atol=1e-12, rtol=1e-12)
    t = np.linspace(0, coeffs.period, 17)
    assert np.allclose(cplx.evaluate(t).real, coeffs.evaluate(t), atol=1e-9)


# ---------------------------------------------------------------- DFT & epicycles
complex_vectors = st.integers(1, 128).flatmap(
    lambda m: st.tuples(
        arrays(np.float64, m, elements=finite), arrays(np.float64, m, elements=finite)
    )
)


@given(complex_vectors)
def test_dft_roundtrip_and_numpy_agreement(parts: tuple[np.ndarray, np.ndarray]) -> None:
    z = parts[0] + 1j * parts[1]
    direct = dft(z, "direct")
    assert np.allclose(direct, np.fft.fft(z), atol=1e-8)
    assert np.allclose(idft(direct, "direct"), z, atol=1e-9)
    assert np.allclose(idft(dft(z)), z, atol=1e-12)


@given(complex_vectors)
def test_epicycles_redraw_points(parts: tuple[np.ndarray, np.ndarray]) -> None:
    pts = np.column_stack(parts)
    epi = epicycles_from_points(pts)
    s = np.arange(len(pts)) / len(pts)
    assert np.allclose(epi.evaluate(s), points_to_complex(pts), atol=1e-8)
    assert np.all(np.diff(epi.radii) <= 1e-12)


@given(complex_vectors, st.integers(0, 64))
def test_epicycle_truncation_is_best_by_energy(
    parts: tuple[np.ndarray, np.ndarray], k: int
) -> None:
    """En büyük k çemberi tutmak, ayrık L2 hatasını (Parseval gereği) en aza indirir."""
    pts = np.column_stack(parts)
    m = len(pts)
    assume(m > 1)
    full = epicycles_from_points(pts)
    kept = full.limit(k)
    s = np.arange(m) / m
    err = np.mean(np.abs(kept.evaluate(s) - points_to_complex(pts)) ** 2)
    dropped = full.radii[kept.n_circles :]
    assert err == pytest.approx(np.sum(dropped**2), rel=1e-7, abs=1e-9)
