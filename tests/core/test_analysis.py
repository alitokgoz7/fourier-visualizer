from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from fourier_viz.core.analysis import (
    GIBBS_CONSTANT,
    WILBRAHAM_GIBBS_PEAK,
    ConvergenceResult,
    convergence_study,
    cumulative_energy,
    estimate_rate,
    gibbs_overshoot,
    inner_product,
    l2_error,
    l2_error_from_energy,
    max_error,
    orthogonality_matrix,
    parseval_check,
    signal_energy,
    spectrum,
    terms_for_energy,
)
from fourier_viz.core.series import RealCoefficients
from fourier_viz.core.signals import (
    SIGNAL_LIBRARY,
    Jump,
    SignalError,
    build_signal,
    from_function,
    pulse_train,
    sawtooth_wave,
    square_wave,
    triangle_wave,
)

TWO_PI = 2.0 * np.pi
CONTINUOUS = ["triangle", "half_rectified", "full_rectified", "abs_sine", "parabolic"]
DISCONTINUOUS = ["square", "sawtooth", "pulse"]


# ---------------------------------------------------------------- constants
def test_gibbs_constants() -> None:
    assert abs(GIBBS_CONSTANT - 0.0894898722) < 1e-9
    assert abs(WILBRAHAM_GIBBS_PEAK - 1.1789797444) < 1e-9
    assert abs(GIBBS_CONSTANT - 0.0895) < 5e-5  # ≈ %8.95


# ---------------------------------------------------------------- orthogonality
class TestOrthogonality:
    @pytest.mark.parametrize(("n", "m"), [(1, 2), (2, 5), (3, 7), (10, 11)])
    def test_distinct_harmonics_are_orthogonal(self, n: int, m: int) -> None:
        cos_n = lambda t: np.cos(n * t)  # noqa: E731
        cos_m = lambda t: np.cos(m * t)  # noqa: E731
        sin_n = lambda t: np.sin(n * t)  # noqa: E731
        sin_m = lambda t: np.sin(m * t)  # noqa: E731
        assert abs(inner_product(cos_n, cos_m)) < 1e-12
        assert abs(inner_product(sin_n, sin_m)) < 1e-12
        assert abs(inner_product(cos_n, sin_m)) < 1e-12
        assert abs(inner_product(sin_n, cos_n)) < 1e-12
        assert inner_product(cos_n, cos_n) == pytest.approx(1.0)
        assert inner_product(sin_m, sin_m) == pytest.approx(1.0)

    def test_gram_matrix_is_identity(self) -> None:
        gram = orthogonality_matrix(15, period=3.0)
        assert gram.shape == (30, 30)
        assert np.allclose(gram, np.eye(30), atol=1e-12)

    def test_gram_invalid(self) -> None:
        with pytest.raises(ValueError, match="harmonik"):
            orthogonality_matrix(0)

    def test_inner_product_custom_period(self) -> None:
        w = 2 * np.pi / 5.0
        val = inner_product(
            lambda t: np.cos(w * t), lambda t: np.cos(w * t), period=5.0, t_start=0.0
        )
        assert val == pytest.approx(1.0)


# ---------------------------------------------------------------- Parseval
class TestParseval:
    @pytest.mark.parametrize("key", sorted(SIGNAL_LIBRARY))
    def test_parseval_holds(self, key: str) -> None:
        sig = build_signal(key, amplitude=1.3, period=4.0)
        coeffs = sig.analytic_coefficients(20000)
        result = parseval_check(sig, coeffs)
        assert result.coefficient_energy == pytest.approx(result.complex_energy, rel=1e-12)
        # Kuyruk enerjisi 1/N ile azalır; 20000 terimde < 1e-4 göreli.
        assert result.residual >= -1e-12
        assert result.captured_fraction == pytest.approx(1.0, abs=1e-4)

    def test_known_energies(self) -> None:
        assert signal_energy(square_wave(2.0)) == pytest.approx(4.0)
        assert signal_energy(triangle_wave()) == pytest.approx(1.0 / 3.0)
        assert signal_energy(sawtooth_wave()) == pytest.approx(1.0 / 3.0)

    def test_trig_polynomial_exact(self) -> None:
        coeffs = RealCoefficients(np.array([2.0, 1.0, 0.0, 3.0]), np.array([0.0, 0.5, 2.0, 0.0]))
        sig = from_function(coeffs.evaluate)
        result = parseval_check(sig, sig.coefficients(10))
        assert result.residual == pytest.approx(0.0, abs=1e-12)
        assert result.signal_energy == pytest.approx(1.0 + 0.5 * (1 + 0.25 + 4 + 9))

    def test_zero_signal(self) -> None:
        sig = from_function(np.zeros_like)
        result = parseval_check(sig, sig.coefficients(3))
        assert result.captured_fraction == 1.0


# ---------------------------------------------------------------- errors & convergence
class TestErrors:
    @pytest.mark.parametrize("key", CONTINUOUS)
    def test_l2_monotone_decreasing_for_continuous(self, key: str) -> None:
        sig = build_signal(key)
        coeffs = sig.coefficients(60)
        errors = [l2_error(sig, coeffs, n) for n in range(0, 61)]
        assert all(b <= a + 1e-12 for a, b in pairwise(errors))
        assert errors[-1] < 0.05 * errors[0]

    @pytest.mark.parametrize("key", CONTINUOUS + DISCONTINUOUS)
    def test_convergence_study_matches_individual(self, key: str) -> None:
        sig = build_signal(key)
        coeffs = sig.coefficients(64)
        ns = [1, 2, 4, 8, 16, 32, 64]
        study = convergence_study(sig, coeffs, ns)
        assert isinstance(study, ConvergenceResult)
        assert list(study.n_values) == ns
        individual = [l2_error(sig, coeffs, n, samples=16 * 65) for n in ns]
        assert np.allclose(study.l2, individual, rtol=1e-8, atol=1e-12)
        assert all(b <= a + 1e-12 for a, b in pairwise(study.l2))
        assert np.allclose(study.max_abs[-1], max_error(sig, coeffs, 64))

    def test_l2_matches_parseval_formula(self) -> None:
        sig = triangle_wave()
        coeffs = sig.coefficients(40)
        energy = signal_energy(sig)
        for n in (0, 1, 5, 40):
            assert l2_error(sig, coeffs, n) == pytest.approx(
                l2_error_from_energy(energy, coeffs, n), rel=1e-6, abs=1e-12
            )
        assert l2_error_from_energy(energy, coeffs) == pytest.approx(
            l2_error(sig, coeffs), rel=1e-6
        )

    def test_rates(self) -> None:
        ns = [8, 16, 32, 64, 128, 256]
        tri = triangle_wave()
        tri_study = convergence_study(tri, tri.analytic_coefficients(256), ns)
        assert tri_study.l2_rate() == pytest.approx(-1.5, abs=0.1)  # 1/n² katsayılar
        assert tri_study.max_rate() == pytest.approx(-1.0, abs=0.15)
        sq = square_wave()
        sq_study = convergence_study(sq, sq.analytic_coefficients(256), ns)
        assert sq_study.l2_rate() == pytest.approx(-0.5, abs=0.1)  # 1/n katsayılar
        # Süreksizlikte maksimum hata azalmaz (düzgün olmayan yakınsama)
        assert sq_study.max_abs.min() > 0.9

    def test_max_error_with_exclusion(self) -> None:
        sq = square_wave()
        coeffs = sq.analytic_coefficients(200)
        assert max_error(sq, coeffs) > 0.9
        assert max_error(sq, coeffs, exclude=0.3) < 0.02
        study = convergence_study(sq, coeffs, [50, 200], exclude=0.3)
        assert study.max_abs[1] < study.max_abs[0]
        with pytest.raises(SignalError, match="Dışlama"):
            max_error(sq, coeffs, exclude=10.0)
        with pytest.raises(SignalError, match="Dışlama"):
            convergence_study(sq, coeffs, [5], exclude=10.0)

    def test_smoothing_in_study(self) -> None:
        sq = square_wave()
        coeffs = sq.analytic_coefficients(100)
        plain = convergence_study(sq, coeffs, [100])
        fejer = convergence_study(sq, coeffs, [100], smoothing="fejer")
        assert fejer.smoothing == "fejer"
        assert fejer.l2[0] > plain.l2[0]  # yumuşatma L2'de en iyi değildir

    def test_invalid_study(self) -> None:
        sq = square_wave()
        with pytest.raises(ValueError, match="N değeri"):
            convergence_study(sq, sq.coefficients(5), [])

    def test_estimate_rate(self) -> None:
        n = np.array([1, 2, 4, 8, 16])
        assert estimate_rate(n, 3.0 * n**-2.0) == pytest.approx(-2.0)
        assert np.isnan(estimate_rate([1], [1.0]))
        assert np.isnan(estimate_rate([1, 2], [0.0, 0.0]))


# ---------------------------------------------------------------- Gibbs
class TestGibbs:
    @pytest.mark.parametrize("n_terms", [200, 500, 1000])
    def test_square_wave_overshoot_is_gibbs_constant(self, n_terms: int) -> None:
        sq = square_wave()
        result = gibbs_overshoot(sq, sq.analytic_coefficients(n_terms))
        assert result.ratio == pytest.approx(GIBBS_CONSTANT, abs=2e-4)
        assert result.percent == pytest.approx(8.95, abs=0.03)
        assert abs(result.relative_difference) < 3e-3
        assert result.peak_value == pytest.approx(WILBRAHAM_GIBBS_PEAK, abs=5e-4)
        # Tepe, sıçramaya yaklaşık T/(2N) uzaklıkta (daralır ama yüksekliği azalmaz)
        assert abs(result.peak_time - result.jump.location) == pytest.approx(
            np.pi / n_terms, rel=0.05
        )

    def test_uses_central_jump_and_scales_with_amplitude(self) -> None:
        sq = square_wave(3.0, period=2.0)
        result = gibbs_overshoot(sq, sq.analytic_coefficients(400))
        assert result.jump.location == pytest.approx(0.0)
        assert result.jump.size == pytest.approx(6.0)
        assert result.overshoot == pytest.approx(6.0 * GIBBS_CONSTANT, abs=3e-3)

    def test_numeric_coefficients_also_show_gibbs(self) -> None:
        sq = square_wave()
        result = gibbs_overshoot(sq, sq.coefficients(300))
        assert result.ratio == pytest.approx(GIBBS_CONSTANT, abs=5e-4)

    @pytest.mark.parametrize("n_terms", [10, 50, 300])
    def test_fejer_removes_overshoot(self, n_terms: int) -> None:
        sq = square_wave()
        result = gibbs_overshoot(sq, sq.analytic_coefficients(n_terms), smoothing="fejer")
        assert result.overshoot <= 1e-12
        assert result.percent == 0.0
        # Fejér ortalaması tüm aralıkta aralığın içinde kalır: -1 ≤ σ_N ≤ 1
        t = np.linspace(-np.pi, np.pi, 5001)
        values = sq.analytic_coefficients(n_terms).evaluate(t, smoothing="fejer")
        assert values.max() <= 1.0 + 1e-12
        assert values.min() >= -1.0 - 1e-12

    def test_lanczos_reduces_overshoot(self) -> None:
        sq = square_wave()
        coeffs = sq.analytic_coefficients(300)
        plain = gibbs_overshoot(sq, coeffs)
        lanczos = gibbs_overshoot(sq, coeffs, smoothing="lanczos")
        assert 0.0 < lanczos.ratio < plain.ratio / 5
        assert lanczos.ratio == pytest.approx(0.0119, abs=1e-3)

    def test_downward_jump_sawtooth(self) -> None:
        saw = sawtooth_wave()
        result = gibbs_overshoot(saw, saw.analytic_coefficients(2000))
        assert result.jump.size < 0
        assert result.peak_time < result.jump.location
        assert result.ratio == pytest.approx(GIBBS_CONSTANT, abs=1e-3)

    def test_pulse_narrow_window(self) -> None:
        pulse = pulse_train(duty=0.1)
        result = gibbs_overshoot(pulse, pulse.analytic_coefficients(1000))
        assert result.ratio == pytest.approx(GIBBS_CONSTANT, abs=2e-3)

    def test_explicit_jump_and_small_n(self) -> None:
        sq = square_wave()
        jump = Jump(0.0, -1.0, 1.0)
        result = gibbs_overshoot(sq, sq.analytic_coefficients(1), jump=jump)
        assert result.n_terms == 1
        assert result.peak_value == pytest.approx(4 / np.pi, rel=1e-6)
        zero_jump = gibbs_overshoot(sq, sq.analytic_coefficients(5), jump=Jump(1.0, 1.0, 1.0))
        assert zero_jump.ratio == 0.0

    def test_no_discontinuity(self) -> None:
        tri = triangle_wave()
        with pytest.raises(SignalError, match="süreksizlik"):
            gibbs_overshoot(tri, tri.coefficients(10))


# ---------------------------------------------------------------- spectrum & energy
class TestSpectrum:
    def test_square_spectrum(self) -> None:
        spec = spectrum(square_wave().analytic_coefficients(9))
        assert np.array_equal(spec.harmonics, np.arange(10))
        assert np.allclose(spec.frequencies, np.arange(10) / TWO_PI)
        assert spec.amplitude[1] == pytest.approx(4 / np.pi)
        assert spec.amplitude[2] == 0.0
        assert spec.phase[1] == pytest.approx(-np.pi / 2)  # sin = cos(x - π/2)
        assert spec.phase[2] == 0.0  # sıfır genlikte faz 0 kabul edilir
        assert np.allclose(spec.power[1], (4 / np.pi) ** 2 / 2)
        assert list(spec.dominant(3)) == [1, 3, 5]

    def test_dc_power(self) -> None:
        spec = spectrum(RealCoefficients(np.array([4.0, 0.0]), np.zeros(2)))
        assert spec.amplitude[0] == 2.0
        assert spec.power[0] == 4.0

    def test_cumulative_energy_and_terms(self) -> None:
        sq = square_wave()
        coeffs = sq.analytic_coefficients(200)
        cum = cumulative_energy(coeffs)
        assert cum.shape == (201,)
        assert np.all(np.diff(cum) >= 0)
        assert cum[1] == pytest.approx(8 / np.pi**2)  # ilk harmonik ≈ %81
        assert terms_for_energy(coeffs, 1.0, 0.8) == 1
        assert terms_for_energy(coeffs, 1.0, 0.9) == 3
        assert terms_for_energy(coeffs, 1.0, 1.0) == -1
        assert terms_for_energy(coeffs, 0.0) == 0
        with pytest.raises(ValueError, match="Oran"):
            terms_for_energy(coeffs, 1.0, 0.0)
