"""Performans testleri: temel işlemler makul sürede (< 1 sn) tamamlanmalı."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest

from fourier_viz.core.analysis import convergence_study
from fourier_viz.core.complex_dft import dft, epicycles_from_points
from fourier_viz.core.paths import build_shape, resample_by_arclength
from fourier_viz.core.series import partial_sum, partial_sums
from fourier_viz.core.signals import square_wave

pytestmark = pytest.mark.perf


def best_time(func: Callable[[], object], repeats: int = 3) -> float:
    """Isınma sonrası en iyi süre (paylaşımlı CI makinelerinde gürültüyü azaltır)."""
    func()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times)


def test_partial_sum_n500() -> None:
    coeffs = square_wave().analytic_coefficients(500)
    t = np.linspace(-np.pi, np.pi, 2000)
    assert best_time(lambda: partial_sum(coeffs, t, 500)) < 1.0


def test_numeric_coefficients_n500() -> None:
    sig = square_wave()
    assert best_time(lambda: sig.coefficients(500)) < 1.0


def test_all_partial_sums_for_animation() -> None:
    coeffs = square_wave().analytic_coefficients(500)
    t = np.linspace(-np.pi, np.pi, 1000)
    ns = list(range(1, 501, 10))
    assert best_time(lambda: partial_sums(coeffs, t, ns, "lanczos")) < 1.0


@pytest.mark.parametrize("method", ["fft", "direct"])
def test_dft_2000_points(method: str) -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=2000) + 1j * rng.normal(size=2000)
    assert best_time(lambda: dft(z, method)) < 1.0  # type: ignore[arg-type]


def test_epicycles_2000_points() -> None:
    pts = resample_by_arclength(build_shape("heart"), 2000)

    def run() -> None:
        epi = epicycles_from_points(pts)
        epi.joints(np.linspace(0, 1, 200, endpoint=False))

    assert best_time(run) < 1.0


def test_convergence_study_up_to_500() -> None:
    sig = square_wave()
    coeffs = sig.analytic_coefficients(500)
    ns = np.unique(np.geomspace(1, 500, 30).astype(int))
    assert best_time(lambda: convergence_study(sig, coeffs, ns), repeats=1) < 2.0
