"""Katsayıların ve epicycle'ların CSV/JSON olarak dışa (ve içe) aktarılması."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from fourier_viz.core.complex_dft import EpicycleSet, real_to_complex
from fourier_viz.core.series import RealCoefficients, SeriesError

CONVENTION: Final[str] = (
    "f(t) ~ a0/2 + sum_{n>=1} [a_n cos(n w0 t) + b_n sin(n w0 t)], w0 = 2*pi/T; "
    "c_n = (a_n - i b_n)/2, c_0 = a0/2"
)
"""Dosyalara yazılan katsayı sözleşmesi açıklaması."""

_FLOAT_FORMAT: Final[str] = "{:.17g}"


def _fmt(value: float) -> str:
    return _FLOAT_FORMAT.format(float(value))


def coefficients_to_csv(coeffs: RealCoefficients) -> str:
    """Gerçek katsayıları CSV metnine çevirir.

    Sütunlar: ``n, a_n, b_n, amplitude, phase_rad, c_n_real, c_n_imag`` (``n = 0..N``).
    """
    cplx = real_to_complex(coeffs)
    amplitude = coeffs.amplitude()
    phase = coeffs.phase()
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["n", "a_n", "b_n", "amplitude", "phase_rad", "c_n_real", "c_n_imag"])
    for n in range(coeffs.n_terms + 1):
        c = cplx[n]
        writer.writerow(
            [
                n,
                _fmt(coeffs.a[n]),
                _fmt(coeffs.b[n]),
                _fmt(amplitude[n]),
                _fmt(phase[n]),
                _fmt(c.real),
                _fmt(c.imag),
            ]
        )
    return buffer.getvalue()


def coefficients_to_json(
    coeffs: RealCoefficients, metadata: Mapping[str, Any] | None = None
) -> str:
    """Gerçek ve karmaşık katsayıları, sözleşme ve üst verilerle birlikte JSON'a çevirir."""
    cplx = real_to_complex(coeffs)
    payload = {
        "convention": CONVENTION,
        "period": coeffs.period,
        "n_terms": coeffs.n_terms,
        "a": coeffs.a.tolist(),
        "b": coeffs.b.tolist(),
        "complex": {
            "n": cplx.frequencies.tolist(),
            "real": cplx.c.real.tolist(),
            "imag": cplx.c.imag.tolist(),
        },
        "metadata": dict(metadata or {}),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def coefficients_from_json(text: str) -> RealCoefficients:
    """:func:`coefficients_to_json` çıktısını geri okur.

    Raises:
        SeriesError: JSON geçersizse veya gerekli alanlar eksikse.
    """
    try:
        payload = json.loads(text)
        return RealCoefficients(
            np.asarray(payload["a"], dtype=np.float64),
            np.asarray(payload["b"], dtype=np.float64),
            float(payload["period"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SeriesError):
            raise
        raise SeriesError(f"Katsayı JSON dosyası okunamadı: {exc}") from exc


def epicycles_to_csv(epicycles: EpicycleSet) -> str:
    """Epicycle zincirini CSV'ye çevirir (ilk satır DC merkezi, ``rank = 0``)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["rank", "frequency", "radius", "phase_rad", "c_real", "c_imag"])
    center = epicycles.center
    writer.writerow(
        [0, 0, _fmt(abs(center)), _fmt(np.angle(center)), _fmt(center.real), _fmt(center.imag)]
    )
    for rank, circle in enumerate(epicycles.circles(), start=1):
        c = circle.coefficient
        writer.writerow(
            [
                rank,
                circle.frequency,
                _fmt(circle.radius),
                _fmt(circle.phase),
                _fmt(c.real),
                _fmt(c.imag),
            ]
        )
    return buffer.getvalue()


def epicycles_to_json(epicycles: EpicycleSet, metadata: Mapping[str, Any] | None = None) -> str:
    """Epicycle zincirini JSON'a çevirir."""
    payload = {
        "convention": "z(s) = c_0 + sum_j c_j exp(2*pi*i*k_j*s), s in [0, 1)",
        "n_samples": epicycles.n_samples,
        "center": [epicycles.center.real, epicycles.center.imag],
        "circles": [
            {
                "frequency": circle.frequency,
                "radius": circle.radius,
                "phase": circle.phase,
                "re": circle.coefficient.real,
                "im": circle.coefficient.imag,
            }
            for circle in epicycles.circles()
        ],
        "metadata": dict(metadata or {}),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
