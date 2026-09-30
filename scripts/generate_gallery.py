"""Galeri görsellerini üretir ve temel matematiksel sağlamaları yazdırır.

Kullanım::

    python scripts/generate_gallery.py                  # docs/gallery altına
    python scripts/generate_gallery.py --out /tmp/g --no-animations --dark

Her hazır sinyal için bir "kart" (iki periyotluk sinyal + S₃ ve S₂₅ kısmi toplamları + genlik
spektrumu), her hazır şekil için epicycle ilerleme paneli (5, 20, 100 çember), Gibbs ve
yakınsama grafikleri ve isteğe bağlı iki GIF animasyonu üretilir.

Betik ayrıca her sinyal için sayısal ve analitik katsayılar arasındaki en büyük farkı, Gibbs
aşımını ve epicycle hatalarının çember sayısıyla azaldığını denetler; bir denetim başarısız
olursa çıkış kodu 1 olur.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # paket kurulmadan da çalışsın
    sys.path.insert(0, str(ROOT))

from fourier_viz.core.analysis import (  # noqa: E402
    GIBBS_CONSTANT,
    convergence_study,
    gibbs_overshoot,
    l2_error,
)
from fourier_viz.core.complex_dft import epicycles_from_points, points_to_complex  # noqa: E402
from fourier_viz.core.paths import (  # noqa: E402
    SHAPE_LIBRARY,
    build_shape,
    normalize_path,
    resample_by_arclength,
)
from fourier_viz.core.series import partial_sums  # noqa: E402
from fourier_viz.core.signals import (  # noqa: E402
    SIGNAL_LIBRARY,
    PeriodicSignal,
    build_signal,
    from_expression,
    square_wave,
    triangle_wave,
)
from fourier_viz.viz import animation, static  # noqa: E402
from fourier_viz.viz.theme import DARK, LIGHT, Palette  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "gallery"
CARD_TERMS = (3, 25)
SPECTRUM_TERMS = 25
CIRCLE_COUNTS = (5, 20, 100)
EXPRESSION_EXAMPLE = "t**2 * sin(3*t)"


@dataclass
class Check:
    """Tek bir sağlama satırı."""

    name: str
    value: str
    ok: bool


def _save(fig: object, path: Path, dpi: int) -> Path:
    path.write_bytes(static.figure_to_png_bytes(fig, dpi=dpi))  # type: ignore[arg-type]
    return path


def _signal_card(
    signal: PeriodicSignal, out: Path, stem: str, palette: Palette, dpi: int
) -> tuple[Path, float]:
    coeffs = signal.coefficients(max(*CARD_TERMS, SPECTRUM_TERMS))
    t = np.linspace(signal.start - signal.period / 2, signal.start + 1.5 * signal.period, 2400)
    sums = partial_sums(coeffs, t, CARD_TERMS)
    fig = static.signal_card(
        t,
        signal(t),
        {f"$S_{{{n}}}(t)$": row for n, row in zip(CARD_TERMS, sums, strict=True)},
        np.arange(SPECTRUM_TERMS + 1),
        coeffs.truncate(SPECTRUM_TERMS).amplitude(),
        title=signal.label,
        subtitle=f"İki periyot, T = {signal.period / np.pi:g}π · spektrum n ≤ {SPECTRUM_TERMS}",
        palette=palette,
    )
    l2 = l2_error(signal, coeffs, CARD_TERMS[-1])
    return _save(fig, out / f"{stem}.png", dpi), l2


def signal_gallery(out: Path, palette: Palette, dpi: int) -> tuple[list[Path], list[Check]]:
    """Her hazır sinyal ve bir ifade örneği için kart üretir; katsayı sağlamalarını yapar."""
    files: list[Path] = []
    checks: list[Check] = []
    for key in SIGNAL_LIBRARY:
        signal = build_signal(key)
        path, l2 = _signal_card(signal, out, f"signal_{key}", palette, dpi)
        files.append(path)
        numeric = signal.coefficients(60)
        analytic = signal.analytic_coefficients(60)
        diff = float(
            max(np.abs(numeric.a - analytic.a).max(), np.abs(numeric.b - analytic.b).max())
        )
        checks.append(
            Check(f"{signal.label}: max |sayısal − analitik| (n ≤ 60)", f"{diff:.1e}", diff < 1e-8)
        )
        checks.append(
            Check(f"{signal.label}: L2 hatası, N = {CARD_TERMS[-1]}", f"{l2:.4f}", l2 < 0.25)
        )
    expression = from_expression(EXPRESSION_EXAMPLE)
    path, _ = _signal_card(expression, out, "signal_expression", palette, dpi)
    files.append(path)
    coeffs = expression.coefficients(30)
    checks.append(
        Check(
            f"f(t) = {EXPRESSION_EXAMPLE}: tek fonksiyon → max |aₙ|",
            f"{np.abs(coeffs.a).max():.1e}",
            float(np.abs(coeffs.a).max()) < 1e-9,
        )
    )
    return files, checks


def shape_gallery(out: Path, palette: Palette, dpi: int) -> tuple[list[Path], list[Check]]:
    """Her hazır şekil için 5/20/100 çemberli epicycle ilerleme paneli üretir."""
    files: list[Path] = []
    checks: list[Check] = []
    for key, info in SHAPE_LIBRARY.items():
        points = normalize_path(resample_by_arclength(build_shape(key), 512))
        full = epicycles_from_points(points)
        sets = [full.limit(k) for k in CIRCLE_COUNTS]
        fig = static.epicycle_progression(points, sets, title=f"{info.label} — epicycle ile çizim")
        files.append(_save(fig, out / f"shape_{key}.png", dpi))
        s = np.arange(len(points)) / len(points)
        target = points_to_complex(points)
        errors = [float(np.sqrt(np.mean(np.abs(e.evaluate(s) - target) ** 2))) for e in sets]
        full_error = float(np.max(np.abs(full.evaluate(s) - target)))
        monotone = all(b <= a + 1e-12 for a, b in pairwise(errors))
        checks.append(
            Check(
                f"{info.label}: RMS hata {'/'.join(map(str, CIRCLE_COUNTS))} çember",
                " → ".join(f"{e:.3g}" for e in errors),
                monotone,
            )
        )
        checks.append(
            Check(
                f"{info.label}: tüm çemberlerle max sapma", f"{full_error:.1e}", full_error < 1e-9
            )
        )
    return files, checks


def analysis_gallery(out: Path, palette: Palette, dpi: int) -> tuple[list[Path], list[Check]]:
    """Gibbs yakın çekimi (yumuşatmalarla) ve kare/üçgen log-log yakınsama grafikleri."""
    files: list[Path] = []
    checks: list[Check] = []
    square = square_wave()
    n_terms = 50
    coeffs = square.analytic_coefficients(n_terms)
    t = np.linspace(-0.6, 0.6, 2000)
    curves = {
        f"$S_{{{n_terms}}}$ (yumuşatmasız)": coeffs.evaluate(t),
        "Fejér": coeffs.evaluate(t, smoothing="fejer"),
        "Lanczos σ": coeffs.evaluate(t, smoothing="lanczos"),
    }
    fig = static.comparison_plot(
        t, square(t), curves, title=f"Gibbs olayı: kare dalga, N = {n_terms}", palette=palette
    )
    ax = fig.axes[0]
    level = 1 + 2 * GIBBS_CONSTANT
    ax.axhline(level, color=palette.muted, lw=0.8)
    ax.text(
        -0.58,
        level + 0.02,
        f"teorik tepe 1 + 2·{GIBBS_CONSTANT:.4f}",
        color=palette.muted,
        fontsize=8,
    )
    files.append(_save(fig, out / "gibbs.png", dpi))
    for method, limit in (("none", None), ("fejer", 0.0), ("lanczos", 0.02)):
        result = gibbs_overshoot(square, square.analytic_coefficients(1000), smoothing=method)  # type: ignore[arg-type]
        ok = abs(result.ratio - GIBBS_CONSTANT) < 1e-3 if limit is None else result.ratio <= limit
        checks.append(
            Check(f"Kare dalga Gibbs aşımı (N = 1000, {method})", f"%{100 * result.ratio:.3f}", ok)
        )

    ns = np.unique(np.geomspace(1, 400, 25).round().astype(int)).tolist()
    tri = triangle_wave()
    sq_study = convergence_study(square, square.analytic_coefficients(400), ns)
    tri_study = convergence_study(tri, tri.analytic_coefficients(400), ns)
    fig = static.error_plot(
        ns,
        {"Kare dalga (L2)": sq_study.l2, "Üçgen dalga (L2)": tri_study.l2},
        title="Yakınsama hızı: L2 hatası (log-log)",
        palette=palette,
    )
    files.append(_save(fig, out / "convergence.png", dpi))
    checks.append(
        Check(
            "Kare dalga L2 eğimi (≈ −0.5)",
            f"{sq_study.l2_rate():.3f}",
            abs(sq_study.l2_rate() + 0.5) < 0.1,
        )
    )
    checks.append(
        Check(
            "Üçgen dalga L2 eğimi (≈ −1.5)",
            f"{tri_study.l2_rate():.3f}",
            abs(tri_study.l2_rate() + 1.5) < 0.1,
        )
    )
    return files, checks


def animation_gallery(out: Path, palette: Palette) -> list[Path]:
    """README için iki küçük GIF: kalp epicycle'ı ve kare dalga yakınsaması."""
    points = normalize_path(resample_by_arclength(build_shape("heart"), 256))
    epi = epicycles_from_points(points, 40)
    heart = out / "epicycle_heart.gif"
    heart.write_bytes(
        animation.epicycle_animation(
            epi,
            "gif",
            n_frames=72,
            fps=18,
            target=points,
            size_px=360,
            max_drawn_circles=40,
            palette=palette,
        )
    )
    square = square_wave()
    coeffs = square.analytic_coefficients(60)
    t = np.linspace(-np.pi, np.pi, 800)
    ns = [1, 3, 5, 7, 9, 15, 25, 40, 60]
    conv = out / "convergence_square.gif"
    conv.write_bytes(
        animation.convergence_animation(
            t,
            square(t),
            partial_sums(coeffs, t, ns),
            ns,
            "gif",
            fps=2,
            title="Kare dalga",
            palette=palette,
        )
    )
    return [heart, conv]


def main(argv: Sequence[str] | None = None) -> int:
    """Komut satırı giriş noktası; başarılıysa 0 döndürür."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="çıktı klasörü")
    parser.add_argument("--dpi", type=int, default=110, help="PNG çözünürlüğü")
    parser.add_argument("--dark", action="store_true", help="koyu tema paleti kullan")
    parser.add_argument("--no-animations", action="store_true", help="GIF üretme")
    args = parser.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    palette = DARK if args.dark else LIGHT
    files: list[Path] = []
    checks: list[Check] = []
    for producer in (signal_gallery, shape_gallery, analysis_gallery):
        produced, produced_checks = producer(out, palette, args.dpi)
        files += produced
        checks += produced_checks
    if not args.no_animations:
        files += animation_gallery(out, palette)

    print(f"{len(files)} dosya yazıldı → {out}")
    width = max(len(c.name) for c in checks)
    for check in checks:
        mark = "✓" if check.ok else "✗"
        print(f"  {mark} {check.name:<{width}}  {check.value}")
    failed = [c for c in checks if not c.ok]
    if failed:
        print(f"{len(failed)} sağlama başarısız!", file=sys.stderr)
        return 1
    print("Tüm sağlamalar geçti.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
