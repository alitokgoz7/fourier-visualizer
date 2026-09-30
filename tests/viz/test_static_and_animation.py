from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from fourier_viz.core.complex_dft import epicycles_from_points
from fourier_viz.core.paths import build_shape, normalize_path, resample_by_arclength
from fourier_viz.core.series import SeriesError, harmonic_terms, partial_sums
from fourier_viz.core.signals import square_wave
from fourier_viz.export import (
    coefficients_from_json,
    coefficients_to_csv,
    coefficients_to_json,
    epicycles_to_csv,
    epicycles_to_json,
)
from fourier_viz.viz import animation, static
from fourier_viz.viz.theme import DARK

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
T = np.linspace(-np.pi, np.pi, 300)
SQ = square_wave()
COEFFS = SQ.coefficients(9)


@pytest.fixture(scope="module")
def heart_epicycles() -> tuple[np.ndarray, object]:
    pts = normalize_path(resample_by_arclength(build_shape("heart"), 64))
    return pts, epicycles_from_points(pts, 12)


def png_size(data: bytes) -> tuple[int, int]:
    assert data.startswith(PNG_MAGIC)
    with Image.open(io.BytesIO(data)) as im:
        return im.size


class TestStatic:
    def test_signal_plot_png(self) -> None:
        fig = static.signal_plot(T, SQ(T), COEFFS.evaluate(T), 9, title="Kare")
        width, height = png_size(static.figure_to_png_bytes(fig, dpi=50))
        assert width > height > 0

    def test_other_plots(self, heart_epicycles: tuple[np.ndarray, object]) -> None:
        pts, epi = heart_epicycles
        figures = [
            static.signal_plot(T, SQ(T), None, 0, palette=DARK),
            static.comparison_plot(T, SQ(T), {"Fejér": COEFFS.evaluate(T, smoothing="fejer")},
                                   title="k", x_range=(-1, 1)),
            static.spectrum_plot(np.arange(10), COEFFS.amplitude(), title="s"),
            static.spectrum_plot(np.arange(10), COEFFS.amplitude(), log_y=True),
            static.error_plot([1, 2, 4], {"L2": [1, 0.5, 0.25]}, title="e"),
            static.epicycle_snapshot(epi, target=pts, title="kalp"),  # type: ignore[arg-type]
            static.epicycle_snapshot(epi, show_circles=False, palette=DARK),  # type: ignore[arg-type]
            static.harmonics_plot(T, harmonic_terms(COEFFS, T, 3)[1:], [1, 2, 3], title="h"),
            static.harmonics_plot(T, harmonic_terms(COEFFS, T, 9)[1:], list(range(1, 10))),
        ]  # fmt: skip
        for fig in figures:
            png_size(static.figure_to_png_bytes(fig, dpi=40))


class TestAnimation:
    def test_ffmpeg_detected(self) -> None:
        assert animation.ffmpeg_available()
        assert animation.ffmpeg_path()

    @pytest.mark.slow
    def test_epicycle_gif(self, heart_epicycles: tuple[np.ndarray, object]) -> None:
        pts, epi = heart_epicycles
        data = animation.epicycle_animation(
            epi,
            "gif",
            n_frames=6,
            fps=10,
            target=pts,
            size_px=160,
            trail_fraction=0.5,  # type: ignore[arg-type]
        )
        assert data[:6] == b"GIF89a"
        with Image.open(io.BytesIO(data)) as im:
            assert im.n_frames == 6
            assert im.size == (160, 160)

    @pytest.mark.slow
    def test_epicycle_mp4(self, heart_epicycles: tuple[np.ndarray, object]) -> None:
        _, epi = heart_epicycles
        data = animation.epicycle_animation(epi, "mp4", n_frames=6, size_px=160)  # type: ignore[arg-type]
        assert data[4:8] == b"ftyp"

    @pytest.mark.slow
    def test_convergence_gif_and_mp4(self) -> None:
        ns = [1, 3, 9]
        sums = partial_sums(COEFFS, T, ns)
        gif = animation.convergence_animation(T, SQ(T), sums, ns, "gif", size=(3, 2))
        assert gif[:6] == b"GIF89a"
        with Image.open(io.BytesIO(gif)) as im:
            assert im.n_frames == 3
        mp4 = animation.convergence_animation(T, SQ(T), sums, ns, "mp4", size=(3.2, 2.4))
        assert mp4[4:8] == b"ftyp"

    def test_errors(self, heart_epicycles: tuple[np.ndarray, object]) -> None:
        _, epi = heart_epicycles
        with pytest.raises(animation.AnimationError, match="Bilinmeyen"):
            animation.epicycle_animation(epi, "avi", n_frames=2, size_px=50)  # type: ignore[arg-type]
        with pytest.raises(animation.AnimationError, match="N değeri"):
            animation.convergence_animation(T, SQ(T), np.zeros((2, T.size)), [1])

    def test_mp4_without_ffmpeg(
        self, monkeypatch: pytest.MonkeyPatch, heart_epicycles: tuple[np.ndarray, object]
    ) -> None:
        _, epi = heart_epicycles
        monkeypatch.setattr(animation, "ffmpeg_path", lambda: None)
        assert not animation.ffmpeg_available()
        with pytest.raises(animation.AnimationError, match="ffmpeg"):
            animation.epicycle_animation(epi, "mp4", n_frames=2, size_px=50)  # type: ignore[arg-type]

    def test_system_ffmpeg_preferred(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(animation.shutil, "which", lambda _: "/usr/bin/ffmpeg")
        assert animation.ffmpeg_path() == "/usr/bin/ffmpeg"


class TestExport:
    def test_coefficients_csv(self) -> None:
        text = coefficients_to_csv(COEFFS)
        lines = text.strip().splitlines()
        assert lines[0] == "n,a_n,b_n,amplitude,phase_rad,c_n_real,c_n_imag"
        assert len(lines) == 11
        row1 = [float(v) for v in lines[2].split(",")]
        assert row1[0] == 1
        assert row1[2] == pytest.approx(4 / np.pi)
        assert row1[6] == pytest.approx(-2 / np.pi)  # c₁ = (a₁ − i b₁)/2

    def test_coefficients_json_roundtrip(self) -> None:
        text = coefficients_to_json(COEFFS, {"signal": "Kare dalga"})
        payload = json.loads(text)
        assert payload["metadata"]["signal"] == "Kare dalga"
        assert payload["n_terms"] == 9
        assert payload["complex"]["n"][0] == -9
        back = coefficients_from_json(text)
        assert back.allclose(COEFFS, atol=0, rtol=0)
        assert "Kare dalga" in text  # ensure_ascii=False

    @pytest.mark.parametrize(
        "bad",
        [
            "not json",
            "{}",
            '{"a": [1], "b": [0, 1], "period": 1}',
            '{"a": [1], "b": [0], "period": -1}',
        ],
    )
    def test_coefficients_json_errors(self, bad: str) -> None:
        with pytest.raises(SeriesError):
            coefficients_from_json(bad)

    def test_epicycles_export(self, heart_epicycles: tuple[np.ndarray, object]) -> None:
        _, epi = heart_epicycles
        csv_text = epicycles_to_csv(epi)  # type: ignore[arg-type]
        lines = csv_text.strip().splitlines()
        assert lines[0].startswith("rank,frequency,radius")
        assert len(lines) == 1 + 1 + 12
        payload = json.loads(epicycles_to_json(epi, {"shape": "kalp"}))  # type: ignore[arg-type]
        assert len(payload["circles"]) == 12
        assert payload["n_samples"] == 64
        radii = [c["radius"] for c in payload["circles"]]
        assert radii == sorted(radii, reverse=True)
