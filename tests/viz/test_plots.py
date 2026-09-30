from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from fourier_viz.core.complex_dft import epicycles_from_points
from fourier_viz.core.paths import build_shape, normalize_path, resample_by_arclength
from fourier_viz.core.series import harmonic_terms, partial_sums
from fourier_viz.core.signals import square_wave
from fourier_viz.viz import plots
from fourier_viz.viz.theme import DARK, LIGHT, get_palette

T = np.linspace(-np.pi, np.pi, 400)
SQ = square_wave()
COEFFS = SQ.coefficients(20)


@pytest.fixture(scope="module")
def epicycles() -> tuple[np.ndarray, object]:
    pts = normalize_path(resample_by_arclength(build_shape("heart"), 128))
    return pts, epicycles_from_points(pts, 30)


class TestTheme:
    def test_palette_selection(self) -> None:
        assert get_palette("dark") is DARK
        assert get_palette("light") is LIGHT
        assert get_palette(None) is LIGHT

    def test_categorical_slots_not_cycled(self) -> None:
        assert LIGHT.color(0) == "#2a78d6"
        with pytest.raises(IndexError, match="Diğer"):
            LIGHT.color(8)

    def test_ordinal_ramp(self) -> None:
        assert LIGHT.ordinal(0) == []
        ramp = LIGHT.ordinal(5)
        assert len(ramp) == 5
        assert ramp[0] == "#86b6ef"  # açık modda 250 adımından başlar
        assert ramp[-1] == "#104281"
        dark = DARK.ordinal(3)
        assert dark[0] == "#184f95"  # koyu modda 600'den daha koyu değil

    def test_rgba(self) -> None:
        assert LIGHT.rgba("#ff8000", 0.5) == "rgba(255, 128, 0, 0.5)"


class TestSignalFigures:
    def test_signal_figure_basic(self) -> None:
        fig = plots.signal_figure(T, SQ(T), COEFFS.evaluate(T), 20, title="Kare")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert fig.data[0].line.color == LIGHT.ink_secondary
        assert fig.data[1].line.color == LIGHT.color(0)
        assert "N = 20" in fig.data[1].name
        assert fig.layout.title.text == "Kare"

    def test_signal_without_approximation(self) -> None:
        fig = plots.signal_figure(T, SQ(T), None, 0)
        assert len(fig.data) == 1

    def test_harmonic_layer(self) -> None:
        terms = harmonic_terms(COEFFS, T, 5)
        fig = plots.signal_figure(
            T, SQ(T), COEFFS.evaluate(T, 5), 5, harmonics=terms, harmonic_indices=range(6)
        )
        assert len(fig.data) == 2 + 6
        assert fig.data[2].name == "a₀/2 (sabit)"
        assert fig.data[3].name == "n = 1"
        assert fig.data[3].xaxis == "x2"  # alt panelde

    def test_many_harmonics_hide_legend(self) -> None:
        terms = harmonic_terms(COEFFS, T, 20)[1:]
        fig = plots.signal_figure(T, SQ(T), None, 20, harmonics=terms, palette=DARK)
        assert not fig.data[1].showlegend

    def test_convergence_animation(self) -> None:
        ns = [1, 3, 5, 11]
        sums = partial_sums(COEFFS, T, ns)
        fig = plots.convergence_animation_figure(T, SQ(T), sums, ns)
        assert len(fig.frames) == 4
        assert [f.name for f in fig.frames] == ["1", "3", "5", "11"]
        assert np.allclose(fig.frames[-1].data[0].y, sums[-1])
        assert fig.layout.updatemenus[0].buttons[0].label == "▶ Oynat"
        assert len(fig.layout.sliders[0].steps) == 4

    def test_convergence_animation_validates(self) -> None:
        with pytest.raises(ValueError, match="N değeri"):
            plots.convergence_animation_figure(T, SQ(T), np.zeros((2, T.size)), [1])


class TestSpectrum:
    @pytest.mark.parametrize("style", ["bar", "stem"])
    def test_styles(self, style: str) -> None:
        n = np.arange(21)
        fig = plots.spectrum_figure(n, COEFFS.amplitude(), style=style)  # type: ignore[arg-type]
        assert len(fig.data) >= 1
        assert fig.layout.yaxis.type == "linear"

    def test_log_scale_drops_zeros(self) -> None:
        n = np.arange(21)
        fig = plots.spectrum_figure(n, COEFFS.amplitude(), style="bar", log_y=True)
        assert fig.layout.yaxis.type == "log"
        assert len(fig.data[0].x) == 10  # yalnızca tek harmonikler (çiftler ≈ 0)
        stem = plots.spectrum_figure(n, COEFFS.amplitude(), style="stem", log_y=True)
        assert np.all(np.asarray(stem.data[1].y) > 0)

    def test_phase(self) -> None:
        fig = plots.spectrum_figure(np.arange(21), COEFFS.phase(), phase=True, log_y=True)
        assert fig.layout.yaxis.type == "linear"
        assert list(fig.layout.yaxis.ticktext) == ["−π", "−π/2", "0", "π/2", "π"]

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="aynı uzunlukta"):
            plots.spectrum_figure([1, 2], [1.0])
        with pytest.raises(ValueError, match="stili"):
            plots.spectrum_figure([1], [1.0], style="pie")  # type: ignore[arg-type]


class TestOtherFigures:
    def test_error_figure(self) -> None:
        fig = plots.error_figure([1, 2, 4], {"L2": [1.0, 0.5, 0.0], "Maks.": [1.0, 1.0, 1.0]})
        assert len(fig.data) == 2
        assert fig.layout.xaxis.type == "log"
        assert np.isnan(fig.data[0].y[-1])  # log ölçekte sıfır çizilmez
        lin = plots.error_figure([1, 2], {"L2": [1.0, 0.0]}, log_log=False)
        assert lin.data[0].y[-1] == 0.0

    def test_comparison_figure(self) -> None:
        curves = {"Yok": COEFFS.evaluate(T), "Fejér": COEFFS.evaluate(T, smoothing="fejer")}
        fig = plots.comparison_figure(
            T, SQ(T), curves,
            x_range=(-0.5, 0.5),
            reference_levels={"teorik": 1.179},
            peaks={"Yok": (0.1, 1.18), "bilinmeyen": (0.2, 1.0)},
        )  # fmt: skip
        assert len(fig.data) == 1 + 2 + 2
        assert fig.data[2].line.color == LIGHT.color(1)
        assert tuple(fig.layout.xaxis.range) == (-0.5, 0.5)
        assert len(fig.layout.shapes) == 1

    def test_heatmap(self) -> None:
        fig = plots.heatmap_figure(np.eye(3), ["a", "b", "c"])
        assert fig.data[0].z.shape == (3, 3)
        dark = plots.heatmap_figure(np.eye(2), ["a", "b"], palette=DARK)
        assert dark.data[0].colorscale[0][1] == DARK.sequential[-1]

    def test_drawing_canvas(self) -> None:
        fig = plots.drawing_canvas_figure()
        assert fig.layout.dragmode == "lasso"
        assert len(fig.data[0].x) == 41 * 41
        with_path = plots.drawing_canvas_figure(np.array([[0, 0], [1, 0], [0, 1]]))
        assert len(with_path.data) == 2
        assert len(with_path.data[1].x) == 4  # kapatılmış


class TestEpicycleFigure:
    def test_frames_and_traces(self, epicycles: tuple[np.ndarray, object]) -> None:
        pts, epi = epicycles
        fig = plots.epicycle_figure(epi, n_frames=30, target=pts)  # type: ignore[arg-type]
        assert len(fig.data) == 5
        assert len(fig.frames) == 30
        assert list(fig.frames[0].traces) == [1, 2, 3, 4]
        assert fig.layout.yaxis.scaleanchor == "x"
        # Son karede iz neredeyse tüm yolu kapsar
        assert len(fig.frames[-1].data[2].x) > len(fig.frames[1].data[2].x)

    def test_hide_circles(self, epicycles: tuple[np.ndarray, object]) -> None:
        _, epi = epicycles
        fig = plots.epicycle_figure(epi, n_frames=10, show_circles=False)  # type: ignore[arg-type]
        assert len(fig.frames[0].data[0].x) == 0
        assert not fig.data[1].showlegend

    def test_frames_geometry(self, epicycles: tuple[np.ndarray, object]) -> None:
        _, epi = epicycles
        geo = plots.epicycle_frames(epi, n_frames=20, trail_fraction=0.25)  # type: ignore[arg-type]
        assert geo["joints"].shape == (20, 31)
        window = round(0.25 * 80)
        assert all(len(tr) == window + 1 for tr in geo["trail"])
        # İz, kalem ucunda biter
        assert np.allclose(geo["trail"][7][-1], geo["joints"][7, -1])
        full = plots.epicycle_frames(epi, n_frames=20)  # type: ignore[arg-type]
        assert len(full["trail"][0]) == 2

    @pytest.mark.parametrize(("frames", "trail"), [(1, 1.0), (10, 0.0), (10, 1.5)])
    def test_invalid(self, epicycles: tuple[np.ndarray, object], frames: int, trail: float) -> None:
        _, epi = epicycles
        with pytest.raises(ValueError):
            plots.epicycle_frames(epi, frames, trail)  # type: ignore[arg-type]

    def test_single_point_epicycles(self) -> None:
        epi = epicycles_from_points(np.array([[0.0, 0.0], [1.0, 0.0]]), 0)
        fig = plots.epicycle_figure(epi, n_frames=4)
        assert len(fig.frames) == 4
