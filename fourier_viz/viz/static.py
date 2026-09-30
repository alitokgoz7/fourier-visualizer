"""Matplotlib ile statik grafikler ve PNG dışa aktarma.

``pyplot`` kullanılmaz: :class:`matplotlib.figure.Figure` doğrudan oluşturulur. Bu, Streamlit'in
çok iş parçacıklı ortamında güvenlidir ve global durum bırakmaz. Renkler
:mod:`fourier_viz.viz.theme` paletinden gelir (varsayılan açık tema).
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from fourier_viz.core.complex_dft import EpicycleSet
from fourier_viz.viz.theme import LIGHT, Palette


def new_figure(
    palette: Palette = LIGHT, size: tuple[float, float] = (8.0, 4.5), dpi: int = 110
) -> tuple[Figure, Axes]:
    """Palete göre biçimlendirilmiş tek eksenli bir figür oluşturur."""
    fig = Figure(figsize=size, dpi=dpi, facecolor=palette.surface, layout="constrained")
    ax = fig.add_subplot()
    style_axes(ax, palette)
    return fig, ax


def set_figure_title(fig: Figure, title: str | None, palette: Palette = LIGHT) -> None:
    """Başlığı figür düzeyinde sola hizalı yazar; eksen üstündeki lejantla çakışmaz."""
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=palette.ink)


def style_axes(ax: Axes, palette: Palette = LIGHT) -> None:
    """Saç teli kılavuzlar, sessiz eksenler ve palet yazı renkleri uygular."""
    ax.set_facecolor(palette.surface)
    ax.grid(True, color=palette.grid, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(palette.axis)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=palette.muted, labelcolor=palette.ink_secondary, labelsize=9)
    ax.xaxis.label.set_color(palette.ink_secondary)
    ax.yaxis.label.set_color(palette.ink_secondary)
    ax.title.set_color(palette.ink)


def place_legend(ax: Axes, palette: Palette) -> None:
    """Lejantı çizim alanının üstüne, sağa hizalı tek satırda yerleştirir (veriyle çakışmaz)."""
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        labels,
        frameon=False,
        fontsize=9,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=min(len(labels), 4),
        borderaxespad=0.2,
    )
    for text in legend.get_texts():
        text.set_color(palette.ink_secondary)


def signal_plot(
    t: ArrayLike,
    original: ArrayLike,
    approximation: ArrayLike | None,
    n_terms: int,
    *,
    title: str | None = None,
    palette: Palette = LIGHT,
    size: tuple[float, float] = (8.0, 4.5),
) -> Figure:
    """Özgün sinyal ve :math:`S_N` kısmi toplamı."""
    fig, ax = new_figure(palette, size)
    ax.plot(t, original, color=palette.ink_secondary, lw=2, label="f(t) — özgün")
    if approximation is not None:
        ax.plot(t, approximation, color=palette.color(0), lw=2, label=f"$S_N(t)$, N = {n_terms}")
    ax.set_xlabel("t")
    set_figure_title(fig, title, palette)
    place_legend(ax, palette)
    return fig


def comparison_plot(
    t: ArrayLike,
    original: ArrayLike,
    curves: Mapping[str, ArrayLike],
    *,
    title: str | None = None,
    x_range: tuple[float, float] | None = None,
    palette: Palette = LIGHT,
    size: tuple[float, float] = (8.0, 4.5),
) -> Figure:
    """Birden çok yaklaşımın (ör. yumuşatmaların) özgün sinyalle karşılaştırması."""
    fig, ax = new_figure(palette, size)
    ax.plot(t, original, color=palette.ink_secondary, lw=2, label="f(t) — özgün")
    for slot, (name, values) in enumerate(curves.items()):
        ax.plot(t, values, color=palette.color(slot), lw=2, label=name)
    if x_range:
        ax.set_xlim(*x_range)
    ax.set_xlabel("t")
    set_figure_title(fig, title, palette)
    place_legend(ax, palette)
    return fig


def spectrum_plot(
    harmonics: ArrayLike,
    amplitude: ArrayLike,
    *,
    log_y: bool = False,
    title: str | None = None,
    y_label: str = "Genlik $A_n$",
    palette: Palette = LIGHT,
    size: tuple[float, float] = (8.0, 3.6),
) -> Figure:
    """Gövde (stem) genlik spektrumu."""
    fig, ax = new_figure(palette, size)
    n = np.asarray(harmonics, dtype=np.float64)
    a = np.asarray(amplitude, dtype=np.float64)
    if log_y:
        keep = a > 1e-15 * max(1.0, float(np.abs(a).max(initial=0.0)))
        n, a = n[keep], a[keep]
        ax.set_yscale("log")
    base = float(a.min()) / 10.0 if (log_y and a.size) else 0.0
    ax.vlines(n, base, a, color=palette.color(0), lw=1.5)
    ax.scatter(
        n, a, s=36, color=palette.color(0), edgecolors=palette.surface, linewidths=1.5, zorder=3
    )
    ax.set_xlabel("Harmonik n")
    ax.set_ylabel(y_label)
    set_figure_title(fig, title, palette)
    return fig


def error_plot(
    n_values: ArrayLike,
    series: Mapping[str, ArrayLike],
    *,
    title: str | None = None,
    palette: Palette = LIGHT,
    size: tuple[float, float] = (7.0, 4.2),
) -> Figure:
    """Log-log hata eğrileri."""
    fig, ax = new_figure(palette, size)
    n = np.asarray(n_values, dtype=np.float64)
    for slot, (name, values) in enumerate(series.items()):
        y = np.asarray(values, dtype=np.float64)
        y = np.where(y > 0, y, np.nan)
        ax.loglog(
            n, y, "-o", color=palette.color(slot), lw=2, ms=4.5, mec=palette.surface, label=name
        )
    ax.set_xlabel("Terim sayısı N")
    ax.set_ylabel("Hata")
    set_figure_title(fig, title, palette)
    place_legend(ax, palette)
    return fig


def draw_epicycles(
    ax: Axes,
    epicycles: EpicycleSet,
    s: float,
    trail: np.ndarray | None,
    *,
    target: ArrayLike | None = None,
    show_circles: bool = True,
    max_drawn_circles: int = 80,
    palette: Palette = LIGHT,
) -> None:
    """Verilen eksene tek bir anın epicycle zincirini, izi ve hedef şekli çizer."""
    joints = epicycles.joints(np.array([s]))[0]
    if target is not None:
        tp = np.asarray(target, dtype=np.float64)
        ring = np.vstack([tp, tp[:1]])
        ax.plot(ring[:, 0], ring[:, 1], color=palette.muted, lw=1.2, alpha=0.45)
    if show_circles:
        theta = np.linspace(0, 2 * np.pi, 64)
        count = min(epicycles.n_circles, max_drawn_circles)
        for center, radius in zip(joints[:count], epicycles.radii[:count], strict=True):
            ax.plot(
                center.real + radius * np.cos(theta),
                center.imag + radius * np.sin(theta),
                color=palette.muted,
                lw=0.7,
                alpha=0.55,
            )
    ax.plot(joints.real, joints.imag, "-o", color=palette.ink_secondary, lw=1.0, ms=2)
    if trail is not None and len(trail):
        ax.plot(trail.real, trail.imag, color=palette.color(0), lw=2)
    ax.scatter(
        [joints[-1].real], [joints[-1].imag], s=60, color=palette.color(1),
        edgecolors=palette.surface, linewidths=2, zorder=4,
    )  # fmt: skip
    ax.set_aspect("equal")
    ax.axis("off")


def epicycle_snapshot(
    epicycles: EpicycleSet,
    *,
    s: float = 0.3,
    target: ArrayLike | None = None,
    show_circles: bool = True,
    title: str | None = None,
    palette: Palette = LIGHT,
    size: tuple[float, float] = (6.0, 6.0),
    trail_points: int = 2000,
) -> Figure:
    """Epicycle zincirinin ``s`` anındaki görüntüsü ve tam iz (galeri/PNG için)."""
    fig = Figure(figsize=size, dpi=110, facecolor=palette.surface, layout="constrained")
    ax = fig.add_subplot()
    ax.set_facecolor(palette.surface)
    trail = epicycles.evaluate(np.linspace(0.0, 1.0, trail_points + 1))
    draw_epicycles(
        ax, epicycles, s, trail, target=target, show_circles=show_circles, palette=palette
    )
    set_figure_title(fig, title, palette)
    return fig


def harmonics_plot(
    t: ArrayLike,
    terms: ArrayLike,
    indices: Sequence[int],
    *,
    title: str | None = None,
    palette: Palette = LIGHT,
    size: tuple[float, float] = (8.0, 3.6),
) -> Figure:
    """Tek tek harmonik terimleri (sıralı tek tonlu rampa)."""
    fig, ax = new_figure(palette, size)
    rows = np.atleast_2d(np.asarray(terms, dtype=np.float64))
    colors = palette.ordinal(len(indices))
    for values, n, color in zip(rows, indices, colors, strict=True):
        ax.plot(t, values, color=color, lw=1.5, label=f"n = {n}")
    ax.set_xlabel("t")
    set_figure_title(fig, title, palette)
    if len(indices) <= 8:
        place_legend(ax, palette)
    return fig


def signal_card(
    t: ArrayLike,
    original: ArrayLike,
    sums: Mapping[str, ArrayLike],
    harmonics: ArrayLike,
    amplitude: ArrayLike,
    *,
    title: str,
    subtitle: str | None = None,
    palette: Palette = LIGHT,
    size: tuple[float, float] = (9.0, 6.2),
) -> Figure:
    """Galeri kartı: üstte sinyal ve kısmi toplamlar, altta genlik spektrumu (iki ayrı panel)."""
    fig = Figure(figsize=size, dpi=110, facecolor=palette.surface, layout="constrained")
    top, bottom = fig.subplots(2, 1, height_ratios=[1.7, 1.0])
    style_axes(top, palette)
    style_axes(bottom, palette)
    top.plot(t, original, color=palette.ink_secondary, lw=2, label="f(t) — özgün")
    for slot, (name, values) in enumerate(sums.items()):
        top.plot(t, values, color=palette.color(slot), lw=1.8, label=name)
    top.set_xlabel("t")
    set_figure_title(fig, title, palette)
    if subtitle:
        top.set_title(subtitle, loc="left", fontsize=9, color=palette.muted)
    place_legend(top, palette)
    n = np.asarray(harmonics, dtype=np.float64)
    a = np.asarray(amplitude, dtype=np.float64)
    bottom.vlines(n, 0.0, a, color=palette.color(0), lw=1.5)
    bottom.scatter(n, a, s=30, color=palette.color(0), edgecolors=palette.surface,
                   linewidths=1.5, zorder=3)  # fmt: skip
    bottom.set_xlabel("Harmonik n")
    bottom.set_ylabel("Genlik $A_n$")
    return fig


def epicycle_progression(
    points: ArrayLike,
    sets: Sequence[EpicycleSet],
    *,
    title: str | None = None,
    s: float = 0.35,
    palette: Palette = LIGHT,
    panel_size: float = 3.6,
    max_drawn_circles: int = 30,
) -> Figure:
    """Aynı şeklin artan çember sayılarıyla yeniden çizimi (yan yana paneller)."""
    count = len(sets)
    if count == 0:
        raise ValueError("En az bir epicycle kümesi gereklidir.")
    fig = Figure(
        figsize=(panel_size * count, panel_size + 0.6),
        dpi=110,
        facecolor=palette.surface,
        layout="constrained",
    )
    axes = np.atleast_1d(fig.subplots(1, count))
    for ax, epi in zip(axes, sets, strict=True):
        ax.set_facecolor(palette.surface)
        trail = epi.evaluate(np.linspace(0.0, 1.0, 1501))
        draw_epicycles(
            ax, epi, s, trail, target=points, max_drawn_circles=max_drawn_circles, palette=palette
        )
        ax.set_title(
            f"{epi.n_circles} çember · enerji %{100 * epi.energy_fraction():.2f}",
            fontsize=10,
            color=palette.ink_secondary,
        )
    # Panel yüksekliğini içeriğin en-boy oranına göre ayarla (geniş şekillerde boşluk kalmasın).
    aspect = max(ax.dataLim.height / max(ax.dataLim.width, 1e-12) for ax in axes)
    fig.set_size_inches(panel_size * count, panel_size * float(np.clip(aspect, 0.45, 1.25)) + 0.7)
    set_figure_title(fig, title, palette)
    return fig


def figure_to_png_bytes(fig: Figure, dpi: int = 150) -> bytes:
    """Figürü PNG baytlarına çevirir."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    return buffer.getvalue()
