"""Plotly ile interaktif grafikler.

Bu modüldeki fonksiyonlar yalnızca **hazır hesaplanmış** dizileri çizer (matematik çekirdekte
yapılır) ve :class:`plotly.graph_objects.Figure` döndürür. Arka plan, yazı ve kılavuz renkleri
bilinçli olarak ayarlanmaz: Streamlit'te ``theme="streamlit"`` ile gösterildiğinde bunlar açık/koyu
temaya göre otomatik gelir. Seri renkleri :mod:`fourier_viz.viz.theme` paletinden sabit sırayla
alınır.

Tasarım kuralları: çizgiler 2 px, işaretçiler ≥ 8 px ve yüzey renginde halkalı, iki veya daha
fazla seride lejant her zaman var, tek eksen (asla çift y ekseni), özgün sinyal nötr referans
rengiyle, yaklaşımlar kategorik renklerle çizilir.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from plotly.subplots import make_subplots

from fourier_viz.core.complex_dft import EpicycleSet
from fourier_viz.core.types import FloatArray
from fourier_viz.viz.theme import FONT_FAMILY, LIGHT, Palette

LINE_WIDTH: Final[float] = 2.0
MARKER_SIZE: Final[int] = 9
MAX_DRAWN_CIRCLES: Final[int] = 80
CIRCLE_RESOLUTION: Final[int] = 48

SpectrumStyle = Literal["bar", "stem"]


# ====================================================================== ortak düzen
def _base_layout(fig: go.Figure, *, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left"} if title else None,
        height=height,
        margin={"l": 56, "r": 24, "t": 64 if title else 40, "b": 48},
        font={"family": FONT_FAMILY},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={"font": {"family": FONT_FAMILY}},
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, zeroline=False, ticks="outside", ticklen=4)
    fig.update_yaxes(showgrid=True, gridwidth=1, zeroline=False, ticks="outside", ticklen=4)
    return fig


def _line(
    x: ArrayLike,
    y: ArrayLike,
    name: str,
    color: str,
    *,
    width: float = LINE_WIDTH,
    hover: str | None = None,
    **kwargs: Any,
) -> go.Scatter:
    return go.Scatter(
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        mode="lines",
        name=name,
        line={"color": color, "width": width, "shape": "linear"},
        hovertemplate=hover or f"{name}<br>t=%{{x:.4g}}<br>değer=%{{y:.4g}}<extra></extra>",
        **kwargs,
    )


# ====================================================================== sinyal ve kısmi toplam
def signal_figure(
    t: ArrayLike,
    original: ArrayLike,
    approximation: ArrayLike | None,
    n_terms: int,
    *,
    harmonics: ArrayLike | None = None,
    harmonic_indices: Sequence[int] | None = None,
    approximation_label: str | None = None,
    palette: Palette = LIGHT,
    title: str | None = None,
    height: int = 460,
) -> go.Figure:
    r"""Özgün sinyal ile kısmi toplamı aynı grafikte çizer; isterse harmonikleri alt panelde.

    Args:
        t: Zaman noktaları.
        original: :math:`f(t)`.
        approximation: :math:`S_N(t)` (``None`` ise çizilmez).
        n_terms: :math:`N` (etiket için).
        harmonics: ``(K, len(t))`` biçiminde tek tek harmonik terimleri (katman seçeneği).
        harmonic_indices: Her satırın harmonik numarası :math:`n`.
        approximation_label: Kısmi toplam için özel etiket.
        palette: Renk paleti.
        title: Başlık.
        height: Piksel yüksekliği.
    """
    has_layer = harmonics is not None and np.asarray(harmonics).size > 0
    if has_layer:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.62, 0.38],
            vertical_spacing=0.08,
            subplot_titles=("Sinyal ve kısmi toplam", "Tek tek harmonikler"),
        )
        height = max(height, 620)
    else:
        fig = go.Figure()
    row = {"row": 1, "col": 1} if has_layer else {}
    fig.add_trace(_line(t, original, "f(t) — özgün", palette.ink_secondary), **row)
    if approximation is not None:
        label = approximation_label or f"S<sub>N</sub>(t), N = {n_terms}"
        fig.add_trace(_line(t, approximation, label, palette.color(0)), **row)
    if has_layer:
        rows = np.atleast_2d(np.asarray(harmonics, dtype=np.float64))
        indices = list(harmonic_indices or range(1, rows.shape[0] + 1))
        colors = palette.ordinal(len(indices))
        for k, (values, n) in enumerate(zip(rows, indices, strict=True)):
            name = "a₀/2 (sabit)" if n == 0 else f"n = {n}"
            fig.add_trace(
                _line(
                    t,
                    values,
                    name,
                    colors[k],
                    width=1.5,
                    legendgroup="harmonics",
                    legendgrouptitle={"text": "Harmonikler"} if k == 0 else None,
                    showlegend=len(indices) <= 12,
                ),
                row=2,
                col=1,
            )
        fig.update_xaxes(title_text="t", row=2, col=1)
    else:
        fig.update_xaxes(title_text="t")
    _base_layout(fig, title=title, height=height)
    fig.update_layout(hovermode="x unified")
    return fig


def convergence_animation_figure(
    t: ArrayLike,
    original: ArrayLike,
    sums: ArrayLike,
    n_values: Sequence[int],
    *,
    palette: Palette = LIGHT,
    frame_duration: int = 350,
    title: str | None = None,
    height: int = 480,
) -> go.Figure:
    r"""N arttıkça :math:`S_N`'in sinyale yakınsamasını gösteren oynatılabilir animasyon.

    Args:
        t: Zaman noktaları.
        original: :math:`f(t)`.
        sums: ``(len(n_values), len(t))`` kısmi toplamlar.
        n_values: Her karenin :math:`N` değeri.
        palette: Renk paleti.
        frame_duration: Kare süresi (ms).
        title: Başlık.
        height: Yükseklik.
    """
    sums_arr = np.atleast_2d(np.asarray(sums, dtype=np.float64))
    if sums_arr.shape[0] != len(n_values) or len(n_values) == 0:
        raise ValueError("Her N değeri için bir kısmi toplam satırı gereklidir.")
    t_arr = np.asarray(t, dtype=np.float64)
    fig = go.Figure(
        data=[
            _line(t_arr, original, "f(t) — özgün", palette.ink_secondary),
            _line(t_arr, sums_arr[0], "S<sub>N</sub>(t)", palette.color(0)),
        ]
    )
    fig.frames = [
        go.Frame(
            name=str(n),
            data=[go.Scatter(x=t_arr, y=row)],
            traces=[1],
            layout={"title": {"text": f"{title or 'Yakınsama'} — N = {n}"}},
        )
        for n, row in zip(n_values, sums_arr, strict=True)
    ]
    finite = np.concatenate([np.asarray(original, dtype=np.float64).ravel(), sums_arr.ravel()])
    finite = finite[np.isfinite(finite)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (-1.0, 1.0)
    pad = 0.08 * (hi - lo or 1.0)
    _base_layout(fig, title=f"{title or 'Yakınsama'} — N = {n_values[0]}", height=height)
    fig.update_yaxes(range=[lo - pad, hi + pad])
    fig.update_xaxes(title_text="t")
    fig.update_layout(
        margin={"b": 130},
        updatemenus=[_play_buttons(frame_duration)],
        sliders=[_frame_slider([str(n) for n in n_values], "N = ", frame_duration)],
    )
    return fig


def _play_buttons(frame_duration: int, *, y: float = -0.2, redraw: bool = False) -> dict[str, Any]:
    return {
        "type": "buttons",
        "direction": "left",
        "showactive": False,
        "x": 0.0,
        "y": y,
        "xanchor": "left",
        "yanchor": "top",
        "pad": {"r": 8, "t": 8},
        "buttons": [
            {
                "label": "▶ Oynat",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": frame_duration, "redraw": redraw},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                        "mode": "immediate",
                    },
                ],
            },
            {
                "label": "⏸ Duraklat",
                "method": "animate",
                "args": [
                    [None],
                    {"frame": {"duration": 0, "redraw": redraw}, "mode": "immediate"},
                ],
            },
        ],
    }


def _frame_slider(
    names: Sequence[str],
    prefix: str,
    frame_duration: int,
    *,
    y: float = -0.2,
    show_labels: bool = True,
) -> dict[str, Any]:
    return {
        "active": 0,
        "x": 0.2,
        "len": 0.8,
        "y": y,
        "yanchor": "top",
        "pad": {"t": 8},
        "ticklen": 4 if show_labels else 0,
        "minorticklen": 2 if show_labels else 0,
        "currentvalue": {
            "prefix": prefix,
            "visible": show_labels,
            "xanchor": "right",
            "font": {"size": 12},
        },
        "steps": [
            {
                "label": name if show_labels else "",
                "method": "animate",
                "args": [
                    [name],
                    {
                        "frame": {"duration": frame_duration, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
            }
            for name in names
        ],
    }


# ====================================================================== spektrum
def spectrum_figure(
    x: ArrayLike,
    values: ArrayLike,
    *,
    style: SpectrumStyle = "stem",
    log_y: bool = False,
    x_label: str = "Harmonik n",
    y_label: str = "Genlik Aₙ",
    palette: Palette = LIGHT,
    title: str | None = None,
    height: int = 380,
    phase: bool = False,
) -> go.Figure:
    r"""Genlik veya faz spektrumu (çubuk ya da gövde/stem).

    Logaritmik ölçekte sıfır (veya negatif) değerler çizilemeyeceğinden atlanır. Faz grafiğinde
    y ekseni :math:`[-\pi, \pi]` aralığında π katlarıyla etiketlenir.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(values, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x ve değer dizileri aynı uzunlukta olmalıdır.")
    if log_y and not phase:
        keep = y_arr > 1e-15 * max(1.0, float(np.nanmax(np.abs(y_arr), initial=0.0)))
        x_arr, y_arr = x_arr[keep], y_arr[keep]
    color = palette.color(0)
    hover = f"{x_label}=%{{x}}<br>{y_label}=%{{y:.5g}}<extra></extra>"
    fig = go.Figure()
    if style == "bar":
        fig.add_trace(
            go.Bar(
                x=x_arr,
                y=y_arr,
                marker={"color": color, "cornerradius": 4, "line": {"width": 0}},
                name=y_label,
                hovertemplate=hover,
            )
        )
        fig.update_layout(bargap=0.35)
    elif style == "stem":
        base = 0.0
        if log_y and y_arr.size:
            base = float(y_arr.min()) / 10.0
        stems_x = np.repeat(x_arr, 3)
        stems_y = np.column_stack([np.full_like(y_arr, base), y_arr, np.full_like(y_arr, np.nan)])
        fig.add_trace(
            go.Scatter(
                x=stems_x,
                y=stems_y.ravel(),
                mode="lines",
                line={"color": color, "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_arr,
                y=y_arr,
                mode="markers",
                marker={
                    "color": color,
                    "size": MARKER_SIZE,
                    "line": {"color": palette.surface, "width": 2},
                },
                name=y_label,
                hovertemplate=hover,
                showlegend=False,
            )
        )
    else:
        raise ValueError(f"Bilinmeyen spektrum stili: {style!r} ('bar' veya 'stem').")
    _base_layout(fig, title=title, height=height)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label, type="log" if (log_y and not phase) else "linear")
    if phase:
        fig.update_yaxes(
            range=[-np.pi * 1.1, np.pi * 1.1],
            tickvals=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            ticktext=["−π", "−π/2", "0", "π/2", "π"],
        )
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


# ====================================================================== hata ve yakınsama
def error_figure(
    n_values: ArrayLike,
    series: Mapping[str, ArrayLike],
    *,
    palette: Palette = LIGHT,
    title: str | None = None,
    log_log: bool = True,
    y_label: str = "Hata",
    height: int = 420,
    highlight_n: float | None = None,
    highlight_label: str | None = None,
) -> go.Figure:
    """N'e göre hata eğrileri (varsayılan log-log). Seriler sabit renk sırasıyla çizilir.

    ``highlight_n`` verilirse o N'de ince dikey bir referans çizgisi ve etiket eklenir. (Plotly'de
    log eksende şekiller veri biriminde, açıklamalar ise log10 biriminde konumlandırılır.)
    """
    n_arr = np.asarray(n_values, dtype=np.float64)
    fig = go.Figure()
    for slot, (name, values) in enumerate(series.items()):
        y = np.asarray(values, dtype=np.float64)
        if log_log:
            y = np.where(y > 0, y, np.nan)
        fig.add_trace(
            go.Scatter(
                x=n_arr,
                y=y,
                mode="lines+markers",
                name=name,
                line={"color": palette.color(slot), "width": LINE_WIDTH},
                marker={
                    "color": palette.color(slot),
                    "size": 7,
                    "line": {"color": palette.surface, "width": 1.5},
                },
                hovertemplate=f"{name}<br>N=%{{x}}<br>hata=%{{y:.4g}}<extra></extra>",
            )
        )
    _base_layout(fig, title=title, height=height)
    axis_type = "log" if log_log else "linear"
    log_ticks = {"dtick": "D2"} if log_log else {}  # log eksende yalnızca 1-2-5 etiketleri
    fig.update_xaxes(title_text="Terim sayısı N", type=axis_type, **log_ticks)
    fig.update_yaxes(title_text=y_label, type=axis_type, exponentformat="power", **log_ticks)
    fig.update_layout(hovermode="x unified")
    if highlight_n is not None and highlight_n > 0:
        fig.add_shape(
            type="line", xref="x", yref="paper", x0=highlight_n, x1=highlight_n, y0=0, y1=1,
            line={"color": palette.muted, "width": 1},
        )  # fmt: skip
        fig.add_annotation(
            x=float(np.log10(highlight_n)) if log_log else float(highlight_n),
            xref="x", y=1.0, yref="paper", xanchor="left", yanchor="top", showarrow=False,
            text=highlight_label or f"N = {highlight_n:g}",
            font={"color": palette.muted, "size": 11},
        )  # fmt: skip
    return fig


def gibbs_trend_figure(
    n_values: ArrayLike,
    percents: ArrayLike,
    theoretical_percent: float,
    *,
    palette: Palette = LIGHT,
    title: str | None = None,
    height: int = 360,
) -> go.Figure:
    """Ölçülen Gibbs aşımının (%) N'e göre değişimi ve teorik sınır çizgisi."""
    fig = go.Figure(
        go.Scatter(
            x=np.asarray(n_values, dtype=np.float64),
            y=np.asarray(percents, dtype=np.float64),
            mode="lines+markers",
            name="Ölçülen aşım",
            line={"color": palette.color(0), "width": LINE_WIDTH},
            marker={
                "size": 8,
                "color": palette.color(0),
                "line": {"color": palette.surface, "width": 2},
            },
            hovertemplate="N=%{x}<br>aşım=%{y:.3f}%<extra></extra>",
        )
    )
    fig.add_hline(
        y=theoretical_percent,
        line={"color": palette.muted, "width": 1},
        annotation_text=f"teorik ≈ %{theoretical_percent:.2f}",
        annotation_position="bottom right",
        annotation_font={"color": palette.muted, "size": 11},
    )
    _base_layout(fig, title=title, height=height)
    ticks = [float(n) for n in np.asarray(n_values, dtype=np.float64)]
    shown = ticks if len(ticks) <= 8 else ticks[:: max(1, len(ticks) // 7)]
    fig.update_xaxes(
        title_text="Terim sayısı N",
        type="log",
        tickvals=shown,
        ticktext=[f"{v:g}" for v in shown],
        minor={"showgrid": False},
    )
    fig.update_yaxes(title_text="Aşım (sıçramanın %'si)")
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


def comparison_figure(
    t: ArrayLike,
    original: ArrayLike,
    curves: Mapping[str, ArrayLike],
    *,
    palette: Palette = LIGHT,
    title: str | None = None,
    x_range: tuple[float, float] | None = None,
    reference_levels: Mapping[str, float] | None = None,
    peaks: Mapping[str, tuple[float, float]] | None = None,
    height: int = 440,
) -> go.Figure:
    """Özgün sinyal ile birden çok yaklaşımı (ör. yumuşatma yöntemleri) karşılaştırır.

    Args:
        t: Zaman noktaları.
        original: Özgün sinyal (nötr referans rengi).
        curves: Ad → değerler (kategorik renkler sabit sırayla).
        palette: Renk paleti.
        title: Başlık.
        x_range: Yakınlaştırma aralığı (ör. süreksizlik çevresi).
        reference_levels: Ad → y seviyesi; ince yatay referans çizgileri (ör. teorik Gibbs tepesi).
        peaks: Ad → (t, y); ölçülen tepe noktalarını işaretler (eğriyle aynı renk).
        height: Yükseklik.
    """
    fig = go.Figure()
    fig.add_trace(_line(t, original, "f(t) — özgün", palette.ink_secondary))
    names = list(curves)
    for slot, name in enumerate(names):
        fig.add_trace(_line(t, curves[name], name, palette.color(slot)))
    for name, (pt, pv) in (peaks or {}).items():
        slot = names.index(name) if name in names else 0
        fig.add_trace(
            go.Scatter(
                x=[pt],
                y=[pv],
                mode="markers",
                name=f"{name} tepe",
                showlegend=False,
                marker={
                    "color": palette.color(slot),
                    "size": 10,
                    "line": {"color": palette.surface, "width": 2},
                },
                hovertemplate=f"{name} tepe<br>t=%{{x:.4g}}<br>değer=%{{y:.5g}}<extra></extra>",
            )
        )
    for label, level in (reference_levels or {}).items():
        fig.add_hline(
            y=level,
            line={"color": palette.muted, "width": 1},
            annotation_text=label,
            annotation_position="top left",
            annotation_font={"color": palette.muted, "size": 11},
        )
    _base_layout(fig, title=title, height=height)
    fig.update_xaxes(title_text="t", range=list(x_range) if x_range else None)
    fig.update_layout(hovermode="x unified")
    return fig


def heatmap_figure(
    matrix: ArrayLike,
    labels: Sequence[str],
    *,
    palette: Palette = LIGHT,
    title: str | None = None,
    height: int = 460,
) -> go.Figure:
    """Tek tonlu sıralı rampa ile ısı haritası (ör. ortogonallik Gram matrisi)."""
    values = np.asarray(matrix, dtype=np.float64)
    ramp = list(palette.sequential)
    if palette.mode == "dark":
        ramp = ramp[::-1]
    scale = [[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)]
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=list(labels),
            y=list(labels),
            colorscale=scale,
            zmin=min(0.0, float(values.min())),
            zmax=max(1.0, float(values.max())),
            xgap=2,
            ygap=2,
            hovertemplate="⟨%{y}, %{x}⟩ = %{z:.2e}<extra></extra>",
            colorbar={"thickness": 12, "outlinewidth": 0},
        )
    )
    _base_layout(fig, title=title, height=height)
    fig.update_xaxes(showgrid=False, ticks="")
    fig.update_yaxes(showgrid=False, ticks="", autorange="reversed", scaleanchor="x")
    return fig


# ====================================================================== epicycle
def _circle_polyline(centers: np.ndarray, radii: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Birden çok çemberi NaN ile ayrılmış tek bir çoklu çizgiye çevirir."""
    theta = np.linspace(0.0, 2.0 * np.pi, CIRCLE_RESOLUTION)
    ring = np.exp(1j * theta)
    pts = centers[:, None] + radii[:, None] * ring[None, :]
    pts = np.concatenate([pts, np.full((len(radii), 1), np.nan + 1j * np.nan)], axis=1).ravel()
    return pts.real, pts.imag


def epicycle_frames(
    epicycles: EpicycleSet,
    n_frames: int = 120,
    trail_fraction: float = 1.0,
    trail_oversample: int = 4,
    max_drawn_circles: int = MAX_DRAWN_CIRCLES,
) -> dict[str, Any]:
    """Epicycle animasyonu için tüm karelerin geometrisini hesaplar (Plotly ve Matplotlib ortak).

    Returns:
        ``s`` (kare zamanları), ``joints`` (``(F, K+1)`` karmaşık), ``trail`` (her kare için
        karmaşık iz dizileri listesi), ``circle_count`` ve ``path`` (yoğun uç yolu) anahtarları.
    """
    if n_frames < 2:
        raise ValueError("En az 2 kare gereklidir.")
    if not 0.0 < trail_fraction <= 1.0:
        raise ValueError("İz uzunluğu 0 ile 1 arasında olmalıdır (0 hariç).")
    s = np.arange(n_frames) / n_frames
    joints = epicycles.joints(s)
    dense = n_frames * trail_oversample
    path = epicycles.evaluate(np.arange(dense) / dense)
    window = max(2, round(trail_fraction * dense))
    trails = []
    for k in range(n_frames):
        head = k * trail_oversample
        if trail_fraction >= 1.0:
            idx = np.arange(0, head + 1)
        else:
            idx = np.arange(head - window + 1, head + 1) % dense
        trails.append(np.concatenate([path[idx], joints[k, -1:]]))
    return {
        "s": s,
        "joints": joints,
        "trail": trails,
        "circle_count": min(epicycles.n_circles, max_drawn_circles),
        "path": path,
    }


def epicycle_figure(
    epicycles: EpicycleSet,
    *,
    n_frames: int = 120,
    trail_fraction: float = 1.0,
    show_circles: bool = True,
    target: ArrayLike | None = None,
    frame_duration: int = 40,
    max_drawn_circles: int = MAX_DRAWN_CIRCLES,
    palette: Palette = LIGHT,
    title: str | None = None,
    height: int = 620,
) -> go.Figure:
    """Dönen çemberlerin şekli çizdiği oynatılabilir Plotly animasyonu.

    İzler: 0 hedef şekil (soluk), 1 çemberler, 2 kollar ve eklemler, 3 iz, 4 kalem ucu.

    Args:
        epicycles: Çember zinciri.
        n_frames: Bir tam tur için kare sayısı.
        trail_fraction: İzin tam yola oranı (1 → şekil baştan çizilir; <1 → kuyruklu yıldız).
        show_circles: Çemberleri göster.
        target: Karşılaştırma için hedef nokta dizisi ``(M, 2)``.
        frame_duration: Kare süresi (ms) — hız ayarı.
        max_drawn_circles: Çizilecek en fazla çember (performans için; kollar hepsi için çizilir).
        palette: Renk paleti.
        title: Başlık.
        height: Yükseklik.
    """
    geo = epicycle_frames(epicycles, n_frames, trail_fraction, max_drawn_circles=max_drawn_circles)
    joints: np.ndarray = geo["joints"]
    count: int = geo["circle_count"] if show_circles else 0
    radii = epicycles.radii[:count]

    def circles_trace(k: int) -> go.Scatter:
        if count == 0:
            return go.Scatter(x=[], y=[])
        cx, cy = _circle_polyline(joints[k, :count], radii)
        return go.Scatter(x=cx.astype(np.float32), y=cy.astype(np.float32))

    def arms_trace(k: int) -> go.Scatter:
        pts = joints[k]
        return go.Scatter(x=pts.real.astype(np.float32), y=pts.imag.astype(np.float32))

    def trail_trace(k: int) -> go.Scatter:
        pts = geo["trail"][k]
        return go.Scatter(x=pts.real.astype(np.float32), y=pts.imag.astype(np.float32))

    def tip_trace(k: int) -> go.Scatter:
        tip = joints[k, -1]
        return go.Scatter(x=[tip.real], y=[tip.imag])

    target_pts = np.asarray(target, dtype=np.float64) if target is not None else np.empty((0, 2))
    if target_pts.size:
        target_pts = np.vstack([target_pts, target_pts[:1]])
    fig = go.Figure(
        data=[
            go.Scatter(
                x=target_pts[:, 0] if target_pts.size else [],
                y=target_pts[:, 1] if target_pts.size else [],
                mode="lines",
                name="Hedef şekil",
                line={"color": palette.rgba(palette.muted, 0.45), "width": 1.5},
                hoverinfo="skip",
            ),
            go.Scatter(
                circles_trace(0),
                mode="lines",
                name="Çemberler",
                line={"color": palette.rgba(palette.muted, 0.55), "width": 1},
                hoverinfo="skip",
                showlegend=count > 0,
            ),
            go.Scatter(
                arms_trace(0),
                mode="lines+markers",
                name="Vektörler",
                line={"color": palette.ink_secondary, "width": 1.2},
                marker={"size": 3, "color": palette.ink_secondary},
                hoverinfo="skip",
            ),
            go.Scatter(
                trail_trace(0),
                mode="lines",
                name="Çizilen iz",
                line={"color": palette.color(0), "width": LINE_WIDTH},
                hoverinfo="skip",
            ),
            go.Scatter(
                tip_trace(0),
                mode="markers",
                name="Kalem ucu",
                marker={
                    "color": palette.color(1),
                    "size": 10,
                    "line": {"color": palette.surface, "width": 2},
                },
                hoverinfo="skip",
            ),
        ]
    )
    fig.frames = [
        go.Frame(
            name=str(k),
            data=[circles_trace(k), arms_trace(k), trail_trace(k), tip_trace(k)],
            traces=[1, 2, 3, 4],
        )
        for k in range(n_frames)
    ]
    extent = _epicycle_extent(joints, geo["path"], target_pts, radii)
    _base_layout(fig, title=title, height=height)
    fig.update_xaxes(range=extent[0], visible=False, showgrid=False)
    fig.update_yaxes(range=extent[1], visible=False, showgrid=False, scaleanchor="x")
    fig.update_layout(
        hovermode=False,
        margin={"l": 8, "r": 8, "b": 70},
        updatemenus=[_play_buttons(frame_duration, y=-0.02)],
        sliders=[
            _frame_slider(
                [str(k) for k in range(n_frames)], "", frame_duration, y=-0.02, show_labels=False
            )
        ],
    )
    return fig


def _epicycle_extent(
    joints: np.ndarray, path: np.ndarray, target: np.ndarray, radii: FloatArray
) -> tuple[list[float], list[float]]:
    """Tüm karelerde çizilen her şeyi (eklemler, iz, hedef, çemberler) kapsayan en dar kutu."""
    xs = [joints.real.ravel(), path.real]
    ys = [joints.imag.ravel(), path.imag]
    if target.size:
        xs.append(target[:, 0])
        ys.append(target[:, 1])
    if radii.size:
        centers = joints[:, : radii.size]
        xs += [(centers.real - radii).ravel(), (centers.real + radii).ravel()]
        ys += [(centers.imag - radii).ravel(), (centers.imag + radii).ravel()]
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    x_lo, x_hi = float(x.min()), float(x.max())
    y_lo, y_hi = float(y.min()), float(y.max())
    pad = 0.04 * max(x_hi - x_lo, y_hi - y_lo, 1e-9)
    return [x_lo - pad, x_hi + pad], [y_lo - pad, y_hi + pad]


def drawing_canvas_figure(
    points: ArrayLike | None = None,
    *,
    grid_size: int = 41,
    palette: Palette = LIGHT,
    height: int = 480,
) -> go.Figure:
    """Kement (lasso) seçimiyle serbest çizim için tuval.

    Kement seçimi yalnızca veri noktası içeren grafiklerde çalıştığından ``[-1, 1]²``
    üzerinde soluk bir nokta ızgarası çizilir. Kullanıcının kement yolu Streamlit seçim
    olayında ``lasso[0]["x"]``, ``lasso[0]["y"]`` olarak döner.
    """
    g = np.linspace(-1.0, 1.0, grid_size)
    gx, gy = np.meshgrid(g, g)
    fig = go.Figure(
        go.Scatter(
            x=gx.ravel(),
            y=gy.ravel(),
            mode="markers",
            marker={"size": 3, "color": palette.rgba(palette.muted, 0.35)},
            hoverinfo="skip",
            name="tuval",
            showlegend=False,
        )
    )
    if points is not None:
        pts = np.asarray(points, dtype=np.float64)
        if pts.size:
            ring = np.vstack([pts, pts[:1]])
            fig.add_trace(
                go.Scatter(
                    x=ring[:, 0],
                    y=ring[:, 1],
                    mode="lines",
                    name="Çizimin",
                    line={"color": palette.color(0), "width": LINE_WIDTH},
                    hoverinfo="skip",
                )
            )
    _base_layout(fig, height=height)
    fig.update_xaxes(range=[-1.1, 1.1], visible=False, fixedrange=True)
    fig.update_yaxes(range=[-1.1, 1.1], visible=False, fixedrange=True, scaleanchor="x")
    fig.update_layout(dragmode="lasso", margin={"l": 8, "r": 8, "t": 8, "b": 8}, showlegend=False)
    return fig
