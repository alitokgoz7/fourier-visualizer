"""Matplotlib ile epicycle ve yakınsama animasyonları; GIF/MP4 dışa aktarma.

GIF, Pillow ile; MP4 ise ``ffmpeg`` ile yazılır. Sistemde ``ffmpeg`` yoksa ``imageio-ffmpeg``
paketinin getirdiği gömülü ikili kullanılır. Animasyonlar bellek içi bayt olarak döndürülür
(Streamlit indirme düğmesi için).
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib import animation, rcParams
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from fourier_viz.core.complex_dft import EpicycleSet
from fourier_viz.viz.plots import epicycle_extent, epicycle_frames
from fourier_viz.viz.static import draw_epicycles, place_legend, style_axes
from fourier_viz.viz.theme import LIGHT, Palette

AnimationFormat = Literal["gif", "mp4"]


class AnimationError(RuntimeError):
    """Animasyon üretilemediğinde (ör. ffmpeg bulunamadı) fırlatılır."""


def ffmpeg_path() -> str | None:
    """Kullanılabilir ``ffmpeg`` ikilisinin yolu (yoksa ``None``)."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        path: str = imageio_ffmpeg.get_ffmpeg_exe()
        return path
    except Exception:  # pragma: no cover - paket/ikili eksikse
        return None


def ffmpeg_available() -> bool:
    """MP4 dışa aktarma mümkün mü."""
    return ffmpeg_path() is not None


def _make_writer(fmt: AnimationFormat, fps: int) -> animation.AbstractMovieWriter:
    """Biçime uygun yazıcıyı oluşturur (animasyon kurulmadan önce doğrulama için)."""
    if fmt == "gif":
        return animation.PillowWriter(fps=fps)
    if fmt == "mp4":
        path = ffmpeg_path()
        if path is None:
            raise AnimationError("MP4 için ffmpeg bulunamadı; GIF olarak dışa aktarmayı deneyin.")
        rcParams["animation.ffmpeg_path"] = path
        return animation.FFMpegWriter(
            fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-preset", "fast"]
        )
    raise AnimationError(f"Bilinmeyen animasyon biçimi: {fmt!r} ('gif' veya 'mp4').")


def _save(
    anim: animation.FuncAnimation,
    writer: animation.AbstractMovieWriter,
    fmt: AnimationFormat,
    dpi: int,
) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / f"animation.{fmt}"
        anim.save(str(target), writer=writer, dpi=dpi)
        return target.read_bytes()


def epicycle_animation(
    epicycles: EpicycleSet,
    fmt: AnimationFormat = "gif",
    *,
    n_frames: int = 90,
    fps: int = 20,
    trail_fraction: float = 1.0,
    show_circles: bool = True,
    target: ArrayLike | None = None,
    max_drawn_circles: int = 60,
    size_px: int = 480,
    palette: Palette = LIGHT,
) -> bytes:
    """Epicycle çiziminin GIF/MP4 animasyonunu üretir.

    Args:
        epicycles: Çember zinciri.
        fmt: ``"gif"`` veya ``"mp4"``.
        n_frames: Tam tur için kare sayısı.
        fps: Saniyedeki kare.
        trail_fraction: İz uzunluğu oranı.
        show_circles: Çemberleri çiz.
        target: Hedef şekil noktaları (soluk arka plan).
        max_drawn_circles: En fazla çizilecek çember.
        size_px: Kare boyutu (piksel, kare görüntü).
        palette: Renk paleti.

    Returns:
        Dosya baytları.
    """
    writer = _make_writer(fmt, fps)
    geo = epicycle_frames(epicycles, n_frames, trail_fraction, max_drawn_circles=max_drawn_circles)
    dpi = 100
    fig = Figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi, facecolor=palette.surface)
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.96))
    target_pts = np.asarray(target, dtype=np.float64) if target is not None else np.empty((0, 2))
    drawn_radii = epicycles.radii[: geo["circle_count"]] if show_circles else epicycles.radii[:0]
    (lo_x, hi_x), (lo_y, hi_y) = epicycle_extent(
        geo["joints"], geo["path"], target_pts, drawn_radii
    )
    half = 0.5 * max(hi_x - lo_x, hi_y - lo_y, 1e-9)
    cx, cy = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2

    def draw(k: int) -> list[Artist]:
        ax.clear()
        ax.set_facecolor(palette.surface)
        draw_epicycles(
            ax, epicycles, float(geo["s"][k]), geo["trail"][k], target=target,
            show_circles=show_circles, max_drawn_circles=max_drawn_circles, palette=palette,
        )  # fmt: skip
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=n_frames, interval=1000 / fps, blit=False)
    return _save(anim, writer, fmt, dpi)


def convergence_animation(
    t: ArrayLike,
    original: ArrayLike,
    sums: ArrayLike,
    n_values: Sequence[int],
    fmt: AnimationFormat = "gif",
    *,
    fps: int = 4,
    title: str = "Fourier serisinin yakınsaması",
    size: tuple[float, float] = (7.2, 4.0),
    palette: Palette = LIGHT,
) -> bytes:
    """N arttıkça kısmi toplamın sinyale yakınsamasını gösteren GIF/MP4."""
    sums_arr = np.atleast_2d(np.asarray(sums, dtype=np.float64))
    if sums_arr.shape[0] != len(n_values) or len(n_values) == 0:
        raise AnimationError("Her N değeri için bir kısmi toplam satırı gereklidir.")
    writer = _make_writer(fmt, fps)
    dpi = 100
    fig = Figure(figsize=size, dpi=dpi, facecolor=palette.surface)
    ax = fig.add_subplot()
    style_axes(ax, palette)
    t_arr = np.asarray(t, dtype=np.float64)
    orig = np.asarray(original, dtype=np.float64)
    ax.plot(t_arr, orig, color=palette.ink_secondary, lw=2, label="f(t) — özgün")
    (line,) = ax.plot(t_arr, sums_arr[0], color=palette.color(0), lw=2, label="$S_N(t)$")
    lo = float(min(orig.min(), sums_arr.min()))
    hi = float(max(orig.max(), sums_arr.max()))
    pad = 0.08 * (hi - lo or 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("t")
    place_legend(ax, palette)
    heading = ax.set_title(f"{title} — N = {n_values[0]}", loc="left", fontsize=11)
    fig.tight_layout()

    def draw(k: int) -> list[Artist]:
        line.set_ydata(sums_arr[k])
        heading.set_text(f"{title} — N = {n_values[k]}")
        return [line, heading]

    anim = animation.FuncAnimation(fig, draw, frames=len(n_values), interval=1000 / fps)
    return _save(anim, writer, fmt, dpi)
