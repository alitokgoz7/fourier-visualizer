r"""2D kapalı yollar: hazır şekiller, CSV ayrıştırma ve eşit yay uzunluğuna göre yeniden örnekleme.

Epicycle çizimi için bir yol :math:`z_m = x_m + i y_m` noktalarıyla verilir ve DFT, noktaların
eşit "zaman" aralıklarıyla örneklendiğini varsayar. Noktalar eğri boyunca düzensiz dağılmışsa
(ör. bir poligonun köşeleri) çizim hızı değişken olur ve yüksek frekanslı gereksiz terimler
ortaya çıkar. Bu yüzden yollar **eşit yay uzunluğuna** göre yeniden örneklenir:

.. math::

    s_j = \sum_{i<j} \lVert p_{i+1} - p_i \rVert, \qquad
    L = s_M \;(\text{kapalı yol için son nokta ilk noktaya bağlanır}),

ve yeni noktalar :math:`s = kL/M'` (:math:`k = 0..M'-1`) konumlarında doğrusal
interpolasyonla bulunur.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import ArrayLike

from fourier_viz.core.types import FloatArray

MAX_POINTS: Final[int] = 200_000
"""Bir yolda izin verilen en fazla nokta sayısı."""

MAX_TEXT_LENGTH: Final[int] = 5_000_000
"""Yüklenen CSV/SVG metninin en fazla karakter sayısı."""


class PathError(ValueError):
    """Yol/şekil hataları (Türkçe, kullanıcıya gösterilebilir)."""


# ====================================================================== temel işlemler
def as_points(points: ArrayLike) -> FloatArray:
    r"""Girdiyi ``(M, 2)`` biçimli ``float64`` diziye çevirir ve doğrular.

    Karmaşık diziler :math:`(\operatorname{Re}, \operatorname{Im})` olarak yorumlanır.

    Raises:
        PathError: Boş, yanlış biçimli, NaN/sonsuz içeren veya çok büyük girdi.
    """
    arr = np.asarray(points)
    if np.iscomplexobj(arr):
        arr = np.column_stack([arr.real.ravel(), arr.imag.ravel()])
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        raise PathError("Yol boş: en az iki farklı nokta gereklidir.")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise PathError(f"Noktalar (x, y) çiftleri olmalıdır; verilen biçim {arr.shape}.")
    if arr.shape[0] > MAX_POINTS:
        raise PathError(f"Yol en fazla {MAX_POINTS} nokta içerebilir.")
    if not np.all(np.isfinite(arr)):
        raise PathError("Yol NaN veya sonsuz koordinat içeriyor.")
    return arr


def remove_duplicates(points: ArrayLike, closed: bool = True, tol: float = 1e-12) -> FloatArray:
    """Ardışık tekrar eden noktaları (ve kapalı yolda ilk noktayı tekrarlayan son noktayı) atar."""
    pts = as_points(points)
    scale = max(1.0, float(np.abs(pts).max()))
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > tol * scale
    pts = pts[keep]
    if closed and len(pts) > 1 and np.linalg.norm(pts[-1] - pts[0]) <= tol * scale:
        pts = pts[:-1]
    return pts


def cumulative_arclength(points: ArrayLike, closed: bool = True) -> FloatArray:
    r"""Kümülatif yay uzunluğu :math:`s_0 = 0, s_1, \dots`.

    Kapalı yolda son nokta ilk noktaya bağlanır; dönen dizinin uzunluğu :math:`M + 1` olur ve son
    eleman toplam uzunluk :math:`L`'dir. Açık yolda uzunluk :math:`M`'dir.
    """
    pts = as_points(points)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def path_length(points: ArrayLike, closed: bool = True) -> float:
    """Yolun toplam uzunluğu."""
    return float(cumulative_arclength(points, closed)[-1])


def resample_by_arclength(points: ArrayLike, n_points: int, closed: bool = True) -> FloatArray:
    """Yolu eşit yay uzunluğu aralıklarıyla ``n_points`` noktaya yeniden örnekler.

    Kapalı yolda :math:`s_k = kL/n` (:math:`k = 0..n-1`) kullanılır; böylece son noktadan ilk
    noktaya olan mesafe de diğer aralıklara eşittir ve ilk nokta korunur. Açık yolda uç noktalar
    dahil :math:`s_k = kL/(n-1)` kullanılır.

    Args:
        points: ``(M, 2)`` noktalar veya karmaşık dizi.
        n_points: Yeni nokta sayısı (en az 2).
        closed: Yol kapalı mı.

    Returns:
        ``(n_points, 2)`` dizi.

    Raises:
        PathError: Yol boşsa, tek (farklı) noktadan oluşuyorsa veya uzunluğu sıfırsa.
    """
    if int(n_points) < 2:
        raise PathError("Yeniden örnekleme için en az 2 nokta istenmelidir.")
    if int(n_points) > MAX_POINTS:
        raise PathError(f"En fazla {MAX_POINTS} nokta istenebilir.")
    pts = remove_duplicates(points, closed)
    if len(pts) < 2:
        raise PathError("Yol tek bir noktadan oluşuyor: en az iki farklı nokta gereklidir.")
    s = cumulative_arclength(pts, closed)
    total = s[-1]
    if total <= 0.0:  # pragma: no cover - remove_duplicates sonrası erişilemez
        raise PathError("Yolun uzunluğu sıfır.")
    ring = np.vstack([pts, pts[:1]]) if closed else pts
    targets = np.linspace(0.0, total, int(n_points), endpoint=not closed)
    x = np.interp(targets, s, ring[:, 0])
    y = np.interp(targets, s, ring[:, 1])
    return np.column_stack([x, y])


def normalize_path(points: ArrayLike, half_size: float = 1.0) -> FloatArray:
    """Yolu sınırlayıcı kutusunun merkezine taşır ve en büyük yarı-boyutu ``half_size`` yapar."""
    pts = as_points(points)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = 0.5 * (lo + hi)
    extent = float(0.5 * (hi - lo).max())
    if extent == 0.0:
        raise PathError("Yolun genişliği sıfır; normalleştirilemez.")
    normalized: FloatArray = (pts - center) * (half_size / extent)
    return normalized


def reverse_path(points: ArrayLike) -> FloatArray:
    """Dolanım yönünü tersine çevirir (ilk nokta korunur)."""
    pts = as_points(points)
    return np.vstack([pts[:1], pts[:0:-1]])


# ====================================================================== hazır şekiller
def _param(n: int) -> FloatArray:
    if n < 3:
        raise PathError("Şekil için en az 3 nokta gereklidir.")
    return np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)


def heart(n: int = 1000) -> FloatArray:
    r"""Kalp eğrisi.

    :math:`x = 16\sin^3 t`, :math:`y = 13\cos t - 5\cos 2t - 2\cos 3t - \cos 4t`.
    """
    t = _param(n)
    x = 16.0 * np.sin(t) ** 3
    y = 13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t)
    return np.column_stack([x, y])


def star(n_tips: int = 5, inner_ratio: float = 0.45) -> FloatArray:
    """Düzgün yıldız poligonu (köşeler; tepe yukarıda, saat yönünün tersine).

    Args:
        n_tips: Uç sayısı (en az 3).
        inner_ratio: İç yarıçap / dış yarıçap (0–1).
    """
    k = int(n_tips)
    if k < 3:
        raise PathError("Yıldızın en az 3 ucu olmalıdır.")
    if not 0.0 < inner_ratio < 1.0:
        raise PathError("İç yarıçap oranı 0 ile 1 arasında olmalıdır.")
    angles = np.pi / 2 + np.arange(2 * k) * np.pi / k
    radii = np.where(np.arange(2 * k) % 2 == 0, 1.0, float(inner_ratio))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def infinity(n: int = 1000) -> FloatArray:
    r"""Sonsuz işareti (Bernoulli lemniskatı).

    :math:`x = \dfrac{\cos t}{1 + \sin^2 t}`, :math:`y = \dfrac{\sin t\cos t}{1 + \sin^2 t}`.
    """
    t = _param(n)
    denom = 1.0 + np.sin(t) ** 2
    return np.column_stack([np.cos(t) / denom, np.sin(t) * np.cos(t) / denom])


def square(n: int = 4) -> FloatArray:
    """Kare poligon (köşeler, saat yönünün tersine). ``n`` yok sayılır; imza uyumu içindir."""
    del n
    return np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])


def epitrochoid(
    big_r: float = 5.0, small_r: float = 1.0, distance: float = 2.0, n: int = 2000
) -> FloatArray:
    r"""Epitrokoid: :math:`R` yarıçaplı çemberin dışında yuvarlanan :math:`r` yarıçaplı çemberdeki
    merkezden :math:`d` uzaklıktaki noktanın izi.

    .. math::

        x = (R + r)\cos t - d\cos\!\Big(\frac{R+r}{r}t\Big), \quad
        y = (R + r)\sin t - d\sin\!\Big(\frac{R+r}{r}t\Big).

    :math:`R/r` rasyonel (:math:`p/q` en sade) ise eğri :math:`t \in [0, 2\pi q)` sonra kapanır.
    """  # noqa: D205
    if big_r <= 0 or small_r <= 0:
        raise PathError("Epitrokoid yarıçapları pozitif olmalıdır.")
    if distance < 0:
        raise PathError("Epitrokoid uzaklığı negatif olamaz.")
    ratio = _rational(big_r / small_r)
    turns = ratio[1]
    t = np.linspace(0.0, 2.0 * np.pi * turns, int(n) * turns, endpoint=False)
    k = (big_r + small_r) / small_r
    x = (big_r + small_r) * np.cos(t) - distance * np.cos(k * t)
    y = (big_r + small_r) * np.sin(t) - distance * np.sin(k * t)
    return np.column_stack([x, y])


def _rational(value: float, max_denominator: int = 12) -> tuple[int, int]:
    """Oranı küçük paydalı bir kesre yaklaştırır (eğrinin kapanması için tur sayısı)."""
    from fractions import Fraction

    frac = Fraction(value).limit_denominator(max_denominator)
    if frac.numerator == 0:
        raise PathError("Epitrokoid yarıçap oranı çok küçük.")
    return frac.numerator, frac.denominator


def circle(n: int = 400) -> FloatArray:
    """Birim çember (tek bir epicycle ile tam çizilir)."""
    t = _param(n)
    return np.column_stack([np.cos(t), np.sin(t)])


# ====================================================================== CSV
_CSV_SPLIT = re.compile(r"[\s,;]+")


def parse_csv_points(text: str) -> FloatArray:
    """CSV/metin içeriğinden ``x, y`` noktalarını okur.

    * Ayırıcı olarak virgül, noktalı virgül, sekme veya boşluk kabul edilir.
    * Noktalı virgülle ayrılmış satırlarda ondalık virgül (``1,5;2,25``) desteklenir.
    * ``#`` ile başlayan satırlar ve boş satırlar atlanır; ilk satır sayısal değilse başlık sayılır.
    * Satırda ikiden fazla sütun varsa ilk iki sütun kullanılır.

    Raises:
        PathError: Geçersiz satır, eksik sütun veya hiç nokta yoksa (satır numarasıyla).
    """
    if len(text) > MAX_TEXT_LENGTH:
        raise PathError("Dosya çok büyük.")
    rows: list[tuple[float, float]] = []
    header_allowed = True
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip().lstrip("﻿")
        if not line or line.startswith("#"):
            continue
        if ";" in line:
            fields = [f.strip().replace(",", ".") for f in line.split(";")]
        else:
            fields = [f for f in _CSV_SPLIT.split(line) if f]
        try:
            if len(fields) < 2:
                raise ValueError
            x, y = float(fields[0]), float(fields[1])
        except ValueError:
            if header_allowed:
                header_allowed = False
                continue
            raise PathError(
                f"{lineno}. satır okunamadı: {raw.strip()[:40]!r}. Her satırda 'x, y' bekleniyor."
            ) from None
        header_allowed = False
        if not (math.isfinite(x) and math.isfinite(y)):
            raise PathError(f"{lineno}. satırda sonlu olmayan değer var.")
        rows.append((x, y))
    if not rows:
        raise PathError("Dosyada hiç nokta bulunamadı. Beklenen biçim: her satırda 'x, y'.")
    return as_points(rows)


# ====================================================================== kayıt defteri
@dataclass(frozen=True)
class ShapeParam:
    """Arayüzde gösterilecek bir şekil parametresi."""

    name: str
    label: str
    default: float
    minimum: float
    maximum: float
    step: float
    integer: bool = False


@dataclass(frozen=True)
class ShapeInfo:
    """Kütüphanedeki bir şeklin tanımı."""

    key: str
    label: str
    factory: Callable[..., FloatArray]
    params: tuple[ShapeParam, ...] = ()


SHAPE_LIBRARY: Final[Mapping[str, ShapeInfo]] = MappingProxyType(
    {
        info.key: info
        for info in (
            ShapeInfo("heart", "Kalp", heart),
            ShapeInfo(
                "star",
                "Yıldız",
                star,
                (
                    ShapeParam("n_tips", "Uç sayısı", 5, 3, 12, 1, integer=True),
                    ShapeParam("inner_ratio", "İç yarıçap oranı", 0.45, 0.1, 0.9, 0.05),
                ),
            ),
            ShapeInfo("infinity", "Sonsuz işareti", infinity),
            ShapeInfo("square", "Kare", square),
            ShapeInfo(
                "epitrochoid",
                "Epitrokoid",
                epitrochoid,
                (
                    ShapeParam("big_r", "Sabit çember R", 5, 1, 8, 1, integer=True),
                    ShapeParam("small_r", "Yuvarlanan çember r", 1, 1, 5, 1, integer=True),
                    ShapeParam("distance", "Kalem uzaklığı d", 2.0, 0.0, 3.0, 0.1),
                ),
            ),
            ShapeInfo("circle", "Çember", circle),
        )
    }
)
"""Hazır şekillerin kayıt defteri (anahtar → :class:`ShapeInfo`)."""


def build_shape(key: str, **params: float) -> FloatArray:
    """Kütüphaneden şekil üretir.

    Raises:
        PathError: Anahtar veya parametre geçersizse.
    """
    info = SHAPE_LIBRARY.get(key)
    if info is None:
        raise PathError(f"Bilinmeyen şekil: {key!r}. Geçerli: {', '.join(SHAPE_LIBRARY)}.")
    specs = {p.name: p for p in info.params}
    unknown = set(params) - set(specs)
    if unknown:
        raise PathError(f"'{info.label}' için bilinmeyen parametre(ler): {sorted(unknown)}.")
    kwargs = {k: (int(v) if specs[k].integer else float(v)) for k, v in params.items()}
    return info.factory(**kwargs)
