r"""SVG ``path`` verisi (``d`` özniteliği) ayrıştırıcı.

Desteklenen komutlar (büyük harf mutlak, küçük harf göreli koordinat):

* ``M/m`` taşı, ``L/l`` çizgi, ``H/h`` yatay, ``V/v`` dikey, ``Z/z`` kapat
* ``C/c`` kübik Bézier, ``S/s`` yumuşak kübik (önceki kontrol noktası yansıtılır)
* ``Q/q`` kuadratik Bézier, ``T/t`` yumuşak kuadratik
* ``A/a`` eliptik yay (SVG 1.1 Ek F.6.5 uç nokta → merkez dönüşümü)

Kübik Bézier: :math:`B(s) = (1-s)^3P_0 + 3(1-s)^2 s P_1 + 3(1-s)s^2 P_2 + s^3 P_3`.

Güvenlik: Bir XML ayrıştırıcısı **kullanılmaz** (XXE / "billion laughs" riskleri yoktur);
tam bir SVG dosyası verilirse ``d="..."`` öznitelikleri basit bir düzenli ifadeyle çıkarılır.
Dönüşümler (``transform``), ``<circle>``/``<rect>`` gibi öğeler desteklenmez.
"""

from __future__ import annotations

import math
import re
from typing import Final

import numpy as np

from fourier_viz.core.paths import MAX_TEXT_LENGTH, PathError, as_points
from fourier_viz.core.types import FloatArray

_NUMBER: Final = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_SEPARATORS: Final = re.compile(r"[\s,]*")
_COMMANDS: Final[str] = "MmLlHhVvCcSsQqTtAaZz"
_ARITY: Final[dict[str, int]] = {
    "M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0,
}  # fmt: skip
_D_ATTRIBUTE: Final = re.compile(r"""\sd\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE)

MAX_SEGMENTS: Final[int] = 20_000


class _Tokenizer:
    """``d`` metninde imleçle ilerleyen basit tarayıcı (bayraklar tek karakter olabilir)."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def _skip(self) -> None:
        match = _SEPARATORS.match(self.text, self.pos)
        if match:
            self.pos = match.end()

    def at_end(self) -> bool:
        self._skip()
        return self.pos >= len(self.text)

    def peek_command(self) -> str | None:
        self._skip()
        if self.pos < len(self.text) and self.text[self.pos] in _COMMANDS:
            return self.text[self.pos]
        return None

    def command(self) -> str:
        cmd = self.peek_command()
        if cmd is None:
            snippet = self.text[self.pos : self.pos + 10]
            raise PathError(f"SVG yolunda komut bekleniyordu, bulunan: {snippet!r}.")
        self.pos += 1
        return cmd

    def number(self) -> float:
        self._skip()
        match = _NUMBER.match(self.text, self.pos)
        if match is None:
            char = self.text[self.pos : self.pos + 1]
            if char.isalpha() and char not in _COMMANDS:
                raise PathError(f"Bilinmeyen SVG komutu: {char!r}.")
            snippet = self.text[self.pos : self.pos + 10] or "metin sonu"
            raise PathError(f"SVG yolunda sayı bekleniyordu, bulunan: {snippet!r}.")
        self.pos = match.end()
        return float(match.group())

    def flag(self) -> bool:
        self._skip()
        if self.pos < len(self.text) and self.text[self.pos] in "01":
            value = self.text[self.pos] == "1"
            self.pos += 1
            return value
        raise PathError("SVG yay (A) komutunda 0/1 bayrağı bekleniyordu.")

    def has_number(self) -> bool:
        self._skip()
        return _NUMBER.match(self.text, self.pos) is not None


def _cubic(p0: complex, p1: complex, p2: complex, p3: complex, samples: int) -> list[complex]:
    s = np.linspace(0.0, 1.0, samples + 1)[1:]
    pts = (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * p1 + 3 * (1 - s) * s**2 * p2 + s**3 * p3
    return list(pts)


def _quadratic(p0: complex, p1: complex, p2: complex, samples: int) -> list[complex]:
    s = np.linspace(0.0, 1.0, samples + 1)[1:]
    pts = (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s**2 * p2
    return list(pts)


def _angle(ux: float, uy: float, vx: float, vy: float) -> float:
    return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)


def arc_points(
    start: complex,
    rx: float,
    ry: float,
    rotation_deg: float,
    large_arc: bool,
    sweep: bool,
    end: complex,
    samples: int,
) -> list[complex]:
    """SVG eliptik yayını örnekler (SVG 1.1, Ek F.6.5 / F.6.6).

    Yarıçaplardan biri 0 ise düz çizgi, uç noktalar aynıysa boş liste döner; yarıçaplar yayı
    çizmeye yetmiyorsa ölçeklenir.
    """
    if start == end:
        return []
    rx, ry = abs(rx), abs(ry)
    if rx == 0.0 or ry == 0.0:
        return [end]
    phi = math.radians(rotation_deg % 360.0)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    dx, dy = (start.real - end.real) / 2.0, (start.imag - end.imag) / 2.0
    x1p = cos_p * dx + sin_p * dy
    y1p = -sin_p * dx + cos_p * dy
    lam = x1p**2 / rx**2 + y1p**2 / ry**2
    if lam > 1.0:
        scale = math.sqrt(lam)
        rx, ry = rx * scale, ry * scale
    num = rx**2 * ry**2 - rx**2 * y1p**2 - ry**2 * x1p**2
    den = rx**2 * y1p**2 + ry**2 * x1p**2
    coef = math.sqrt(max(0.0, num / den)) * (-1.0 if large_arc == sweep else 1.0)
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    cx = cos_p * cxp - sin_p * cyp + (start.real + end.real) / 2.0
    cy = sin_p * cxp + cos_p * cyp + (start.imag + end.imag) / 2.0
    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx, vy = (-x1p - cxp) / rx, (-y1p - cyp) / ry
    theta1 = _angle(1.0, 0.0, ux, uy)
    delta = _angle(ux, uy, vx, vy)
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi
    theta = theta1 + delta * np.linspace(0.0, 1.0, samples + 1)[1:]
    x = cos_p * rx * np.cos(theta) - sin_p * ry * np.sin(theta) + cx
    y = sin_p * rx * np.cos(theta) + cos_p * ry * np.sin(theta) + cy
    points = list(x + 1j * y)
    points[-1] = end  # yuvarlama hatasını gider
    return points


def parse_path_data(d: str, samples_per_segment: int = 24) -> list[FloatArray]:
    """SVG ``d`` metnini alt yollara (``(K, 2)`` dizileri) ayırarak örnekler.

    Args:
        d: SVG yol verisi, ör. ``"M 0 0 L 10 0 C 20 0, 20 10, 10 10 Z"``.
        samples_per_segment: Eğri (Bézier/yay) başına örnek sayısı.

    Returns:
        Alt yolların listesi (SVG koordinatlarında; y ekseni aşağı doğru).

    Raises:
        PathError: Sözdizimi hatası veya hiç çizim komutu yoksa.
    """
    if len(d) > MAX_TEXT_LENGTH:
        raise PathError("SVG yol verisi çok büyük.")
    samples = max(2, int(samples_per_segment))
    tok = _Tokenizer(d)
    subpaths: list[list[complex]] = []
    current: list[complex] = []
    pos = 0j
    start = 0j
    last_ctrl: complex | None = None
    last_cmd = ""
    cmd = ""
    segments = 0

    def point(rel: bool) -> complex:
        x, y = tok.number(), tok.number()
        return complex(x, y) + (pos if rel else 0j)

    while not tok.at_end():
        explicit = tok.peek_command()
        if explicit is not None:
            cmd = tok.command()
        elif not cmd or cmd in "Zz":
            raise PathError("SVG yolu bir komutla (ör. 'M') başlamalıdır.")
        upper, rel = cmd.upper(), cmd.islower()
        segments += 1
        if segments > MAX_SEGMENTS:
            raise PathError(f"SVG yolu en fazla {MAX_SEGMENTS} segment içerebilir.")

        if upper == "Z":
            if current:
                current.append(start)
                subpaths.append(current)
                current = []
            pos = start
            last_ctrl = None
            last_cmd = "Z"
            continue
        if upper == "M":
            if current:
                subpaths.append(current)
            pos = point(rel)
            start = pos
            current = [pos]
            last_ctrl = None
            last_cmd = "M"
            cmd = "l" if rel else "L"  # M sonrasındaki koordinat çiftleri örtük L'dir
            continue
        if not current:
            current = [pos]
        new_ctrl: complex | None = None
        if upper == "L":
            pos = point(rel)
            current.append(pos)
        elif upper == "H":
            x = tok.number()
            pos = complex(x + (pos.real if rel else 0.0), pos.imag)
            current.append(pos)
        elif upper == "V":
            y = tok.number()
            pos = complex(pos.real, y + (pos.imag if rel else 0.0))
            current.append(pos)
        elif upper in "CS":
            if upper == "C":
                c1 = point(rel)
            else:
                c1 = 2 * pos - last_ctrl if last_ctrl is not None and last_cmd in "CS" else pos
            c2, end = point(rel), point(rel)
            current.extend(_cubic(pos, c1, c2, end, samples))
            pos, new_ctrl = end, c2
        elif upper in "QT":
            if upper == "Q":
                c1 = point(rel)
            else:
                c1 = 2 * pos - last_ctrl if last_ctrl is not None and last_cmd in "QT" else pos
            end = point(rel)
            current.extend(_quadratic(pos, c1, end, samples))
            pos, new_ctrl = end, c1
        else:  # "A"
            rx, ry, rot = tok.number(), tok.number(), tok.number()
            large, sweep = tok.flag(), tok.flag()
            end = point(rel)
            current.extend(arc_points(pos, rx, ry, rot, large, sweep, end, samples))
            pos = end
        last_ctrl = new_ctrl
        last_cmd = upper
        if tok.peek_command() is None and not tok.has_number() and not tok.at_end():
            snippet = tok.text[tok.pos : tok.pos + 10]
            raise PathError(f"SVG yolunda beklenmeyen karakter: {snippet!r}.")

    if current:
        subpaths.append(current)
    result = [as_points(np.array(sp)) for sp in subpaths if len(sp) >= 2]
    if not result:
        raise PathError("SVG yolunda çizilebilir bir segment bulunamadı.")
    return result


def extract_path_data(svg_text: str) -> list[str]:
    """Tam bir SVG belgesinden tüm ``d="..."`` özniteliklerini çıkarır (XML ayrıştırmadan)."""
    if len(svg_text) > MAX_TEXT_LENGTH:
        raise PathError("SVG dosyası çok büyük.")
    return [
        m.group(1) if m.group(1) is not None else m.group(2)
        for m in _D_ATTRIBUTE.finditer(svg_text)
    ]


def svg_to_points(text: str, samples_per_segment: int = 24, flip_y: bool = True) -> FloatArray:
    """SVG belgesini veya çıplak ``d`` metnini tek bir nokta dizisine çevirir.

    Tüm ``path`` öğeleri ve alt yollar sırayla birleştirilir. SVG'de y ekseni aşağı baktığı
    için ``flip_y=True`` iken y koordinatları ters çevrilir.

    Raises:
        PathError: Hiç yol verisi bulunamazsa veya ayrıştırma başarısızsa.
    """
    stripped = text.strip()
    if not stripped:
        raise PathError("SVG içeriği boş.")
    if "<" in stripped:
        datas = extract_path_data(stripped)
        if not datas:
            raise PathError("SVG dosyasında 'd' öznitelikli bir <path> öğesi bulunamadı.")
    else:
        datas = [stripped]
    parts = [sub for d in datas for sub in parse_path_data(d, samples_per_segment)]
    points = np.vstack(parts)
    if flip_y:
        points[:, 1] *= -1.0
    return points
