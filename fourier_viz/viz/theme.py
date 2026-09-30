"""Açık/koyu tema uyumlu, renk körlüğü açısından doğrulanmış renk paleti.

Kategorik yuvalar sabit sırayla kullanılır (asla döngüsel değil); ilk dört yuva her iki modda
bitişik çiftler için CVD ΔE ≥ 8 ve normal görüş ΔE ≥ 15 eşiklerini geçer (``validate_palette``
betiğiyle doğrulanmıştır). Açık modda aqua ve sarı yüzeye karşı 3:1'in altında kaldığından bu
renkler yalnızca lejant/doğrudan etiket ya da tablo görünümüyle birlikte kullanılır.

Harmonik indisi gibi **sıralı** büyüklükler tek tonlu (mavi) sıralı rampa ile kodlanır.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

ThemeMode = Literal["light", "dark"]

FONT_FAMILY: Final[str] = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

_SEQUENTIAL_BLUE: Final[tuple[str, ...]] = (
    "#cde2fb",  # 100
    "#b7d3f6",  # 150
    "#9ec5f4",  # 200
    "#86b6ef",  # 250
    "#6da7ec",  # 300
    "#5598e7",  # 350
    "#3987e5",  # 400
    "#2a78d6",  # 450
    "#256abf",  # 500
    "#1c5cab",  # 550
    "#184f95",  # 600
    "#104281",  # 650
    "#0d366b",  # 700
)


@dataclass(frozen=True)
class Palette:
    """Bir tema modunun tüm renk rolleri.

    Attributes:
        mode: ``"light"`` veya ``"dark"``.
        surface: Grafik yüzeyi.
        page: Sayfa zemini.
        ink: Birincil metin.
        ink_secondary: İkincil metin / referans eğri (özgün sinyal).
        muted: Eksen etiketleri, yardımcı çizgiler (çemberler).
        grid: Kılavuz çizgileri (saç teli).
        axis: Taban çizgisi / eksen.
        series: Sabit sıralı kategorik renkler.
        sequential: Tek tonlu sıralı rampa (açıktan koyuya).
    """

    mode: ThemeMode
    surface: str
    page: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    sequential: tuple[str, ...] = _SEQUENTIAL_BLUE

    def color(self, slot: int) -> str:
        """``slot``. kategorik renk (0 tabanlı).

        Raises:
            IndexError: 8'den fazla seri istenirse (renkler döngüsel kullanılmaz).
        """
        if not 0 <= slot < len(self.series):
            raise IndexError(
                f"Kategorik palet {len(self.series)} yuvalıdır; fazla seriler 'Diğer' "
                "grubuna katlanmalı veya küçük çoklu grafiklere bölünmelidir."
            )
        return self.series[slot]

    def ordinal(self, count: int) -> list[str]:
        """Sıralı (ordinal) ``count`` renk: açık modda 250–650, koyu modda 250–600 adımları.

        Yüzeye en yakın adım 2:1 karşıtlığı korur; sıra küçükten büyüğe koyulaşır (açık mod) /
        açılır (koyu mod) ki büyük indisler zeminden ayrışsın.
        """
        if count <= 0:
            return []
        lo, hi = (3, 11) if self.mode == "light" else (3, 10)
        steps = self.sequential[lo : hi + 1]
        if self.mode == "dark":
            steps = steps[::-1]
        idx = np.linspace(0, len(steps) - 1, count).round().astype(int)
        return [steps[i] for i in idx]

    def rgba(self, color: str, alpha: float) -> str:
        """``#rrggbb`` rengini ``rgba(r, g, b, alpha)`` dizgesine çevirir."""
        value = color.lstrip("#")
        r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha:g})"


LIGHT: Final[Palette] = Palette(
    mode="light",
    surface="#fcfcfb",
    page="#f9f9f7",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
            "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
)  # fmt: skip

DARK: Final[Palette] = Palette(
    mode="dark",
    surface="#1a1a19",
    page="#0d0d0d",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500",
            "#d55181", "#008300", "#9085e9", "#e66767"),
)  # fmt: skip


def get_palette(mode: str | None) -> Palette:
    """Tema moduna göre palet (bilinmeyen/``None`` → açık tema)."""
    return DARK if mode == "dark" else LIGHT
