"""Arayüz genelinde paylaşılan sabitler ve küçük yardımcılar."""

from __future__ import annotations

from typing import Final

import streamlit as st

TAB_SERIES: Final[str] = "📈 Fourier Serisi"
TAB_EPICYCLES: Final[str] = "🌀 Epicycles"
TAB_SPECTRUM: Final[str] = "📊 Spektrum"
TAB_CONVERGENCE: Final[str] = "📉 Yakınsama ve Hata"
TAB_LEARN: Final[str] = "📚 Öğren"
TAB_EXPORT: Final[str] = "💾 Dışa Aktar"

TABS: Final[tuple[str, ...]] = (
    TAB_SERIES,
    TAB_EPICYCLES,
    TAB_SPECTRUM,
    TAB_CONVERGENCE,
    TAB_LEARN,
    TAB_EXPORT,
)

TAB_STATE_KEY: Final[str] = "main_tab"
"""Etkin sekmenin tutulduğu ``st.session_state`` anahtarı."""


def go_to_tab(label: str) -> None:
    """Etkin sekmeyi değiştirir (buton geri çağrısı olarak kullanılır)."""
    st.session_state[TAB_STATE_KEY] = label


def show_error(exc: Exception, context: str | None = None) -> None:
    """Kullanıcı dostu Türkçe hata kutusu gösterir."""
    prefix = f"**{context}:** " if context else ""
    st.error(f"{prefix}{exc}", icon="⚠️")
