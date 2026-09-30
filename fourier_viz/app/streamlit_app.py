"""Fourier Serisi Görselleştirici — Streamlit giriş noktası.

Çalıştırma::

    streamlit run fourier_viz/app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # Paket kurulmadan (pip install -e .) doğrudan çalıştırmayı da destekle.
    import fourier_viz  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - ortam bağımlı
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from fourier_viz.app.common import TAB_STATE_KEY, TABS
from fourier_viz.app.sidebar import render_sidebar
from fourier_viz.app.tabs import TAB_RENDERERS
from fourier_viz.viz.theme import get_palette

PAGE_CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
</style>
"""


def main() -> None:
    """Uygulamayı çizer."""
    st.set_page_config(
        page_title="Fourier Serisi Görselleştirici",
        page_icon="〰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    settings = render_sidebar()
    st.title("Fourier Serisi Görselleştirici")
    st.caption(
        "Periyodik sinyalleri sinüs ve kosinüslere ayırın, dönen çemberlerle şekil çizin, "
        "yakınsamayı, Parseval özdeşliğini ve Gibbs olayını keşfedin."
    )
    palette = get_palette(st.context.theme.type)
    tabs = st.tabs(list(TABS), key=TAB_STATE_KEY, on_change="rerun")
    for tab, render in zip(tabs, TAB_RENDERERS, strict=True):
        with tab:
            if tab.open:
                render(settings, palette)


if __name__ == "__main__":
    main()
