"""Streamlit AppTest ile arayüz duman testleri."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from fourier_viz.app.common import (
    TAB_CONVERGENCE,
    TAB_EPICYCLES,
    TAB_EXPORT,
    TAB_LEARN,
    TAB_SERIES,
    TAB_SPECTRUM,
    TAB_STATE_KEY,
    TABS,
)
from fourier_viz.app.learn import SECTIONS
from fourier_viz.core.paths import SHAPE_LIBRARY
from fourier_viz.core.signals import SIGNAL_LIBRARY

APP = str(Path(__file__).resolve().parents[2] / "fourier_viz" / "app" / "streamlit_app.py")
TIMEOUT = 120


def open_app(tab: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    if tab is not None:
        at.session_state[TAB_STATE_KEY] = tab
    return at.run()


def assert_ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def metric_value(at: AppTest, label: str) -> str:
    for metric in at.metric:
        if metric.label == label:
            return str(metric.value)
    raise AssertionError(f"Metrik bulunamadı: {label}")


def error_texts(at: AppTest) -> list[str]:
    return [str(e.value) for e in at.error]


# ---------------------------------------------------------------- açılış ve sekmeler
def test_app_opens_without_errors() -> None:
    at = open_app()
    assert_ok(at)
    assert not at.error
    assert at.title[0].value == "Fourier Serisi Görselleştirici"
    assert [t.label for t in at.tabs] == list(TABS)
    assert at.sidebar.header[0].value == "⚙️ Ayarlar"


@pytest.mark.parametrize("tab", TABS)
def test_every_tab_renders(tab: str) -> None:
    at = open_app(tab)
    assert_ok(at)
    assert not at.error


def test_tab_switch_back_and_forth() -> None:
    at = open_app()
    for tab in (*TABS, TAB_SERIES):
        at.session_state[TAB_STATE_KEY] = tab
        at.run()
        assert_ok(at)


# ---------------------------------------------------------------- Fourier serisi sekmesi
def test_slider_changes_update_metrics() -> None:
    at = open_app(TAB_SERIES)
    l2_small = float(metric_value(at, "L2 hatası (RMS)"))
    at.slider(key="n_terms").set_value(200).run()
    assert_ok(at)
    l2_large = float(metric_value(at, "L2 hatası (RMS)"))
    assert metric_value(at, "Terim sayısı N") == "200"
    assert l2_large < l2_small
    # Kare dalga: L2² = 1 − (8/π²) Σ_{tek n ≤ N} 1/n²
    n = np.arange(1, 200, 2)
    expected = np.sqrt(1 - 8 / np.pi**2 * np.sum(1 / n**2))
    assert l2_large == pytest.approx(expected, rel=2e-3)
    at.slider(key="n_terms").set_value(1).run()
    assert_ok(at)
    at.slider(key="n_terms").set_value(500).run()
    assert_ok(at)


@pytest.mark.parametrize("key", sorted(SIGNAL_LIBRARY))
def test_every_signal_renders(key: str) -> None:
    at = open_app(TAB_SERIES)
    at.selectbox(key="signal_key").select(key).run()
    assert_ok(at)
    assert not at.error
    assert at.subheader[0].value == SIGNAL_LIBRARY[key].label
    for tab in (TAB_SPECTRUM, TAB_CONVERGENCE):
        at.session_state[TAB_STATE_KEY] = tab
        at.run()
        assert_ok(at)


def test_signal_parameters_and_options() -> None:
    at = open_app(TAB_SERIES)
    at.selectbox(key="signal_key").select("pulse").run()
    at.slider(key="param_pulse_duty").set_value(0.1).run()
    at.slider(key="param_pulse_period").set_value(1.0).run()
    at.slider(key="param_pulse_amplitude").set_value(2.5).run()
    assert_ok(at)
    at.toggle(key="use_analytic").set_value(True).run()
    assert_ok(at)
    for smoothing in ("fejer", "lanczos", "none"):
        at.radio(key="smoothing").set_value(smoothing).run()
        assert_ok(at)
    at.toggle(key="show_harmonics").set_value(True).run()
    at.slider(key="harmonic_count").set_value(12).run()
    assert_ok(at)


@pytest.mark.parametrize("method", ["trapezoid", "simpson", "gauss"])
def test_integration_methods(method: str) -> None:
    at = open_app(TAB_SERIES)
    at.selectbox(key="method").select(method).run()
    assert_ok(at)
    at.toggle(key="samples_auto").set_value(False).run()
    at.number_input(key="samples").set_value(2048).run()
    assert_ok(at)
    assert not at.error


def test_quad_method_limits_n() -> None:
    at = open_app(TAB_SERIES)
    at.slider(key="n_terms").set_value(300).run()
    at.selectbox(key="method").select("quad").run()
    assert_ok(at)
    assert at.slider(key="n_terms").value == 100
    assert at.slider(key="n_terms").max == 100


def test_coefficient_comparison_table_shows_small_difference() -> None:
    at = open_app(TAB_SERIES)
    successes = [s.value for s in at.success]
    assert successes, "Karşılaştırma özeti bulunamadı"
    worst = float(successes[0].split("fark: ")[1].split(" ")[0])
    assert worst < 1e-8


# ---------------------------------------------------------------- kullanıcı ifadesi
def test_valid_expression() -> None:
    at = open_app(TAB_SERIES)
    at.selectbox(key="signal_key").select("expression").run()
    assert_ok(at)
    assert not at.error
    at.text_input(key="expression").input("exp(-t**2) * cos(5*t)").run()
    assert_ok(at)
    assert not at.error
    at.radio(key="expr_interval").set_value("[0, T)").run()
    at.slider(key="expr_period").set_value(1.0).run()
    assert_ok(at)
    assert not at.error
    at.selectbox(key="expression_example").select("abs(t)").run()
    assert at.text_input(key="expression").value == "abs(t)"


@pytest.mark.parametrize(
    ("expression", "fragment"),
    [
        ("3t", "Sözdizimi"),
        ("__import__('os').system('ls')", "öznitelik"),
        ("__import__('os')", "'_' ile başlayan"),
        ("os.system('ls')", "öznitelik"),
        ("1/t", "sonsuz"),
        ("log(t)", "NaN"),
        ("foo(t)", "Bilinmeyen"),
        ("", "boş"),
    ],
)
def test_invalid_expression_shows_turkish_error(expression: str, fragment: str) -> None:
    at = open_app(TAB_SERIES)
    at.selectbox(key="signal_key").select("expression").run()
    at.text_input(key="expression").input(expression).run()
    assert_ok(at)
    errors = error_texts(at)
    assert errors, "Hata mesajı gösterilmedi"
    assert fragment in errors[0]
    assert "Sinyal oluşturulamadı" in errors[0]
    for tab in TABS:  # hiçbir sekme çökmemeli
        at.session_state[TAB_STATE_KEY] = tab
        at.run()
        assert_ok(at)


# ---------------------------------------------------------------- epicycles
@pytest.mark.parametrize("key", sorted(SHAPE_LIBRARY))
def test_every_shape_renders(key: str) -> None:
    at = open_app(TAB_EPICYCLES)
    at.selectbox(key="shape_key").select(key).run()
    assert_ok(at)
    assert not at.error


def test_epicycle_controls() -> None:
    at = open_app(TAB_EPICYCLES)
    at.select_slider(key="n_points").set_value(512).run()
    at.slider(key="n_circles").set_value(200).run()
    at.slider(key="speed").set_value(2.0).run()
    at.slider(key="trail").set_value(30).run()
    at.select_slider(key="n_frames").set_value(60).run()
    at.toggle(key="show_circles").set_value(False).run()
    assert_ok(at)
    assert metric_value(at, "Çember sayısı") == "200 / 511"
    at.slider(key="n_circles").set_value(1).run()
    assert_ok(at)
    at.selectbox(key="shape_key").select("star").run()
    at.slider(key="shape_star_n_tips").set_value(7).run()
    assert_ok(at)


def test_drawing_without_points_shows_hint() -> None:
    at = open_app(TAB_EPICYCLES)
    at.radio(key="shape_source").set_value("drawing").run()
    assert_ok(at)
    assert any("Henüz bir çizim yok" in i.value for i in at.info)


def test_drawing_with_points() -> None:
    at = open_app(TAB_EPICYCLES)
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    at.session_state["drawn_points"] = np.column_stack([np.cos(theta), 0.5 * np.sin(theta)])
    at.radio(key="shape_source").set_value("drawing").run()
    assert_ok(at)
    assert not at.error
    assert metric_value(at, "Çember sayısı").endswith("/ 255")
    at.button(key="clear_drawing").click().run()
    assert_ok(at)
    assert "drawn_points" not in at.session_state


def test_upload_pasted_svg_and_csv() -> None:
    at = open_app(TAB_EPICYCLES)
    at.radio(key="shape_source").set_value("upload").run()
    assert_ok(at)
    assert any("Bir dosya yükleyin" in i.value for i in at.info)
    at.text_area(key="shape_text").input("M 0 0 C 40 -40, 80 40, 120 0 S 80 -80, 0 0 Z").run()
    assert_ok(at)
    assert not at.error
    at.text_area(key="shape_text").input("x,y\n0,0\n1,0\n1,1\n0,1").run()
    assert_ok(at)
    assert not at.error
    at.text_area(key="shape_text").input("M 0 0 L").run()
    assert_ok(at)
    assert any("Şekil okunamadı" in e for e in error_texts(at))
    at.text_area(key="shape_text").input("1,2").run()
    assert any("tek bir nokta" in e for e in error_texts(at))


# ---------------------------------------------------------------- spektrum ve yakınsama
def test_spectrum_view_options() -> None:
    at = open_app(TAB_SPECTRUM)
    at.session_state["spectrum_view"] = "Çift taraflı (|cₙ|, arg cₙ)"
    at.session_state["spectrum_style"] = "Çubuk"
    at.run()
    assert_ok(at)
    at.toggle(key="spectrum_log").set_value(True).run()
    assert_ok(at)
    assert metric_value(at, "%99 enerji için gereken N") == "41"


def test_convergence_tab_gibbs_metrics() -> None:
    at = open_app(TAB_CONVERGENCE)
    at.slider(key="n_terms").set_value(400).run()
    assert_ok(at)
    measured = float(metric_value(at, "Ölçülen aşım (yumuşatmasız)").lstrip("%"))
    assert measured == pytest.approx(8.949, abs=0.01)
    assert float(metric_value(at, "Fejér ile aşım").lstrip("%")) == 0.0
    assert float(metric_value(at, "L2 hata eğimi")) == pytest.approx(-0.5, abs=0.06)
    at.toggle(key="exclude_jumps").set_value(True).run()
    assert_ok(at)


def test_continuous_signal_has_no_gibbs() -> None:
    at = open_app(TAB_CONVERGENCE)
    at.selectbox(key="signal_key").select("triangle").run()
    assert_ok(at)
    assert any("sürekli" in i.value for i in at.info)


# ---------------------------------------------------------------- öğren ve dışa aktar
def test_learn_links_switch_tabs() -> None:
    at = open_app(TAB_LEARN)
    assert len(at.latex) + len(at.markdown) > len(SECTIONS)
    for section in SECTIONS:
        at.session_state[TAB_STATE_KEY] = TAB_LEARN
        at.run()
        at.button(key=f"learn_go_{section.key}").click().run()
        assert_ok(at)
        assert at.session_state[TAB_STATE_KEY] == section.target_tab


def test_learn_gram_slider() -> None:
    at = open_app(TAB_LEARN)
    at.slider(key="gram_n").set_value(8).run()
    assert_ok(at)


def test_export_tab_buttons() -> None:
    at = open_app(TAB_EXPORT)
    assert_ok(at)
    at.session_state["export_theme"] = "Koyu"
    at.run()
    assert_ok(at)
    at.radio(key="shape_source").set_value("upload").run()
    assert_ok(at)
    assert any("Epicycle dışa aktarımı" in i.value for i in at.info)
