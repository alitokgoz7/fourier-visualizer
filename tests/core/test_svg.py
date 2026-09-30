from __future__ import annotations

import time

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from fourier_viz.core.paths import PathError, path_length
from fourier_viz.core.svg import (
    MAX_SEGMENTS,
    arc_points,
    extract_path_data,
    parse_path_data,
    svg_to_points,
)


def test_lines_and_close() -> None:
    (sub,) = parse_path_data("M0 0 L10 0 L10 10 Z")
    assert np.allclose(sub, [[0, 0], [10, 0], [10, 10], [0, 0]])


def test_relative_equals_absolute() -> None:
    absolute = parse_path_data("M 1 1 L 4 1 L 4 5 H 1 V 1 Z")[0]
    relative = parse_path_data("m1,1 l3,0 l0,4 h-3 v-4 z")[0]
    assert np.allclose(absolute, relative)


def test_implicit_lineto_after_moveto() -> None:
    (sub,) = parse_path_data("M0 0 10 0 10 10")
    assert np.allclose(sub, [[0, 0], [10, 0], [10, 10]])
    (rel,) = parse_path_data("m1 1 2 0 0 2")
    assert np.allclose(rel, [[1, 1], [3, 1], [3, 3]])


def test_repeated_command_parameters() -> None:
    (sub,) = parse_path_data("M0 0 L1 0 2 0 3 0")
    assert np.allclose(sub[:, 0], [0, 1, 2, 3])


def test_compact_number_syntax() -> None:
    (sub,) = parse_path_data("M-1-2L.5.5l1e1-1E0")
    assert np.allclose(sub, [[-1, -2], [0.5, 0.5], [10.5, -0.5]])


def test_cubic_bezier_endpoints_and_shape() -> None:
    (sub,) = parse_path_data("M0 0 C 0 10, 10 10, 10 0", samples_per_segment=100)
    assert np.allclose(sub[0], [0, 0])
    assert np.allclose(sub[-1], [10, 0])
    # B(1/2) = (0 + 3·0 + 3·10 + 10)/8, (0 + 30 + 30 + 0)/8
    mid = sub[50]
    assert np.allclose(mid, [5.0, 7.5])


def test_smooth_cubic_reflects_control_point() -> None:
    a = parse_path_data("M0 0 C0 10 10 10 10 0 S 20 -10 20 0", 10)[0]
    b = parse_path_data("M0 0 C0 10 10 10 10 0 C 10 -10 20 -10 20 0", 10)[0]
    assert np.allclose(a, b)
    # Önceki komut C/S değilse kontrol noktası mevcut noktadır
    c = parse_path_data("M0 0 S 10 10 10 0", 10)[0]
    d = parse_path_data("M0 0 C 0 0 10 10 10 0", 10)[0]
    assert np.allclose(c, d)


def test_quadratic_and_smooth_quadratic() -> None:
    (sub,) = parse_path_data("M0 0 Q 5 10 10 0", 2)
    assert np.allclose(sub, [[0, 0], [5, 5], [10, 0]])
    a = parse_path_data("M0 0 Q 5 10 10 0 T 20 0", 4)[0]
    b = parse_path_data("M0 0 Q 5 10 10 0 Q 15 -10 20 0", 4)[0]
    assert np.allclose(a, b)
    rel = parse_path_data("m0 0 q 5 10 10 0 t 10 0", 4)[0]
    assert np.allclose(a, rel)
    c = parse_path_data("M0 0 T 10 0", 2)[0]  # kontrol noktası = mevcut nokta → düz çizgi
    assert np.allclose(c[:, 1], 0.0)


class TestArcs:
    def test_semicircle_on_circle(self) -> None:
        (sub,) = parse_path_data("M0 0 A5 5 0 0 1 10 0", 32)
        assert np.allclose(np.hypot(sub[:, 0] - 5, sub[:, 1]), 5.0)
        assert np.allclose(sub[-1], [10, 0])
        assert sub[16, 1] == pytest.approx(-5.0)

    def test_sweep_flag_changes_side(self) -> None:
        up = parse_path_data("M0 0 A5 5 0 0 1 10 0", 8)[0]
        down = parse_path_data("M0 0 A5 5 0 0 0 10 0", 8)[0]
        assert up[4, 1] < 0 < down[4, 1]

    def test_large_arc_flag(self) -> None:
        small = parse_path_data("M0 0 A10 10 0 0 1 10 0", 64)[0]
        large = parse_path_data("M0 0 A10 10 0 1 1 10 0", 64)[0]
        assert path_length(large, closed=False) > 3 * path_length(small, closed=False)

    def test_compact_flags(self) -> None:
        a = parse_path_data("M0 0 a5 5 0 0110 0", 8)[0]
        b = parse_path_data("M0 0 a 5 5 0 0 1 10 0", 8)[0]
        assert np.allclose(a, b)

    def test_radius_scaled_up_when_too_small(self) -> None:
        (sub,) = parse_path_data("M0 0 A1 1 0 0 1 10 0", 16)
        assert np.allclose(np.hypot(sub[:, 0] - 5, sub[:, 1]), 5.0)

    def test_rotated_ellipse(self) -> None:
        pts = arc_points(0j, 4, 2, 30, False, True, complex(4, 3), 32)
        assert pts[-1] == complex(4, 3)
        assert len(pts) == 32

    def test_degenerate_arcs(self) -> None:
        assert arc_points(1 + 1j, 5, 5, 0, False, True, 1 + 1j, 8) == []
        assert arc_points(0j, 0, 5, 0, False, True, 3 + 0j, 8) == [3 + 0j]


def test_multiple_subpaths() -> None:
    subs = parse_path_data("M0 0 L1 0 L1 1 Z M5 5 L6 5 L6 6 Z")
    assert len(subs) == 2
    assert np.allclose(subs[1][0], [5, 5])
    subs2 = parse_path_data("M0 0 L1 0 M5 5 L6 5")
    assert len(subs2) == 2


def test_lineto_after_close_starts_from_start() -> None:
    (a, b) = parse_path_data("M1 1 L2 1 L2 2 Z L 0 5")
    assert np.allclose(b[0], [1, 1])
    assert np.allclose(b[1], [0, 5])
    assert len(a) == 4


@pytest.mark.parametrize(
    ("d", "match"),
    [
        ("", "çizilebilir"),
        ("M 0 0", "çizilebilir"),
        ("10 10", "komutla"),
        ("Z 10 10", "komutla"),
        ("M 0 0 L 10", "sayı bekleniyordu"),
        ("M 0 0 X 5 5", "Bilinmeyen SVG komutu"),
        ("M 0 0 L 5 5 !", "beklenmeyen"),
        ("M0 0 A 5 5 0 2 1 10 0", "bayrağı"),
    ],
)
def test_errors(d: str, match: str) -> None:
    with pytest.raises(PathError, match=match):
        parse_path_data(d)


def test_segment_limit() -> None:
    with pytest.raises(PathError, match="segment"):
        parse_path_data("M0 0" + " L1 1" * (MAX_SEGMENTS + 1))


def test_extract_and_full_svg_document() -> None:
    svg = """<?xml version="1.0"?>
    <!DOCTYPE svg [<!ENTITY lol "lol">]>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
      <path id='a' d='M0 0 H 10 V 10 H 0 Z'/>
      <path fill="red" d="M 20 20 L 30 20 L 30 30 Z" />
    </svg>"""
    assert extract_path_data(svg) == ["M0 0 H 10 V 10 H 0 Z", "M 20 20 L 30 20 L 30 30 Z"]
    pts = svg_to_points(svg)
    assert pts.shape == (9, 2)
    assert np.allclose(pts[2], [10, -10])  # y ekseni çevrildi
    raw = svg_to_points("M0 0 H 10 V 10", flip_y=False)
    assert np.allclose(raw[-1], [10, 10])


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "boş"),
        ("   ", "boş"),
        ("<svg><rect/></svg>", "<path>"),
        ("<svg>" + "a" * 5_000_001, "büyük"),
    ],
)
def test_svg_to_points_errors(text: str, match: str) -> None:
    with pytest.raises(PathError, match=match):
        svg_to_points(text)


def test_huge_path_data_rejected() -> None:
    with pytest.raises(PathError, match="büyük"):
        parse_path_data("M0 0 " + "L1 1 " * 1_000_001)


@given(st.text(alphabet="MmLlHhVvCcSsQqTtAaZz0123456789.,-e ", max_size=60))
def test_fuzz_never_crashes_unexpectedly(d: str) -> None:
    start = time.perf_counter()
    try:
        subs = parse_path_data(d, 4)
    except PathError:
        pass
    else:
        for sub in subs:
            assert np.all(np.isfinite(sub))
    assert time.perf_counter() - start < 1.0
