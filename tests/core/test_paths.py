from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from fourier_viz.core.complex_dft import epicycles_from_points, points_to_complex
from fourier_viz.core.paths import (
    MAX_POINTS,
    SHAPE_LIBRARY,
    PathError,
    as_points,
    build_shape,
    circle,
    cumulative_arclength,
    epitrochoid,
    heart,
    infinity,
    normalize_path,
    parse_csv_points,
    path_length,
    remove_duplicates,
    resample_by_arclength,
    reverse_path,
    square,
    star,
)

SQUARE = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])


def chord_lengths(points: np.ndarray, closed: bool = True) -> np.ndarray:
    ring = np.vstack([points, points[:1]]) if closed else points
    return np.linalg.norm(np.diff(ring, axis=0), axis=1)


# ---------------------------------------------------------------- basics
class TestBasics:
    def test_as_points_accepts_complex_and_lists(self) -> None:
        assert as_points([[1, 2], [3, 4]]).dtype == np.float64
        assert np.allclose(as_points(np.array([1 + 2j, 3 - 1j])), [[1, 2], [3, -1]])

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (np.zeros((0, 2)), "boş"),
            ([], "boş"),
            (np.zeros((3, 3)), "çiftleri"),
            (np.zeros(4), "çiftleri"),
            ([[0.0, np.nan], [1.0, 1.0]], "NaN"),
            ([[0.0, np.inf], [1.0, 1.0]], "NaN"),
        ],
    )
    def test_as_points_errors(self, value: object, match: str) -> None:
        with pytest.raises(PathError, match=match):
            as_points(value)  # type: ignore[arg-type]

    def test_too_many_points(self) -> None:
        with pytest.raises(PathError, match="en fazla"):
            as_points(np.zeros((MAX_POINTS + 1, 2)))

    def test_remove_duplicates(self) -> None:
        pts = [[0, 0], [0, 0], [1, 0], [1, 1], [1, 1], [0, 0]]
        assert np.allclose(remove_duplicates(pts), [[0, 0], [1, 0], [1, 1]])
        assert np.allclose(remove_duplicates(pts, closed=False), [[0, 0], [1, 0], [1, 1], [0, 0]])

    def test_lengths(self) -> None:
        assert path_length(SQUARE) == pytest.approx(8.0)
        assert path_length(SQUARE, closed=False) == pytest.approx(6.0)
        assert np.allclose(cumulative_arclength(SQUARE), [0, 2, 4, 6, 8])
        assert np.allclose(cumulative_arclength(SQUARE, closed=False), [0, 2, 4, 6])

    def test_normalize(self) -> None:
        norm = normalize_path(SQUARE * 5 + 7)
        assert np.allclose(norm.min(axis=0), -1)
        assert np.allclose(norm.max(axis=0), 1)
        with pytest.raises(PathError, match="sıfır"):
            normalize_path([[1, 1], [1, 1]])

    def test_reverse(self) -> None:
        rev = reverse_path(SQUARE)
        assert np.allclose(rev[0], SQUARE[0])
        assert np.allclose(rev[1], SQUARE[-1])


# ---------------------------------------------------------------- resampling
class TestResample:
    def test_square_exactly_equal_spacing(self) -> None:
        res = resample_by_arclength(SQUARE, 16)
        assert res.shape == (16, 2)
        assert np.allclose(chord_lengths(res), 0.5)
        assert np.allclose(res[0], SQUARE[0])  # ilk nokta korunur

    def test_closed_path_preserved(self) -> None:
        res = resample_by_arclength(heart(), 500)
        # Kapanış aralığı (son → ilk) diğer aralıklarla aynı
        gaps = chord_lengths(res)
        assert gaps[-1] == pytest.approx(np.median(gaps), rel=0.05)
        assert path_length(res) == pytest.approx(path_length(heart()), rel=1e-3)

    @pytest.mark.parametrize("key", sorted(SHAPE_LIBRARY))
    def test_equal_arclength_spacing(self, key: str) -> None:
        shape = build_shape(key)
        dense = resample_by_arclength(shape, 20000)
        res = resample_by_arclength(shape, 400)
        # Her yeni noktanın özgün yol üzerindeki yay konumu k·L/n olmalı:
        # yoğun örneklemedeki en yakın noktanın indeksi ≈ k·50
        idx = np.array([np.argmin(np.linalg.norm(dense - p, axis=1)) for p in res[::20]])
        expected = np.arange(0, 20000, 1000)
        assert np.max(np.abs(idx - expected)) <= 2

    def test_open_path_includes_endpoints(self) -> None:
        line = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        res = resample_by_arclength(line, 4, closed=False)
        assert np.allclose(res[:, 0], [0, 1, 2, 3])
        assert np.allclose(chord_lengths(res, closed=False), 1.0)

    def test_duplicate_first_point_at_end_handled(self) -> None:
        closed = np.vstack([SQUARE, SQUARE[:1]])
        assert np.allclose(resample_by_arclength(closed, 8), resample_by_arclength(SQUARE, 8))

    @pytest.mark.parametrize(
        ("pts", "n", "match"),
        [
            (np.zeros((0, 2)), 10, "boş"),
            ([[1.0, 2.0]], 10, "tek bir nokta"),
            ([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], 10, "tek bir nokta"),
            (SQUARE, 1, "en az 2"),
            (SQUARE, MAX_POINTS + 1, "(?i)en fazla"),
        ],
    )
    def test_invalid(self, pts: object, n: int, match: str) -> None:
        with pytest.raises(PathError, match=match):
            resample_by_arclength(pts, n)  # type: ignore[arg-type]

    @given(
        arrays(np.float64, st.tuples(st.integers(3, 30), st.just(2)),
               elements=st.floats(-100, 100, allow_nan=False)),
        st.integers(4, 200),
    )  # fmt: skip
    def test_property_equal_spacing_on_polyline(self, pts: np.ndarray, n: int) -> None:
        try:
            res = resample_by_arclength(pts, n)
        except PathError:
            return  # tüm noktalar aynı
        assert res.shape == (n, 2)
        # Yeni noktaların özgün kapalı poligon üzerindeki yay konumları eşit aralıklı olmalı:
        # yeniden örneklenen yolun uzunluğu özgün uzunluğu aşmaz (kısa kesen kirişler).
        assert path_length(res) <= path_length(pts) * (1 + 1e-9) + 1e-9
        assert np.allclose(res[0], remove_duplicates(pts)[0])


# ---------------------------------------------------------------- shapes
class TestShapes:
    @pytest.mark.parametrize("key", sorted(SHAPE_LIBRARY))
    def test_all_shapes_build_and_are_finite(self, key: str) -> None:
        pts = build_shape(key)
        assert pts.ndim == 2
        assert pts.shape[1] == 2
        assert np.all(np.isfinite(pts))
        assert len(remove_duplicates(pts)) >= 3

    def test_shape_specifics(self) -> None:
        assert heart().shape == (1000, 2)
        assert np.allclose(heart(4)[0], [0.0, 13 - 5 - 2 - 1])
        s = star(6, 0.5)
        assert s.shape == (12, 2)
        assert np.allclose(np.linalg.norm(s, axis=1)[::2], 1.0)
        assert np.allclose(np.linalg.norm(s, axis=1)[1::2], 0.5)
        assert np.allclose(s[0], [0.0, 1.0])
        inf = infinity(400)
        assert np.allclose(inf[0], [1.0, 0.0])
        assert np.allclose(inf, inf * [1, 1])  # sonlu
        assert square().shape == (4, 2)
        assert np.allclose(np.linalg.norm(circle(100), axis=1), 1.0)

    def test_epitrochoid_closes(self) -> None:
        pts = epitrochoid(3, 2, 0.5, n=500)
        assert pts.shape == (1000, 2)  # R/r = 3/2 → 2 tur
        nxt = epitrochoid(3, 2, 0.5, n=500)
        assert np.allclose(pts, nxt)
        # Kapanış: son nokta ile ilk nokta arası tipik adım boyunda
        gaps = chord_lengths(pts)
        assert gaps[-1] < 3 * np.median(gaps)

    @pytest.mark.parametrize(
        ("call", "match"),
        [
            (lambda: star(2), "en az 3 ucu"),
            (lambda: star(5, 1.2), "İç yarıçap"),
            (lambda: epitrochoid(-1, 1, 1), "pozitif"),
            (lambda: epitrochoid(3, 1, -1), "negatif"),
            (lambda: epitrochoid(0.0001, 100, 1), "çok küçük"),
            (lambda: heart(2), "en az 3"),
            (lambda: build_shape("triangle"), "Bilinmeyen şekil"),
            (lambda: build_shape("heart", size=3), "bilinmeyen parametre"),
        ],
    )
    def test_shape_errors(self, call: object, match: str) -> None:
        with pytest.raises(PathError, match=match):
            call()  # type: ignore[operator]

    def test_build_shape_casts_integer_params(self) -> None:
        pts = build_shape("star", n_tips=6.0, inner_ratio=0.3)
        assert pts.shape == (12, 2)
        assert build_shape("epitrochoid", big_r=4.0, small_r=1.0, distance=1.0).shape[1] == 2

    @pytest.mark.parametrize("key", sorted(SHAPE_LIBRARY))
    def test_epicycles_redraw_shape(self, key: str) -> None:
        pts = resample_by_arclength(build_shape(key), 256)
        epi = epicycles_from_points(pts)
        s = np.arange(256) / 256
        assert np.allclose(epi.evaluate(s), points_to_complex(pts), atol=1e-9)


# ---------------------------------------------------------------- CSV
class TestCSV:
    def test_basic_with_header(self) -> None:
        text = "x,y\n0,0\n1,0\n1,1\n"
        assert np.allclose(parse_csv_points(text), [[0, 0], [1, 0], [1, 1]])

    def test_separators_and_comments(self) -> None:
        text = "﻿# yorum\n\n0 0\n1\t2\n3;4\n5 , 6 , 99\n"
        assert np.allclose(parse_csv_points(text), [[0, 0], [1, 2], [3, 4], [5, 6]])

    def test_turkish_decimal_comma(self) -> None:
        assert np.allclose(parse_csv_points("x;y\n1,5;2,25\n-0,5;3"), [[1.5, 2.25], [-0.5, 3.0]])

    def test_scientific(self) -> None:
        assert np.allclose(parse_csv_points("1e-3, -2E2"), [[1e-3, -200.0]])

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("", "hiç nokta"),
            ("x,y\n", "hiç nokta"),
            ("x,y\n1,2\nabc,3\n", "3. satır"),
            ("1,2\n3\n", "2. satır"),
            ("1,2\nnan,3\n", "sonlu"),
            ("1," * 3_000_000, "çok büyük"),
        ],
    )
    def test_errors(self, text: str, match: str) -> None:
        with pytest.raises(PathError, match=match):
            parse_csv_points(text)
