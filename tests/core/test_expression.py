from __future__ import annotations

import time

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from fourier_viz.core.expression import (
    ALLOWED_FUNCTIONS,
    CONSTANTS,
    MAX_LENGTH,
    ExpressionError,
    evaluate_expression,
    parse_expression,
)

T = np.linspace(-np.pi, np.pi, 101)


# ---------------------------------------------------------------- valid expressions
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("t**2 * sin(3*t)", T**2 * np.sin(3 * T)),
        ("t^2 * sin(3*t)", T**2 * np.sin(3 * T)),
        ("t^2 + 1", T**2 + 1),  # ^ üs olarak, + 'dan önce
        ("-t + 2*pi", -T + 2 * np.pi),
        ("+t", T),
        ("π * t", np.pi * T),
        ("2 × t − 1", 2 * T - 1),
        ("e**cos(t)", np.e ** np.cos(T)),
        ("tau / 2", np.full_like(T, np.pi)),
        ("abs(sin(t))", np.abs(np.sin(T))),
        ("exp(-t**2) * cos(5*t)", np.exp(-(T**2)) * np.cos(5 * T)),
        ("sqrt(abs(t))", np.sqrt(np.abs(T))),
        ("t % 1", T % 1),
        ("where(t > 0, 1, -1)", np.where(T > 0, 1.0, -1.0)),
        ("1 if t > 0 else 0", (T > 0).astype(float)),
        ("(t > -1) * (t < 1)", ((T > -1) & (T < 1)).astype(float)),
        ("-1 < t < 1", ((T > -1) & (T < 1)).astype(float)),
        ("(t > 0) and (t < 2)", ((T > 0) & (T < 2)).astype(float)),
        ("(t < -2) or (t > 2)", ((T < -2) | (T > 2)).astype(float)),
        ("(t >= 0) & (t <= 1)", ((T >= 0) & (T <= 1)).astype(float)),
        ("(t == 0) | (t != 0)", np.ones_like(T)),
        ("max(sin(t), 0)", np.maximum(np.sin(T), 0)),
        ("min(t, 1)", np.minimum(T, 1)),
        ("clip(t, -1, 1)", np.clip(T, -1, 1)),
        ("heaviside(t)", np.heaviside(T, 0.5)),
        ("sign(t)", np.sign(T)),
        ("atan2(sin(t), cos(t))", np.arctan2(np.sin(T), np.cos(T))),
        ("sinc(t)", np.sinc(T / np.pi)),
        ("square(t)", np.sign(np.sin(T))),
        ("saw(t)", (T + np.pi) % (2 * np.pi) - np.pi),
        ("triangle(0)", np.ones_like(T)),
        ("floor(t) + ceil(t) + round(t)", np.floor(T) + np.ceil(T) + np.round(T)),
        ("log(exp(t)) + ln(1) + log10(10) + log2(4)", T + 3),
        ("cosh(t) - sinh(t) - exp(-t) + tanh(0)", np.zeros_like(T)),
        ("mod(t, 2) + hypot(3, 4) + cbrt(8)", np.mod(T, 2) + 7),
        ("1e-3 * t", 1e-3 * T),
    ],
)
def test_valid_expressions(text: str, expected: np.ndarray) -> None:
    assert np.allclose(evaluate_expression(text, T), expected, atol=1e-12)


def test_constant_expression_broadcasts() -> None:
    expr = parse_expression("3")
    assert not expr.uses_variable
    assert expr(np.zeros((2, 3))).shape == (2, 3)
    assert parse_expression("sin(t)").uses_variable


def test_trig_functions_all_available() -> None:
    x = np.linspace(-0.9, 0.9, 11)
    for name in ("sin", "cos", "tan", "asin", "acos", "atan", "arcsin", "arccos", "arctan"):
        func = getattr(np, {"asin": "arcsin", "acos": "arccos", "atan": "arctan"}.get(name, name))
        assert np.allclose(evaluate_expression(f"{name}(t)", x), func(x))


def test_metadata() -> None:
    expr = parse_expression("  t^2  ")
    assert expr.source == "t^2"
    assert expr.normalized == "t**2"
    assert "sin" in ALLOWED_FUNCTIONS
    assert ALLOWED_FUNCTIONS["where"] == 3
    assert set(CONSTANTS) == {"pi", "e", "tau"}


# ---------------------------------------------------------------- malicious / invalid inputs
@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('echo pwned')",
        "__import__",
        "os.system('ls')",
        "os",
        "sys.exit()",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "t.__class__",
        "sin.__globals__",
        "(lambda: 1)()",
        "eval('1+1')",
        "exec('x=1')",
        "compile('1', 'a', 'eval')",
        "getattr(t, 'real')",
        "globals()",
        "[t for t in range(10)]",
        "{'a': 1}",
        "{1, 2}",
        "(1, 2)",
        "[1, 2][0]",
        "'string'",
        "b'bytes'",
        "f'{t}'",
        "(x := 5)",
        "sin(*[t])",
        "sin(x=t)",
        "_t",
        "_(t)",
        "t[0]",
        "1j * t",
        "True + t",
        "None",
        "...",
        "not t",
        "~t",
        "t // 2",
        "t << 2",
        "t in t",
        "t is t",
        "x + 1",
        "sin",
        "sin(t, t)",
        "where(t, 1)",
        "sin(t)(t)",
        "numpy.sin(t)",
        "np.sin(t)",
    ],
)
def test_rejected(text: str) -> None:
    with pytest.raises(ExpressionError):
        parse_expression(text)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "boş"),
        ("   ", "boş"),
        ("3t", "Sözdizimi"),
        ("sin(t", "Sözdizimi"),
        ("os.system('x')", "öznitelik"),
        ("__import__('os')", "_"),
        ("foo(t)", "Bilinmeyen veya izin verilmeyen fonksiyon"),
        ("y * 2", "Bilinmeyen ad"),
        ("sin", "fonksiyondur"),
        ("sin(t, 2)", "1 argüman"),
        ("sin(x=t)", "anahtar kelime"),
        ("'a'", "gerçek sayı"),
        ("t // 2", "tam bölme"),
        ("sin.x(t)", "doğrudan"),
        ("not t", "not"),
        ("t in t", "'in'"),
    ],
)
def test_error_messages_are_turkish_and_helpful(text: str, match: str) -> None:
    with pytest.raises(ExpressionError, match=match):
        parse_expression(text)


def test_length_and_complexity_limits() -> None:
    with pytest.raises(ExpressionError, match="çok uzun"):
        parse_expression("t+" * MAX_LENGTH + "t")
    with pytest.raises(ExpressionError, match="karmaşık"):
        parse_expression("+".join(["t"] * 200))
    with pytest.raises(ExpressionError):
        parse_expression("(" * 240 + "t" + ")" * 240)


def test_null_byte_rejected() -> None:
    with pytest.raises(ExpressionError):
        parse_expression("t\x00+1")


def test_non_string_rejected() -> None:
    with pytest.raises(ExpressionError, match="metin"):
        parse_expression(123)  # type: ignore[arg-type]


def test_huge_integer_power_does_not_hang() -> None:
    start = time.perf_counter()
    expr = parse_expression("9**9**9**9")
    with pytest.raises(ExpressionError, match="sonsuz"):
        expr.evaluate(T)
    assert time.perf_counter() - start < 1.0


def test_huge_literal() -> None:
    with pytest.raises(ExpressionError, match="çok büyük"):
        parse_expression("1" * 400)


# ---------------------------------------------------------------- non-finite values
@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("1/t", "sonsuz"),
        ("log(t)", "NaN"),
        ("sqrt(t)", "NaN"),
        ("tan(t)", None),  # tan π/2 civarında büyük ama sonlu olabilir
        ("0/0 + t", "NaN"),
    ],
)
def test_nonfinite_detected(text: str, kind: str | None) -> None:
    t = np.array([-1.0, 0.0, 1.0])
    expr = parse_expression(text)
    raw = expr(t)  # denetimsiz çağrı hata vermez
    assert raw.shape == t.shape
    if kind is None:
        return
    with pytest.raises(ExpressionError, match=kind):
        expr.evaluate(t)


# ---------------------------------------------------------------- property-based tests
_ATOMS = st.sampled_from(["t", "pi", "e", "1", "2.5", "0.3"])
_UNARY = st.sampled_from(["sin", "cos", "tanh", "atan", "abs"])
_BINARY = st.sampled_from(["+", "-", "*"])


@st.composite
def safe_expressions(draw: st.DrawFn, depth: int = 3) -> str:
    if depth == 0 or draw(st.booleans()):
        return draw(_ATOMS)
    if draw(st.booleans()):
        return f"{draw(_UNARY)}({draw(safe_expressions(depth - 1))})"
    left = draw(safe_expressions(depth - 1))
    right = draw(safe_expressions(depth - 1))
    return f"({left} {draw(_BINARY)} {right})"


@given(safe_expressions())
def test_matches_reference_numpy_evaluation(text: str) -> None:
    namespace = {
        "t": T, "pi": np.pi, "e": np.e,
        "sin": np.sin, "cos": np.cos, "tanh": np.tanh, "atan": np.arctan, "abs": np.abs,
    }  # fmt: skip
    # Test oracle'ı: yalnızca testte, üretilen güvenli dilbilgisiyle sınırlı eval.
    expected = np.broadcast_to(eval(text, {"__builtins__": {}}, namespace), T.shape)
    assert np.allclose(evaluate_expression(text, T), expected, equal_nan=True)


@given(st.text(max_size=80))
def test_arbitrary_text_never_crashes(text: str) -> None:
    try:
        expr = parse_expression(text)
    except ExpressionError:
        return
    result = expr(T)
    assert result.shape == T.shape


@given(
    st.sampled_from(["__import__", "os", "sys", "eval", "exec", "open", "getattr", "globals"]),
    st.sampled_from(["", "()", "('os')", ".system", ".path"]),
)
def test_dangerous_names_always_rejected(name: str, suffix: str) -> None:
    with pytest.raises(ExpressionError):
        parse_expression(name + suffix)
