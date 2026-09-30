r"""Güvenli matematiksel ifade ayrıştırıcı (``eval`` kullanmaz).

Kullanıcının yazdığı ``"t**2 * sin(3*t)"`` gibi bir ifade, Python'un :mod:`ast` modülüyle
**yalnızca ayrıştırılır**; ardından sözdizimi ağacı bir *beyaz liste* ile denetlenir ve izin
verilen düğümler NumPy işlemlerinden oluşan bir kapanışa (closure) derlenir. Hiçbir aşamada
``eval``/``exec``/``compile`` ile kod çalıştırılmaz.

İzin verilenler
    * Değişken: ``t``
    * Sabitler: ``pi`` (``π``), ``e``, ``tau`` (:math:`2\pi`) ve sayılar
      (ör. ``2``, ``0.5``, ``1e-3``)
    * İşlemler: ``+ - * / % **`` (``^`` de üs olarak kabul edilir), tekli ``+``/``-``
    * Karşılaştırmalar: ``< <= > >= == !=`` (sonuç 1.0/0.0), ``and``/``or``/``&``/``|``
    * Koşullu ifade: ``a if koşul else b``
    * Fonksiyonlar: :data:`ALLOWED_FUNCTIONS` sözlüğündeki adlar

Reddedilenler
    Öznitelik erişimi (``os.system``), indeksleme, ``lambda``, dizgeler, liste/sözlük,
    anahtar kelime argümanları, ``_`` ile başlayan adlar, bilinmeyen adlar/fonksiyonlar ve
    bunların dışındaki her şey. Aşırı uzun/karmaşık ifadeler sınırlandırılır. Tüm aritmetik
    ``float64`` ile yapılır; ``9**9**9**9`` gibi girdiler tam sayı patlamasına yol açmaz,
    ``inf`` üretir ve değerlendirme sırasında yakalanır.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import ArrayLike

from fourier_viz.core.types import FloatArray

MAX_LENGTH: Final[int] = 500
"""İfadenin en fazla karakter sayısı."""

MAX_NODES: Final[int] = 300
"""Sözdizimi ağacındaki en fazla düğüm sayısı."""

VARIABLE: Final[str] = "t"
"""İfadedeki tek serbest değişkenin adı."""

Value = FloatArray | np.float64
Evaluator = Callable[[FloatArray], Value]


class ExpressionError(ValueError):
    """İfade ayrıştırma/değerlendirme hatası (mesajlar kullanıcıya gösterilebilir, Türkçe)."""


def _heaviside(x: FloatArray) -> FloatArray:
    return np.heaviside(x, 0.5)


def _as_bool(x: Value) -> np.ndarray:
    mask: np.ndarray = np.asarray(x) != 0.0
    return mask


def _where(cond: Value, a: Value, b: Value) -> FloatArray:
    return np.where(_as_bool(cond), a, b).astype(np.float64)


def _sinc(x: FloatArray) -> FloatArray:
    r"""Normalize edilmemiş sinc: :math:`\sin(x)/x` (``sinc(0) = 1``)."""
    return np.sinc(np.asarray(x) / np.pi)


def _saw(x: FloatArray) -> FloatArray:
    r"""Periyodu :math:`2\pi` olan testere: :math:`(x+\pi) \bmod 2\pi - \pi`."""
    return np.mod(np.asarray(x) + np.pi, 2 * np.pi) - np.pi


def _square(x: FloatArray) -> FloatArray:
    r"""Kare dalga :math:`\operatorname{sign}(\sin x)`."""
    return np.sign(np.sin(x))


def _triangle(x: FloatArray) -> FloatArray:
    """Üçgen dalga (tepe 1, :math:`x=0`'da 1)."""
    return 1.0 - 2.0 * np.abs(_saw(x)) / np.pi


_FUNCTIONS: dict[str, tuple[Callable[..., Value], int]] = {
    # ad: (fonksiyon, argüman sayısı)
    "sin": (np.sin, 1),
    "cos": (np.cos, 1),
    "tan": (np.tan, 1),
    "asin": (np.arcsin, 1),
    "acos": (np.arccos, 1),
    "atan": (np.arctan, 1),
    "arcsin": (np.arcsin, 1),
    "arccos": (np.arccos, 1),
    "arctan": (np.arctan, 1),
    "atan2": (np.arctan2, 2),
    "sinh": (np.sinh, 1),
    "cosh": (np.cosh, 1),
    "tanh": (np.tanh, 1),
    "exp": (np.exp, 1),
    "log": (np.log, 1),
    "ln": (np.log, 1),
    "log10": (np.log10, 1),
    "log2": (np.log2, 1),
    "sqrt": (np.sqrt, 1),
    "cbrt": (np.cbrt, 1),
    "abs": (np.abs, 1),
    "sign": (np.sign, 1),
    "floor": (np.floor, 1),
    "ceil": (np.ceil, 1),
    "round": (np.round, 1),
    "heaviside": (_heaviside, 1),
    "step": (_heaviside, 1),
    "sinc": (_sinc, 1),
    "saw": (_saw, 1),
    "square": (_square, 1),
    "triangle": (_triangle, 1),
    "mod": (np.mod, 2),
    "min": (np.minimum, 2),
    "max": (np.maximum, 2),
    "hypot": (np.hypot, 2),
    "where": (_where, 3),
    "clip": (np.clip, 3),
}

ALLOWED_FUNCTIONS: Final[Mapping[str, int]] = MappingProxyType(
    {name: arity for name, (_, arity) in _FUNCTIONS.items()}
)
"""İzin verilen fonksiyon adları ve argüman sayıları."""

CONSTANTS: Final[Mapping[str, float]] = MappingProxyType(
    {"pi": float(np.pi), "e": float(np.e), "tau": float(2 * np.pi)}
)
"""İzin verilen sabitler."""

_BINARY_OPS: dict[type[ast.operator], Callable[[Value, Value], Value]] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
    ast.Mod: np.mod,
    ast.BitAnd: lambda x, y: np.logical_and(_as_bool(x), _as_bool(y)).astype(np.float64),
    ast.BitOr: lambda x, y: np.logical_or(_as_bool(x), _as_bool(y)).astype(np.float64),
}

_COMPARE_OPS: dict[type[ast.cmpop], Callable[[Value, Value], np.ndarray]] = {
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
}

_NODE_DESCRIPTIONS: dict[str, str] = {
    "Attribute": "öznitelik erişimi (ör. 'os.system')",
    "Subscript": "indeksleme (ör. 'a[0]')",
    "Lambda": "lambda fonksiyonu",
    "List": "liste",
    "Tuple": "demet (tuple)",
    "Dict": "sözlük",
    "Set": "küme",
    "ListComp": "liste üreteci",
    "SetComp": "küme üreteci",
    "DictComp": "sözlük üreteci",
    "GeneratorExp": "üreteç ifadesi",
    "JoinedStr": "f-string",
    "NamedExpr": "atama ifadesi (:=)",
    "Starred": "yıldızlı argüman",
    "Await": "await",
    "Yield": "yield",
    "YieldFrom": "yield from",
    "Slice": "dilimleme",
    "FloorDiv": "tam bölme (//)",
    "MatMult": "matris çarpımı (@)",
    "LShift": "bit kaydırma (<<)",
    "RShift": "bit kaydırma (>>)",
    "BitXor": "bit XOR",
    "Invert": "bit tersleme (~)",
    "Not": "'not' işleci",
    "Is": "'is' karşılaştırması",
    "IsNot": "'is not' karşılaştırması",
    "In": "'in' karşılaştırması",
    "NotIn": "'not in' karşılaştırması",
}

_TEXT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("^", "**"),
    ("π", "pi"),
    ("τ", "tau"),
    ("×", "*"),
    ("·", "*"),
    ("÷", "/"),
    ("−", "-"),  # Unicode eksi işareti
)


def _describe(node: ast.AST | type) -> str:
    name = node.__name__ if isinstance(node, type) else type(node).__name__
    return _NODE_DESCRIPTIONS.get(name, name)


@dataclass(frozen=True)
class CompiledExpression:
    """Derlenmiş, güvenli ve vektörleştirilmiş matematik ifadesi.

    Attributes:
        source: Kullanıcının yazdığı özgün metin.
        normalized: Ayrıştırılan normalleştirilmiş metin (``^`` → ``**`` vb.).
        uses_variable: İfade ``t`` değişkenini içeriyor mu (içermiyorsa sabittir).
    """

    source: str
    normalized: str
    uses_variable: bool
    _evaluator: Evaluator = field(repr=False, compare=False)

    def __call__(self, t: ArrayLike) -> FloatArray:
        """İfadeyi ``t`` noktalarında değerlendirir (NaN/sonsuz değerler denetlenmez).

        Returns:
            ``t`` ile aynı biçimde ``float64`` dizi.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        with np.errstate(all="ignore"):
            raw = self._evaluator(t_arr)
        return np.broadcast_to(np.asarray(raw, dtype=np.float64), t_arr.shape).copy()

    def evaluate(self, t: ArrayLike) -> FloatArray:
        """İfadeyi değerlendirir ve tüm değerlerin sonlu olduğunu doğrular.

        Raises:
            ExpressionError: NaN veya sonsuz değer üretilirse (ör. ``1/t``, ``t=0``'da).
        """
        t_arr = np.asarray(t, dtype=np.float64)
        values = self(t_arr)
        bad = ~np.isfinite(values)
        if np.any(bad):
            where = np.broadcast_to(t_arr, values.shape)[bad]
            kind = "NaN (tanımsız)" if np.isnan(values[bad][0]) else "sonsuz"
            raise ExpressionError(
                f"'{self.source}' ifadesi t ≈ {where[0]:.4g} noktasında {kind} değer üretiyor. "
                "Lütfen ifadenin seçilen aralığın tamamında tanımlı olduğundan emin olun."
            )
        return values


class _Compiler:
    """Beyaz listeli AST → NumPy kapanış derleyicisi."""

    def __init__(self) -> None:
        self.uses_variable = False

    def compile(self, node: ast.AST) -> Evaluator:
        method = getattr(self, f"_compile_{type(node).__name__}", None)
        if method is None:
            raise ExpressionError(f"İzin verilmeyen ifade öğesi: {_describe(node)}.")
        result: Evaluator = method(node)
        return result

    def _compile_Expression(self, node: ast.Expression) -> Evaluator:
        return self.compile(node.body)

    def _compile_Constant(self, node: ast.Constant) -> Evaluator:
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ExpressionError(
                f"Yalnızca gerçek sayı sabitlerine izin verilir (bulunan: {value!r})."
            )
        try:
            number = np.float64(float(value))
        except OverflowError as exc:
            raise ExpressionError("Sayı sabiti çok büyük.") from exc
        return lambda _t: number

    def _compile_Name(self, node: ast.Name) -> Evaluator:
        name = node.id
        if name.startswith("_"):
            raise ExpressionError(f"Güvenlik nedeniyle '_' ile başlayan adlar yasaktır: '{name}'.")
        if name == VARIABLE:
            self.uses_variable = True
            return lambda t: t
        if name in CONSTANTS:
            constant = np.float64(CONSTANTS[name])
            return lambda _t: constant
        if name in _FUNCTIONS:
            raise ExpressionError(f"'{name}' bir fonksiyondur; parantezle çağırın, ör. {name}(t).")
        raise ExpressionError(
            f"Bilinmeyen ad: '{name}'. Değişken olarak yalnızca 't' ve sabit olarak "
            f"{', '.join(CONSTANTS)} kullanılabilir."
        )

    def _compile_UnaryOp(self, node: ast.UnaryOp) -> Evaluator:
        operand = self.compile(node.operand)
        if isinstance(node.op, ast.USub):
            return lambda t: np.negative(operand(t))
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ExpressionError(f"İzin verilmeyen tekli işlem: {_describe(node.op)}.")

    def _compile_BinOp(self, node: ast.BinOp) -> Evaluator:
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"İzin verilmeyen işlem: {_describe(node.op)}.")
        left = self.compile(node.left)
        right = self.compile(node.right)
        return lambda t: op(left(t), right(t))

    def _compile_BoolOp(self, node: ast.BoolOp) -> Evaluator:
        parts = [self.compile(v) for v in node.values]
        combine = np.logical_and if isinstance(node.op, ast.And) else np.logical_or

        def evaluate(t: FloatArray) -> Value:
            result = _as_bool(parts[0](t))
            for part in parts[1:]:
                result = combine(result, _as_bool(part(t)))
            return np.asarray(result, dtype=np.float64)

        return evaluate

    def _compile_Compare(self, node: ast.Compare) -> Evaluator:
        ops: list[Callable[[Value, Value], np.ndarray]] = []
        for op in node.ops:
            func = _COMPARE_OPS.get(type(op))
            if func is None:
                raise ExpressionError(f"İzin verilmeyen karşılaştırma: {_describe(op)}.")
            ops.append(func)
        operands = [self.compile(node.left), *(self.compile(c) for c in node.comparators)]

        def evaluate(t: FloatArray) -> Value:
            values = [operand(t) for operand in operands]
            result = np.asarray(True)
            for func, lhs, rhs in zip(ops, values[:-1], values[1:], strict=True):
                result = np.logical_and(result, func(lhs, rhs))
            return np.asarray(result, dtype=np.float64)

        return evaluate

    def _compile_IfExp(self, node: ast.IfExp) -> Evaluator:
        cond = self.compile(node.test)
        body = self.compile(node.body)
        orelse = self.compile(node.orelse)
        return lambda t: _where(cond(t), body(t), orelse(t))

    def _compile_Call(self, node: ast.Call) -> Evaluator:
        if not isinstance(node.func, ast.Name):
            raise ExpressionError(
                "Yalnızca izin verilen fonksiyonlar doğrudan adlarıyla çağrılabilir "
                f"({_describe(node.func)} kullanılamaz)."
            )
        name = node.func.id
        if name.startswith("_"):
            raise ExpressionError(f"Güvenlik nedeniyle '_' ile başlayan adlar yasaktır: '{name}'.")
        if name not in _FUNCTIONS:
            raise ExpressionError(
                f"Bilinmeyen veya izin verilmeyen fonksiyon: '{name}'. "
                f"İzin verilenler: {', '.join(sorted(_FUNCTIONS))}."
            )
        if node.keywords:
            raise ExpressionError(f"'{name}' fonksiyonunda anahtar kelime argümanı kullanılamaz.")
        func, arity = _FUNCTIONS[name]
        if len(node.args) != arity:
            raise ExpressionError(
                f"'{name}' fonksiyonu {arity} argüman bekler, {len(node.args)} verildi."
            )
        args = [self.compile(arg) for arg in node.args]
        return lambda t: func(*(arg(t) for arg in args))


def _normalize(text: str) -> str:
    normalized = text.strip()
    for old, new in _TEXT_REPLACEMENTS:
        normalized = normalized.replace(old, new)
    return normalized


def parse_expression(text: str) -> CompiledExpression:
    """Metni güvenli şekilde ayrıştırır ve vektörleştirilmiş bir ifadeye derler.

    Args:
        text: Ör. ``"t**2 * sin(3*t)"``, ``"where(t > 0, 1, -1)"``, ``"abs(sin(t))"``.

    Returns:
        :class:`CompiledExpression`.

    Raises:
        ExpressionError: Sözdizimi hatası, izin verilmeyen öğe, bilinmeyen ad, yanlış argüman
            sayısı veya uzunluk/karmaşıklık sınırı aşımında (Türkçe açıklamalı).
    """
    if not isinstance(text, str):
        raise ExpressionError("İfade bir metin (str) olmalıdır.")
    if len(text) > MAX_LENGTH:
        raise ExpressionError(f"İfade çok uzun (en fazla {MAX_LENGTH} karakter).")
    normalized = _normalize(text)
    if not normalized:
        raise ExpressionError("İfade boş olamaz. Örnek: t**2 * sin(3*t)")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        where = f" ({exc.offset}. karakter civarı)" if exc.offset else ""
        raise ExpressionError(
            f"Sözdizimi hatası{where}. Çarpma için '*' kullanmayı unutmayın (ör. 3*t)."
        ) from exc
    except (ValueError, MemoryError, RecursionError) as exc:  # pragma: no cover - savunma
        raise ExpressionError("İfade ayrıştırılamadı (çok karmaşık veya geçersiz).") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ExpressionError(f"İfade çok karmaşık (en fazla {MAX_NODES} sözdizimi öğesi).")
    compiler = _Compiler()
    try:
        evaluator = compiler.compile(tree)
    except RecursionError as exc:  # pragma: no cover - MAX_LENGTH ile pratikte erişilemez
        raise ExpressionError("İfade çok derin iç içe geçmiş.") from exc
    return CompiledExpression(
        source=text.strip(),
        normalized=normalized,
        uses_variable=compiler.uses_variable,
        _evaluator=evaluator,
    )


def evaluate_expression(text: str, t: ArrayLike) -> FloatArray:
    """Kısayol: ifadeyi ayrıştırır ve sonlu değerler üretmesini şart koşarak değerlendirir."""
    return parse_expression(text).evaluate(t)
