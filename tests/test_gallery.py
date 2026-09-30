"""Galeri betiği ve galeri için kullanılan statik figürler."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from fourier_viz.core.complex_dft import epicycles_from_points
from fourier_viz.core.paths import SHAPE_LIBRARY, build_shape, normalize_path, resample_by_arclength
from fourier_viz.core.signals import SIGNAL_LIBRARY, square_wave
from fourier_viz.viz import static
from fourier_viz.viz.theme import DARK, LIGHT

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_gallery.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_gallery", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass'lar modülü sys.modules'ta arar
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_gallery_script_generates_all_images(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_script()
    code = module.main(["--out", str(tmp_path), "--no-animations", "--dpi", "40"])
    output = capsys.readouterr().out
    assert code == 0, output
    names = {p.name for p in tmp_path.iterdir()}
    for key in SIGNAL_LIBRARY:
        assert f"signal_{key}.png" in names
    for key in SHAPE_LIBRARY:
        assert f"shape_{key}.png" in names
    assert {"signal_expression.png", "gibbs.png", "convergence.png"} <= names
    assert "Tüm sağlamalar geçti." in output
    assert "✗" not in output


@pytest.mark.slow
def test_gallery_animations_and_dark(tmp_path: Path) -> None:
    module = load_script()
    files = module.animation_gallery(tmp_path, LIGHT)
    assert [f.name for f in files] == ["epicycle_heart.gif", "convergence_square.gif"]
    assert all(f.read_bytes()[:6] == b"GIF89a" for f in files)
    produced, checks = module.analysis_gallery(tmp_path, DARK, 40)
    assert all(check.ok for check in checks)
    assert len(produced) == 2


def test_gallery_reports_failed_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_script()

    def failing(out: Path, palette: object, dpi: int) -> tuple[list[Path], list[object]]:
        return [], [module.Check("yapay başarısız sağlama", "x", False)]

    monkeypatch.setattr(module, "signal_gallery", failing)
    monkeypatch.setattr(module, "shape_gallery", failing)
    monkeypatch.setattr(module, "analysis_gallery", failing)
    code = module.main(["--out", str(tmp_path), "--no-animations", "--dark"])
    captured = capsys.readouterr()
    assert code == 1
    assert "✗ yapay başarısız sağlama" in captured.out
    assert "başarısız" in captured.err


def test_signal_card_and_progression() -> None:
    sq = square_wave()
    coeffs = sq.coefficients(10)
    t = np.linspace(-np.pi, np.pi, 200)
    fig = static.signal_card(
        t, sq(t), {"S3": coeffs.evaluate(t, 3)}, np.arange(11), coeffs.amplitude(),
        title="Kare", subtitle="alt başlık",
    )  # fmt: skip
    assert len(fig.axes) == 2
    assert fig.get_suptitle() == "Kare"
    assert static.figure_to_png_bytes(fig, dpi=30).startswith(b"\x89PNG")

    pts = normalize_path(resample_by_arclength(build_shape("infinity"), 128))
    full = epicycles_from_points(pts)
    prog = static.epicycle_progression(pts, [full.limit(3), full.limit(30)], title="∞")
    width, height = prog.get_size_inches()
    assert len(prog.axes) == 2
    assert height < width / 2  # geniş şekil → kısa figür
    assert "enerji %" in prog.axes[0].get_title()
    with pytest.raises(ValueError, match="En az bir"):
        static.epicycle_progression(pts, [])
