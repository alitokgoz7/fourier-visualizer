"""Çekirdek modüllerde paylaşılan tip takma adları."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
"""Gerçek değerli ``float64`` NumPy dizisi."""

ComplexArray: TypeAlias = NDArray[np.complex128]
"""Karmaşık değerli ``complex128`` NumPy dizisi."""

IntArray: TypeAlias = NDArray[np.int64]
"""Tam sayı ``int64`` NumPy dizisi."""
