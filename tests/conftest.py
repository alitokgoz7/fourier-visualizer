"""Ortak pytest ayarları ve hypothesis profilleri."""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "default",
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci",
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
