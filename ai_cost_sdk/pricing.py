"""Pricing registry loader."""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from typing import Tuple


def load_pricing(
    snapshot_id: str = "openai-2025-09",
) -> Tuple[dict, str]:
    """Load pricing snapshot bundled with the package."""
    package = pkg_resources.files(__package__) / "pricing"
    path = package / f"{snapshot_id}.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data, snapshot_id
