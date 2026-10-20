"""Loads config/settings.json, the single file that holds every threshold.

The front-end prototype reads the same file, so the report, the pipeline and
the interface cannot drift apart. Change values in the JSON, not here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"


def load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


SETTINGS = load_settings()
