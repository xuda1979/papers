"""audit_artifacts.py

Small helpers for writing machine-auditable JSON artifacts.

We keep this module tiny and dependency-free; artifacts are intended to be
picked up by `certificate_runner_v2.py` and included in bundles.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


def write_json_artifact(path: str, payload: Dict[str, Any]) -> str:
    """Write a JSON artifact with minimal standard metadata.

    - Ensures parent directory exists.
    - Adds `generated_utc` timestamp if not present.

    Returns the absolute path written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if "generated_utc" not in payload:
        payload = dict(payload)
        payload["generated_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return os.path.abspath(path)
