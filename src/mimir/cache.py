"""Content-addressed cache keys — the LOCKED canonicalization from docs/DESIGN.md §3.

canonical_json/cache_key implement the locked formula byte-for-byte and must not
change without updating DESIGN.md §3 and docs/PROGRESS.md. build_payload is the
single normalization boundary: everything that could fork a key for equal request
content (int-vs-float temperature, absent-vs-empty system) is resolved here.
"""

import hashlib
import json
import math
from typing import Any


def canonical_json(payload: Any) -> bytes:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text.encode("utf-8")


def cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _require_int(name: str, value: int) -> int:
    # bool is an int subclass and would serialize as true/false, forking keys.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {value!r}")
    return value


def build_payload(
    *,
    model: str,
    system: str | None = None,
    user: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    sample_index: int,
) -> dict[str, Any]:
    """Build the six-key payload hashed by cache_key.

    `system` and `user` must be fully rendered text, never templates. `seed` and
    `sample_index` are Mimir-internal (cache differentiation only) and are never
    sent to a provider API.
    """
    if isinstance(temperature, bool) or not isinstance(temperature, int | float):
        raise TypeError(f"temperature must be an int or float, got {temperature!r}")
    temperature = float(temperature)
    if not math.isfinite(temperature):
        raise ValueError(f"temperature must be finite, got {temperature!r}")
    if temperature == 0.0:
        temperature = 0.0  # -0.0 would serialize differently and fork keys for equal values
    return {
        "model": model,
        "system": system if system is not None else "",
        "messages": [{"role": "user", "content": user}],
        "params": {
            "temperature": temperature,
            "max_tokens": _require_int("max_tokens", max_tokens),
        },
        "seed": _require_int("seed", seed),
        "sample_index": _require_int("sample_index", sample_index),
    }
