"""Shared, reusable utilities (DRY core for every layer).

Centralizes logic that was duplicated across modules — text normalization,
numeric digit extraction, file-byte resolution, JSON config loading (cached +
auto-invalidated on file change), deterministic hashing, a single condition
evaluator reused by ontology + risk rules, and a config-driven retry wrapper.

Nothing here is domain-hardcoded: rules/prompts/tokens live in config files; this
module only provides the plumbing to load and apply them deterministically.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

_WS = re.compile(r"\s+")
_NON_DIGIT = re.compile(r"[^\d]")
# Unicode-aware word tokens (letters incl. ä/ü/é…, digits; excludes underscore),
# so non-English text isn't fragmented. ASCII tokenization is unchanged.
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: Any) -> list:
    """Lowercase word tokens — the single tokenizer reused by the embedding hasher,
    BM25 keyword index, and the heuristic reranker (DRY). Unicode-aware."""
    return _TOKEN.findall(str(text).lower())

# Project root /config directory (bundled config files live here).
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


def bundled_config(name: str) -> str:
    """Absolute path to a bundled config file under <project_root>/config."""
    return os.path.join(_CONFIG_DIR, name)


# --------------------------------------------------------------------------- #
# Text / number helpers
# --------------------------------------------------------------------------- #
def normalize_text(s: Any) -> str:
    """Lowercase + collapse whitespace for tolerant comparison."""
    return _WS.sub(" ", str(s)).strip().lower()


def digits(s: Any) -> str:
    return _NON_DIGIT.sub("", str(s))


def num_digits(v: Any) -> str:
    """Canonical digit string for a number (drops trailing .0 on integral floats)."""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return digits(v)


def sha1_hex(*parts: Any, length: Optional[int] = None) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    digest = h.hexdigest()
    return digest[:length] if length else digest


# --------------------------------------------------------------------------- #
# File bytes (single resolver used by intake + document_ai)
# --------------------------------------------------------------------------- #
def resolve_bytes(file: Dict[str, Any]) -> Optional[bytes]:
    """Resolve a file dict to raw bytes via content_bytes | content_b64 | path."""
    if file.get("content_bytes") is not None:
        return file["content_bytes"]
    b64 = file.get("content_b64")
    if b64:
        return base64.b64decode(b64)
    path = file.get("path")
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            return fh.read()
    return None


# --------------------------------------------------------------------------- #
# JSON config loading (cached, auto-invalidated on file mtime change)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=64)
def _read_json(path: str, _mtime: float) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_json_config(path: str, default: Any, key: Optional[str] = None) -> Any:
    """Load a JSON config file. Returns `default` on any error. If `key` is given,
    returns that top-level key (or `default`). Cached but re-reads when the file
    changes (mtime in the cache key), so config edits take effect."""
    try:
        data = _read_json(path, os.path.getmtime(path))
    except Exception:
        return default
    if key is not None and isinstance(data, dict):
        return data.get(key, default)
    return data


# --------------------------------------------------------------------------- #
# Single condition evaluator (reused by ontology rules + risk rules)
# --------------------------------------------------------------------------- #
def values_equivalent(a: Any, b: Any, dtype: str, abs_tol: float = 1e-9, rel_tol: float = 0.0) -> bool:
    """Are two values 'materially the same' for the given type? Numbers compare with
    absolute + relative tolerance (config-driven); booleans by identity; strings by
    normalized equality. Used to decide whether duplicate extractions actually agree."""
    if dtype in ("number", "integer"):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return normalize_text(a) == normalize_text(b)
        return abs(fa - fb) <= max(abs_tol, rel_tol * max(abs(fa), abs(fb)))
    if dtype == "boolean":
        return bool(a) == bool(b)
    return normalize_text(a) == normalize_text(b)


def condition_matches(when: Dict[str, Any], value: Any) -> bool:
    """Evaluate a config 'when' clause against a value.
    Supports: equals, gte, lte, gt, lt, present, in."""
    if value is None:
        return when.get("present") is False
    if "equals" in when:
        return value == when["equals"]
    if "in" in when:
        return value in when["in"]
    is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
    if "gte" in when:
        return is_num and value >= when["gte"]
    if "lte" in when:
        return is_num and value <= when["lte"]
    if "gt" in when:
        return is_num and value > when["gt"]
    if "lt" in when:
        return is_num and value < when["lt"]
    if "present" in when:
        return bool(when["present"])
    return False


# --------------------------------------------------------------------------- #
# Boolean coercion tokens (config-driven, not hardcoded)
# --------------------------------------------------------------------------- #
def load_value_tokens(path: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Return (true_tokens, false_tokens) from config/value_tokens.json (or an
    override path). Falls back to a minimal built-in set only if the file is
    unreadable, so the engine never crashes on a bad config."""
    path = path or bundled_config("value_tokens.json")
    data = load_json_config(path, default={})
    true_tokens = [normalize_text(t) for t in data.get("true", ["yes", "true", "1"])]
    false_tokens = [normalize_text(t) for t in data.get("false", ["no", "false", "0"])]
    return true_tokens, false_tokens


# --------------------------------------------------------------------------- #
# Config-driven retry wrapper (retry count + backoff from config)
# --------------------------------------------------------------------------- #
def with_retry(
    fn: Callable[[], Any],
    attempts: int = 1,
    backoff_seconds: float = 0.0,
    exceptions: tuple = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call fn, retrying on the given exceptions. attempts/backoff come from config.
    Deterministic output (only timing varies); raises the last error on exhaustion."""
    attempts = max(1, int(attempts))
    last_exc: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return fn()
        except exceptions as exc:
            last_exc = exc
            if i < attempts - 1 and backoff_seconds > 0:
                sleep(backoff_seconds)
    raise last_exc  # type: ignore[misc]
