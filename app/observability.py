"""Structured logging + lightweight run metrics (architecture: Monitoring).

Config-driven (nebag_LOG_*). JSON or text formatter. Metrics are plain dicts so
they serialize into state/result; timings are wall-clock and are deliberately
EXCLUDED from the audit output_hash (determinism is unaffected).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("submission_id", "node", "duration_ms", "event"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        return json.dumps(payload, default=str)


def configure_logging(settings: Any) -> logging.Logger:
    """Configure the 'nebag' logger once from settings. Idempotent."""
    logger = logging.getLogger("nebag")
    if getattr(logger, "_nebag_configured", False):
        return logger
    if not getattr(settings, "log_enabled", True):
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger._nebag_configured = True  # type: ignore[attr-defined]
        return logger
    level = getattr(logging, str(getattr(settings, "log_level", "WARNING")).upper(), logging.WARNING)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    if str(getattr(settings, "log_format", "text")).lower() == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s nebag.%(name)s: %(message)s"))
    logger.handlers = [handler]
    logger.propagate = False
    logger._nebag_configured = True  # type: ignore[attr-defined]
    return logger


def get_logger(name: str = "nebag") -> logging.Logger:
    return logging.getLogger(name if name.startswith("nebag") else f"nebag.{name}")


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def record_node(state: Dict[str, Any], node: str, duration_ms: float) -> None:
    """Accumulate per-node timing into state['metrics'] (JSON-serializable)."""
    metrics = state.setdefault("metrics", {})
    metrics.setdefault("node_ms", {})[node] = round(duration_ms, 2)
