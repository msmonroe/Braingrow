"""
instrumentation.py — Optional timing and error tracing for BrainGrow.

ENABLE
------
Set the environment variable before launching:

    BRAINGROW_TRACE=1 python main.py

Or flip the fallback value in ENABLED below for persistent dev use.

DISABLE
-------
Unset the env var (or set it to "0").  When disabled, @traced is a no-op
that returns the original function unchanged — zero runtime overhead.

OUTPUT
------
Log lines go to stderr by default.  To redirect to a file:

    BRAINGROW_TRACE=1 BRAINGROW_LOG=braingrow.log python main.py
"""

from __future__ import annotations

import functools
import logging
import os
import time
import traceback
from typing import Callable

# ---------------------------------------------------------------------------
# Toggle — single point of control
# ---------------------------------------------------------------------------

ENABLED: bool = os.environ.get("BRAINGROW_TRACE", "0") == "1"

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

log = logging.getLogger("braingrow")

if not log.handlers:
    _log_path = os.environ.get("BRAINGROW_LOG", "")
    _handler: logging.Handler = (
        logging.FileHandler(_log_path) if _log_path
        else logging.StreamHandler()
    )
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  braingrow  %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(_handler)
    log.setLevel(logging.DEBUG if ENABLED else logging.WARNING)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def traced(fn: Callable) -> Callable:
    """Decorator: time the call, log entry/exit, capture and re-raise exceptions.

    When ENABLED is False the decorator returns the original function object
    untouched — no wrapper overhead whatsoever.
    """
    if not ENABLED:
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        label = fn.__qualname__
        t0 = time.perf_counter()
        log.debug("→ %s", label)
        try:
            result = fn(*args, **kwargs)
            ms = (time.perf_counter() - t0) * 1_000
            log.debug("✓ %s  %.1f ms", label, ms)
            return result
        except Exception:
            ms = (time.perf_counter() - t0) * 1_000
            log.error(
                "✗ %s  FAILED after %.1f ms\n%s",
                label, ms, traceback.format_exc(),
            )
            raise

    return wrapper


def log_event(msg: str, *args: object) -> None:
    """Emit a structured INFO line — silently discarded when ENABLED is False."""
    if ENABLED:
        log.info(msg, *args)
