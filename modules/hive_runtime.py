"""Runtime helpers for keeping hive intelligence fresh in shared consumers."""

from __future__ import annotations

from typing import Any, Callable, Optional


def _safe_log(log: Optional[Callable[..., Any]], message: str, *, level: str) -> None:
    if log is None:
        return
    try:
        log(message, level=level)
    except TypeError:
        try:
            log(message)
        except Exception:
            pass
    except Exception:
        pass


def refresh_hive_runtime(*, hive_hints: Any, hive_router: Any, log: Optional[Callable[..., Any]] = None) -> None:
    """Refresh shared hive state used by fee and routing logic.

    The helper intentionally fails open. Hint polling, askrene layer refresh,
    fleet balance refresh, and route cache invalidation are all best-effort.
    """
    if hive_hints is not None:
        try:
            hive_hints.poll()
        except Exception as exc:
            _safe_log(log, f"Hive hint refresh failed: {exc}", level="warn")

    if hive_router is None:
        return

    try:
        refreshed = bool(hive_router.refresh_layer())
    except Exception as exc:
        _safe_log(log, f"Shared hive router refresh failed: {exc}", level="warn")
        return

    if not refreshed:
        return

    try:
        hive_router.refresh_fleet_balances()
    except Exception as exc:
        _safe_log(log, f"Fleet balance refresh failed: {exc}", level="warn")

    try:
        hive_router.clear_route_cache()
    except Exception as exc:
        _safe_log(log, f"Hive route cache clear failed: {exc}", level="warn")
