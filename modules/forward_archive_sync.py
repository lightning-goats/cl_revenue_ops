"""Bounded, read-only synchronization of canonical CLN forward evidence."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from .forward_archive import (
    ARCHIVE_SCHEMA_VERSION,
    ForwardArchiveError,
    ForwardArchiveStore,
)


class ForwardArchiveSyncError(RuntimeError):
    """Raised when a read-only forward synchronization cycle fails closed."""


@dataclass(frozen=True, slots=True)
class SyncResult:
    observed_at_ns: int
    created_live_max: int
    updated_live_max: int
    created_pages: int
    updated_pages: int
    touched_dates: tuple[int, ...]

    caught_up: bool
    backlog_family: Optional[str]

class ForwardArchiveSynchronizer:
    """Synchronize independent CLN created/updated cursor families."""

    PAGE_LIMIT = 500
    MAX_PAGES_PER_FAMILY = 200

    def __init__(self, rpc: Any, store: ForwardArchiveStore, log: Any):
        self.rpc = rpc
        self.store = store
        self.log = log

    def _check_schema(self) -> None:
        for family in ("created", "updated"):
            version = int(self.store.get_sync_state(family)["schema_version"])
            if version != ARCHIVE_SCHEMA_VERSION:
                raise ForwardArchiveSyncError(
                    f"unsupported archive schema version {version}"
                )

    def _live_max(self, family: str) -> int:
        payload = self.rpc.wait(
            subsystem="forwards",
            indexname=family,
            nextvalue=0,
        )
        if (
            not isinstance(payload, Mapping)
            or payload.get("subsystem") != "forwards"
            or family not in payload
        ):
            raise ForwardArchiveSyncError(
                f"wait forwards/{family} returned malformed payload"
            )
        value = payload[family]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ForwardArchiveSyncError(
                f"wait forwards/{family} returned invalid index"
            )
        return value

    def _validate_live_cursor(self, family: str, live_max: int) -> None:
        next_index = int(self.store.get_sync_state(family)["next_index"])
        if next_index > live_max + 1:
            raise ForwardArchiveSyncError(
                f"{family} cursor {next_index} exceeds live maximum {live_max}"
            )

    @staticmethod
    def _records_through_live_max(
        family: str,
        records: list[Any],
        live_max: int,
    ) -> list[Mapping[str, Any]]:
        # listforwards is live, so a row can arrive after the wait snapshot.
        # Defer such rows to the next cycle instead of making a busy node
        # perpetually fail a bounded snapshot.
        index_key = f"{family}_index"
        bounded: list[Mapping[str, Any]] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ForwardArchiveSyncError(
                    f"{family} page record: expected object"
                )
            value = record.get(index_key)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ForwardArchiveSyncError(
                    f"{family} page record has invalid {index_key}"
                )
            if value <= live_max:
                bounded.append(record)
        return bounded

    def _page_family(
        self,
        family: str,
        live_max: int,
        observed_at_ns: int,
    ) -> tuple[int, set[int], bool]:
        pages = 0
        touched_dates: set[int] = set()
        next_index = int(self.store.get_sync_state(family)["next_index"])
        if next_index > live_max or (next_index == 0 and live_max == 0):
            result = self.store.apply_page(
                family,
                [],
                observed_at_ns=observed_at_ns,
                live_max_index=live_max,
            )
            touched_dates.update(result.touched_dates)
            return pages, touched_dates, True

        while next_index <= live_max:
            if pages >= self.MAX_PAGES_PER_FAMILY:
                return pages, touched_dates, False
            # CLN indices are one-based.  start=0 is a special full view; for
            # updated ordering it includes records that were never updated and
            # therefore have no updated_index.  Begin both cursor families at 1.
            request_index = max(1, next_index)
            payload = self.rpc.listforwards(
                index=family,
                start=request_index,
                limit=self.PAGE_LIMIT,
            )
            if not isinstance(payload, Mapping):
                raise ForwardArchiveSyncError(
                    f"listforwards {family} returned malformed payload"
                )
            records = payload.get("forwards")
            if not isinstance(records, list):
                raise ForwardArchiveSyncError(
                    f"listforwards {family} returned malformed forwards"
                )
            snapshot_records = self._records_through_live_max(
                family,
                records,
                live_max,
            )
            try:
                result = self.store.apply_page(
                    family,
                    snapshot_records,
                    observed_at_ns=observed_at_ns,
                    live_max_index=live_max,
                )
            except ForwardArchiveError as exc:
                raise ForwardArchiveSyncError(str(exc)) from exc
            pages += 1
            touched_dates.update(result.touched_dates)
            if result.next_index <= next_index:
                raise ForwardArchiveSyncError(
                    f"{family} page did not advance cursor {next_index}"
                )
            next_index = result.next_index
        return pages, touched_dates, True

    def sync_once(self, now_ns: Optional[int] = None) -> SyncResult:
        """Run one bounded cycle using only ``wait`` and ``listforwards``."""
        observed_at_ns = time.time_ns() if now_ns is None else int(now_ns)
        if observed_at_ns < 0:
            raise ForwardArchiveSyncError("now_ns must be non-negative")
        error_families = ("created", "updated")
        try:
            self._check_schema()
            created_live_max = self._live_max("created")
            updated_live_max = self._live_max("updated")
            self._validate_live_cursor("created", created_live_max)
            self._validate_live_cursor("updated", updated_live_max)

            error_families = ("created",)
            created_pages, created_dates, created_caught_up = self._page_family(
                "created", created_live_max, observed_at_ns
            )
            error_families = ("updated",)
            if not created_caught_up:
                return SyncResult(
                    observed_at_ns=observed_at_ns,
                    created_live_max=created_live_max,
                    updated_live_max=updated_live_max,
                    created_pages=created_pages,
                    updated_pages=0,
                    touched_dates=tuple(sorted(created_dates)),
                    caught_up=False,
                    backlog_family="created",
                )
            updated_pages, updated_dates, updated_caught_up = self._page_family(
                "updated", updated_live_max, observed_at_ns
            )
            touched_dates = created_dates | updated_dates
            if not updated_caught_up:
                return SyncResult(
                    observed_at_ns=observed_at_ns,
                    created_live_max=created_live_max,
                    updated_live_max=updated_live_max,
                    created_pages=created_pages,
                    updated_pages=updated_pages,
                    touched_dates=tuple(sorted(touched_dates)),
                    caught_up=False,
                    backlog_family="updated",
                )
            observed_seconds = observed_at_ns // 1_000_000_000
            current_day = observed_seconds - (observed_seconds % 86400)
            if current_day >= 86400:
                touched_dates.add(current_day - 86400)
            if touched_dates:
                error_families = ("created", "updated")
                self.store.rebuild_days(
                    sorted(touched_dates),
                    checked_at_ns=observed_at_ns,
                )
            return SyncResult(
                observed_at_ns=observed_at_ns,
                created_live_max=created_live_max,
                updated_live_max=updated_live_max,
                created_pages=created_pages,
                updated_pages=updated_pages,
                caught_up=True,
                backlog_family=None,
                touched_dates=tuple(sorted(touched_dates)),
            )
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            for family in error_families:
                try:
                    self.store.record_sync_error(
                        family,
                        message,
                        observed_at_ns=observed_at_ns,
                    )
                except Exception:
                    pass
            if isinstance(exc, ForwardArchiveSyncError):
                raise
            raise ForwardArchiveSyncError(message) from exc
