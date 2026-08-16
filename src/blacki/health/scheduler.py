"""Bounded background synchronization for active Google Health connections."""

from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .service import GoogleHealthService

logger = logging.getLogger(__name__)


class GoogleHealthScheduler:
    """Run read-only synchronization on a configurable interval."""

    def __init__(self, service: GoogleHealthService) -> None:
        self.service = service
        self.scheduler = AsyncIOScheduler()
        self._running = False

    async def start(self) -> None:
        """Start the scheduler without blocking application startup on a sync."""
        if self._running:
            return
        self.scheduler.add_job(
            self._sync_all,
            trigger=IntervalTrigger(hours=self.service.config.sync_interval_hours),
            id="google_health_sync",
            name="Synchronize Google Health records",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=3600,
        )
        self.scheduler.start()
        self._running = True
        logger.info(
            "Google Health scheduler started (every %d hours)",
            self.service.config.sync_interval_hours,
        )

    async def stop(self) -> None:
        """Stop the scheduler and wait for a running job to finish."""
        if not self._running:
            return
        self.scheduler.shutdown(wait=True)
        self._running = False
        logger.info("Google Health scheduler stopped")

    async def _sync_all(self) -> None:
        """Run a scheduled sync while keeping provider failures isolated."""
        try:
            results = await self.service.sync_all()
        except Exception:
            logger.exception("Google Health scheduled sync failed")
            return
        logger.info(
            "Google Health scheduled sync finished for %d connection(s)",
            len(results),
        )
