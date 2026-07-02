import asyncio
import logging
from datetime import UTC, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.error import TelegramError

from bot.telegram_bot import (
    send_creation_failure,
)
from config.settings import Settings, get_settings
from src.exceptions import PipelineError
from src.graphs.creation_pipeline import build_creation_pipeline
from src.graphs.learning_pipeline import build_learning_pipeline
from src.nodes.research import research_viral_content
from src.persistence import create_checkpointer, create_store
from src.store.knowledge_base import KnowledgeBase
from src.tools.apify_client import get_threads_scraper
from src.tools.embeddings import EmbeddingClient
from src.tools.hackernews_client import get_hackernews_client
from src.tools.threads_api import get_threads_client

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(
        self,
        settings: Settings | None = None,
        store=None,
        checkpointer=None,
        bot_app=None,
        telegram_chat_id: str = "",
    ):
        self.settings = settings or get_settings()
        self.store = store or create_store(self.settings)
        self.checkpointer = checkpointer or create_checkpointer(self.settings)
        self.bot_app = bot_app
        self.telegram_chat_id = telegram_chat_id or self.settings.telegram_chat_id
        self._scheduler = AsyncIOScheduler()
        self._creation_cycle = 0
        self._learning_cycle = 0
        self._cycle_lock = asyncio.Lock()
        self._paused = False
        self.kb = KnowledgeBase(store=self.store, account_id=self.settings.account_id)

        self._threads_client = get_threads_client(self.settings)
        self._hn_client = get_hackernews_client(self.settings)
        self._scraper = get_threads_scraper(self.settings)
        self._embedding_client = EmbeddingClient()

    def setup_schedules(self) -> None:
        for hour in [8, 12, 18]:
            self._scheduler.add_job(
                self.run_creation_pipeline,
                "cron",
                hour=hour,
                minute=0,
                timezone="Europe/Warsaw",
                id=f"creation_{hour}",
                replace_existing=True,
            )

        self._scheduler.add_job(
            self.run_learning_pipeline,
            "cron",
            hour=6,
            minute=0,
            timezone="Europe/Warsaw",
            id="learning",
            replace_existing=True,
        )

        self._scheduler.add_job(
            self.refresh_threads_token,
            "cron",
            day_of_week="sun",
            hour=5,
            minute=0,
            timezone="Europe/Warsaw",
            id="threads_token_refresh",
            replace_existing=True,
        )

    async def run_creation_pipeline(self) -> dict | None:
        async with self._cycle_lock:
            self._creation_cycle += 1
            cycle = self._creation_cycle

        logger.info("Starting creation pipeline cycle #%d", cycle)

        graph = build_creation_pipeline(
            self.settings,
            self.store,
            threads_client=self._threads_client,
            hn=self._hn_client,
            scraper=self._scraper,
            embedding_client=self._embedding_client,
        )
        compiled = graph.compile(checkpointer=self.checkpointer)

        thread_id = f"creation_{cycle}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "current_follower_count": 0,
            "target_follower_count": self.settings.target_followers,
            "goal_reached": False,
            "viral_posts": [],
            "extracted_patterns": [],
            "generated_variants": [],
            "ranked_posts": [],
            "selected_post": None,
            "human_decision": None,
            "human_edited_content": None,
            "human_feedback": None,
            "published_post": None,
            "cycle_number": cycle,
            "errors": [],
        }

        try:
            result = None
            async for event in compiled.astream(initial_state, config):
                result = event
                logger.info("Creation pipeline event: %s", list(event.keys()))

            state = await compiled.aget_state(config)
            values = (state.values if state else None) or (result or {})
            if values.get("goal_reached"):
                logger.info(
                    "Creation cycle #%d completed early: "
                    "follower goal already reached (%d/%d followers)",
                    cycle,
                    values.get("current_follower_count", 0),
                    values.get("target_follower_count", 0),
                )
            elif not values.get("selected_post") or not values.get("published_post"):
                errors = values.get("errors", [])
                await self._send_creation_failure_telegram(cycle, errors)

            logger.info("Creation pipeline cycle #%d completed", cycle)
            return result
        except Exception as e:
            raise PipelineError(f"Creation pipeline cycle #{cycle} failed: {e}") from e

    async def _send_creation_failure_telegram(self, cycle: int, errors: list) -> None:
        if not self.bot_app or not self.telegram_chat_id:
            logger.warning("Cannot send failure notification: bot_app or chat_id not configured")
            return

        next_run_time = ""
        try:
            jobs = self._scheduler.get_jobs()
            creation_jobs = [j for j in jobs if j.id.startswith("creation_") and j.next_run_time]
            if creation_jobs:
                soonest = min(creation_jobs, key=lambda j: j.next_run_time)
                next_run_time = str(soonest.next_run_time)
        except (AttributeError, ValueError, TypeError):
            logger.debug("Could not determine next run time", exc_info=True)

        error_strings = [str(e) for e in errors]
        try:
            await send_creation_failure(
                app=self.bot_app,
                chat_id=self.telegram_chat_id,
                cycle_number=cycle,
                errors=error_strings,
                next_run_time=next_run_time,
            )
            logger.info("Creation failure notification sent for cycle #%d", cycle)
        except TelegramError as e:
            logger.error("Failed to send creation failure notification: %s", e)

    async def run_learning_pipeline(self) -> dict | None:
        async with self._cycle_lock:
            self._learning_cycle += 1
            cycle = self._learning_cycle

        logger.info("Starting learning pipeline cycle #%d", cycle)

        graph = build_learning_pipeline(
            self.settings,
            self.store,
            threads_client=self._threads_client,
        )
        compiled = graph.compile(checkpointer=self.checkpointer)

        config = {
            "configurable": {
                "thread_id": f"learning_{cycle}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            }
        }

        initial_state = {
            "posts_to_check": [],
            "collected_metrics": [],
            "performance_analysis": None,
            "pattern_updates": [],
            "new_strategy": None,
            "cycle_number": cycle,
            "errors": [],
        }

        try:
            result = None
            async for event in compiled.astream(initial_state, config):
                result = event
                logger.info("Learning pipeline event: %s", list(event.keys()))

            logger.info("Learning pipeline cycle #%d completed", cycle)
            return result
        except Exception as e:
            raise PipelineError(f"Learning pipeline cycle #{cycle} failed: {e}") from e

    async def run_research_only(self) -> list[dict]:
        minimal_state = {
            "viral_posts": [],
            "errors": [],
        }

        result = await research_viral_content(
            minimal_state, hn=self._hn_client, scraper=self._scraper, kb=self.kb
        )
        return result.get("viral_posts", [])

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def creation_cycle(self) -> int:
        return self._creation_cycle

    @property
    def learning_cycle(self) -> int:
        return self._learning_cycle

    def get_scheduled_jobs(self) -> list[dict]:
        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append(
                {
                    "id": job.id,
                    "next_run_time": str(job.next_run_time) if job.next_run_time else None,
                    "paused": job.next_run_time is None,
                }
            )
        return jobs

    def pause_all_jobs(self) -> None:
        for job in self._scheduler.get_jobs():
            job.pause()
        self._paused = True
        logger.info("All scheduled jobs paused")

    def resume_all_jobs(self) -> None:
        for job in self._scheduler.get_jobs():
            job.resume()
        self._paused = False
        logger.info("All scheduled jobs resumed")

    def reschedule_creation_jobs(self, posting_times: list[str]) -> None:
        for job in self._scheduler.get_jobs():
            if job.id.startswith("creation_"):
                job.remove()

        for time_str in posting_times:
            parts = time_str.split(":")
            try:
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
            except (ValueError, IndexError):
                logger.warning("Skipping invalid posting time: %s", time_str)
                continue
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                logger.warning("Skipping out-of-range posting time: %s", time_str)
                continue
            self._scheduler.add_job(
                self.run_creation_pipeline,
                "cron",
                hour=hour,
                minute=minute,
                timezone="Europe/Warsaw",
                id=f"creation_{hour}_{minute:02d}",
                replace_existing=True,
            )

        logger.info("Rescheduled creation jobs for times: %s", posting_times)

    async def _bootstrap_threads_token(self) -> None:
        if not self.settings.is_production:
            return
        try:
            stored = await self.kb.get_threads_access_token()
        except Exception:
            logger.exception("Failed to load threads token from KB; using env value")
            return
        if stored:
            self._threads_client.set_access_token(stored)
            logger.info("Loaded threads access token from KB")
        elif self.settings.threads_access_token:
            try:
                await self.kb.save_threads_access_token(self.settings.threads_access_token)
                logger.info("Seeded threads access token from env into KB")
            except Exception:
                logger.exception("Failed to seed threads access token into KB")

    async def refresh_threads_token(self) -> None:
        logger.info("Refreshing Threads long-lived access token")
        try:
            new_token = await self._threads_client.refresh_long_lived_token()
        except Exception as e:
            logger.exception("Threads token refresh failed")
            await self._send_token_refresh_failure_telegram(str(e))
            return
        if not new_token:
            logger.warning("Threads token refresh returned empty token")
            await self._send_token_refresh_failure_telegram("empty token in response")
            return
        try:
            await self.kb.save_threads_access_token(new_token)
        except Exception:
            logger.exception("Failed to persist refreshed threads token")
            await self._send_token_refresh_failure_telegram("persist failed")
            return
        logger.info("Threads access token refreshed and persisted")

    async def _send_token_refresh_failure_telegram(self, reason: str) -> None:
        if not self.bot_app or not self.telegram_chat_id:
            return
        try:
            await self.bot_app.bot.send_message(
                chat_id=self.telegram_chat_id,
                text=f"Threads token refresh failed: {reason}. Generate a new token manually.",
            )
        except TelegramError as e:
            logger.error("Failed to send token refresh failure notification: %s", e)

    async def start(self) -> None:
        await self._bootstrap_threads_token()
        self.setup_schedules()
        self._scheduler.start()
        logger.info("Orchestrator started with scheduled jobs")

    async def stop(self) -> None:
        self._scheduler.shutdown()
        for client in (self._threads_client, self._hn_client, self._scraper):
            try:
                await client.close()
            except Exception:  # best-effort resource cleanup
                logger.exception("Error closing %s", type(client).__name__)
        logger.info("Orchestrator stopped")
