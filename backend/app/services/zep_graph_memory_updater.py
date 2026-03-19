"""
Graph memory update service.

Dynamically updates agent activities from simulations into the Graphiti
knowledge graph. Keeps the same public interface as the original Zep-based
implementation.
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.logger import get_logger
from ..utils.graphiti_client import get_graphiti_sync, run_async

logger = get_logger("mirofish.graph_memory_updater")


@dataclass
class AgentActivity:
    """Agent activity record."""

    platform: str  # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str  # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str

    def to_episode_text(self) -> str:
        """Convert one activity into natural language for graph ingestion."""
        action_descriptions = {
            "CREATE_POST": self._describe_create_post,
            "LIKE_POST": self._describe_like_post,
            "DISLIKE_POST": self._describe_dislike_post,
            "REPOST": self._describe_repost,
            "QUOTE_POST": self._describe_quote_post,
            "FOLLOW": self._describe_follow,
            "CREATE_COMMENT": self._describe_create_comment,
            "LIKE_COMMENT": self._describe_like_comment,
            "DISLIKE_COMMENT": self._describe_dislike_comment,
            "SEARCH_POSTS": self._describe_search,
            "SEARCH_USER": self._describe_search_user,
            "MUTE": self._describe_mute,
        }
        describe_func = action_descriptions.get(self.action_type, self._describe_generic)
        description = describe_func()
        return f"{self.agent_name}: {description}"

    def _describe_create_post(self) -> str:
        content = self.action_args.get("content", "")
        if content:
            return f'Posted: "{content}"'
        return "Posted."

    def _describe_like_post(self) -> str:
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        if post_content and post_author:
            return f'Liked a post by {post_author}: "{post_content}"'
        if post_content:
            return f'Liked a post: "{post_content}"'
        if post_author:
            return f"Liked a post by {post_author}."
        return "Liked a post."

    def _describe_dislike_post(self) -> str:
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        if post_content and post_author:
            return f'Disliked a post by {post_author}: "{post_content}"'
        if post_content:
            return f'Disliked a post: "{post_content}"'
        if post_author:
            return f"Disliked a post by {post_author}."
        return "Disliked a post."

    def _describe_repost(self) -> str:
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        if original_content and original_author:
            return f'Reposted from {original_author}: "{original_content}"'
        if original_content:
            return f'Reposted: "{original_content}"'
        if original_author:
            return f"Reposted from {original_author}."
        return "Reposted."

    def _describe_quote_post(self) -> str:
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        quote_content = self.action_args.get("quote_content", "") or self.action_args.get("content", "")

        if original_content and original_author:
            base = f'Quoted {original_author}: "{original_content}"'
        elif original_content:
            base = f'Quoted a post: "{original_content}"'
        elif original_author:
            base = f"Quoted a post by {original_author}."
        else:
            base = "Quoted a post."

        if quote_content:
            base = f'{base} Commented: "{quote_content}"'
        return base

    def _describe_follow(self) -> str:
        target_user_name = self.action_args.get("target_user_name", "")
        if target_user_name:
            return f"Followed user {target_user_name}."
        return "Followed a user."

    def _describe_create_comment(self) -> str:
        content = self.action_args.get("content", "")
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")

        if content:
            if post_content and post_author:
                return f'Commented on {post_author}\'s post "{post_content}": "{content}"'
            if post_content:
                return f'Commented on post "{post_content}": "{content}"'
            if post_author:
                return f'Commented on {post_author}\'s post: "{content}"'
            return f'Commented: "{content}"'
        return "Posted a comment."

    def _describe_like_comment(self) -> str:
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")
        if comment_content and comment_author:
            return f'Liked a comment by {comment_author}: "{comment_content}"'
        if comment_content:
            return f'Liked a comment: "{comment_content}"'
        if comment_author:
            return f"Liked a comment by {comment_author}."
        return "Liked a comment."

    def _describe_dislike_comment(self) -> str:
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")
        if comment_content and comment_author:
            return f'Disliked a comment by {comment_author}: "{comment_content}"'
        if comment_content:
            return f'Disliked a comment: "{comment_content}"'
        if comment_author:
            return f"Disliked a comment by {comment_author}."
        return "Disliked a comment."

    def _describe_search(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("keyword", "")
        return f'Searched posts for "{query}"' if query else "Searched posts."

    def _describe_search_user(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("username", "")
        return f'Searched users for "{query}"' if query else "Searched users."

    def _describe_mute(self) -> str:
        target_user_name = self.action_args.get("target_user_name", "")
        if target_user_name:
            return f"Muted user {target_user_name}."
        return "Muted a user."

    def _describe_generic(self) -> str:
        return f"Executed action {self.action_type}."


class GraphMemoryUpdater:
    """
    Graph memory updater.

    Monitors activity events and sends them to Graphiti in platform-based batches.
    """

    BATCH_SIZE = 5
    PLATFORM_DISPLAY_NAMES = {
        "twitter": "world 1",
        "reddit": "world 2",
    }
    SEND_INTERVAL = 0.5
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    def __init__(self, graph_id: str):
        self.graph_id = graph_id
        self.graphiti = get_graphiti_sync()

        self._activity_queue: Queue = Queue()
        self._platform_buffers: Dict[str, List[AgentActivity]] = {
            "twitter": [],
            "reddit": [],
        }
        self._buffer_lock = threading.Lock()

        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        self._total_activities = 0
        self._total_sent = 0
        self._total_items_sent = 0
        self._failed_count = 0
        self._skipped_count = 0

        logger.info(
            "GraphMemoryUpdater initialized: graph_id=%s, batch_size=%s",
            graph_id,
            self.BATCH_SIZE,
        )

    def _get_platform_display_name(self, platform: str) -> str:
        return self.PLATFORM_DISPLAY_NAMES.get(platform.lower(), platform)

    def start(self):
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"GraphMemoryUpdater-{self.graph_id[:8]}",
        )
        self._worker_thread.start()
        logger.info("GraphMemoryUpdater started: graph_id=%s", self.graph_id)

    def stop(self):
        self._running = False
        self._flush_remaining()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)

        logger.info(
            "GraphMemoryUpdater stopped: graph_id=%s total_activities=%s "
            "batches_sent=%s items_sent=%s failed=%s skipped=%s",
            self.graph_id,
            self._total_activities,
            self._total_sent,
            self._total_items_sent,
            self._failed_count,
            self._skipped_count,
        )

    def add_activity(self, activity: AgentActivity):
        """Add an activity to the queue."""
        if activity.action_type == "DO_NOTHING":
            self._skipped_count += 1
            return

        self._activity_queue.put(activity)
        self._total_activities += 1
        logger.debug(
            "Added activity to queue: %s - %s",
            activity.agent_name,
            activity.action_type,
        )

    def add_activity_from_dict(self, data: Dict[str, Any], platform: str):
        """Build and enqueue an AgentActivity from parsed log data."""
        if "event_type" in data:
            return

        activity = AgentActivity(
            platform=platform,
            agent_id=data.get("agent_id", 0),
            agent_name=data.get("agent_name", ""),
            action_type=data.get("action_type", ""),
            action_args=data.get("action_args", {}),
            round_num=data.get("round", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )
        self.add_activity(activity)

    def _worker_loop(self):
        """Background worker loop that batches by platform."""
        while self._running or not self._activity_queue.empty():
            try:
                activity = self._activity_queue.get(timeout=1)
            except Empty:
                continue
            except Exception as exc:
                logger.error("Worker loop error while reading queue: %s", exc)
                time.sleep(1)
                continue

            batch: List[AgentActivity] = []
            platform = activity.platform.lower()

            try:
                with self._buffer_lock:
                    if platform not in self._platform_buffers:
                        self._platform_buffers[platform] = []
                    self._platform_buffers[platform].append(activity)

                    if len(self._platform_buffers[platform]) >= self.BATCH_SIZE:
                        batch = self._platform_buffers[platform][: self.BATCH_SIZE]
                        self._platform_buffers[platform] = self._platform_buffers[platform][
                            self.BATCH_SIZE:
                        ]

                if batch:
                    self._send_batch_activities(batch, platform)
                    time.sleep(self.SEND_INTERVAL)
            except Exception as exc:
                logger.error("Worker loop error while processing activity: %s", exc)
                time.sleep(1)

    def _send_batch_activities(self, activities: List[AgentActivity], platform: str):
        """Send one platform batch to the Graphiti graph."""
        if not activities:
            return

        combined_text = "\n".join(activity.to_episode_text() for activity in activities)

        for attempt in range(self.MAX_RETRIES):
            try:
                run_async(
                    self.graphiti.add_episode(
                        name=f"simulation_{platform}_batch",
                        episode_body=combined_text,
                        source_description=f"MiroFish {platform} simulation activity",
                        reference_time=datetime.now(),
                        source="text",
                        group_id=self.graph_id,
                    )
                )
                self._total_sent += 1
                self._total_items_sent += len(activities)
                display_name = self._get_platform_display_name(platform)
                logger.info(
                    "Sent %s activities for %s to graph %s",
                    len(activities),
                    display_name,
                    self.graph_id,
                )
                logger.debug("Batch preview: %s...", combined_text[:200])
                return
            except Exception as exc:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(
                        "Batch send failed (attempt %s/%s): %s",
                        attempt + 1,
                        self.MAX_RETRIES,
                        exc,
                    )
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(
                        "Batch send failed after %s retries: %s",
                        self.MAX_RETRIES,
                        exc,
                    )
                    self._failed_count += 1

    def _flush_remaining(self):
        """Drain queue and send all remaining buffered activities."""
        while not self._activity_queue.empty():
            try:
                activity = self._activity_queue.get_nowait()
                platform = activity.platform.lower()
                with self._buffer_lock:
                    if platform not in self._platform_buffers:
                        self._platform_buffers[platform] = []
                    self._platform_buffers[platform].append(activity)
            except Empty:
                break

        remaining_batches: List[tuple[str, List[AgentActivity]]] = []
        with self._buffer_lock:
            for platform, buffer in self._platform_buffers.items():
                if buffer:
                    remaining_batches.append((platform, list(buffer)))
            for platform in self._platform_buffers:
                self._platform_buffers[platform] = []

        for platform, buffer in remaining_batches:
            display_name = self._get_platform_display_name(platform)
            logger.info(
                "Sending remaining %s activities for %s",
                len(buffer),
                display_name,
            )
            self._send_batch_activities(buffer, platform)

    def get_stats(self) -> Dict[str, Any]:
        with self._buffer_lock:
            buffer_sizes = {p: len(b) for p, b in self._platform_buffers.items()}

        return {
            "graph_id": self.graph_id,
            "batch_size": self.BATCH_SIZE,
            "total_activities": self._total_activities,
            "batches_sent": self._total_sent,
            "items_sent": self._total_items_sent,
            "failed_count": self._failed_count,
            "skipped_count": self._skipped_count,
            "queue_size": self._activity_queue.qsize(),
            "buffer_sizes": buffer_sizes,
            "running": self._running,
        }


class GraphMemoryManager:
    """Manages one GraphMemoryUpdater per simulation."""

    _updaters: Dict[str, GraphMemoryUpdater] = {}
    _lock = threading.Lock()
    _stop_all_done = False

    @classmethod
    def create_updater(cls, simulation_id: str, graph_id: str) -> GraphMemoryUpdater:
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()

            updater = GraphMemoryUpdater(graph_id)
            updater.start()
            cls._updaters[simulation_id] = updater

            logger.info(
                "Created graph memory updater: simulation_id=%s graph_id=%s",
                simulation_id,
                graph_id,
            )
            return updater

    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[GraphMemoryUpdater]:
        return cls._updaters.get(simulation_id)

    @classmethod
    def stop_updater(cls, simulation_id: str):
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
                del cls._updaters[simulation_id]
                logger.info("Graph memory updater stopped: simulation_id=%s", simulation_id)

    @classmethod
    def stop_all(cls):
        if cls._stop_all_done:
            return
        cls._stop_all_done = True

        with cls._lock:
            if cls._updaters:
                for simulation_id, updater in list(cls._updaters.items()):
                    try:
                        updater.stop()
                    except Exception as exc:
                        logger.error(
                            "Failed to stop updater: simulation_id=%s error=%s",
                            simulation_id,
                            exc,
                        )
                cls._updaters.clear()
            logger.info("All graph memory updaters stopped")

    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        return {sim_id: updater.get_stats() for sim_id, updater in cls._updaters.items()}


# Backwards-compatible aliases
ZepGraphMemoryUpdater = GraphMemoryUpdater
ZepGraphMemoryManager = GraphMemoryManager
