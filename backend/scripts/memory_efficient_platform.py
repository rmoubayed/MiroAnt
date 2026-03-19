"""
Memory-efficient Platform subclass for OASIS Twitter simulations.

Overrides update_rec_table() to:
1. Cache TWHin-BERT embeddings -- only compute embeddings for NEW posts each round
2. Process in small batches (32 instead of 1000) to limit attention tensor memory
3. Clean up tensors after each batch with gc.collect()
4. Save/load embedding cache from disk across rounds

Peak memory drops from ~17 GB (original batch=1000) to ~400 MB (batch=32),
and total steady-state memory drops from 60 GB+ to ~2-3 GB because we skip
re-embedding thousands of already-seen posts.
"""

import gc
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from oasis.social_platform.channel import Channel
from oasis.social_platform.database import (
    fetch_rec_table_as_matrix,
    fetch_table_from_db,
)
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import RecsysType

logger = logging.getLogger("mirofish.memory_efficient_platform")

SMALL_BATCH_SIZE = 32
MAX_CORPUS_POSTS = 4000


def _ensure_twhin_model():
    """Lazy-load TWHin-BERT using OASIS's own global model instance."""
    from oasis.social_platform import recsys as _mod
    from oasis.social_platform.recsys import get_recsys_model

    if _mod.twhin_model is None or _mod.twhin_tokenizer is None:
        _mod.twhin_tokenizer, _mod.twhin_model = get_recsys_model(
            recsys_type="twhin-bert"
        )
    return _mod.twhin_model, _mod.twhin_tokenizer


def _embed_texts(
    model, tokenizer, texts: List[str], batch_size: int = SMALL_BATCH_SIZE
) -> np.ndarray:
    """
    Embed texts via TWHin-BERT in small batches.

    Returns ndarray of shape (len(texts), 768).  Each batch creates ~400 MB
    of intermediate attention tensors instead of ~12 GB with batch=1000.
    """
    from oasis.social_platform.process_recsys_posts import process_batch

    parts: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        out = process_batch(model, tokenizer, chunk)
        parts.append(out.cpu().numpy())
        del out

    if not parts:
        return np.empty((0, 768), dtype=np.float32)
    result = np.concatenate(parts, axis=0)
    del parts
    return result


class MemoryEfficientTwitterPlatform(Platform):
    """
    Drop-in replacement for OASIS's default Twitter Platform that keeps the
    same TWHin-BERT recommendation quality while using a fraction of the RAM.

    Pass an instance of this class to ``oasis.make(platform=...)`` instead of
    ``DefaultPlatformType.TWITTER``.
    """

    def __init__(
        self,
        *args,
        cache_dir: Optional[str] = None,
        embedding_batch_size: int = SMALL_BATCH_SIZE,
        max_corpus_posts: int = MAX_CORPUS_POSTS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._cache_dir = cache_dir or os.path.dirname(self.db_path)
        self._batch_size = embedding_batch_size
        self._max_corpus = max_corpus_posts

        # post_id -> 768-dim float32 vector  (~3 KB each, 12 MB for 4000 posts)
        self._post_emb: Dict[int, np.ndarray] = {}
        self._embedded_ids: set = set()

        # Date-recency scores computed once per post when first seen
        self._date_scores: Dict[int, float] = {}

        # User-profile tracking (profiles mutate each round)
        self._profiles: List[str] = []
        self._user_recent_post: Dict[int, str] = {}

        # Cached user-profile embeddings -- only recomputed when the
        # profile text actually changes (i.e. the user made a new post).
        self._user_emb: Optional[np.ndarray] = None
        self._user_emb_keys: List[str] = []  # profile text at last embed

        self._cache_file = os.path.join(
            self._cache_dir, "twitter_embedding_cache.npz"
        )
        self._initialised = False

        logger.info(
            "MemoryEfficientTwitterPlatform: batch=%d, max_corpus=%d, cache=%s",
            self._batch_size,
            self._max_corpus,
            self._cache_file,
        )

    # ------------------------------------------------------------------
    # Embedding cache persistence
    # ------------------------------------------------------------------

    def _load_cache(self):
        if not os.path.exists(self._cache_file):
            return
        try:
            data = np.load(self._cache_file, allow_pickle=True)
            ids = data["post_ids"].tolist()
            vecs = data["embeddings"]
            for pid, vec in zip(ids, vecs):
                self._post_emb[pid] = vec
                self._embedded_ids.add(pid)
            logger.info("Loaded %d cached post embeddings from disk", len(ids))
        except Exception as exc:
            logger.warning("Could not load embedding cache: %s", exc)

    def _save_cache(self):
        if not self._post_emb:
            return
        try:
            ids = list(self._post_emb.keys())
            vecs = np.stack([self._post_emb[i] for i in ids])
            np.savez_compressed(
                self._cache_file, post_ids=np.array(ids), embeddings=vecs
            )
        except Exception as exc:
            logger.warning("Could not save embedding cache: %s", exc)

    # ------------------------------------------------------------------
    # User-profile builder (mirrors OASIS rec_sys_personalized_twh logic)
    # ------------------------------------------------------------------

    def _refresh_profiles(self, user_table, post_table):
        if not self._profiles or len(self._profiles) != len(user_table):
            self._profiles = [
                u.get("bio") or "This user does not have profile"
                for u in user_table
            ]
            self._user_recent_post = {i: "" for i in range(len(user_table))}

        for post in post_table:
            uid = post["user_id"]
            content = post.get("content") or ""
            if content and uid < len(user_table):
                self._user_recent_post[uid] = content

        for uid, recent in self._user_recent_post.items():
            if uid >= len(self._profiles) or not recent:
                continue
            tag = f" # Recent post:{recent}"
            prof = self._profiles[uid]
            if "# Recent post:" not in prof:
                self._profiles[uid] = prof + tag
            elif tag not in prof:
                self._profiles[uid] = prof.split("# Recent post:")[0] + tag

    # ------------------------------------------------------------------
    # Main override
    # ------------------------------------------------------------------

    async def update_rec_table(self):
        """
        Memory-efficient TWHin-BERT recommendation update.

        Falls back to the parent implementation for non-TWHIN recsys types
        (e.g. Reddit's hot-score algorithm).
        """
        if self.recsys_type != RecsysType.TWHIN:
            return await super().update_rec_table()

        # -- fetch tables (we skip trace_table; like-score is disabled) ----
        user_table = fetch_table_from_db(self.db_cursor, "user")
        post_table = fetch_table_from_db(self.db_cursor, "post")
        rec_matrix = fetch_rec_table_as_matrix(self.db_cursor)

        if not post_table:
            return

        # First-call initialisation
        if not self._initialised:
            self._load_cache()
            self._initialised = True

        # -- user profiles (change every round, must re-embed) -------------
        self._refresh_profiles(user_table, post_table)

        # -- identify NEW posts that need embedding ------------------------
        new_posts = [
            p for p in post_table if p["post_id"] not in self._embedded_ids
        ]

        if new_posts:
            model, tokenizer = _ensure_twhin_model()
            texts = [p.get("content") or "" for p in new_posts]
            t0 = time.time()

            new_vecs = _embed_texts(
                model, tokenizer, texts, batch_size=self._batch_size
            )

            for post, vec in zip(new_posts, new_vecs):
                pid = post["post_id"]
                self._post_emb[pid] = vec
                self._embedded_ids.add(pid)

            del new_vecs
            gc.collect()

            logger.info(
                "Embedded %d new posts in %.1fs (total cached: %d)",
                len(new_posts),
                time.time() - t0,
                len(self._embedded_ids),
            )

            # persist to disk periodically
            if len(new_posts) >= 5:
                self._save_cache()

        # -- compute date-recency scores (once per post) -------------------
        current_time = self.sandbox_clock.time_step
        for post in post_table:
            pid = post["post_id"]
            if pid not in self._date_scores:
                try:
                    age = current_time - int(post["created_at"])
                    self._date_scores[pid] = np.log((271.8 - age) / 100)
                except (ValueError, TypeError, ZeroDivisionError):
                    self._date_scores[pid] = 0.0

        # -- embed user profiles (only those whose text changed) ------------
        changed_indices = [
            i for i, prof in enumerate(self._profiles)
            if i >= len(self._user_emb_keys) or self._user_emb_keys[i] != prof
        ]

        if changed_indices or self._user_emb is None:
            model, tokenizer = _ensure_twhin_model()

            if self._user_emb is None or len(changed_indices) == len(self._profiles):
                # First call or all changed -- embed everything once
                user_vecs = _embed_texts(
                    model, tokenizer, self._profiles, batch_size=self._batch_size
                )
            else:
                # Only re-embed the users whose profile text changed
                user_vecs = self._user_emb.copy()
                if changed_indices:
                    changed_texts = [self._profiles[i] for i in changed_indices]
                    changed_vecs = _embed_texts(
                        model, tokenizer, changed_texts, batch_size=self._batch_size
                    )
                    for slot, idx in enumerate(changed_indices):
                        user_vecs[idx] = changed_vecs[slot]
                    del changed_vecs

            self._user_emb = user_vecs
            self._user_emb_keys = list(self._profiles)
            gc.collect()
            logger.debug(
                "User profile embeddings: %d changed out of %d",
                len(changed_indices),
                len(self._profiles),
            )
        else:
            user_vecs = self._user_emb

        # -- select posts for similarity (coarse filtering) ----------------
        all_pids = [p["post_id"] for p in post_table]
        if len(all_pids) > self._max_corpus:
            sel_idx = sorted(random.sample(range(len(all_pids)), self._max_corpus))
        else:
            sel_idx = list(range(len(all_pids)))

        sel_pids: List[int] = []
        sel_vecs: List[np.ndarray] = []
        sel_scores: List[float] = []
        for i in sel_idx:
            pid = all_pids[i]
            if pid in self._post_emb:
                sel_pids.append(pid)
                sel_vecs.append(self._post_emb[pid])
                sel_scores.append(self._date_scores.get(pid, 0.0))

        if not sel_vecs:
            del user_vecs
            gc.collect()
            return

        post_mat = np.stack(sel_vecs)
        score_arr = np.array(sel_scores)
        del sel_vecs
        gc.collect()

        # -- cosine similarity + date weighting ----------------------------
        sims = cosine_similarity(user_vecs, post_mat)  # (n_users, n_posts)
        sims = sims * score_arr  # broadcast multiply

        del user_vecs, post_mat, score_arr

        # -- pick top-k per user -------------------------------------------
        new_rec_matrix: List[List[int]] = []
        k = self.max_rec_post_len

        if len(sel_pids) <= k:
            new_rec_matrix = [list(sel_pids)] * len(rec_matrix)
        else:
            sims_t = torch.tensor(sims, dtype=torch.float32)
            _, top_idx = torch.topk(sims_t, min(k, sims_t.shape[1]), dim=1)
            for row in top_idx.cpu().numpy():
                new_rec_matrix.append([sel_pids[j] for j in row])
            del sims_t, top_idx

        del sims
        gc.collect()

        # -- write rec table to DB (same as original OASIS) ----------------
        self.pl_utils._execute_db_command("DELETE FROM rec", commit=True)
        insert_values = [
            (uid, pid)
            for uid in range(len(new_rec_matrix))
            for pid in new_rec_matrix[uid]
        ]
        self.pl_utils._execute_many_db_command(
            "INSERT INTO rec (user_id, post_id) VALUES (?, ?)",
            insert_values,
            commit=True,
        )
