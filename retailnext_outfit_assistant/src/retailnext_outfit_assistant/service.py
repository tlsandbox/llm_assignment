from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from retailnext_outfit_assistant.catalog import CatalogIndex, build_or_load_index, unique_article_types
from retailnext_outfit_assistant.db import RetailNextDB
from retailnext_outfit_assistant.openai_utils import (
    OpenAIConfig,
    analyze_outfit_image,
    make_client,
    text_embedding,
    transcribe_audio,
)
from retailnext_outfit_assistant.retrieval import top_k_cosine


_GITHUB_IMAGE_PREFIX = (
    "https://raw.githubusercontent.com/openai/openai-cookbook/main/"
    "examples/data/sample_clothes/sample_images"
)
_LOGGER = logging.getLogger(__name__)
_OPENAI_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="openai-call")
_GENERIC_KEYWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "i",
    "in",
    "is",
    "item",
    "look",
    "my",
    "need",
    "of",
    "on",
    "outfit",
    "please",
    "search",
    "shirt",
    "shirts",
    "show",
    "style",
    "t",
    "tee",
    "tshirt",
    "tshirts",
    "the",
    "this",
    "to",
    "want",
    "wants",
    "wife",
    "wives",
    "husband",
    "woman",
    "women",
    "man",
    "men",
    "girl",
    "girls",
    "boy",
    "boys",
    "her",
    "him",
}


class OutfitAssistantService:
    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or Path(__file__).resolve().parents[2]
        self.data_dir = self.root_dir / "data" / "sample_clothes"
        self.cache_dir = self.root_dir / "data" / "cache"
        self.image_dir = self.data_dir / "sample_images"
        self.db = RetailNextDB(self.root_dir / "data" / "retailnext_demo.db")

        self.cfg = OpenAIConfig.from_env()
        self.client = None
        self._embedding_cache: dict[str, np.ndarray] = {}
        self.search_timeout_seconds = self._env_timeout("RN_AI_SEARCH_TIMEOUT_SECONDS", 25.0)
        self.image_timeout_seconds = self._env_timeout("RN_AI_IMAGE_TIMEOUT_SECONDS", 50.0)
        self.match_timeout_seconds = self._env_timeout("RN_AI_MATCH_TIMEOUT_SECONDS", 20.0)
        self.transcribe_timeout_seconds = self._env_timeout("RN_AI_TRANSCRIBE_TIMEOUT_SECONDS", 25.0)
        self.request_timeout_seconds = self._env_timeout("RN_AI_REQUEST_TIMEOUT_SECONDS", 20.0)

        self.index: CatalogIndex = build_or_load_index(self.data_dir, self.cache_dir)
        self.article_types = unique_article_types(self.index.items)
        self._id_to_row = {item.id: i for i, item in enumerate(self.index.items)}

        self.db.upsert_catalog(self.index.items, self.image_dir)

    @property
    def ai_enabled(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def _ensure_client(self):
        if not self.ai_enabled:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        if self.client is None:
            self.client = make_client()
        return self.client

    @staticmethod
    def _env_timeout(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = float(raw)
        except Exception:
            return default
        return value if value > 0 else default

    @staticmethod
    def _remaining_timeout(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("AI time budget was exceeded.")
        return remaining

    def _run_with_timeout(self, operation: str, fn, timeout_seconds: float):
        safe_timeout = max(1.0, float(timeout_seconds))
        future = _OPENAI_EXECUTOR.submit(fn)
        try:
            return future.result(timeout=safe_timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise RuntimeError(f"{operation} timed out after {int(round(safe_timeout))}s.") from exc
        except Exception as exc:
            raise RuntimeError(f"{operation} failed: {exc}") from exc

    @staticmethod
    def _normalize_text(value: Any) -> str:
        text = str(value or "").strip().lower()
        chars = [ch if ch.isalnum() or ch.isspace() else " " for ch in text]
        return " ".join("".join(chars).split())

    @staticmethod
    def _tokenize(value: Any) -> list[str]:
        return [token for token in OutfitAssistantService._normalize_text(value).split() if token]

    @staticmethod
    def _compact(value: Any) -> str:
        return "".join(ch for ch in OutfitAssistantService._normalize_text(value) if ch.isalnum())

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = str(value or "").strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
        return out

    @staticmethod
    def _score_status(score: float) -> str:
        if score >= 0.99:
            return "match"
        if score >= 0.49:
            return "partial"
        return "miss"

    @staticmethod
    def _status_label(status: str) -> str:
        mapping = {"match": "Match", "partial": "Partial", "miss": "Missing"}
        return mapping.get(status, "Missing")

    def _extract_intent(self, session: dict[str, Any]) -> dict[str, Any]:
        query_text = str(session.get("query_text") or "").strip()
        query_tokens = self._tokenize(query_text)

        gender = ""
        women_tokens = {"woman", "women", "wife", "female", "ladies", "lady", "girl", "girls", "her"}
        men_tokens = {"man", "men", "husband", "male", "gentleman", "gentlemen", "boy", "boys", "him"}
        if any(token in women_tokens for token in query_tokens):
            gender = "Women"
        elif any(token in men_tokens for token in query_tokens):
            gender = "Men"

        article_hints: list[str] = []
        query_norm = self._normalize_text(query_text)
        query_compact_tokens = {self._compact(token) for token in query_tokens}
        for article in self.article_types:
            article_norm = self._normalize_text(article)
            article_compact = self._compact(article)
            if not article_norm:
                continue
            if f" {article_norm} " in f" {query_norm} ":
                article_hints.append(article)
                continue
            if article_compact in query_compact_tokens:
                article_hints.append(article)
                continue
            singular = article_compact[:-1] if article_compact.endswith("s") else article_compact
            if singular and singular in query_compact_tokens:
                article_hints.append(article)

        image_summary: dict[str, Any] = {}
        raw_image_summary = session.get("image_summary")
        if raw_image_summary:
            try:
                parsed = json.loads(str(raw_image_summary))
                if isinstance(parsed, dict):
                    image_summary = parsed
            except Exception:
                image_summary = {}

        image_gender = str(image_summary.get("gender") or "").strip()
        if self._normalize_text(image_gender) in {"unknown", ""}:
            image_gender = ""

        image_colors = image_summary.get("colors", [])
        image_article_types = image_summary.get("article_types", [])
        occasion = str(image_summary.get("occasion") or "").strip()
        if self._normalize_text(occasion) in {"unknown", ""}:
            occasion = ""

        if image_gender:
            gender = image_gender

        if isinstance(image_article_types, list):
            article_hints.extend([str(value).strip() for value in image_article_types if str(value).strip()])
        article_hints = self._dedupe(article_hints)

        color_hints: list[str] = []
        if isinstance(image_colors, list):
            color_hints.extend([str(value).strip() for value in image_colors if str(value).strip()])
        if not color_hints:
            # Query color hint extraction from known catalog colors.
            catalog_colors = {item.base_colour for item in self.index.items if item.base_colour}
            for color in sorted(catalog_colors):
                if f" {self._normalize_text(color)} " in f" {query_norm} ":
                    color_hints.append(color)
        color_hints = self._dedupe(color_hints)

        usage_hints: list[str] = []
        if occasion:
            usage_hints.append(occasion)
        usage_hints = self._dedupe(usage_hints)

        motif_keywords = [
            token
            for token in query_tokens
            if len(token) >= 4 and token not in _GENERIC_KEYWORDS and token not in {"women", "woman", "men", "man"}
        ]

        return {
            "query_text": query_text,
            "gender": gender,
            "article_hints": article_hints,
            "color_hints": color_hints,
            "usage_hints": usage_hints,
            "motif_keywords": self._dedupe(motif_keywords),
        }

    def _signal_gender(self, intent: dict[str, Any], product: dict[str, Any]) -> dict[str, Any]:
        expected = str(intent.get("gender") or "").strip()
        actual = str(product.get("gender") or "").strip()
        if not expected:
            return {
                "attribute": "Gender",
                "expected": "Not specified",
                "actual": actual or "Unknown",
                "status": "not_specified",
                "score": None,
                "weight": 0.0,
                "note": "No gender intent was detected from your input.",
                "reason": "No gender signal in the query/image intent.",
                "matched_values": [],
                "missing_values": [],
            }
        status = "match" if self._normalize_text(expected) == self._normalize_text(actual) else "miss"
        score = 1.0 if status == "match" else 0.0
        if status == "match":
            reason = f"Gender matched exactly ({expected} vs {actual or 'Unknown'})."
            matched_values = [expected]
            missing_values: list[str] = []
        else:
            reason = (
                f"Gender mismatch: expected {expected}, but product gender is {actual or 'Unknown'}."
            )
            matched_values = []
            missing_values = [expected]
        return {
            "attribute": "Gender",
            "expected": expected,
            "actual": actual or "Unknown",
            "status": status,
            "score": score,
            "weight": 0.22,
            "note": "Gender alignment is a strong filtering signal.",
            "reason": reason,
            "matched_values": matched_values,
            "missing_values": missing_values,
        }

    def _signal_article_type(self, intent: dict[str, Any], product: dict[str, Any]) -> dict[str, Any]:
        raw_hints = [str(value).strip() for value in intent.get("article_hints", []) if str(value).strip()]
        hints = [(raw_hint, self._normalize_text(raw_hint)) for raw_hint in raw_hints]
        actual = str(product.get("article_type") or "").strip()
        actual_norm = self._normalize_text(actual)
        if not hints:
            return {
                "attribute": "Article Type",
                "expected": "Not specified",
                "actual": actual or "Unknown",
                "status": "not_specified",
                "score": None,
                "weight": 0.0,
                "note": "No explicit article type was detected from your input.",
                "reason": "No article type signal in the query/image intent.",
                "matched_values": [],
                "missing_values": [],
            }

        exact_matches: list[str] = []
        partial_matches: list[str] = []
        missing_hints: list[str] = []
        for raw_hint, hint_norm in hints:
            if not hint_norm:
                continue
            if hint_norm == actual_norm:
                exact_matches.append(raw_hint)
                continue
            if hint_norm in actual_norm or actual_norm in hint_norm:
                partial_matches.append(raw_hint)
                continue
            missing_hints.append(raw_hint)

        if exact_matches:
            score = 1.0
            status = "match"
            related = f" Related intent keyword(s): {', '.join(partial_matches)}." if partial_matches else ""
            reason = (
                f"Exact article type match: requested {', '.join(exact_matches)} and product type is "
                f"{actual or 'Unknown'}.{related}"
            )
        elif partial_matches:
            score = 0.6
            status = "partial"
            missing_note = (
                f" Missing requested type(s): {', '.join(missing_hints)}."
                if missing_hints
                else ""
            )
            reason = (
                f"Partial article type match: requested {', '.join(partial_matches)} is related to "
                f"product type {actual or 'Unknown'}.{missing_note}"
            )
        else:
            score = 0.0
            status = "miss"
            reason = (
                f"Article type missing: requested {', '.join(raw_hints)}, but product type is {actual or 'Unknown'}."
            )

        matched_values = exact_matches + partial_matches
        if not missing_hints and status == "miss":
            missing_hints = raw_hints

        return {
            "attribute": "Article Type",
            "expected": ", ".join(raw_hints),
            "actual": actual or "Unknown",
            "status": status,
            "score": score,
            "weight": 0.28,
            "note": "Article type fit is the strongest product-structure signal.",
            "reason": reason,
            "matched_values": self._dedupe(matched_values),
            "missing_values": self._dedupe(missing_hints),
        }

    def _signal_color(self, intent: dict[str, Any], product: dict[str, Any]) -> dict[str, Any]:
        raw_hints = [str(value).strip() for value in intent.get("color_hints", []) if str(value).strip()]
        hints = [(raw_hint, self._normalize_text(raw_hint)) for raw_hint in raw_hints]
        actual = str(product.get("base_colour") or "").strip()
        actual_norm = self._normalize_text(actual)
        if not hints:
            return {
                "attribute": "Color",
                "expected": "Not specified",
                "actual": actual or "Unknown",
                "status": "not_specified",
                "score": None,
                "weight": 0.0,
                "note": "No explicit color preference was detected.",
                "reason": "No color signal in the query/image intent.",
                "matched_values": [],
                "missing_values": [],
            }

        exact_matches: list[str] = []
        partial_matches: list[str] = []
        missing_hints: list[str] = []
        for raw_hint, hint_norm in hints:
            if hint_norm == actual_norm:
                exact_matches.append(raw_hint)
                continue
            if hint_norm and (hint_norm in actual_norm or actual_norm in hint_norm):
                partial_matches.append(raw_hint)
                continue
            missing_hints.append(raw_hint)

        if exact_matches:
            score = 1.0
            status = "match"
            reason = f"Color matched exactly ({', '.join(exact_matches)})."
        elif partial_matches:
            score = 0.6
            status = "partial"
            reason = (
                f"Color partially matched via related tone(s): {', '.join(partial_matches)} vs "
                f"product color {actual or 'Unknown'}."
            )
        else:
            score = 0.0
            status = "miss"
            reason = f"Color missing: requested {', '.join(raw_hints)}, product color is {actual or 'Unknown'}."

        return {
            "attribute": "Color",
            "expected": ", ".join(raw_hints),
            "actual": actual or "Unknown",
            "status": status,
            "score": score,
            "weight": 0.15,
            "note": "Color contributes to style similarity when provided.",
            "reason": reason,
            "matched_values": self._dedupe(exact_matches + partial_matches),
            "missing_values": self._dedupe(missing_hints),
        }

    def _signal_usage(self, intent: dict[str, Any], product: dict[str, Any]) -> dict[str, Any]:
        raw_hints = [str(value).strip() for value in intent.get("usage_hints", []) if str(value).strip()]
        hints = [(raw_hint, self._normalize_text(raw_hint)) for raw_hint in raw_hints]
        actual = str(product.get("usage") or "").strip()
        actual_norm = self._normalize_text(actual)
        if not hints:
            return {
                "attribute": "Occasion / Usage",
                "expected": "Not specified",
                "actual": actual or "Unknown",
                "status": "not_specified",
                "score": None,
                "weight": 0.0,
                "note": "No occasion/usage intent was detected.",
                "reason": "No occasion/usage signal in the query/image intent.",
                "matched_values": [],
                "missing_values": [],
            }

        exact_matches: list[str] = []
        partial_matches: list[str] = []
        missing_hints: list[str] = []
        for raw_hint, hint_norm in hints:
            if hint_norm == actual_norm:
                exact_matches.append(raw_hint)
                continue
            if hint_norm and (hint_norm in actual_norm or actual_norm in hint_norm):
                partial_matches.append(raw_hint)
                continue
            missing_hints.append(raw_hint)

        if exact_matches:
            score = 1.0
            status = "match"
            reason = f"Usage/occasion matched exactly ({', '.join(exact_matches)})."
        elif partial_matches:
            score = 0.5
            status = "partial"
            reason = (
                f"Usage partially matched via related term(s): {', '.join(partial_matches)} vs "
                f"product usage {actual or 'Unknown'}."
            )
        else:
            score = 0.0
            status = "miss"
            reason = f"Usage missing: requested {', '.join(raw_hints)}, product usage is {actual or 'Unknown'}."

        return {
            "attribute": "Occasion / Usage",
            "expected": ", ".join(raw_hints),
            "actual": actual or "Unknown",
            "status": status,
            "score": score,
            "weight": 0.1,
            "note": "Usage helps refine whether the item fits the occasion.",
            "reason": reason,
            "matched_values": self._dedupe(exact_matches + partial_matches),
            "missing_values": self._dedupe(missing_hints),
        }

    def _signal_motif(self, intent: dict[str, Any], product: dict[str, Any]) -> dict[str, Any]:
        raw_keywords = [str(value).strip() for value in intent.get("motif_keywords", []) if str(value).strip()]
        keywords = [self._normalize_text(value) for value in raw_keywords]
        product_blob = " ".join(
            [
                str(product.get("name") or ""),
                str(product.get("article_type") or ""),
                str(product.get("base_colour") or ""),
                str(product.get("usage") or ""),
            ]
        )
        product_norm = self._normalize_text(product_blob)
        if not keywords:
            return {
                "attribute": "Style Keyword",
                "expected": "Not specified",
                "actual": str(product.get("name") or ""),
                "status": "not_specified",
                "score": None,
                "weight": 0.0,
                "note": "No unique style keyword (motif/theme) was detected.",
                "reason": "No specific style keyword (motif/theme) was provided.",
                "matched_values": [],
                "missing_values": [],
                "matched_keywords": [],
                "unmatched_keywords": [],
            }

        matched = [keyword for keyword in keywords if keyword and keyword in product_norm]
        unmatched = [keyword for keyword in keywords if keyword not in matched]
        ratio = len(matched) / max(1, len(keywords))
        status = "match" if ratio >= 0.99 else ("partial" if ratio >= 0.25 else "miss")
        normalized_to_raw = {self._normalize_text(value): value for value in raw_keywords}
        matched_raw = [normalized_to_raw.get(keyword, keyword) for keyword in matched]
        unmatched_raw = [normalized_to_raw.get(keyword, keyword) for keyword in unmatched]

        if status == "match":
            reason = f"All style keyword(s) matched in product metadata: {', '.join(matched_raw)}."
        elif status == "partial":
            reason = (
                f"Partial style keyword match: matched {', '.join(matched_raw)}; "
                f"missing {', '.join(unmatched_raw)}."
            )
        else:
            reason = f"Style keyword missing: none of these were found in product metadata: {', '.join(unmatched_raw)}."

        return {
            "attribute": "Style Keyword",
            "expected": ", ".join(raw_keywords),
            "actual": ", ".join(matched_raw) if matched_raw else "No direct keyword match",
            "status": status,
            "score": ratio,
            "weight": 0.25,
            "note": "Keywords capture motif/theme intent such as 'sakura'.",
            "reason": reason,
            "matched_values": matched_raw,
            "missing_values": unmatched_raw,
            "matched_keywords": matched_raw,
            "unmatched_keywords": unmatched_raw,
        }

    def _heuristic_breakdown(self, session: dict[str, Any], product: dict[str, Any]) -> dict[str, Any]:
        intent = self._extract_intent(session)
        checks = [
            self._signal_gender(intent, product),
            self._signal_article_type(intent, product),
            self._signal_color(intent, product),
            self._signal_usage(intent, product),
            self._signal_motif(intent, product),
        ]

        active_checks = [check for check in checks if check.get("score") is not None and check.get("weight", 0) > 0]
        total_weight = sum(float(check["weight"]) for check in active_checks)
        weighted_score = (
            sum(float(check["score"]) * float(check["weight"]) for check in active_checks) / total_weight
            if total_weight > 0
            else 0.0
        )

        confidence = round(0.2 + 0.7 * weighted_score, 2)
        confidence = max(0.2, min(0.9, confidence))

        if weighted_score >= 0.78:
            verdict = "Strong match"
        elif weighted_score >= 0.62:
            verdict = "Good match"
        elif weighted_score >= 0.45:
            verdict = "Possible match"
        else:
            verdict = "Weak match"

        matched = [check for check in active_checks if check["status"] == "match"]
        partial = [check for check in active_checks if check["status"] == "partial"]
        missed = [check for check in active_checks if check["status"] == "miss"]

        matched_labels = ", ".join(check["attribute"] for check in matched) or "none"
        partial_labels = ", ".join(check["attribute"] for check in partial) or "none"
        missed_labels = ", ".join(check["attribute"] for check in missed) or "none"

        explanation = (
            f"Score explanation: {int(round(confidence * 100))}% because matched signals = {matched_labels}; "
            f"partial signals = {partial_labels}; missing signals = {missed_labels}."
        )
        signal_detail_parts: list[str] = []
        for check in active_checks:
            reason = str(check.get("reason") or "").strip()
            if reason:
                signal_detail_parts.append(f"{check.get('attribute')}: {reason}")
        if signal_detail_parts:
            explanation += " Details: " + " ".join(signal_detail_parts)

        details: list[dict[str, Any]] = []
        for check in checks:
            status = check.get("status")
            label = "Not specified" if status == "not_specified" else self._status_label(str(status))
            details.append(
                {
                    "attribute": check.get("attribute"),
                    "expected": check.get("expected"),
                    "actual": check.get("actual"),
                    "status": label,
                    "score": None if check.get("score") is None else round(float(check["score"]), 2),
                    "weight": round(float(check.get("weight", 0.0)), 2),
                    "note": check.get("note"),
                    "reason": check.get("reason"),
                    "matched_values": check.get("matched_values", []),
                    "missing_values": check.get("missing_values", []),
                }
            )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "rationale": explanation,
            "intent": intent,
            "details": details,
            "matched_count": len(matched),
            "active_count": len(active_checks),
            "weighted_score": round(weighted_score, 3),
        }

    def _embed(self, query: str, *, timeout_seconds: float | None = None) -> np.ndarray:
        cleaned = query.strip()
        if cleaned in self._embedding_cache:
            return self._embedding_cache[cleaned]

        client = self._ensure_client()
        embedding_values = self._run_with_timeout(
            "Embedding request",
            lambda: text_embedding(client, cleaned, self.cfg.embedding_model),
            timeout_seconds if timeout_seconds is not None else self.request_timeout_seconds,
        )
        embedding = np.asarray(embedding_values, dtype=np.float32)
        self._embedding_cache[cleaned] = embedding
        return embedding

    def _rank_from_queries(
        self,
        queries: list[str],
        *,
        top_k: int,
        deadline: float | None = None,
    ) -> list[tuple[int, float]]:
        query_scores: dict[int, float] = {}

        for query in queries[:6]:
            cleaned = query.strip()
            if not cleaned:
                continue

            call_timeout = self.request_timeout_seconds
            if deadline is not None:
                call_timeout = min(call_timeout, self._remaining_timeout(deadline))
            emb = self._embed(cleaned, timeout_seconds=call_timeout)
            idx, scores = top_k_cosine(emb, self.index.embeddings, self.index.norms, k=max(top_k * 8, top_k))
            for row_id, score in zip(idx.tolist(), scores.tolist()):
                previous = query_scores.get(row_id)
                if previous is None or score > previous:
                    query_scores[row_id] = float(score)

        ranked_rows = sorted(query_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [(self.index.items[row_idx].id, score) for row_idx, score in ranked_rows]

    def _random_recommendations(self, top_k: int) -> list[tuple[int, float]]:
        rows = self.db.list_random_products(top_k)
        return [(int(row["id"]), 0.0) for row in rows]

    def _build_public_product(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "name": row["name"],
            "gender": row["gender"],
            "master_category": row["master_category"],
            "sub_category": row["sub_category"],
            "article_type": row["article_type"],
            "base_colour": row["base_colour"],
            "season": row["season"],
            "year": row["year"],
            "usage": row["usage"],
            "image_url": f"/api/image/{int(row['id'])}",
        }

    def home_feed(self, *, limit: int = 24, gender: str | None = None) -> list[dict[str, Any]]:
        rows = self.db.list_random_products(limit, gender=gender)
        return [self._build_public_product(row) for row in rows]

    def transcribe_voice(self, *, audio_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
        if not audio_bytes:
            raise ValueError("Audio payload is empty.")

        client = self._ensure_client()
        text = self._run_with_timeout(
            "Transcription request",
            lambda: transcribe_audio(
                client=client,
                audio_bytes=audio_bytes,
                filename=filename,
                model=self.cfg.audio_model,
            ),
            self.transcribe_timeout_seconds,
        )
        if not text:
            raise ValueError("No speech detected. Please try again and speak clearly.")

        return {
            "text": text,
            "model": self.cfg.audio_model,
            "ai_powered": True,
        }

    def search_by_text(self, *, query: str, shopper_name: str = "Bob", top_k: int = 10) -> dict[str, Any]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Please enter a search query.")

        ai_powered = False
        if self.ai_enabled:
            deadline = time.monotonic() + self.search_timeout_seconds
            try:
                ranked = self._rank_from_queries([cleaned], top_k=top_k, deadline=deadline)
                note = "Outfit Assistant AI powered by OpenAI generated these recommendations from your search."
                ai_powered = True
            except Exception:
                _LOGGER.warning("Falling back to random recommendations for text search due to OpenAI unavailability.")
                ranked = self._random_recommendations(top_k)
                note = "OpenAI search is unavailable right now. Showing fallback recommendations so you can continue."
        else:
            ranked = self._random_recommendations(top_k)
            note = "OpenAI key missing, so recommendations are random placeholders for UI demo."

        session_id = self.db.create_session(
            shopper_name=shopper_name,
            source="natural-language-query-search",
            query_text=cleaned,
            image_summary=None,
        )
        self.db.store_recommendations(session_id, ranked)

        payload = self.get_personalized(session_id)
        payload["assistant_note"] = note
        payload["ai_powered"] = ai_powered
        return payload

    def search_by_image(
        self,
        *,
        image_bytes: bytes,
        shopper_name: str = "Bob",
        top_k: int = 10,
    ) -> dict[str, Any]:
        ai_powered = False
        if self.ai_enabled:
            deadline = time.monotonic() + self.image_timeout_seconds
            try:
                client = self._ensure_client()
                analysis = self._run_with_timeout(
                    "Vision analysis request",
                    lambda: analyze_outfit_image(
                        client=client,
                        image_bytes=image_bytes,
                        article_types=self.article_types,
                        model=self.cfg.vision_model,
                    ),
                    min(self.request_timeout_seconds, self._remaining_timeout(deadline)),
                )
                queries = [q for q in analysis.get("search_queries", []) if isinstance(q, str) and q.strip()]
                if not queries:
                    fallback_query = " ".join(
                        [
                            str(analysis.get("gender", "")),
                            str(analysis.get("occasion", "")),
                            " ".join(analysis.get("colors", [])),
                            " ".join(analysis.get("article_types", [])),
                        ]
                    ).strip()
                    if fallback_query:
                        queries = [fallback_query]

                ranked = (
                    self._rank_from_queries(queries, top_k=top_k, deadline=deadline)
                    if queries
                    else self._random_recommendations(top_k)
                )
                image_summary = json.dumps(analysis)
                note = "Outfit Assistant AI powered by OpenAI analyzed your image and found matching items."
                ai_powered = True
            except Exception:
                _LOGGER.warning("Falling back to random recommendations for image search due to OpenAI unavailability.")
                analysis = {
                    "error": "openai_unavailable",
                    "search_queries": [],
                }
                ranked = self._random_recommendations(top_k)
                image_summary = json.dumps(analysis)
                note = "Image analysis is unavailable right now. Showing fallback recommendations so you can continue."
        else:
            analysis = {
                "error": "OPENAI_API_KEY not configured",
                "search_queries": [],
            }
            ranked = self._random_recommendations(top_k)
            image_summary = json.dumps(analysis)
            note = "OpenAI key missing, so image flow returned random placeholder recommendations."

        session_id = self.db.create_session(
            shopper_name=shopper_name,
            source="image-upload-match",
            query_text=None,
            image_summary=image_summary,
        )
        self.db.store_recommendations(session_id, ranked)

        payload = self.get_personalized(session_id)
        payload["assistant_note"] = note
        payload["image_analysis"] = analysis
        payload["ai_powered"] = ai_powered
        return payload

    def get_personalized(self, session_id: str) -> dict[str, Any]:
        session = self.db.get_session(session_id)
        if not session:
            raise KeyError("Recommendation session not found.")

        rows = self.db.get_recommendations(session_id)
        products: list[dict[str, Any]] = []
        for row in rows:
            product = self._build_public_product(row)
            product["rank"] = int(row["rank_position"])
            product["score"] = float(row["score"])
            if row.get("match_verdict"):
                product["match"] = {
                    "verdict": row["match_verdict"],
                    "rationale": row["match_rationale"],
                    "confidence": row["match_confidence"],
                }
            products.append(product)

        return {
            "session": session,
            "recommendations": products,
        }

    def _heuristic_match(self, session: dict[str, Any], product: dict[str, Any]) -> tuple[str, str, float]:
        assessment = self._heuristic_breakdown(session, product)
        return (
            str(assessment["verdict"]),
            str(assessment["rationale"]),
            float(assessment["confidence"]),
        )

    def check_match(self, *, session_id: str, product_id: int) -> dict[str, Any]:
        session = self.db.get_session(session_id)
        if not session:
            raise KeyError("Recommendation session not found.")

        product = self.db.get_product(product_id)
        if not product:
            raise KeyError("Product not found.")
        heuristic_assessment = self._heuristic_breakdown(session, product)

        ai_powered = False
        if self.ai_enabled:
            client = self._ensure_client()
            prompt = (
                "You are RetailNext's Outfit Assistant. Evaluate if the recommended product is a good match for Bob.\n"
                "Return JSON only with keys: verdict, rationale, confidence.\n"
                "verdict must be one of: Strong match, Good match, Possible match, Weak match.\n"
                "confidence must be a number between 0 and 1.\n\n"
                f"SESSION_SOURCE: {session.get('source')}\n"
                f"SHOPPER_QUERY: {session.get('query_text')}\n"
                f"IMAGE_SUMMARY_JSON: {session.get('image_summary')}\n\n"
                "PRODUCT:\n"
                f"- id: {product['id']}\n"
                f"- name: {product['name']}\n"
                f"- gender: {product['gender']}\n"
                f"- article_type: {product['article_type']}\n"
                f"- base_colour: {product['base_colour']}\n"
                f"- usage: {product['usage']}\n"
                f"- season: {product['season']}\n"
                f"- year: {product['year']}\n"
            )

            try:
                resp = self._run_with_timeout(
                    "Match scoring request",
                    lambda: client.responses.create(
                        model=self.cfg.chat_model,
                        input=prompt,
                        temperature=0.2,
                    ),
                    self.match_timeout_seconds,
                )
                raw = (resp.output_text or "").strip()
                parsed = json.loads(raw)
                verdict = str(parsed.get("verdict", "Possible match"))
                llm_rationale = str(parsed.get("rationale", "The item has partial alignment with Bob's intent.")).strip()
                confidence = parsed.get("confidence")
                confidence = float(confidence) if confidence is not None else 0.5
                ai_powered = True
                rationale = (
                    f"{llm_rationale} "
                    f"Heuristic check: {heuristic_assessment['rationale']}"
                ).strip()
            except Exception:
                _LOGGER.warning("Falling back to heuristic match scoring due to OpenAI unavailability.")
                verdict = str(heuristic_assessment["verdict"])
                rationale = str(heuristic_assessment["rationale"])
                confidence = float(heuristic_assessment["confidence"])
        else:
            verdict = str(heuristic_assessment["verdict"])
            rationale = str(heuristic_assessment["rationale"])
            confidence = float(heuristic_assessment["confidence"])

        self.db.store_match_check(
            session_id=session_id,
            product_id=product_id,
            verdict=verdict,
            rationale=rationale,
            confidence=confidence,
        )

        return {
            "session_id": session_id,
            "product_id": product_id,
            "verdict": verdict,
            "rationale": rationale,
            "confidence": confidence,
            "ai_powered": ai_powered,
            "judgement_details": heuristic_assessment,
        }

    def image_path_for_product(self, product_id: int) -> Path | None:
        path = self.image_dir / f"{product_id}.jpg"
        if path.exists():
            return path
        return None

    def fallback_image_url(self, product_id: int) -> str:
        return f"{_GITHUB_IMAGE_PREFIX}/{product_id}.jpg"

    def stats(self) -> dict[str, Any]:
        details = self.db.stats()
        details["ai_enabled"] = self.ai_enabled
        details["catalog_items_in_memory"] = len(self.index.items)
        return details
