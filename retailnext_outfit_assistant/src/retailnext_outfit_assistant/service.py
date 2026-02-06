from __future__ import annotations

import json
import os
from pathlib import Path
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

    def _embed(self, query: str) -> np.ndarray:
        cleaned = query.strip()
        if cleaned in self._embedding_cache:
            return self._embedding_cache[cleaned]

        client = self._ensure_client()
        embedding = np.asarray(text_embedding(client, cleaned, self.cfg.embedding_model), dtype=np.float32)
        self._embedding_cache[cleaned] = embedding
        return embedding

    def _rank_from_queries(self, queries: list[str], *, top_k: int) -> list[tuple[int, float]]:
        query_scores: dict[int, float] = {}

        for query in queries[:6]:
            cleaned = query.strip()
            if not cleaned:
                continue

            emb = self._embed(cleaned)
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
        text = transcribe_audio(
            client=client,
            audio_bytes=audio_bytes,
            filename=filename,
            model=self.cfg.audio_model,
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

        if self.ai_enabled:
            ranked = self._rank_from_queries([cleaned], top_k=top_k)
            note = "Outfit Assistant AI powered by OpenAI generated these recommendations from your search."
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
        payload["ai_powered"] = self.ai_enabled
        return payload

    def search_by_image(
        self,
        *,
        image_bytes: bytes,
        shopper_name: str = "Bob",
        top_k: int = 10,
    ) -> dict[str, Any]:
        if self.ai_enabled:
            client = self._ensure_client()
            analysis = analyze_outfit_image(
                client=client,
                image_bytes=image_bytes,
                article_types=self.article_types,
                model=self.cfg.vision_model,
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

            ranked = self._rank_from_queries(queries, top_k=top_k) if queries else self._random_recommendations(top_k)
            image_summary = json.dumps(analysis)
            note = "Outfit Assistant AI powered by OpenAI analyzed your image and found matching items."
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
        payload["ai_powered"] = self.ai_enabled
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
        intent = " ".join(
            [
                str(session.get("query_text") or ""),
                str(session.get("image_summary") or ""),
            ]
        ).lower()
        score = 0
        signals = [
            str(product.get("article_type", "")).lower(),
            str(product.get("base_colour", "")).lower(),
            str(product.get("usage", "")).lower(),
            str(product.get("gender", "")).lower(),
        ]
        for signal in signals:
            if signal and signal in intent:
                score += 1

        if score >= 3:
            return "Strong match", "Core style attributes align with Bob's current shopping intent.", 0.72
        if score == 2:
            return "Good match", "Several style signals line up and this is a practical option.", 0.61
        if score == 1:
            return "Possible match", "There is partial alignment, but review alternatives before checkout.", 0.5
        return "Weak match", "This item may not fit the current style intent as well as other recommendations.", 0.39

    def check_match(self, *, session_id: str, product_id: int) -> dict[str, Any]:
        session = self.db.get_session(session_id)
        if not session:
            raise KeyError("Recommendation session not found.")

        product = self.db.get_product(product_id)
        if not product:
            raise KeyError("Product not found.")

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

            resp = client.responses.create(
                model=self.cfg.chat_model,
                input=prompt,
                temperature=0.2,
            )
            raw = (resp.output_text or "").strip()
            try:
                parsed = json.loads(raw)
                verdict = str(parsed.get("verdict", "Possible match"))
                rationale = str(parsed.get("rationale", "The item has partial alignment with Bob's intent."))
                confidence = parsed.get("confidence")
                confidence = float(confidence) if confidence is not None else 0.5
            except Exception:
                verdict, rationale, confidence = self._heuristic_match(session, product)
        else:
            verdict, rationale, confidence = self._heuristic_match(session, product)

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
            "ai_powered": self.ai_enabled,
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
