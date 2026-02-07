from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retailnext_outfit_assistant.catalog import CatalogIndex, build_or_load_index, unique_article_types
from retailnext_outfit_assistant.inventory import demo_quantity
from retailnext_outfit_assistant.openai_utils import OpenAIConfig, analyze_outfit_image, craft_answer, make_client, text_embedding
from retailnext_outfit_assistant.retrieval import top_k_cosine


@dataclass(frozen=True)
class RecommendedProduct:
    id: int
    name: str
    gender: str
    article_type: str
    base_colour: str
    usage: str
    season: str
    year: int | None
    image: str | None
    availability: str


class RetailNextAssistant:
    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[2]
        self.data_dir = root / "data" / "sample_clothes"
        self.cache_dir = root / "data" / "cache"
        self.index: CatalogIndex | None = None
        self.article_types: list[str] = []
        self._id_to_row: dict[int, int] = {}
        self._embedding_cache: dict[str, np.ndarray] = {}
        self.cfg = OpenAIConfig.from_env()
        self.client = None

    def _ensure_client(self):
        if self.client is None:
            self.client = make_client()
        return self.client

    def _ensure_index(self) -> CatalogIndex:
        if self.index is None:
            self.index = build_or_load_index(self.data_dir, self.cache_dir)
            self.article_types = unique_article_types(self.index.items)
            self._id_to_row = {item.id: i for i, item in enumerate(self.index.items)}
        return self.index

    def _image_path_or_url(self, product_id: int) -> str | None:
        local = self.data_dir / "sample_images" / f"{product_id}.jpg"
        if local.exists():
            return str(local)
        return (
            "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/"
            f"sample_images/{product_id}.jpg"
        )

    def _rank_with_prefer_newest(self, idx: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Stable re-ranking: score first, then year desc
        index = self._ensure_index()
        years = np.array([index.items[i].year or 0 for i in idx], dtype=np.int32)
        order = np.lexsort((-years, -scores))
        return idx[order], scores[order]

    def _retrieve(self, query: str, top_k: int, prefer_newest: bool) -> list[int]:
        client = self._ensure_client()
        index = self._ensure_index()
        if query in self._embedding_cache:
            emb = self._embedding_cache[query]
        else:
            emb = np.asarray(text_embedding(client, query, self.cfg.embedding_model), dtype=np.float32)
            self._embedding_cache[query] = emb
        idx, scores = top_k_cosine(emb, index.embeddings, index.norms, k=top_k)
        if prefer_newest:
            idx, scores = self._rank_with_prefer_newest(idx, scores)
        return idx.tolist()

    def _extract_product_ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in re.findall(r"\\b\\d{4,6}\\b", text):
            try:
                ids.append(int(match))
            except Exception:
                continue
        return ids

    def _infer_gender(self, user_text: str, analysis_gender: str | None) -> str | None:
        if analysis_gender in {"Men", "Women", "Unisex"}:
            return analysis_gender
        t = user_text.lower()
        if "women" in t or "woman" in t or "ladies" in t:
            return "Women"
        if "men" in t or "man" in t or "mens" in t:
            return "Men"
        return None

    def _infer_usage(self, user_text: str, analysis_occasion: str | None) -> str | None:
        if analysis_occasion in {"Formal", "Casual", "Ethnic", "Party", "Sports"}:
            return analysis_occasion
        t = user_text.lower()
        for key in ("formal", "casual", "ethnic", "party", "sports"):
            if key in t:
                return key.capitalize()
        if "wedding" in t or "interview" in t or "office" in t:
            return "Formal"
        return None

    def _infer_colors(self, user_text: str, analysis_colors: list[str] | None) -> set[str] | None:
        colors: set[str] = set()
        if analysis_colors:
            colors |= {c.strip().lower() for c in analysis_colors if isinstance(c, str) and c.strip()}
        t = user_text.lower()
        for c in ("black", "white", "blue", "red", "green", "yellow", "pink", "purple", "brown", "grey", "gray"):
            if c in t:
                colors.add(c)
        return colors or None

    def _infer_article_types(self, analysis_article_types: list[str] | None) -> set[str] | None:
        if not analysis_article_types:
            return None
        allowed = set(self.article_types)
        chosen = {a for a in analysis_article_types if isinstance(a, str) and a in allowed}
        return chosen or None

    def _matches_filters(
        self,
        *,
        row: int,
        gender: str | None,
        usage: str | None,
        colors: set[str] | None,
        article_types: set[str] | None,
    ) -> bool:
        index = self._ensure_index()
        item = index.items[row]
        if gender and item.gender != gender and gender != "Unisex":
            return False
        if usage and item.usage != usage:
            return False
        if colors and item.base_colour.strip().lower() not in colors:
            return False
        if article_types and item.article_type not in article_types:
            return False
        return True

    def handle_turn(
        self,
        user_text: str,
        image_bytes: bytes | None,
        store_id: str,
        prefer_newest: bool,
        top_k: int,
    ) -> dict:
        # If no API key, fall back to a non-AI message.
        if not os.getenv("OPENAI_API_KEY"):
            return {
                "answer": "Set `OPENAI_API_KEY` to enable recommendations. Then ask again (with or without an image).",
                "products": [],
            }

        debug: dict = {"queries": [], "image_used": bool(image_bytes)}

        try:
            client = self._ensure_client()
            queries: list[str] = []
            analysis = None
            if image_bytes:
                self._ensure_index()
                analysis = analyze_outfit_image(
                    client, image_bytes=image_bytes, article_types=self.article_types, model=self.cfg.vision_model
                )
                queries.extend([q for q in analysis.get("search_queries", []) if isinstance(q, str)])

            if user_text.strip():
                queries.append(user_text.strip())

            # De-dupe while preserving order
            queries = list(dict.fromkeys([q.strip() for q in queries if q.strip()]))
            debug["queries"] = queries
            if analysis is not None:
                debug["image_analysis"] = analysis

            if not queries:
                return {"answer": "Ask a question or upload an image to get started.", "products": [], "debug": debug}

            analysis_gender = analysis.get("gender") if isinstance(analysis, dict) else None
            analysis_occasion = analysis.get("occasion") if isinstance(analysis, dict) else None
            analysis_colors = analysis.get("colors") if isinstance(analysis, dict) else None
            analysis_article_types = analysis.get("article_types") if isinstance(analysis, dict) else None

            gender = self._infer_gender(user_text, analysis_gender if isinstance(analysis_gender, str) else None)
            usage = self._infer_usage(user_text, analysis_occasion if isinstance(analysis_occasion, str) else None)
            colors = self._infer_colors(user_text, analysis_colors if isinstance(analysis_colors, list) else None)
            article_types = self._infer_article_types(
                analysis_article_types if isinstance(analysis_article_types, list) else None
            )

            debug["filters"] = {
                "gender": gender,
                "usage": usage,
                "colors": sorted(colors) if colors else None,
                "article_types": sorted(article_types) if article_types else None,
            }

            self._ensure_index()

            # Retrieve candidates across queries, preserving order and uniqueness
            seen: set[int] = set()
            ranked_ids: list[int] = []

            # Seed with explicit product IDs if present in the user message
            for pid in self._extract_product_ids(user_text):
                row = self._id_to_row.get(pid)
                if row is not None and row not in seen:
                    seen.add(row)
                    ranked_ids.append(row)
            for q in queries[:6]:
                for i in self._retrieve(q, top_k=top_k * 4, prefer_newest=prefer_newest):
                    if i in seen:
                        continue
                    if not self._matches_filters(
                        row=i, gender=gender, usage=usage, colors=colors, article_types=article_types
                    ):
                        continue
                    seen.add(i)
                    ranked_ids.append(i)
                    if len(ranked_ids) >= top_k:
                        break
                if len(ranked_ids) >= top_k:
                    break

            if len(ranked_ids) < min(3, top_k) and any([gender, usage, colors, article_types]):
                debug["filters_relaxed"] = True
                for q in queries[:6]:
                    for i in self._retrieve(q, top_k=top_k * 4, prefer_newest=prefer_newest):
                        if i in seen:
                            continue
                        seen.add(i)
                        ranked_ids.append(i)
                        if len(ranked_ids) >= top_k:
                            break
                    if len(ranked_ids) >= top_k:
                        break

            products: list[RecommendedProduct] = []
            context_lines: list[str] = []
            index = self._ensure_index()
            for i in ranked_ids:
                item = index.items[i]
                qty = demo_quantity(store_id, item.id)
                availability = "Out of stock" if qty == 0 else f"{qty} in stock"
                products.append(
                    RecommendedProduct(
                        id=item.id,
                        name=item.name,
                        gender=item.gender,
                        article_type=item.article_type,
                        base_colour=item.base_colour,
                        usage=item.usage,
                        season=item.season,
                        year=item.year,
                        image=self._image_path_or_url(item.id),
                        availability=availability,
                    )
                )
                context_lines.append(
                    f"- {item.name} (id={item.id}, gender={item.gender}, articleType={item.article_type}, "
                    f"color={item.base_colour}, usage={item.usage}, season={item.season}, year={item.year}, "
                    f"availability={availability})"
                )

            context = "\n".join(context_lines)
            answer = craft_answer(
                client,
                user_text=user_text,
                context=context,
                store_id=store_id,
                prefer_newest=prefer_newest,
                model=self.cfg.chat_model,
            )

            return {"answer": answer, "products": products, "debug": debug}
        except Exception as e:
            return {
                "answer": f"Sorry — I hit an error while generating recommendations: {type(e).__name__}: {e}",
                "products": [],
                "debug": debug,
            }
