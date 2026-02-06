from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


@dataclass(frozen=True)
class OpenAIConfig:
    chat_model: str
    vision_model: str
    embedding_model: str
    audio_model: str

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        return cls(
            chat_model=os.getenv("RN_CHAT_MODEL", "gpt-4o-mini"),
            vision_model=os.getenv("RN_VISION_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("RN_EMBEDDING_MODEL", "text-embedding-3-large"),
            audio_model=os.getenv("RN_AUDIO_MODEL", "gpt-4o-mini-transcribe"),
        )


def make_client() -> OpenAI:
    return OpenAI()


def text_embedding(client: OpenAI, text: str, model: str) -> list[float]:
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding


def transcribe_audio(
    client: OpenAI,
    audio_bytes: bytes,
    *,
    filename: str | None,
    model: str,
) -> str:
    safe_name = Path(filename or "voice.webm").name
    if "." not in safe_name:
        safe_name = f"{safe_name}.webm"

    audio_buffer = io.BytesIO(audio_bytes)
    audio_buffer.name = safe_name

    resp = client.audio.transcriptions.create(
        model=model,
        file=audio_buffer,
    )
    text = getattr(resp, "text", "") or ""
    return text.strip()


def analyze_outfit_image(
    client: OpenAI,
    image_bytes: bytes,
    article_types: list[str],
    model: str,
) -> dict[str, Any]:
    """Turn an image into structured search intent for retrieval."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "You are a retail fashion assistant for a department store.\n"
        "Given the uploaded outfit photo, extract concise attributes that help search an in-store catalog.\n"
        "Return ONLY valid JSON with keys:\n"
        "- gender (string; 'Men'|'Women'|'Unisex' or 'Unknown')\n"
        "- occasion (string; e.g. 'Formal', 'Casual', 'Ethnic', 'Party', or 'Unknown')\n"
        "- colors (array of strings; dominant colors)\n"
        "- article_types (array of strings; choose from the allowed list)\n"
        "- search_queries (array of 3-6 short queries to retrieve matching/up-to-date items)\n"
        f"Allowed article types: {article_types}\n"
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer", "content": "Return only JSON. No markdown, no extra text."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = completion.choices[0].message.content or "{}"
    return json.loads(raw)


def craft_answer(
    client: OpenAI,
    user_text: str,
    context: str,
    store_id: str,
    prefer_newest: bool,
    model: str,
) -> str:
    """Produce an executive-demo friendly response grounded in retrieved context."""
    prompt = (
        "You are RetailNext's in-store Style & Store-Finder assistant.\n"
        "Goals:\n"
        "1) Help customers find updated styles fast.\n"
        "2) Confirm availability at the selected store when provided.\n"
        "3) Be concise and actionable.\n\n"
        "Rules:\n"
        "- Only recommend products that appear in the provided CONTEXT.\n"
        "- If the user asked about an upcoming event, tailor for that occasion.\n"
        "- If prefer_newest is true, bias toward newer styles (higher year) when reasonable.\n"
        "- End with one short follow-up question.\n\n"
        f"SELECTED_STORE_ID: {store_id}\n"
        f"PREFER_NEWEST: {prefer_newest}\n\n"
        f"USER: {user_text}\n\n"
        f"CONTEXT:\n{context}\n"
    )

    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.4,
    )
    return resp.output_text
