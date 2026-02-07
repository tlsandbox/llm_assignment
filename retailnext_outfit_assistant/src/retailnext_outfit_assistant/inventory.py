from __future__ import annotations

import hashlib
from typing import TypedDict


class Store(TypedDict):
    id: str
    name: str


DEFAULT_STORES: list[Store] = [
    {"id": "RNTX-NYC-001", "name": "RetailNext • NYC (5th Ave)"},
    {"id": "RNTX-SEA-002", "name": "RetailNext • Seattle (Downtown)"},
    {"id": "RNTX-ATX-003", "name": "RetailNext • Austin (Domain)"},
]


def demo_quantity(store_id: str, product_id: int) -> int:
    payload = f"{store_id}:{product_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # 0–8 units, deterministic
    return digest[0] % 9

