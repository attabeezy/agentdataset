"""Simple file-based cache for fetched documents and extraction results (§10.8).

Keys are hashed; values are JSON. Cache invalidation is explicit: a changed
content hash, model, prompt version, or schema version produces a distinct key.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


class ContentCache:
    def __init__(self, base_dir: str = ".agentdataset_cache/cache"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: Any) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(str(p).encode("utf-8", errors="replace"))
            h.update(b"|")
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return self.base / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        p = self._path(key)
        p.write_text(json.dumps(value, default=str), encoding="utf-8")

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def invalidate(self, key: str) -> None:
        p = self._path(key)
        try:
            p.unlink()
        except FileNotFoundError:
            pass