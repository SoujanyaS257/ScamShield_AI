"""MongoDB history, with a local JSON file when MongoDB is not configured or reachable."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import HISTORY_PATH, MONGODB_COLLECTION, MONGODB_DB, MONGODB_URI


class DetectionStore:
    def __init__(self, mode: str, warning: str | None = None, path: Path | None = None, collection=None) -> None:
        self.mode = mode
        self.warning = warning
        self.path = path
        self._collection = collection

    def add(self, record: dict) -> None:
        if self._collection is not None:
            self._collection.insert_one(dict(record))
            return
        rows = self._read()
        rows.append(record)
        self._write(rows)

    def list_recent(self, limit: int = 300) -> list[dict]:
        if self._collection is not None:
            cursor = self._collection.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
            return list(cursor)
        rows = self._read()
        rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        return rows[:limit]

    def clear(self) -> None:
        if self._collection is not None:
            self._collection.delete_many({})
            return
        self._write([])

    def _read(self) -> list[dict]:
        if self.path is None or not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []

    def _write(self, rows: list[dict]) -> None:
        if self.path is None:
            raise RuntimeError("JSON history path is not set.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        temporary.replace(self.path)


def open_store(
    mongo_uri: str | None = None,
    fallback_path: Path | None = None,
    database: str | None = None,
    collection_name: str | None = None,
) -> DetectionStore:
    path = fallback_path or HISTORY_PATH
    uri = MONGODB_URI if mongo_uri is None else mongo_uri.strip()
    if not uri:
        return DetectionStore(
            mode="local-json",
            warning="MongoDB URI is empty. History is stored in a local JSON file.",
            path=path,
        )
    try:
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError
    except ImportError as exc:
        return DetectionStore(
            mode="local-json",
            warning=f"pymongo is not installed ({exc}). History is stored in a local JSON file.",
            path=path,
        )
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        collection = client[database or MONGODB_DB][collection_name or MONGODB_COLLECTION]
        return DetectionStore(mode="mongodb", warning=None, path=path, collection=collection)
    except Exception as exc:  # connection errors subclass PyMongoError, plus config mistakes
        hint = exc.__class__.__name__
        try:
            from pymongo.errors import PyMongoError

            if not isinstance(exc, PyMongoError) and not isinstance(exc, OSError):
                hint = str(exc)
        except ImportError:
            hint = str(exc)
        return DetectionStore(
            mode="local-json",
            warning=f"MongoDB is not reachable ({hint}). History is stored in a local JSON file.",
            path=path,
        )
