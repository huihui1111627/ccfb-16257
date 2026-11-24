"""Self-contained storage layer for the document store.

No sqlite, no leveldb, no pickle.  The on-disk format is:

    data.log
        magic  b"DCS1\\x00"                              (5 bytes)
        record*  [u32 length][u32 crc32][u8 kind][payload]
                 kind 1 = document put, payload = UTF-8 JSON document
                 kind 2 = document delete, payload = UTF-8 JSON {"_id": ...}
    manifest.json
        {"format": 1, "collection": ..., "indexes": [...], "records": n}

Recovery rules
--------------
The log is append-only.  On open we replay records in order.  A record whose
length or CRC does not validate is treated as a torn tail: replay stops and
the file is truncated to the last good offset.  Because a put of the same
``_id`` is last-write-wins, replaying is idempotent.

Compaction
----------
``compact()`` writes a fresh log containing only live documents (one put per
document, in ``_id`` order), fsyncs it, then atomically renames it over the
old log.  All writes go through a single lock, so compaction is safe in this
single-process server.
"""

from __future__ import annotations

import json
import os
import struct
import threading
import zlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .errors import DuplicateDocumentError, StorageCorruptionError

MAGIC = b"DCS1\x00"
REC_HEADER = struct.Struct(">IIB")
KIND_PUT = 1
KIND_DEL = 2
FORMAT_VERSION = 1

MANIFEST = "manifest.json"
DATA_LOG = "data.log"


class Storage:
    """Append-only document log plus an in-memory document map."""

    def __init__(self, path: str, collection: str = "products",
                 create: bool = True, load_index_defs: bool = True):
        self.path = os.path.abspath(path)
        self.collection = collection
        self._lock = threading.RLock()
        self._docs: Dict[Any, dict] = {}
        self._order: List[Any] = []
        self._records = 0
        self._log_bytes = 0
        self.manifest: dict = {
            "format": FORMAT_VERSION,
            "collection": collection,
            "indexes": [],
            "records": 0,
        }
        if create:
            os.makedirs(self.path, exist_ok=True)
        self._log_path = os.path.join(self.path, DATA_LOG)
        self._manifest_path = os.path.join(self.path, MANIFEST)
        if os.path.exists(self._manifest_path):
            self._read_manifest()
        elif not create:
            raise StorageCorruptionError(f"no manifest at {self.path}")
        else:
            self._write_manifest()
        if os.path.exists(self._log_path):
            self._replay()
        else:
            with open(self._log_path, "wb") as f:
                f.write(MAGIC)
                f.flush()
                os.fsync(f.fileno())
        self._log_bytes = os.path.getsize(self._log_path)

    # -- manifest ----------------------------------------------------------
    def _read_manifest(self) -> None:
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, ValueError) as exc:
            raise StorageCorruptionError(f"cannot read manifest: {exc}") from exc
        if m.get("format") != FORMAT_VERSION:
            raise StorageCorruptionError(
                f"unsupported storage format {m.get('format')!r}")
        self.manifest = m

    def _write_manifest(self) -> None:
        tmp = self._manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._manifest_path)

    def save_manifest(self, **updates: Any) -> None:
        with self._lock:
            self.manifest.update(updates)
            self._write_manifest()

    # -- replay / append ---------------------------------------------------
    def _replay(self) -> None:
        size = os.path.getsize(self._log_path)
        good_end = 0
        with open(self._log_path, "rb") as f:
            magic = f.read(len(MAGIC))
            if magic != MAGIC:
                raise StorageCorruptionError(f"{self._log_path}: bad magic")
            good_end = len(MAGIC)
            while True:
                header = f.read(REC_HEADER.size)
                if len(header) < REC_HEADER.size:
                    break
                length, crc, kind = REC_HEADER.unpack(header)
                payload = f.read(length)
                if len(payload) < length or zlib.crc32(payload) != crc:
                    break
                try:
                    obj = json.loads(payload.decode("utf-8"))
                except ValueError:
                    break
                if kind == KIND_PUT:
                    self._apply_put(obj)
                elif kind == KIND_DEL:
                    self._apply_del(obj["_id"])
                else:  # pragma: no cover - future
                    break
                self._records += 1
                good_end = f.tell()
        if good_end < size:
            with open(self._log_path, "r+b") as f:
                f.truncate(good_end)
        self.manifest["records"] = self._records
        self._end_offset = good_end

    @property
    def end_offset(self) -> int:
        """Byte offset just past the last valid record."""
        return getattr(self, "_end_offset", len(MAGIC))

    def records_after(self, offset: int):
        """Yield (kind, obj) for valid records starting at byte ``offset``."""
        if offset < len(MAGIC):
            return
        with open(self._log_path, "rb") as f:
            f.seek(offset)
            while True:
                header = f.read(REC_HEADER.size)
                if len(header) < REC_HEADER.size:
                    return
                length, crc, kind = REC_HEADER.unpack(header)
                payload = f.read(length)
                if len(payload) < length or zlib.crc32(payload) != crc:
                    return
                try:
                    obj = json.loads(payload.decode("utf-8"))
                except ValueError:
                    return
                yield kind, obj

    def _apply_put(self, doc: dict) -> None:
        doc_id = doc["_id"]
        if doc_id not in self._docs:
            self._order.append(doc_id)
        self._docs[doc_id] = doc

    def _apply_del(self, doc_id: Any) -> None:
        if doc_id in self._docs:
            del self._docs[doc_id]
        # _order keeps a tombstone id; it is filtered lazily and removed at compaction
        self._dead = getattr(self, "_dead", set())
        self._dead.add(doc_id)

    def append(self, kind: int, obj: dict) -> int:
        payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
        header = REC_HEADER.pack(len(payload), zlib.crc32(payload), kind)
        with self._lock:
            with open(self._log_path, "ab") as f:
                f.write(header)
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            self._records += 1
            self._log_bytes += len(header) + len(payload)
            self._end_offset = self._log_bytes
        return len(header) + len(payload)

    def append_many(self, records: Iterable[Tuple[int, dict]]) -> None:
        """Bulk append without fsync per record (one fsync at the end)."""
        buf = bytearray()
        n = 0
        for kind, obj in records:
            payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8")
            buf += REC_HEADER.pack(len(payload), zlib.crc32(payload), kind)
            buf += payload
            n += 1
        with self._lock:
            with open(self._log_path, "ab") as f:
                f.write(buf)
                f.flush()
                os.fsync(f.fileno())
            self._records += n
            self._log_bytes += len(buf)
            self._end_offset = self._log_bytes

    # -- document access ---------------------------------------------------
    def get(self, doc_id: Any) -> Optional[dict]:
        return self._docs.get(doc_id)

    def contains(self, doc_id: Any) -> bool:
        return doc_id in self._docs

    def __len__(self) -> int:
        return len(self._docs)

    def count(self) -> int:
        return len(self._docs)

    def __contains__(self, doc_id: Any) -> bool:
        return doc_id in self._docs

    def ids(self) -> List[Any]:
        dead = getattr(self, "_dead", set())
        if not dead:
            return list(self._order)
        return [i for i in self._order if i not in dead]

    def all_docs(self) -> Iterable[dict]:
        dead = getattr(self, "_dead", set())
        if not dead:
            return list(self._docs.values())
        return [d for i, d in self._docs.items() if i not in dead]

    def scan(self) -> Iterable[dict]:
        """Full-collection scan in insertion order (tombstones skipped)."""
        dead = getattr(self, "_dead", set())
        docs = self._docs
        for doc_id in self._order:
            if doc_id in dead:
                continue
            d = docs.get(doc_id)
            if d is not None:
                yield d

    def put(self, doc: dict, append: bool = True) -> None:
        with self._lock:
            if append:
                self.append(KIND_PUT, doc)
            self._apply_put(doc)
            dead = getattr(self, "_dead", None)
            if dead and doc["_id"] in dead:
                dead.discard(doc["_id"])

    def put_many(self, docs: List[dict]) -> None:
        with self._lock:
            self.append_many((KIND_PUT, d) for d in docs)
            for d in docs:
                self._apply_put(d)

    def delete(self, doc_id: Any) -> bool:
        with self._lock:
            if doc_id not in self._docs:
                return False
            self.append(KIND_DEL, {"_id": doc_id})
            self._apply_del(doc_id)
            return True

    def bulk_load(self, docs: Iterable[dict], fsync: bool = True) -> int:
        """Write a fresh log from ``docs``, replacing current contents."""
        docs = list(docs)
        ids = set()
        for d in docs:
            if "_id" not in d:
                raise DuplicateDocumentError("every document needs an _id")
            if d["_id"] in ids:
                raise DuplicateDocumentError(f"duplicate _id {d['_id']!r} in bulk load")
            ids.add(d["_id"])
        tmp = self._log_path + ".bulk"
        n = 0
        with open(tmp, "wb") as f:
            f.write(MAGIC)
            for d in docs:
                payload = json.dumps(d, ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")
                f.write(REC_HEADER.pack(len(payload), zlib.crc32(payload), KIND_PUT))
                f.write(payload)
                n += 1
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, self._log_path)
        self._docs = {d["_id"]: d for d in docs}
        self._order = [d["_id"] for d in docs]
        self._dead = set()
        self._records = n
        self._log_bytes = os.path.getsize(self._log_path)
        self._end_offset = self._log_bytes
        self.manifest["records"] = n
        self._write_manifest()
        return n

    def compact(self) -> int:
        with self._lock:
            docs = list(self.scan())
        return self.bulk_load(docs)

    def stats(self) -> dict:
        return {
            "path": self.path,
            "documents": len(self),
            "log_bytes": self._log_bytes,
            "records": self._records,
            "dead_ids": len(getattr(self, "_dead", ())),
        }


class IndexSnapshotStore:
    """Persist index snapshots beside the log, keyed by a log offset.

    A snapshot records the byte offset of the last log record it accounts for.
    On startup, the index is loaded from the snapshot and then the records
    written *after* that offset are replayed into it.  Anything the snapshot
    cannot account for (missing file, bad CRC, offset past the log) causes a
    full rebuild, so the snapshot is a pure performance optimisation and never
    affects correctness.
    """

    def __init__(self, dir_path: str):
        self.dir = os.path.join(dir_path, "idx")
        os.makedirs(self.dir, exist_ok=True)

    def path(self, name: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        return os.path.join(self.dir, safe + ".idx")
