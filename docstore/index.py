"""Multikey indexes over nested arrays of objects.

An ``Index`` owns one definition and maps each *index key* (a tuple with one
encoded component per indexed clause) to the documents holding it.  Keys stay
sorted so range scans and index-driven sorts are cheap.

Multikey rules -- each exercised in ``tests/test_multikey_index.py``:

1. **One entry per array element.**  ``specs.color`` over a 7-element array
   produces 7 index entries for that document.
2. **Empty arrays are real values.**  A leaf array that is empty gets one entry
   with key ``[]``; it is *present*, so ``$exists: true`` and ``$eq: []`` are
   answerable from the index.  An empty array in the *middle* of a path (e.g.
   ``specs`` empty while indexing ``specs.color``) yields no colour values, so
   the path is treated as missing -- consistent with the query matcher.
3. **Missing paths** yield the sentinel ``<missing>`` key unless the index is
   ``sparse``.  Missing and an explicit ``null`` are different keys.
4. **Parallel arrays are rejected.**  A compound index whose clauses fan out at
   two *different* arrays would need the cartesian product of the element sets,
   so a 10x10 pair yields 100 entries per document.  Declaring ``[]`` on two
   clauses with different array paths raises ``PARALLEL_ARRAYS`` at create time;
   a document that crosses two arrays raises it at write time.  The supported
   alternatives are (a) two single-field multikey indexes plus intersection, or
   (b) an explicitly *element-aligned* index, ``align="specs"``, for clauses
   that read fields of the *same* array element.  Alignment yields one entry
   per element, so the entry count equals the element count -- it cannot
   explode.  Without ``align`` even same-array compound indexes are rejected,
   so the safe behaviour is the default and the fast behaviour is opt-in.
5. **A unique index may not index an array.**  A document whose indexed path is
   an array is rejected: per-element uniqueness would mean "only one product in
   the catalogue may be red", which is never what is meant.
6. **Incremental maintenance.**  An update computes the new key set and diffs
   it against the stored old set; only added/removed keys touch the postings.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import values as V
from .errors import (
    DocStoreError,
    DuplicateKeyError,
    IndexError_,
    MultikeyUniqueError,
    ParallelArraysError,
)
from .paths import Clause, parse_clause

Key = Tuple[tuple, ...]
MISSING = V.MISSING_KEY

_EMPTY_ELEM = object()   # identity stand-in for "the empty array" element


# --------------------------------------------------------------------------
# Definition
# --------------------------------------------------------------------------

@dataclass
class IndexDef:
    name: str
    clauses: List[Clause]
    unique: bool = False
    sparse: bool = False
    align: Optional[str] = None
    built_at: Optional[str] = None

    @property
    def fields(self) -> List[str]:
        return [c.field for c in self.clauses]

    @property
    def directions(self) -> List[int]:
        return [-1 if c.descending else 1 for c in self.clauses]

    @property
    def multikey_declared(self) -> bool:
        return self.align is not None or any(c.multikey_declared for c in self.clauses)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "fields": [c.to_dict() for c in self.clauses],
            "unique": self.unique,
            "sparse": self.sparse,
            "align": self.align,
            "built_at": self.built_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IndexDef":
        clauses = []
        for f in d["fields"]:
            text = f["field"] + ("[]" if f.get("multikey") else "")
            if f.get("direction", 1) < 0:
                text = "-" + text
            clauses.append(parse_clause(text))
        return cls(name=d["name"], clauses=clauses, unique=d.get("unique", False),
                   sparse=d.get("sparse", False), align=d.get("align"),
                   built_at=d.get("built_at"))

    def validate_creation(self) -> None:
        if not self.clauses:
            raise IndexError_("an index needs at least one field")
        if len({c.field for c in self.clauses}) != len(self.clauses):
            raise IndexError_(f"index {self.name!r} repeats a field")
        fanout_paths: Set[str] = set()
        for c in self.clauses:
            fp = c.fanout_paths
            if len(fp) > 1:
                raise IndexError_(
                    f"index field {c.field!r} fans out at more than one array level "
                    f"({', '.join(fp)}); nested multikey fan-out is not supported -- "
                    f"index the inner field separately")
            fanout_paths.update(fp)
        if len(fanout_paths) > 1:
            paths = sorted(fanout_paths)
            raise ParallelArraysError(self.name, paths[0], paths[1])
        if self.align is not None:
            if not fanout_paths:
                raise IndexError_(
                    f"index {self.name!r}: align={self.align!r} was given but no field "
                    f"declares '[]'; write e.g. '{self.align}[].color'")
            if fanout_paths != {self.align}:
                raise IndexError_(
                    f"index {self.name!r}: align={self.align!r} but fields fan out at "
                    f"{sorted(fanout_paths)}; every [] must name the aligned array")
            if self.unique:
                raise IndexError_(
                    f"index {self.name!r}: a unique index may not be multikey")
        else:
            declared = [c.field for c in self.clauses if c.multikey_declared]
            if len(declared) > 1:
                raise ParallelArraysError(self.name, declared[0], declared[1])
            if declared and self.unique:
                raise IndexError_(
                    f"index {self.name!r}: a unique index may not be multikey "
                    f"({declared[0]!r} declares [])")


# --------------------------------------------------------------------------
# Element-aware extraction
# --------------------------------------------------------------------------

# A hit records one scalar reachable through a clause path together with the
# identity of every array *element* crossed to reach it.  ``sig`` maps the
# dotted array path -> id(element).  Two clauses whose hits share the same
# (path, element id) pair refer to the same array element and can be aligned.
@dataclass(frozen=True)
class _Hit:
    value: Any
    sig: Tuple[Tuple[str, int], ...]
    is_empty_array: bool = False


def _walk(node: Any, steps: Sequence[Any], i: int, prefix: str,
          sig: Tuple[Tuple[str, int], ...], out: List[_Hit]) -> None:
    if i >= len(steps):
        if isinstance(node, list):
            path = prefix
            if not node:
                out.append(_Hit(V.EMPTY_ARRAY_KEY, sig + ((path, id(_EMPTY_ELEM)),), True))
                return
            for el in node:
                out.append(_Hit(el, sig + ((path, id(el)),)))
            return
        out.append(_Hit(node, sig))
        return
    if isinstance(node, list):
        path = prefix
        if not node:
            return  # empty array in the middle: no values below it
        for el in node:
            _walk(el, steps, i, prefix, sig + ((path, id(el)),), out)
        return
    if not isinstance(node, dict):
        return
    step = steps[i]
    if step.name not in node:
        return
    child = f"{prefix}.{step.name}" if prefix else step.name
    _walk(node[step.name], steps, i + 1, child, sig, out)


def _clause_hits(doc: dict, clause: Clause) -> List[_Hit]:
    out: List[_Hit] = []
    _walk(doc, clause.steps, 0, "", (), out)
    return out


@dataclass
class Extracted:
    keys: Set[Key]
    multikey: bool
    detail: List[dict] = field(default_factory=list)
    sparse_skipped: bool = False
    aligned_at: Optional[str] = None


def extract_keys(doc: dict, idef: IndexDef) -> Extracted:
    hits_by_clause = [_clause_hits(doc, c) for c in idef.clauses]
    detail = []
    for c, hits in zip(idef.clauses, hits_by_clause):
        detail.append({
            "field": c.field,
            "values": [V.describe_value(h.value) for h in hits],
            "array_roots": sorted({p for h in hits for p, _ in h.sig}),
        })

    array_paths: Set[str] = {p for hits in hits_by_clause for h in hits for p, _ in h.sig}

    if idef.sparse and all(not hits for hits in hits_by_clause):
        return Extracted(set(), False, detail, sparse_skipped=True)

    if len(array_paths) > 1:
        paths = sorted(array_paths)
        raise ParallelArraysError(idef.name, paths[0], paths[1])

    if not array_paths:
        return Extracted(_extract_plain(hits_by_clause, idef), False, detail)

    align_path = next(iter(array_paths))
    crossing = sum(1 for hits in hits_by_clause if any(h.sig for h in hits))
    if crossing > 1 and idef.align != align_path:
        a = [c.field for c, hits in zip(idef.clauses, hits_by_clause)
             if any(h.sig for h in hits)]
        raise ParallelArraysError(idef.name, a[0], a[1])
    if idef.align is not None and idef.align != align_path:
        raise IndexError_(
            f"index {idef.name!r} is aligned at {idef.align!r} but this document "
            f"crosses {align_path!r}")
    keys = _extract_aligned(hits_by_clause, idef, align_path)
    return Extracted(keys, True, detail, aligned_at=align_path)


def _extract_plain(hits_by_clause: List[List[_Hit]], idef: IndexDef) -> Set[Key]:
    comps: List[tuple] = []
    for c, hits in zip(idef.clauses, hits_by_clause):
        if not hits:
            comps.append(MISSING)
        elif len(hits) == 1:
            comps.append(_encode(hits[0]))
        else:  # pragma: no cover - no arrays means at most one hit
            raise DocStoreError(f"internal: multiple hits without an array at {c.field}")
    return {tuple(comps)}


def _extract_aligned(hits_by_clause: List[List[_Hit]], idef: IndexDef,
                     align_path: str) -> Set[Key]:
    # group each clause's hits by the element identity at align_path
    groups: Dict[int, Dict[int, List[_Hit]]] = {}
    doc_level: Dict[int, List[_Hit]] = {}
    for ci, hits in enumerate(hits_by_clause):
        for h in hits:
            eid = None
            for p, e in h.sig:
                if p == align_path:
                    eid = e
                    break
            if eid is None:
                doc_level.setdefault(ci, []).append(h)
            else:
                groups.setdefault(eid, {}).setdefault(ci, []).append(h)
    if not groups:
        groups = {id(_EMPTY_ELEM): {}}

    keys: Set[Key] = set()
    for eid, per_clause in groups.items():
        comps: List[tuple] = []
        for ci, c in enumerate(idef.clauses):
            hs = per_clause.get(ci) or doc_level.get(ci) or []
            if not hs:
                comps.append(MISSING)
                continue
            if len(hs) > 1:
                raise ParallelArraysError(
                    idef.name, align_path, f"{align_path}[].{c.field}")
            comps.append(_encode(hs[0]))
        keys.add(tuple(comps))
    return keys


def _encode(hit: _Hit) -> tuple:
    if hit.is_empty_array:
        return V.EMPTY_ARRAY_KEY
    return V.encode_key(hit.value)


# --------------------------------------------------------------------------
# Sorted key store (lazy merge of inserts)
# --------------------------------------------------------------------------

class KeyStore:
    __slots__ = ("_base", "_delta")

    def __init__(self) -> None:
        self._base: List[tuple] = []
        self._delta: Set[tuple] = set()

    def add(self, key: tuple) -> None:
        self._delta.add(key)
        if len(self._delta) > max(2048, len(self._base) // 4):
            self.rebuild()

    def discard(self, key: tuple) -> None:
        self._delta.discard(key)

    def rebuild(self) -> None:
        if self._delta:
            self._base = sorted(set(self._base) | self._delta)
            self._delta.clear()

    def sorted_keys(self) -> List[tuple]:
        self.rebuild()
        return self._base

    def range_keys(self, lo: Optional[tuple], hi: Optional[tuple],
                   lo_inclusive: bool, hi_inclusive: bool) -> List[tuple]:
        keys = self.sorted_keys()
        start = 0 if lo is None else (bisect_left(keys, lo) if lo_inclusive
                                      else bisect_right(keys, lo))
        end = len(keys) if hi is None else (bisect_right(keys, hi) if hi_inclusive
                                            else bisect_left(keys, hi))
        return keys[start:end] if start < end else []

    def first(self) -> Optional[tuple]:
        keys = self.sorted_keys()
        return keys[0] if keys else None

    def last(self) -> Optional[tuple]:
        keys = self.sorted_keys()
        return keys[-1] if keys else None


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------

class Index:
    def __init__(self, idef: IndexDef):
        self.defn = idef
        self.name = idef.name
        self.postings: Dict[Key, Dict[Any, None]] = {}
        self.doc_keys: Dict[Any, Set[Key]] = {}
        self.keys = KeyStore()
        self.entry_count = 0
        self.multikey_docs = 0
        self.entries_added = 0
        self.entries_removed = 0

    def _insert(self, key: Key, doc_id: Any) -> None:
        postings = self.postings.get(key)
        if postings is None:
            postings = {}
            self.postings[key] = postings
            self.keys.add(key)
        postings[doc_id] = None

    def _remove(self, key: Key, doc_id: Any) -> None:
        postings = self.postings.get(key)
        if postings is None:
            return
        postings.pop(doc_id, None)
        if not postings:
            self.keys.discard(key)

    def _check_unique(self, doc_id: Any, keys: Set[Key]) -> None:
        if not self.defn.unique:
            return
        if len(keys) > 1:
            raise MultikeyUniqueError(self.name, doc_id, len(keys))
        if not keys:
            return
        key = next(iter(keys))
        postings = self.postings.get(key)
        if postings:
            other = next(iter(postings))
            if other != doc_id:
                raise DuplicateKeyError(self.name, _show_key(key), other, doc_id)

    # -- writes ------------------------------------------------------------
    def add_document(self, doc: dict) -> None:
        doc_id = doc["_id"]
        ex = extract_keys(doc, self.defn)
        self._check_unique(doc_id, ex.keys)
        for key in ex.keys:
            self._insert(key, doc_id)
        self.doc_keys[doc_id] = ex.keys
        self.entry_count += len(ex.keys)
        self.entries_added += len(ex.keys)
        if ex.multikey:
            self.multikey_docs += 1

    def remove_document(self, doc_id: Any) -> None:
        keys = self.doc_keys.pop(doc_id, None)
        if not keys:
            return
        for key in keys:
            self._remove(key, doc_id)
        self.entry_count -= len(keys)
        self.entries_removed += len(keys)

    def update_document(self, old_doc: Optional[dict], new_doc: dict) -> Tuple[int, int]:
        doc_id = new_doc["_id"]
        if old_doc is None:
            self.add_document(new_doc)
            return (len(self.doc_keys.get(doc_id, ())), 0)
        old = self.doc_keys.get(doc_id, set())
        ex = extract_keys(new_doc, self.defn)
        self._check_unique(doc_id, ex.keys)
        added = ex.keys - old
        removed = old - ex.keys
        for key in removed:
            self._remove(key, doc_id)
        for key in added:
            self._insert(key, doc_id)
        self.doc_keys[doc_id] = ex.keys
        self.entry_count += len(added) - len(removed)
        self.entries_added += len(added)
        self.entries_removed += len(removed)
        return (len(added), len(removed))

    # -- reads -------------------------------------------------------------
    def lookup_equal(self, key: Key) -> List[Any]:
        postings = self.postings.get(key)
        return list(postings) if postings else []

    def scan_range(self, lo: Optional[Key], hi: Optional[Key],
                   lo_inclusive: bool = True, hi_inclusive: bool = True
                   ) -> Tuple[List[Any], int, int]:
        """Return (doc ids, index keys examined, posting rows read)."""
        keys = self.keys.range_keys(lo, hi, lo_inclusive, hi_inclusive)
        out: List[Any] = []
        seen: Set[Any] = set()
        postings_read = 0
        for k in keys:
            postings = self.postings.get(k)
            if not postings:
                continue
            postings_read += len(postings)
            for doc_id in postings:
                if doc_id not in seen:
                    seen.add(doc_id)
                    out.append(doc_id)
        return out, len(keys), postings_read

    def scan_all(self) -> Tuple[List[Any], int, int]:
        return self.scan_range(None, None)

    def entries_for_doc(self, doc_id: Any) -> List[dict]:
        keys = self.doc_keys.get(doc_id, set())
        out = []
        for key in sorted(keys):
            out.append({
                "key": [V.describe_key(k) for k in key],
                "raw_key": [V.key_to_json(k) for k in key],
                "documents_sharing_key": len(self.postings.get(key, {})),
            })
        return out

    def stats(self) -> dict:
        return {
            "name": self.name,
            "fields": self.defn.fields,
            "directions": self.defn.directions,
            "unique": self.defn.unique,
            "sparse": self.defn.sparse,
            "align": self.defn.align,
            "multikey": self.defn.multikey_declared,
            "documents_indexed": len(self.doc_keys),
            "index_entries": sum(len(p) for p in self.postings.values()),
            "distinct_keys": sum(1 for p in self.postings.values() if p),
            "multikey_documents": self.multikey_docs,
            "entries_added": self.entries_added,
            "entries_removed": self.entries_removed,
            "bytes_estimate": _estimate_bytes(self.postings),
        }

    def bulk_build(self, docs: Iterable[dict]) -> int:
        self.postings.clear()
        self.doc_keys.clear()
        self.keys = KeyStore()
        self.entry_count = 0
        self.multikey_docs = 0
        new_keys: Set[tuple] = set()
        for doc in docs:
            ex = extract_keys(doc, self.defn)
            if not ex.keys:
                continue
            if self.defn.unique and len(ex.keys) > 1:
                raise MultikeyUniqueError(self.name, doc["_id"], len(ex.keys))
            for key in ex.keys:
                postings = self.postings.get(key)
                if postings is None:
                    postings = {}
                    self.postings[key] = postings
                    new_keys.add(key)
                postings[doc["_id"]] = None
            self.doc_keys[doc["_id"]] = ex.keys
            self.entry_count += len(ex.keys)
            if ex.multikey:
                self.multikey_docs += 1
        if self.defn.unique:
            for key, postings in self.postings.items():
                if len(postings) > 1:
                    ids = list(postings)
                    raise DuplicateKeyError(self.name, _show_key(key), ids[0], ids[1])
        self.keys._base = sorted(new_keys)
        self.keys._delta = set()
        return self.entry_count

    # -- persistence -------------------------------------------------------
    def save(self, path: str, log_offset: int = 0) -> int:
        """Write postings to ``path`` so a restart need not rebuild.

        Format (all big-endian):
            magic b"DCI1\\0",
            u64 log_offset (last log record this snapshot accounts for),
            u32 n_docs, u32 n_keys, u32 n_entries, u32 multikey_docs,
            then repeated: u32 klen, key JSON, u32 ilen, ids JSON
            terminated by a u32 CRC32 of the body.
        """
        import json as _json
        import struct as _struct
        import zlib as _zlib
        body = bytearray()
        body += _struct.pack(">QIIII", log_offset, len(self.doc_keys),
                             sum(1 for p in self.postings.values() if p),
                             self.entry_count, self.multikey_docs)
        for key in self.keys.sorted_keys():
            postings = self.postings.get(key)
            if not postings:
                continue
            payload = _json.dumps([V.key_to_json(k) for k in key],
                                  separators=(",", ":")).encode("utf-8")
            ids = _json.dumps(list(postings), ensure_ascii=False,
                              separators=(",", ":")).encode("utf-8")
            body += _struct.pack(">II", len(payload), len(ids))
            body += payload
            body += ids
        crc = _zlib.crc32(bytes(body))
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(b"DCI1\x00")
            f.write(body)
            f.write(_struct.pack(">I", crc))
        import os as _os
        _os.replace(tmp, path)
        return len(body)

    def load(self, path: str) -> Optional[int]:
        """Restore postings from ``save`` output.

        Returns the snapshot's log offset on success, ``None`` if unusable.
        """
        import json as _json
        import struct as _struct
        import zlib as _zlib
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return None
        if len(data) < 9 or data[:5] != b"DCI1\x00":
            return None
        body = data[5:-4]
        (crc,) = _struct.unpack(">I", data[-4:])
        if _zlib.crc32(body) != crc:
            return None
        try:
            log_offset, n_docs, n_keys, n_entries, n_multikey = _struct.unpack(
                ">QIIII", body[:24])
            pos = 24
            self.postings.clear()
            self.doc_keys.clear()
            self.keys = KeyStore()
            self.entry_count = 0
            self.multikey_docs = 0
            new_keys: List[tuple] = []
            doc_keys: Dict[Any, Set[Key]] = {}
            for _ in range(n_keys):
                (plen, ilen) = _struct.unpack(">II", body[pos:pos + 8])
                pos += 8
                key = tuple(V.key_from_json(x)
                            for x in _json.loads(body[pos:pos + plen]))
                pos += plen
                ids = _json.loads(body[pos:pos + ilen])
                pos += ilen
                self.postings[key] = {d: None for d in ids}
                new_keys.append(key)
                for d in ids:
                    doc_keys.setdefault(d, set()).add(key)
            self.doc_keys = doc_keys
            self.keys._base = sorted(new_keys)
            self.keys._delta = set()
            self.entry_count = n_entries
            self.multikey_docs = n_multikey
            return log_offset
        except (ValueError, _struct.error, KeyError):
            return None


def _show_key(key: Key) -> Any:
    return V.describe_key(key[0]) if len(key) == 1 else [V.describe_key(k) for k in key]


def _estimate_bytes(postings: Dict[Key, Dict[Any, None]]) -> int:
    # rough: each key tuple ~ 64 bytes + each posting entry 16 bytes
    n_entries = sum(len(p) for p in postings.values())
    return len(postings) * 64 + n_entries * 16
