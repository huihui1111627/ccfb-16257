"""Collection: documents, indexes, writes and queries in one place."""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import values as V
from .errors import (
    DocStoreError,
    DuplicateDocumentError,
    IndexError_,
    QueryError,
)
from .index import Index, IndexDef
from .paths import (
    apply_positional,
    parse_positional_path,
    set_path,
    unset_path,
)
from .planner import (
    Plan,
    build_plan,
    covering_report,
    execute,
    gather_predicates,
)
from .query import (
    array_semantics,
    explain_semantics,
    node_to_dict,
    parse_query,
)
from .storage import Storage


class Collection:
    def __init__(self, path: str, name: str = "products", create: bool = True):
        self.name = name
        self.storage = Storage(path, collection=name, create=create)
        self.indexes: Dict[str, Index] = {}
        from .storage import IndexSnapshotStore
        self._snapshots = IndexSnapshotStore(path)
        self._load_indexes()

    # ------------------------------------------------------------------
    # index management
    # ------------------------------------------------------------------
    def _load_indexes(self) -> None:
        """Load indexes from snapshots, then replay newer log records.

        A snapshot stores the log offset it accounts for.  When it loads, only
        records after that offset need applying; the index already knows each
        document's old keys, and ``Index.update_document`` diffs against those,
        so replaying a document means syncing it to its final stored state.
        A missing/corrupt/stale-by-offset snapshot triggers a full rebuild, so
        the snapshot only ever affects startup speed, never correctness.
        """
        from .storage import KIND_PUT, KIND_DEL
        entries = self.storage.manifest.get("indexes", [])
        if not entries:
            return
        end = self.storage.end_offset
        offsets: Dict[str, int] = {}
        to_rebuild: List[Index] = []
        for d in entries:
            idef = IndexDef.from_dict(d)
            idx = Index(idef)
            off = idx.load(self._snapshots.path(idef.name)) if idef.built_at else None
            if off is None or off > end:
                to_rebuild.append(idx)
            else:
                offsets[idef.name] = off
            self.indexes[idef.name] = idx

        if to_rebuild:
            docs = list(self.storage.scan())
            for idx in to_rebuild:
                idx.bulk_build(docs)
            # everything rebuilt is current as of ``end``
            for idx in to_rebuild:
                offsets[idx.name] = end

        min_off = min(offsets.values()) if offsets else end
        if min_off >= end:
            return
        changed: Set[Any] = set()
        for kind, obj in self.storage.records_after(min_off):
            changed.add(obj["_id"])
        for doc_id in changed:
            if doc_id in self.storage:
                final = self.storage.get(doc_id)
                for idx in self.indexes.values():
                    if idx.doc_keys.get(doc_id) or offsets[idx.name] < end:
                        idx.update_document({"_id": doc_id}, final)
            else:
                for idx in self.indexes.values():
                    idx.remove_document(doc_id)

    def _persist_index(self, idx: Index) -> None:
        """Save one index snapshot at the current log end offset."""
        idx.save(self._snapshots.path(idx.name), self.storage.end_offset)

    def snapshot_all(self) -> int:
        """Persist every index at the current log end offset."""
        off = self.storage.end_offset
        for idx in self.indexes.values():
            idx.save(self._snapshots.path(idx.name), off)
        return len(self.indexes)

    def create_index(self, fields: Sequence[str], name: Optional[str] = None,
                     unique: bool = False, sparse: bool = False,
                     align: Optional[str] = None) -> dict:
        if isinstance(fields, str):
            fields = [fields]
        from .paths import parse_clause, Clause
        clauses = [parse_clause(f) for f in fields]
        if name is None:
            name = "_".join(
                ("-" if c.descending else "") +
                (c.field + ("_mk" if c.multikey_declared else ""))
                for c in clauses
            ) + ("_unique" if unique else "")
        idef = IndexDef(name=name, clauses=clauses, unique=unique, sparse=sparse,
                        align=align)
        if name in self.indexes:
            raise IndexError_(f"index {name!r} already exists")
        idef.validate_creation()
        started = time.perf_counter()
        idx = Index(idef)
        try:
            entries = idx.bulk_build(self.storage.scan())
        except DocStoreError:
            raise
        idef.built_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.indexes[name] = idx
        self.storage.save_manifest(
            indexes=[i.defn.to_dict() for i in self.indexes.values()],
            records=self.storage.manifest.get("records", 0))
        self._persist_index(idx)
        elapsed = (time.perf_counter() - started) * 1000.0
        return {
            "index": name,
            "entries": entries,
            "build_ms": round(elapsed, 2),
            "stats": idx.stats(),
        }

    def drop_index(self, name: str) -> bool:
        if name not in self.indexes:
            return False
        del self.indexes[name]
        self.storage.save_manifest(
            indexes=[i.defn.to_dict() for i in self.indexes.values()])
        return True

    def index_stats(self) -> List[dict]:
        return [idx.stats() for idx in self.indexes.values()]

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def _validate_doc(self, doc: Any) -> None:
        if not isinstance(doc, dict):
            raise DocStoreError("document must be an object")
        if "_id" not in doc:
            raise DocStoreError("document needs an _id")
        doc_id = doc["_id"]
        if isinstance(doc_id, (dict, list)) or doc_id is None:
            raise DocStoreError("_id must be a scalar")
        V.ensure_document_value(doc, "document")

    def _sync_doc(self, old: Optional[dict], new: dict) -> Dict[str, Tuple[int, int]]:
        """Apply an index-visible change to every index.  Returns per-index deltas.

        Index exceptions propagate; the caller must not have written storage yet,
        so a rejection leaves the collection unchanged.
        """
        deltas: Dict[str, Tuple[int, int]] = {}
        applied: List[str] = []
        try:
            for idx in self.indexes.values():
                deltas[idx.name] = idx.update_document(old, new)
                applied.append(idx.name)
        except DocStoreError:
            # roll back indexes already touched
            for nm in applied:
                idx = self.indexes[nm]
                if old is None:
                    idx.remove_document(new["_id"])
                else:
                    idx.update_document(new, old)
            raise
        return deltas

    def insert(self, doc: dict) -> dict:
        self._validate_doc(doc)
        doc = copy.deepcopy(doc)
        doc_id = doc["_id"]
        if self.storage.contains(doc_id):
            raise DuplicateDocumentError(f"_id {doc_id!r} already exists")
        deltas = self._sync_doc(None, doc)
        self.storage.put(doc)
        return {"_id": doc_id, "index_deltas": deltas}

    def insert_many(self, docs: Iterable[dict]) -> int:
        n = 0
        for d in docs:
            self.insert(d)
            n += 1
        return n

    def replace(self, doc_id: Any, doc: dict) -> dict:
        self._validate_doc(doc)
        if doc["_id"] != doc_id:
            raise DocStoreError("_id in replacement must match the target")
        old = self.storage.get(doc_id)
        if old is None:
            raise DocStoreError(f"document {doc_id!r} not found")
        new = copy.deepcopy(doc)
        deltas = self._sync_doc(old, new)
        self.storage.put(new)
        return {"_id": doc_id, "index_deltas": deltas}

    def update(self, doc_id: Any, changes: Dict[str, Any]) -> dict:
        """Update by set/unset paths, including positional ``specs[2].stock``."""
        old = self.storage.get(doc_id)
        if old is None:
            raise DocStoreError(f"document {doc_id!r} not found")
        new = copy.deepcopy(old)
        applied = []
        for path, value in changes.items():
            if path.startswith("$"):
                raise QueryError(f"update operator {path!r} is not supported; give "
                                 f"a path -> value map")
            ppath = parse_positional_path(path)
            if ppath.is_positional:
                changed, reason = apply_positional(new, ppath, "set", value)
                if not changed and not reason.startswith("no-op"):
                    raise DocStoreError(reason)
            else:
                changed, reason = set_path(new, path, value)
                if not changed and reason.startswith("path") and "does not exist" in reason:
                    changed, reason = _force_set(new, path, value)
                elif not changed and not reason.startswith("no-op"):
                    raise DocStoreError(reason)
            applied.append({"path": path, "changed": changed, "reason": reason})
        deltas = self._sync_doc(old, new)
        self.storage.put(new)
        return {"_id": doc_id, "changes": applied, "index_deltas": deltas}

    def unset(self, doc_id: Any, paths: Sequence[str]) -> dict:
        old = self.storage.get(doc_id)
        if old is None:
            raise DocStoreError(f"document {doc_id!r} not found")
        new = copy.deepcopy(old)
        applied = []
        for path in paths:
            ppath = parse_positional_path(path)
            if ppath.is_positional:
                changed, reason = apply_positional(new, ppath, "unset")
            else:
                changed, reason = unset_path(new, path)
            applied.append({"path": path, "changed": changed, "reason": reason})
        deltas = self._sync_doc(old, new)
        self.storage.put(new)
        return {"_id": doc_id, "changes": applied, "index_deltas": deltas}

    def array_push(self, doc_id: Any, array_path: str, element: Any) -> dict:
        old = self.storage.get(doc_id)
        if old is None:
            raise DocStoreError(f"document {doc_id!r} not found")
        new = copy.deepcopy(old)
        from .paths import get_path_one
        ok, arr = get_path_one(new, array_path)
        if not ok:
            _force_set(new, array_path, [element])
        elif not isinstance(arr, list):
            raise DocStoreError(f"{array_path!r} is not an array")
        else:
            arr.append(element)
        deltas = self._sync_doc(old, new)
        self.storage.put(new)
        return {"_id": doc_id, "index_deltas": deltas}

    def array_pop(self, doc_id: Any, array_path: str, index: int = -1) -> dict:
        old = self.storage.get(doc_id)
        if old is None:
            raise DocStoreError(f"document {doc_id!r} not found")
        new = copy.deepcopy(old)
        from .paths import get_path_one
        ok, arr = get_path_one(new, array_path)
        if not ok or not isinstance(arr, list):
            raise DocStoreError(f"{array_path!r} is not an array")
        if not arr:
            raise DocStoreError(f"{array_path!r} is already empty")
        if index < 0:
            index += len(arr)
        if index < 0 or index >= len(arr):
            raise DocStoreError(f"index {index} out of range for {array_path!r}")
        arr.pop(index)
        deltas = self._sync_doc(old, new)
        self.storage.put(new)
        return {"_id": doc_id, "index_deltas": deltas}

    def delete(self, doc_id: Any) -> bool:
        old = self.storage.get(doc_id)
        if old is None:
            return False
        for idx in self.indexes.values():
            idx.remove_document(doc_id)
        self.storage.delete(doc_id)
        return True

    # ------------------------------------------------------------------
    # reads / queries
    # ------------------------------------------------------------------
    def get(self, doc_id: Any) -> Optional[dict]:
        return self.storage.get(doc_id)

    def count(self) -> int:
        return len(self.storage)

    def find(self, query: Any, sort: Optional[Sequence[Any]] = None,
             projection: Optional[Dict[str, Any]] = None,
             limit: int = 0, skip: int = 0, force_scan: bool = False
             ) -> Tuple[List[dict], Plan]:
        tree = parse_query(query)
        sort_spec = _normalize_sort(sort)
        indexes = [] if force_scan else list(self.indexes.values())
        plan = build_plan(tree, indexes, sort_spec, store=self.storage)
        if force_scan:
            plan.mode = "collscan"
            plan.legs = []
            plan.sort_in_memory = bool(sort_spec)
            plan.notes.append("forced full collection scan (reference path)")
        docs, plan = execute(plan, self.storage, projection, limit, skip)
        return docs, plan

    def explain(self, query: Any, sort: Optional[Sequence[Any]] = None,
                projection: Optional[Dict[str, Any]] = None,
                force_scan: bool = False) -> dict:
        tree = parse_query(query)
        sort_spec = _normalize_sort(sort)
        indexes = [] if force_scan else list(self.indexes.values())
        plan = build_plan(tree, indexes, sort_spec, store=self.storage)
        if force_scan:
            plan.mode = "collscan"
            plan.legs = []
            plan.notes.append("forced full collection scan (reference path)")
        _, plan = execute(plan, self.storage, None, 0, 0)
        preds, _ = gather_predicates(tree)
        query_fields = sorted({p.field for p in preds})
        cov = covering_report(plan, projection, query_fields)
        return {
            "plan": plan.to_dict(),
            "plan_text": plan.text(),
            "semantics": explain_semantics(tree),
            "array_semantics": array_semantics(tree),
            "query_ast": node_to_dict(tree),
            "covering": cov,
            "indexes_available": [i.defn.to_dict() for i in self.indexes.values()],
        }

    def doc_index_entries(self, doc_id: Any) -> dict:
        doc = self.storage.get(doc_id)
        if doc is None:
            return {"_id": doc_id, "found": False, "indexes": []}
        out = []
        for idx in self.indexes.values():
            out.append({
                "index": idx.name,
                "fields": idx.defn.fields,
                "multikey": idx.defn.multikey_declared,
                "align": idx.defn.align,
                "entries": idx.entries_for_doc(doc_id),
            })
        return {"_id": doc_id, "found": True, "document": doc, "indexes": out}

    def stats(self) -> dict:
        return {
            "collection": self.name,
            "documents": len(self.storage),
            "storage": self.storage.stats(),
            "indexes": self.index_stats(),
        }

    def compact(self) -> dict:
        """Rewrite the log and re-snapshot every index at the new offsets."""
        before = self.storage.stats()["log_bytes"]
        self.storage.compact()
        self.snapshot_all()
        return {"log_bytes_before": before,
                "log_bytes_after": self.storage.stats()["log_bytes"],
                "documents": len(self.storage)}

    def verify_indexes(self) -> dict:
        """Rebuild each index from scratch and compare with the live one."""
        report = {}
        docs = list(self.storage.scan())
        for name, idx in self.indexes.items():
            fresh = Index(idx.defn)
            try:
                fresh.bulk_build(docs)
            except DocStoreError as exc:
                report[name] = {"ok": False, "error": exc.message}
                continue
            live = {k: set(p) for k, p in idx.postings.items() if p}
            ref = {k: set(p) for k, p in fresh.postings.items() if p}
            if live == ref:
                report[name] = {"ok": True, "keys": len(ref),
                                "entries": sum(len(v) for v in ref.values())}
            else:
                missing = set(ref) - set(live)
                extra = set(live) - set(ref)
                report[name] = {
                    "ok": False,
                    "missing_keys": [V.describe_key(k[0]) for k in list(missing)[:10]],
                    "extra_keys": [V.describe_key(k[0]) for k in list(extra)[:10]],
                }
        return report


def _normalize_sort(sort: Optional[Sequence[Any]]) -> List[Tuple[str, int]]:
    if not sort:
        return []
    out: List[Tuple[str, int]] = []
    for item in sort:
        if isinstance(item, str):
            if item.startswith("-"):
                out.append((_sort_field(item[1:]), -1))
            else:
                out.append((_sort_field(item.lstrip("+")), 1))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            f, d = item
            out.append((_sort_field(str(f)), -1 if d < 0 else 1))
        elif isinstance(item, dict):
            f = item.get("field")
            d = item.get("direction", 1)
            out.append((_sort_field(str(f)), -1 if d < 0 else 1))
        else:
            raise QueryError(f"bad sort item {item!r}")
    return out


def _sort_field(text: str) -> str:
    """Sort fields may be written with index fan-out syntax; [] is a no-op
    for lookups because ``get_path`` already traverses arrays."""
    return text.replace("[]", "")


def _force_set(doc: dict, dotted: str, value: Any) -> Tuple[bool, str]:
    parts = dotted.split(".")
    node = doc
    for part in parts[:-1]:
        if not isinstance(node, dict):
            return False, f"cannot create path through non-object at {part!r}"
        if part not in node:
            node[part] = {}
        node = node[part]
    if not isinstance(node, dict):
        return False, "parent is not an object"
    node[parts[-1]] = value
    return True, "created"
