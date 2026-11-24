"""A small standard-library web UI for the document store.

Run::

    python3 -m docstore.server --data data/catalog --port 8765

Routes
------
GET  /                     the single-page UI
GET  /api/stats            collection + index statistics
GET  /api/query            ?q=<json>&sort=<json>&limit=&skip=&scan=1
GET  /api/explain          same params, returns the plan
GET  /api/entries/<id>     every index entry a document occupies
GET  /api/doc/<id>         the stored document
POST /api/index            {"fields": [...], "name": ..., "align": ...,
                            "unique": false, "sparse": false}
POST /api/drop-index       {"name": ...}
POST /api/import           {"documents": [...]}  (bulk insert)
POST /api/verify           re-check indexes against a fresh build

Everything returns JSON with a ``error`` object on failure; the UI shows it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from .collection import Collection
from .errors import DocStoreError

HERE = os.path.dirname(__file__)
UI_PATH = os.path.join(HERE, "ui.html")

_query_lock = threading.RLock()
_collection: Optional[Collection] = None


def _json_response(handler, status: int, obj: Any) -> None:
    body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _error(handler, status: int, exc: Exception) -> None:
    if isinstance(exc, DocStoreError):
        _json_response(handler, status, {"error": exc.to_dict()})
    else:
        _json_response(handler, status, {
            "error": {"code": "INTERNAL", "message": str(exc),
                      "details": {"traceback": traceback.format_exc().splitlines()[-6:]}}
        })


def _parse_json_param(raw: Optional[str], default: Any) -> Any:
    if raw is None or raw == "":
        return default
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise DocStoreError(f"invalid JSON parameter: {exc}")


class Handler(BaseHTTPRequestHandler):
    server_version = "docstore/1.0"

    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                return self._serve_ui()
            if path == "/api/stats":
                return _json_response(self, 200, _collection.stats())
            if path == "/api/indexes":
                return _json_response(self, 200, {
                    "indexes": [i.defn.to_dict() for i in _collection.indexes.values()],
                    "stats": _collection.index_stats(),
                })
            if path == "/api/query":
                return self._handle_query(qs, explain=False)
            if path == "/api/explain":
                return self._handle_query(qs, explain=True)
            if path.startswith("/api/entries/"):
                doc_id = _decode_id(path[len("/api/entries/"):])
                return _json_response(self, 200, _collection.doc_index_entries(doc_id))
            if path.startswith("/api/doc/"):
                doc_id = _decode_id(path[len("/api/doc/"):])
                doc = _collection.get(doc_id)
                if doc is None:
                    return _json_response(self, 404, {"error": {
                        "code": "NOT_FOUND", "message": f"no document {doc_id!r}"}})
                return _json_response(self, 200, {"document": doc})
            if path == "/api/verify":
                return _json_response(self, 200, {"verify": _collection.verify_indexes()})
            return _json_response(self, 404, {"error": {
                "code": "NOT_FOUND", "message": f"no route {path}"}})
        except Exception as exc:  # noqa: BLE001
            _error(self, 400, exc)

    # ------------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            body = self._read_body()
            if parsed.path == "/api/index":
                with _query_lock:
                    fields = body.get("fields")
                    if not isinstance(fields, list) or not fields:
                        raise DocStoreError("fields must be a non-empty array")
                    info = _collection.create_index(
                        fields, name=body.get("name") or None,
                        unique=bool(body.get("unique")),
                        sparse=bool(body.get("sparse")),
                        align=body.get("align") or None)
                return _json_response(self, 200, {"ok": True, "result": info,
                                                  "stats": _collection.stats()})
            if parsed.path == "/api/drop-index":
                name = body.get("name")
                with _query_lock:
                    ok = _collection.drop_index(name)
                return _json_response(self, 200 if ok else 404,
                                      {"ok": ok, "stats": _collection.stats()})
            if parsed.path == "/api/import":
                docs = body.get("documents")
                if not isinstance(docs, list):
                    raise DocStoreError("documents must be an array")
                with _query_lock:
                    n = _collection.insert_many(docs)
                return _json_response(self, 200, {"ok": True, "inserted": n})
            if parsed.path == "/api/update":
                with _query_lock:
                    res = _collection.update(body["id"], body["changes"])
                return _json_response(self, 200, {"ok": True, "result": res})
            if parsed.path == "/api/delete":
                with _query_lock:
                    ok = _collection.delete(body["id"])
                return _json_response(self, 200, {"ok": ok})
            return _json_response(self, 404, {"error": {
                "code": "NOT_FOUND", "message": f"no route {parsed.path}"}})
        except Exception as exc:  # noqa: BLE001
            _error(self, 400, exc)

    # ------------------------------------------------------------------
    def _handle_query(self, qs: Dict[str, List[str]], explain: bool) -> None:
        def one(key, default=None):
            v = qs.get(key)
            return v[0] if v else default

        q = _parse_json_param(one("q"), {})
        sort = _parse_json_param(one("sort"), None)
        projection = _parse_json_param(one("projection"), None)
        limit = int(one("limit", "0") or 0)
        skip = int(one("skip", "0") or 0)
        force_scan = one("scan") in ("1", "true", "yes")
        with _query_lock:
            if explain:
                out = _collection.explain(q, sort=sort, projection=projection,
                                          force_scan=force_scan)
                return _json_response(self, 200, out)
            docs, plan = _collection.find(q, sort=sort, projection=projection,
                                          limit=limit, skip=skip,
                                          force_scan=force_scan)
            payload = {
                "documents": docs,
                "count": len(docs),
                "plan": plan.to_dict(),
                "plan_text": plan.text(),
                "array_semantics": _collection.explain(
                    q, force_scan=force_scan)["array_semantics"],
            }
        return _json_response(self, 200, payload)

    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise DocStoreError(f"invalid JSON body: {exc}")

    def _serve_ui(self) -> None:
        with open(UI_PATH, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _decode_id(text: str) -> Any:
    from urllib.parse import unquote
    raw = unquote(text)
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def serve(data_path: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    global _collection
    os.makedirs(data_path, exist_ok=True)
    db = os.path.join(data_path, "db")
    _collection = Collection(db)
    if _collection.count() == 0:
        sys.stderr.write(
            "warning: collection is empty; run `python3 -m docstore.bench --count "
            "200000` first, then point --data at the same directory.\n")
    httpd = ThreadingHTTPServer((host, port), Handler)
    sys.stderr.write(f"serving {_collection.count()} documents at http://{host}:{port}/\n")
    try:
        httpd.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        httpd.server_close()
        try:
            n = _collection.snapshot_all()
            sys.stderr.write(f"persisted {n} index snapshot(s) on shutdown\n")
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"snapshot on shutdown failed: {exc}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/catalog")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    import signal

    def _graceful(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _graceful)
    try:
        serve(args.data, args.host, args.port)
    except SystemExit:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
