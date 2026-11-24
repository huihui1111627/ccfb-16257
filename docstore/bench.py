"""Benchmark: generate a catalogue, compare indexed vs full-scan results, time it.

Usage::

    python3 -m docstore.bench --count 300000 --seed 42 --out data/catalog

What it does
------------
1. Generates ``count`` deterministic products (see ``generator.py``).
2. Writes them to ``<out>`` through the real storage layer.
3. Runs the ops query in both semantics -- ``$elemMatch`` (same element) and
   ``$some`` / dotted (independent elements) -- with **no index** and *checks
   the two return different sets*, which is the whole point of the exercise.
4. Builds three indexes, times the build, then runs each query again.
5. Verifies, row by row, that each indexed result set equals the full-scan
   result set for the same query.  Any difference is reported as a failure.
6. Writes a JSON report and a human-readable Markdown report.

Exit code is non-zero if any row differs or any index verifies as stale.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from .collection import Collection
from .generator import generate

OPS_QUERIES = {
    "same_element_elemMatch": {
        "specs": {"$elemMatch": {"color": "red", "size": "XL",
                                 "stock": {"$gt": 0}}}
    },
    "independent_some": {
        "$some": {"specs.color": "red", "specs.size": "XL",
                  "specs.stock": {"$gt": 0}}
    },
    "independent_dotted": {
        "specs.color": "red", "specs.size": "XL", "specs.stock": {"$gt": 0}
    },
    "same_element_with_price": {
        "specs": {"$elemMatch": {"color": "red", "size": "XL",
                                 "stock": {"$gt": 0}, "price": {"$lt": 200}}}
    },
    "independent_range": {
        "$and": [{"specs.price": {"$gt": 100}}, {"specs.price": {"$lt": 120}}]
    },
}

INDEX_SPECS = [
    dict(fields=["specs[].color", "specs[].size", "specs[].stock"], align="specs",
         name="ops_aligned"),
    dict(fields=["specs[].color"], name="color_mk"),
    dict(fields=["specs[].size"], name="size_mk"),
    dict(fields=["specs[].stock"], name="stock_mk"),
    dict(fields=["specs[].price"], name="price_mk"),
    dict(fields=["brand"], name="brand"),
]


def _idset(docs: List[dict]) -> List[Any]:
    return sorted(d["_id"] for d in docs)


def run(count: int, seed: int, out: str, use_indexes: bool = True) -> Dict[str, Any]:
    os.makedirs(out, exist_ok=True)
    db_path = os.path.join(out, "db")
    report: Dict[str, Any] = {
        "count": count, "seed": seed, "db_path": db_path, "steps": [], "ok": True,
    }

    def log(msg: str) -> None:
        print(msg, flush=True)
        report["steps"].append(msg)

    t0 = time.perf_counter()
    if os.path.exists(os.path.join(db_path, "data.log")):
        import shutil
        shutil.rmtree(db_path)
    col = Collection(db_path)
    batch: List[dict] = []
    n = 0
    for doc in generate(count, seed):
        batch.append(doc)
        if len(batch) >= 5000:
            col.storage.put_many(batch)
            n += len(batch)
            batch = []
    if batch:
        col.storage.put_many(batch)
        n += len(batch)
    gen_ms = (time.perf_counter() - t0) * 1000
    log(f"generated and stored {n} documents in {gen_ms:.0f} ms")
    report["documents"] = n
    report["generate_ms"] = round(gen_ms, 1)

    # shape census
    census = {
        "empty_specs": 0, "missing_specs": 0, "non_array_specs": 0,
        "with_specs": 0, "total_elements": 0, "max_len": 0,
    }
    for doc in col.storage.scan():
        specs = doc.get("specs", "__missing__")
        if specs == "__missing__":
            census["missing_specs"] += 1
        elif not isinstance(specs, list):
            census["non_array_specs"] += 1
        elif not specs:
            census["empty_specs"] += 1
        else:
            census["with_specs"] += 1
            census["total_elements"] += len(specs)
            census["max_len"] = max(census["max_len"], len(specs))
    report["census"] = census
    log(f"shape census: {census}")

    # full-scan reference results for every query
    baseline: Dict[str, List[Any]] = {}
    baseline_ms: Dict[str, float] = {}
    audit: Dict[str, Any] = {}
    for name, q in OPS_QUERIES.items():
        t = time.perf_counter()
        docs, plan = col.find(q, force_scan=True)
        ms = (time.perf_counter() - t) * 1000
        baseline[name] = _idset(docs)
        baseline_ms[name] = ms
        audit[name] = {
            "semantics": plan.__dict__.get("sort"),
            "count": len(docs),
            "ms": round(ms, 1),
            "scanned": plan.docs_fetched,
        }
        log(f"full scan {name}: {len(docs)} rows in {ms:.1f} ms "
            f"({plan.docs_fetched} docs scanned)")
    report["baseline"] = audit

    same = baseline["same_element_elemMatch"]
    indep = baseline["independent_some"]
    dotted = baseline["independent_dotted"]
    report["semantics_differ"] = same != indep
    if indep != dotted:
        report["ok"] = False
        log("FAIL: $some and dotted forms disagree")
    log(f"same-element={len(same)} rows, independent={len(indep)} rows, "
        f"difference={len(set(indep) - set(same))} docs that only the "
        f"independent form returns")
    report["rows_same_element"] = len(same)
    report["rows_independent"] = len(indep)
    report["docs_only_independent"] = len(set(indep) - set(same))

    # build indexes
    build_report = {}
    if use_indexes:
        for spec in INDEX_SPECS:
            t = time.perf_counter()
            try:
                info = col.create_index(spec["fields"], name=spec["name"],
                                        align=spec.get("align"))
                ms = (time.perf_counter() - t) * 1000
                build_report[spec["name"]] = {
                    "ok": True, "entries": info["entries"],
                    "distinct_keys": info["stats"]["distinct_keys"],
                    "build_ms": round(ms, 1),
                }
                log(f"built index {spec['name']}: {info['entries']} entries, "
                    f"{info['stats']['distinct_keys']} distinct keys, {ms:.0f} ms")
            except Exception as exc:  # noqa: BLE001 - report, don't crash
                build_report[spec["name"]] = {"ok": False, "error": str(exc)}
                log(f"index {spec['name']} rejected: {exc}")
        report["index_build"] = build_report
        col.snapshot_all()

    # indexed runs + row-by-row comparison
    indexed = {}
    for name, q in OPS_QUERIES.items():
        t = time.perf_counter()
        docs, plan = col.find(q)
        ms = (time.perf_counter() - t) * 1000
        got = _idset(docs)
        ok = got == baseline[name]
        indexed[name] = {
            "count": len(docs),
            "ms": round(ms, 1),
            "mode": plan.mode,
            "index": plan.index_names,
            "scanned": plan.docs_fetched,
            "keys_examined": plan.keys_examined,
            "speedup": round(baseline_ms[name] / ms, 2) if ms else None,
            "matches_full_scan": ok,
        }
        if not ok:
            report["ok"] = False
            log(f"FAIL: indexed {name} differs from full scan "
                f"(missing={len(set(baseline[name]) - set(got))}, "
                f"extra={len(set(got) - set(baseline[name]))})")
        else:
            log(f"indexed {name}: {len(docs)} rows, {ms:.1f} ms, mode={plan.mode} "
                f"index={plan.index_names} scanned={plan.docs_fetched} "
                f"keys={plan.keys_examined} -- matches full scan")
    report["indexed"] = indexed

    verify = col.verify_indexes()
    report["verify"] = verify
    if not all(v.get("ok") for v in verify.values()):
        report["ok"] = False
        log(f"FAIL: index verification: {verify}")

    report_path = os.path.join(out, "bench.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    md_path = os.path.join(out, "BENCH.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_markdown(report))
    log(f"report written to {report_path} and {md_path}")
    return report


def _fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{ms:.1f} ms"


def _markdown(r: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# 基准报告\n")
    lines.append(f"- 文档数：**{r['documents']}**，seed={r['seed']}")
    c = r.get("census", {})
    lines.append(
        f"- 形态：有规格 {c.get('with_specs')}，空数组 {c.get('empty_specs')}，"
        f"缺字段 {c.get('missing_specs')}，specs 非数组 {c.get('non_array_specs')}，"
        f"元素总数 {c.get('total_elements')}，最长数组 {c.get('max_len')}")
    lines.append(
        f"- **同一元素查询 {r.get('rows_same_element')} 条；不同元素查询 "
        f"{r.get('rows_independent')} 条；仅后者返回的文档 "
        f"{r.get('docs_only_independent')} 条**")
    lines.append(f"- 总体校验：{'通过 ✅' if r.get('ok') else '**失败 ❌**'}\n")

    lines.append("## 建索引\n")
    lines.append("| 索引 | 条目数 | 不同键 | 建索引耗时 |")
    lines.append("|---|---:|---:|---:|")
    for name, b in (r.get("index_build") or {}).items():
        if b.get("ok"):
            lines.append(f"| `{name}` | {b['entries']:,} | {b['distinct_keys']:,} | "
                         f"{_fmt_ms(b['build_ms'])} |")
        else:
            lines.append(f"| `{name}` | 拒绝 | — | {b.get('error','')[:40]} |")
    lines.append("")

    base = r.get("baseline", {})
    idx = r.get("indexed", {})
    lines.append("## 查询：全表扫描 vs 走索引\n")
    lines.append("| 查询 | 结果条数 | 全表扫描耗时 | 索引耗时 | 加速 | 索引 | 模式 | "
                 "扫描文档数 | 与全表逐条一致 |")
    lines.append("|---|---:|---:|---:|---:|---|---|---:|:---:|")
    for name in base:
        b = base[name]
        i = idx.get(name, {})
        speed = i.get("speedup")
        speed_txt = f"{speed:.2f}×" if speed else "—"
        ok = "✅" if i.get("matches_full_scan") else "❌"
        lines.append(
            f"| `{name}` | {i.get('count', b['count']):,} | {_fmt_ms(b['ms'])} | "
            f"{_fmt_ms(i.get('ms', 0))} | {speed_txt} | "
            f"{', '.join(i.get('index', [])) or '—'} | {i.get('mode','—')} | "
            f"{i.get('scanned', 0):,} | {ok} |")
    lines.append("")
    verify = r.get("verify", {})
    bad = [k for k, v in verify.items() if not v.get("ok")]
    lines.append("## 索引一致性\n")
    lines.append("每个索引重建后与在线索引逐键对比："
                 + ("全部一致 ✅" if not bad else f"不一致：{bad} ❌") + "\n")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/catalog")
    ap.add_argument("--no-indexes", action="store_true")
    args = ap.parse_args(argv)
    report = run(args.count, args.seed, args.out, use_indexes=not args.no_indexes)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
