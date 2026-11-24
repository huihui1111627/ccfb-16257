"""Requirement 6: the plan says which index, how much was scanned, and why
a covering return was not possible."""

from docstore.planner import covering_report

from .conftest import SAMPLE, make


def test_plan_names_index_and_counters(col):
    make(col, SAMPLE)
    col.create_index(["specs[].color", "specs[].size", "specs[].stock"],
                     align="specs")
    q = {"specs": {"$elemMatch": {"color": "red", "size": "XL",
                                  "stock": {"$gt": 0}}}}
    docs, plan = col.find(q)
    d = plan.to_dict()
    assert d["mode"] == "ixscan"
    assert d["indexes"] == ["specs.color_mk_specs.size_mk_specs.stock_mk"]
    c = d["counters"]
    assert c["documents_fetched"] >= c["documents_matched"] == c["documents_returned"]
    assert c["index_keys_examined"] > 0
    assert c["elapsed_ms"] >= 0


def test_explain_reports_semantics_and_ast(col):
    make(col, SAMPLE)
    col.create_index(["specs[].color", "specs[].size"], align="specs")
    e = col.explain({"$some": {"specs.color": "red", "specs.size": "XL"}})
    assert e["array_semantics"]
    assert e["query_ast"]
    assert "SOME elements" in e["semantics"]


def test_collscan_plan_when_no_index(col):
    make(col, SAMPLE)
    docs, plan = col.find({"specs.color": "red"})
    assert plan.mode == "collscan"
    assert plan.docs_fetched == len(SAMPLE)


def test_covering_blocked_by_multikey(col):
    make(col, SAMPLE)
    col.create_index(["specs[].color"], name="c")
    e = col.explain({"specs.color": "red"}, projection={"specs": 1})
    cov = e["covering"]
    assert cov["possible"] is False
    assert any("multikey" in b for b in cov["blockers"])


def test_covering_blocked_by_full_document(col):
    make(col, SAMPLE)
    col.create_index(["specs[].color"], name="c")
    e = col.explain({"specs.color": "red"})
    assert e["covering"]["possible"] is False
    assert any("full document" in b for b in e["covering"]["blockers"])


def test_covering_reports_only_index_field_with_projection(col):
    make(col, SAMPLE)
    col.create_index(["specs[].color"], name="c")
    e = col.explain({"specs.color": "red"}, projection={"specs": 1})
    # still blocked by multikey but not by "no projection"
    assert not any("no projection" in b for b in e["covering"]["blockers"])


def test_plan_text_has_scan_ranges(col):
    make(col, [{"_id": 1, "price": 5}, {"_id": 2, "price": 15}])
    col.create_index(["price"])
    _, plan = col.find({"price": {"$gt": 10}})
    txt = plan.text()
    assert "scan" in txt and "price" in txt
    assert "counters:" in txt


def test_doc_index_entries_lists_every_key(col):
    make(col, [{"_id": 1, "specs": [
        {"color": "red", "size": "S"},
        {"color": "blue", "size": "XL"},
        {"color": "red", "size": "M"},
    ]}])
    col.create_index(["specs[].color", "specs[].size"], align="specs")
    info = col.doc_index_entries(1)
    entries = info["indexes"][0]["entries"]
    assert len(entries) == 3
    keys = sorted(tuple(e["key"]) for e in entries)
    assert keys == [("blue", "XL"), ("red", "M"), ("red", "S")]


def test_doc_index_entries_missing_document(col):
    make(col, [{"_id": 1}])
    info = col.doc_index_entries("nope")
    assert info["found"] is False
