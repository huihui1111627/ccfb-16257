"""Requirement 3: multikey range-bound merging, safe vs unsafe."""

import pytest

from docstore import QueryError
from docstore.planner import build_plan, build_ranges, gather_predicates
from docstore.query import parse_query

from .conftest import ids, make


RANGE_DOCS = [
    {"_id": 1, "specs": [{"price": 1}, {"price": 30}]},   # outside (10,20) but
                                                            # one elem >10 and
                                                            # one elem <20
    {"_id": 2, "specs": [{"price": 15}]},                  # inside
    {"_id": 3, "specs": [{"price": 5}, {"price": 25}]},    # straddles
]


def test_non_multikey_bounds_merge(col):
    make(col, [{"_id": i, "price": p} for i, p in [(1, 5), (2, 15), (3, 25)]])
    col.create_index(["price"])
    docs, plan = col.find({"price": {"$gt": 10, "$lt": 20}})
    assert plan.mode == "ixscan"
    assert ids(docs) == [2]
    assert any("MERGED" in n or "merged" in n for leg in plan.to_dict()["legs"]
               for n in leg["notes"])


def test_multikey_independent_bounds_must_not_merge(col):
    """The classic miss: [1,30] satisfies >10 and <20 with different elements."""
    make(col, RANGE_DOCS)
    col.create_index(["specs[].price"])
    # two independent conditions ($and) -> separate witnesses -> union scan
    docs, plan = col.find({"$and": [{"specs.price": {"$gt": 10}},
                                    {"specs.price": {"$lt": 20}}]})
    assert ids(docs) == [1, 2, 3]
    assert plan.mode == "ixscan"
    notes = " ".join(n for leg in plan.to_dict()["legs"] for n in leg["notes"])
    assert ("UNIONED" in notes or "SEPARATE" in notes
            or "different element" in notes or "residual" in notes.lower()
            or "range bound" in notes)


def test_multikey_merged_interval_would_miss_document(col):
    """Demonstrate the bug directly: a merged (10,20) scan misses doc 1."""
    make(col, RANGE_DOCS)
    col.create_index(["specs[].price"])
    idx = col.indexes["specs.price_mk"]
    from docstore.values import encode_key
    lo = (encode_key(10),)
    hi = (encode_key(20),)
    merged, _, _ = idx.scan_range(lo, hi, False, False)
    assert 1 not in merged  # merged scan alone would drop doc 1
    # the engine's union scan recovers it
    docs, _ = col.find({"$and": [{"specs.price": {"$gt": 10}},
                                 {"specs.price": {"$lt": 20}}]})
    assert 1 in ids(docs)


def test_elem_match_bounds_are_safe_to_merge(col):
    make(col, RANGE_DOCS)
    col.create_index(["specs[].price"])
    docs, plan = col.find({"specs": {"$elemMatch": {"price": {"$gt": 10,
                                                              "$lt": 20}}}})
    assert ids(docs) == [2]  # doc 1 has no single element in (10,20)
    assert plan.mode == "ixscan"
    notes = " ".join(n for leg in plan.to_dict()["legs"] for n in leg["notes"])
    assert "MERGED" in notes


def test_single_condition_two_bounds_share_one_element(col):
    """One condition object = one element, so bounds merge (documented)."""
    make(col, RANGE_DOCS)
    col.create_index(["specs[].price"])
    docs, plan = col.find({"specs.price": {"$gt": 10, "$lt": 20}})
    assert ids(docs) == [2]
    assert plan.mode == "ixscan"


def test_range_plus_equality_same_element(col):
    make(col, [
        {"_id": 1, "specs": [{"color": "red", "price": 5},
                             {"color": "red", "price": 50}]},
        {"_id": 2, "specs": [{"color": "red", "price": 30}]},
    ])
    col.create_index(["specs[].color", "specs[].price"], align="specs")
    docs, plan = col.find({"specs": {"$elemMatch": {"color": "red",
                                                    "price": {"$gt": 10}}}})
    assert ids(docs) == [1, 2]
    assert plan.mode == "ixscan"
    # equality prefix + range suffix -> one tight interval
    leg = plan.to_dict()["legs"][0]
    assert "['red', 10]" in leg["ranges"][0]


def test_independent_compound_clause_is_residual(col):
    """Two independent witnesses on different clauses must not share a key."""
    make(col, [
        {"_id": 1, "specs": [{"color": "red", "stock": 0},
                             {"color": "blue", "stock": 5}]},
    ])
    col.create_index(["specs[].color", "specs[].stock"], align="specs")
    q = {"$some": {"specs.color": "red", "specs.stock": {"$gt": 0}}}
    docs, plan = col.find(q)
    assert ids(docs) == [1]
    # the aligned entry (red, >0) does not exist; plan must not scan only it
    leg = plan.to_dict()["legs"][0]
    assert "residual" in " ".join(leg["notes"]).lower() or \
        "scope" in " ".join(leg["notes"]).lower()


@pytest.mark.parametrize("q", [
    {"specs.price": {"$gt": 10, "$lt": 20}},
    {"$and": [{"specs.price": {"$gt": 10}}, {"specs.price": {"$lt": 20}}]},
    {"specs": {"$elemMatch": {"price": {"$gt": 10, "$lt": 20}}}},
    {"$or": [{"specs.price": {"$lt": 10}}, {"specs.price": {"$gt": 20}}]},
    {"specs.price": {"$gte": 1, "$lte": 30}},
])
def test_indexed_equals_scan_for_range_queries(col, q):
    make(col, RANGE_DOCS)
    col.create_index(["specs[].price"])
    ref, _ = col.find(q, force_scan=True)
    got, _ = col.find(q)
    assert ids(ref) == ids(got)


def test_contradictory_bounds_yield_nothing_not_everything(col):
    make(col, RANGE_DOCS)
    col.create_index(["specs[].price"])
    docs, _ = col.find({"specs.price": {"$gt": 100, "$lt": 5}})
    assert ids(docs) == []


def test_build_ranges_notes_explain_merge_decision():
    from docstore.index import IndexDef
    from docstore.paths import parse_clause
    idef = IndexDef("t", [parse_clause("specs[].price")])
    preds, _ = gather_predicates(parse_query(
        {"$and": [{"specs.price": {"$gt": 10}}, {"specs.price": {"$lt": 20}}]}))
    ranges, used, notes = build_ranges(idef, preds, multikey=True)
    assert len(ranges) == 2  # two separate ranges, not one intersected
    assert sum(1 for n in notes if "UNIONED" in n or "range bound" in n) >= 1

    preds2, _ = gather_predicates(parse_query(
        {"specs": {"$elemMatch": {"price": {"$gt": 10, "$lt": 20}}}}))
    ranges2, used2, notes2 = build_ranges(idef, preds2, multikey=True)
    assert len(ranges2) == 1
    assert any("MERGED" in n for n in notes2)
