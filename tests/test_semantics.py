"""Requirement 2: the two array semantics are distinct and both correct."""

import pytest

from docstore import QueryError
from docstore.query import explain_semantics, parse_query

from .conftest import SAMPLE, ids, make


OPS_ELEM = {"specs": {"$elemMatch": {"color": "red", "size": "XL",
                                     "stock": {"$gt": 0}}}}
OPS_SOME = {"$some": {"specs.color": "red", "specs.size": "XL",
                      "specs.stock": {"$gt": 0}}}
OPS_DOTTED = {"specs.color": "red", "specs.size": "XL",
              "specs.stock": {"$gt": 0}}


def test_same_element_semantics(col):
    make(col, SAMPLE)
    docs, _ = col.find(OPS_ELEM)
    # only doc 1 has one element that is red AND XL AND in stock
    assert ids(docs) == [1]


def test_independent_semantics_some_and_dotted_agree(col):
    make(col, SAMPLE)
    a, _ = col.find(OPS_SOME)
    b, _ = col.find(OPS_DOTTED)
    assert ids(a) == ids(b) == [1, 2]


def test_semantics_return_different_sets(col):
    make(col, SAMPLE)
    same, _ = col.find(OPS_ELEM)
    indep, _ = col.find(OPS_SOME)
    assert ids(same) != ids(indep)
    assert set(ids(indep)) - set(ids(same)) == {2}


def test_trap_document_is_the_bug_case(col):
    """A red S plus a blue XL product must not be returned by same-element."""
    make(col, [{"_id": "trap", "specs": [
        {"color": "red", "size": "S", "stock": 9},
        {"color": "blue", "size": "XL", "stock": 9},
    ]}])
    same, _ = col.find(OPS_ELEM)
    indep, _ = col.find(OPS_SOME)
    assert ids(same) == []
    assert ids(indep) == ["trap"]


def test_indexed_matches_full_scan_both_semantics(col):
    make(col, SAMPLE)
    col.create_index(["specs[].color", "specs[].size", "specs[].stock"],
                     align="specs")
    for q in (OPS_ELEM, OPS_SOME, OPS_DOTTED):
        ref, _ = col.find(q, force_scan=True)
        got, plan = col.find(q)
        assert ids(ref) == ids(got), (q, plan.text())


def test_semantics_are_reported_in_explain(col):
    make(col, SAMPLE)
    e = col.explain(OPS_ELEM)
    sem = e["array_semantics"]
    assert any(s["semantics"] == "same-element" for s in sem)
    e2 = col.explain(OPS_SOME)
    assert any(s["semantics"] == "independent-elements"
               for s in e2["array_semantics"])


def test_explain_text_names_semantics():
    assert "elemMatch" in explain_semantics(parse_query(OPS_ELEM))
    assert "SOME elements" in explain_semantics(parse_query(OPS_SOME))


def test_ambiguous_top_level_operator_is_rejected():
    with pytest.raises(QueryError):
        parse_query({"$eq": "red"})
    with pytest.raises(QueryError):
        parse_query({"$gt": 5})


def test_every_semantics(col):
    make(col, [
        {"_id": 1, "specs": [{"size": "XL"}, {"size": "XL"}]},
        {"_id": 2, "specs": [{"size": "XL"}, {"size": "S"}]},
        {"_id": 3, "specs": []},
        {"_id": 4},
    ])
    docs, _ = col.find({"$every": {"specs.size": "XL"}})
    assert ids(docs) == [1, 3, 4]  # vacuous truth for empty/missing


def test_ne_on_array_is_set_level(col):
    make(col, [
        {"_id": 1, "specs": [{"color": "red"}, {"color": "blue"}]},
        {"_id": 2, "specs": [{"color": "red"}]},
        {"_id": 3, "specs": [{"color": "blue"}]},
        {"_id": 4},
    ])
    docs, _ = col.find({"specs.color": {"$ne": "red"}})
    assert ids(docs) == [3, 4]


def test_nested_elem_match(col):
    make(col, [
        {"_id": 1, "specs": [{"color": "red",
                              "tags": ["new", "sale"]}]},
        {"_id": 2, "specs": [{"color": "red", "tags": ["new"]}]},
    ])
    docs, _ = col.find({"specs": {"$elemMatch": {"tags": "sale"}}})
    assert ids(docs) == [1]


def test_some_requires_array_root(col):
    with pytest.raises(QueryError):
        parse_query({"$some": {"specs.color": "red", "brand": "acme"}})


def test_index_entries_show_aligned_elements(col):
    make(col, [{"_id": 1, "specs": [
        {"color": "red", "size": "S", "stock": 1},
        {"color": "blue", "size": "XL", "stock": 2},
    ]}])
    col.create_index(["specs[].color", "specs[].size", "specs[].stock"],
                     align="specs")
    info = col.doc_index_entries(1)
    keys = info["indexes"][0]["entries"]
    assert len(keys) == 2
    assert ["red", "S", 1] in [e["key"] for e in keys]
    assert ["blue", "XL", 2] in [e["key"] for e in keys]
