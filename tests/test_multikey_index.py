"""Requirement 1: one index entry per array element; entries follow changes."""

import pytest

from docstore import Collection
from docstore.index import Index, IndexDef, extract_keys
from docstore.paths import parse_clause

from .conftest import SAMPLE, ids, make


def test_one_entry_per_element(col):
    make(col, SAMPLE)
    info = col.create_index(["specs[].color"])
    # doc 2 has two colours -> two entries; doc 1,3,7 one each; doc 4 empty -> 1
    # (the empty-array key); docs 5,6 -> missing-ish keys
    idx = col.indexes[info["index"]]
    assert len(idx.entries_for_doc(2)) == 2
    assert len(idx.entries_for_doc(1)) == 1
    keys2 = {tuple(e["key"]) for e in idx.entries_for_doc(2)}
    assert ("red",) in keys2 and ("blue",) in keys2


def test_entry_count_tracks_elements(col):
    # distinct element values each become an entry; identical duplicates collapse
    # to a single posting (documented dedup).
    palette = ["red", "blue", "green", "black", "white", "pink", "navy", "grey"]
    docs = [{"_id": i, "specs": [{"color": c} for c in palette[:i]]}
            for i in range(1, 6)]
    make(col, docs)
    info = col.create_index(["specs[].color"])
    idx = col.indexes[info["index"]]
    assert info["stats"]["index_entries"] == 1 + 2 + 3 + 4 + 5 == 15


def test_duplicate_elements_collapse_to_one_posting(col):
    make(col, [{"_id": 1, "specs": [{"color": "red"}, {"color": "red"}]}])
    info = col.create_index(["specs[].color"])
    idx = col.indexes[info["index"]]
    assert len(idx.entries_for_doc(1)) == 1
    assert ids(col.find({"specs.color": "red"})[0]) == [1]


def test_index_follows_array_push_and_pop(col):
    make(col, [{"_id": 1, "specs": [{"color": "red", "stock": 1}]}])
    col.create_index(["specs[].color"])
    idx = col.indexes["specs.color_mk"]
    assert len(idx.entries_for_doc(1)) == 1

    col.array_push(1, "specs", {"color": "blue", "stock": 2})
    assert len(idx.entries_for_doc(1)) == 2
    assert ids(col.find({"specs.color": "blue"})[0]) == [1]

    col.array_pop(1, "specs")
    assert len(idx.entries_for_doc(1)) == 1
    assert ids(col.find({"specs.color": "blue"})[0]) == []


def test_index_follows_element_replace_via_positional_update(col):
    make(col, [{"_id": 1, "specs": [{"color": "red"}, {"color": "green"}]}])
    col.create_index(["specs[].color"])
    res = col.update(1, {"specs[0].color": "blue"})
    assert res["index_deltas"]["specs.color_mk"] == (1, 1)
    assert ids(col.find({"specs.color": "blue"})[0]) == [1]
    assert ids(col.find({"specs.color": "red"})[0]) == []


def test_index_follows_whole_array_replacement(col):
    make(col, [{"_id": 1, "specs": [{"color": "red"}]}])
    col.create_index(["specs[].color"])
    col.update(1, {"specs": [{"color": "blue"}, {"color": "blue"},
                             {"color": "green"}]})
    idx = col.indexes["specs.color_mk"]
    assert len(idx.entries_for_doc(1)) == 2  # distinct keys blue, green
    assert ids(col.find({"specs.color": "green"})[0]) == [1]
    assert ids(col.find({"specs.color": "red"})[0]) == []


def test_index_follows_field_removal(col):
    make(col, [{"_id": 1, "specs": [{"color": "red"}, {"color": "blue"}]}])
    col.create_index(["specs[].color"])
    col.unset(1, ["specs[0].color"])
    assert ids(col.find({"specs.color": "red"})[0]) == []
    assert ids(col.find({"specs.color": "blue"})[0]) == [1]


def test_index_follows_document_delete(col):
    make(col, [{"_id": 1, "specs": [{"color": "red"}]}])
    col.create_index(["specs[].color"])
    col.delete(1)
    assert col.indexes["specs.color_mk"].entry_count == 0
    assert ids(col.find({"specs.color": "red"})[0]) == []


def test_empty_array_is_distinct_from_missing(col):
    make(col, [{"_id": 1, "specs": []},
               {"_id": 2},
               {"_id": 3, "specs": [{}]}])
    col.create_index(["specs[].color"])
    idx = col.indexes["specs.color_mk"]
    keys1 = [tuple(e["key"]) for e in idx.entries_for_doc(1)]
    keys2 = [tuple(e["key"]) for e in idx.entries_for_doc(2)]
    keys3 = [tuple(e["key"]) for e in idx.entries_for_doc(3)]
    assert ("<missing>",) in keys1
    assert ("<missing>",) in keys2
    assert ("<missing>",) in keys3
    assert ids(col.find({"specs.color": {"$exists": False}})[0]) == [1, 2, 3]


def test_verify_after_churn(col):
    make(col, SAMPLE)
    col.create_index(["specs[].color", "specs[].size"], align="specs")
    col.array_push(1, "specs", {"color": "black", "size": "M"})
    col.update(2, {"specs[1].color": "pink"})
    col.delete(3)
    col.insert({"_id": 8, "specs": [{"color": "red", "size": "XL"}]})
    report = col.verify_indexes()
    assert all(v["ok"] for v in report.values()), report
