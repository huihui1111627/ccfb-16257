"""Requirement 4: compound indexes with two array fields are rejected."""

import pytest

from docstore import ParallelArraysError, IndexError_, Collection

from .conftest import make


def test_declaring_two_arrays_rejected(col):
    make(col, [{"_id": 1, "tags": ["a"], "specs": [{"color": "red"}]}])
    with pytest.raises(ParallelArraysError) as e:
        col.create_index(["tags[]", "specs[].color"])
    assert e.value.code == "PARALLEL_ARRAYS"
    assert "cross product" in e.value.message


def test_undeclared_parallel_arrays_rejected_on_data(col):
    make(col, [{"_id": 1, "tags": ["a", "b"], "specs": [{"color": "red"},
                                                        {"color": "blue"}]}])
    # neither clause declares [], but both fields hold arrays
    with pytest.raises(ParallelArraysError):
        col.create_index(["tags", "specs.color"])
    assert "specs.color" not in col.indexes


def test_undeclared_parallel_arrays_on_write_is_atomic(col):
    # documents with scalar tags/specs so the index can be created
    make(col, [{"_id": 1, "tags": "a", "specs": {"color": "red"}},
               {"_id": 2, "tags": "b", "specs": {"color": "blue"}}])
    col.create_index(["tags", "specs.color"])
    with pytest.raises(ParallelArraysError):
        col.insert({"_id": 3, "tags": ["a", "b"],
                    "specs": [{"color": "red"}, {"color": "blue"}]})
    # the rejected insert must leave no trace
    assert col.get(3) is None
    assert col.verify_indexes()["tags_specs.color"]["ok"]


def test_parallel_arrays_message_mentions_alternative(col):
    make(col, [{"_id": 1, "a": [1], "b": [2]}])
    with pytest.raises(ParallelArraysError) as e:
        col.create_index(["a[]", "b[]"])
    assert "intersect" in e.value.message


def test_aligned_compound_index_is_allowed_and_does_not_explode(col):
    docs = [{"_id": i, "specs": [{"color": "red", "size": "XL", "stock": j}
                                 for j in range(10)]} for i in range(5)]
    make(col, docs)
    info = col.create_index(["specs[].color", "specs[].size", "specs[].stock"],
                            align="specs")
    # 10 distinct keys, not 10*3 cross products
    assert info["stats"]["distinct_keys"] == 10
    assert info["entries"] == 50


def test_align_requires_bracket_marker(col):
    make(col, [{"_id": 1, "specs": [{"color": "red"}]}])
    with pytest.raises(IndexError_):
        col.create_index(["specs.color"], align="specs")


def test_align_at_wrong_path_rejected(col):
    make(col, [{"_id": 1, "specs": [{"color": "red"}]}])
    with pytest.raises(IndexError_):
        col.create_index(["specs[].color"], align="tags")


def test_unique_index_on_array_rejected(col):
    make(col, [{"_id": 1, "specs": [{"color": "red"}]}])
    with pytest.raises(IndexError_):
        col.create_index(["specs[].color"], unique=True)
    # a unique non-multikey index is fine
    col.create_index(["specs.color"], unique=True)


def test_unique_multikey_rejected_at_write_if_data_is_array(col):
    make(col, [{"_id": 1, "code": "a"}])
    col.create_index(["code"], unique=True)
    with pytest.raises(Exception) as e:
        col.insert({"_id": 2, "code": ["x", "y"]})
    assert "unique" in str(e.value).lower()


def test_two_single_field_indexes_allowed_as_alternative(col):
    make(col, [{"_id": 1, "tags": ["a"], "specs": [{"color": "red"}]}])
    col.create_index(["tags[]"])
    col.create_index(["specs[].color"])
    docs, plan = col.find({"tags": "a", "specs.color": "red"})
    assert [d["_id"] for d in docs] == [1]
    assert plan.mode in ("ixscan", "index_intersection")


def test_intersection_matches_full_scan(col):
    docs = [
        {"_id": 1, "tags": ["a"], "specs": [{"color": "red"}]},
        {"_id": 2, "tags": ["b"], "specs": [{"color": "red"}]},
        {"_id": 3, "tags": ["a"], "specs": [{"color": "blue"}]},
    ]
    make(col, docs)
    col.create_index(["tags[]"])
    col.create_index(["specs[].color"])
    q = {"tags": "a", "specs.color": "red"}
    ref, _ = col.find(q, force_scan=True)
    got, _ = col.find(q)
    assert sorted(d["_id"] for d in ref) == sorted(d["_id"] for d in got) == [1]
