"""Requirement 5: sorting on arrays, unique-in-array, positional updates."""

import pytest

from docstore import DocStoreError, DuplicateKeyError, MultikeyUniqueError

from .conftest import ids, make


SORT_DOCS = [
    {"_id": 1, "specs": [{"price": 30}, {"price": 5}]},   # min 5, max 30
    {"_id": 2, "specs": [{"price": 10}]},                 # 10
    {"_id": 3, "specs": [{"price": 20}, {"price": 25}]},  # min 20, max 25
    {"_id": 4, "specs": []},
    {"_id": 5},
]


# ---- sorting -------------------------------------------------------------

def test_sort_arrays_uses_min_element_ascending(col):
    make(col, SORT_DOCS)
    docs, _ = col.find({}, sort=["specs[].price"])
    prices = [[e["price"] for e in d["specs"]] if isinstance(d.get("specs"), list)
              else None for d in docs]
    # ascending: 1 (min 5), 2 (10), 3 (min 20), then missing/empty
    assert ids(docs)[:3] == [1, 2, 3]


def test_sort_arrays_uses_max_element_descending(col):
    make(col, SORT_DOCS)
    docs, _ = col.find({}, sort=["-specs[].price"])
    # descending by max element: 1 (30), 3 (25), 2 (10)
    assert [d["_id"] for d in docs][:3] == [1, 3, 2]


def test_sort_order_identical_indexed_and_scan(col):
    make(col, SORT_DOCS)
    col.create_index(["specs[].price"])
    for sort in (["specs[].price"], ["-specs[].price"]):
        a, _ = col.find({}, sort=sort, force_scan=True)
        b, plan = col.find({}, sort=sort)
        assert ids(a) == ids(b)
        # tie-break by _id must match too
        assert [d["_id"] for d in a] == [d["_id"] for d in b]


def test_sort_multiple_fields(col):
    make(col, [
        {"_id": 1, "specs": [{"color": "red", "price": 10}]},
        {"_id": 2, "specs": [{"color": "red", "price": 5}]},
        {"_id": 3, "specs": [{"color": "blue", "price": 30}]},
    ])
    docs, _ = col.find({}, sort=["specs[].color", "-specs[].price"])
    assert [d["_id"] for d in docs] == [3, 1, 2]


def test_missing_sorts_first_and_empty_array_matches_array_order(col):
    # Documented order: missing < null < booleans/numbers/strings/objects; an
    # empty array sorts among arrays in array order (an empty array is < [1]
    # because tuples compare shorter-first), which places it after scalars.
    make(col, [
        {"_id": "a", "v": 5},
        {"_id": "b"},
        {"_id": "c", "v": []},
        {"_id": "d", "v": [1]},
    ])
    docs, _ = col.find({}, sort=["v"])
    assert [d["_id"] for d in docs] == ["b", "d", "a", "c"]


# ---- unique in array -----------------------------------------------------

def test_unique_index_rejects_array_document(col):
    make(col, [{"_id": 1, "code": "x"}])
    col.create_index(["code"], unique=True)
    with pytest.raises((MultikeyUniqueError, DocStoreError)):
        col.insert({"_id": 2, "code": ["a", "b"]})
    assert col.get(2) is None


def test_unique_index_rejects_duplicate_scalar(col):
    make(col, [{"_id": 1, "code": "x"}])
    col.create_index(["code"], unique=True)
    with pytest.raises(DuplicateKeyError):
        col.insert({"_id": 2, "code": "x"})


def test_unique_index_allows_update_within_same_doc(col):
    make(col, [{"_id": 1, "code": "x"}])
    col.create_index(["code"], unique=True)
    col.update(1, {"code": "x"})  # no-op, same doc, allowed
    col.update(1, {"code": "y"})
    assert col.find({"code": "y"})[0][0]["_id"] == 1


def test_unique_index_rolls_back_on_rejection(col):
    make(col, [{"_id": 1, "code": "x"}])
    col.create_index(["code"], unique=True)
    with pytest.raises(DuplicateKeyError):
        col.insert({"_id": 2, "code": "x"})
    assert col.count() == 1
    assert not col.storage.contains(2)


# ---- positional updates --------------------------------------------------

def test_positional_update_sets_element_field(col):
    make(col, [{"_id": 1, "specs": [{"stock": 1}, {"stock": 2}, {"stock": 3}]}])
    col.create_index(["specs[].stock"])
    res = col.update(1, {"specs[2].stock": 99})
    assert res["_id"] == 1
    doc = col.get(1)
    assert [e["stock"] for e in doc["specs"]] == [1, 2, 99]
    assert ids(col.find({"specs.stock": 99})[0]) == [1]


def test_positional_update_replaces_whole_element(col):
    make(col, [{"_id": 1, "specs": [{"color": "red"}, {"color": "blue"}]}])
    col.update(1, {"specs[0]": {"color": "green"}})
    assert col.get(1)["specs"][0] == {"color": "green"}


def test_positional_update_out_of_range_is_rejected(col):
    make(col, [{"_id": 1, "specs": [{"stock": 1}]}])
    with pytest.raises(DocStoreError) as e:
        col.update(1, {"specs[5].stock": 9})
    assert "out of range" in str(e.value)
    # never appends
    assert len(col.get(1)["specs"]) == 1


def test_positional_update_on_non_array_is_rejected(col):
    make(col, [{"_id": 1, "specs": {"stock": 1}}])
    with pytest.raises(DocStoreError) as e:
        col.update(1, {"specs[0].stock": 9})
    assert "not an array" in str(e.value)


def test_positional_unset_removes_element(col):
    make(col, [{"_id": 1, "specs": [{"stock": 1}, {"stock": 2}]}])
    col.unset(1, ["specs[0]"])
    assert col.get(1)["specs"] == [{"stock": 2}]


def test_non_positional_update_crossing_array_refused(col):
    make(col, [{"_id": 1, "specs": [{"stock": 1}]}])
    with pytest.raises(DocStoreError):
        col.update(1, {"specs.stock": 2})


def test_sort_consistency_across_index_and_scan_variants(col):
    import random
    rng = random.Random(7)
    for i in range(400):
        col.insert({"_id": i,
                    "specs": [{"price": rng.randint(1, 20)}
                              for _ in range(rng.randint(0, 4))],
                    "rating": rng.randint(1, 5)})
    col.create_index(["specs[].price"])
    col.create_index(["rating"])
    for sort in (["specs[].price"], ["-specs[].price"], ["rating"], ["-rating"],
                 ["rating", "-specs[].price"]):
        q = {"rating": {"$gte": 1}}
        a, _ = col.find(q, sort=sort, force_scan=True)
        b, _ = col.find(q, sort=sort)
        assert [d["_id"] for d in a] == [d["_id"] for d in b], sort
