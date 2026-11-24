"""Randomised churn: indexes must stay byte-identical to a fresh rebuild."""

import random

from .conftest import ids


def test_churn_keeps_indexes_consistent(col):
    rng = random.Random(1234)
    colors = ["red", "blue", "green"]
    sizes = ["S", "XL"]
    for i in range(300):
        col.insert({"_id": i, "specs": [
            {"color": rng.choice(colors), "size": rng.choice(sizes),
             "stock": rng.randint(0, 9)}
            for _ in range(rng.randint(0, 4))]})
    col.create_index(["specs[].color", "specs[].size", "specs[].stock"],
                     align="specs")
    col.create_index(["specs[].color"])
    next_id = 300
    for _ in range(600):
        op = rng.random()
        target = rng.randint(0, next_id - 1)
        try:
            if op < 0.30:
                col.insert({"_id": next_id, "specs": [
                    {"color": rng.choice(colors), "size": rng.choice(sizes),
                     "stock": rng.randint(0, 9)}
                    for _ in range(rng.randint(0, 3))]})
                next_id += 1
            elif op < 0.50:
                col.array_push(target, "specs",
                               {"color": "red", "size": "XL", "stock": 1})
            elif op < 0.65:
                doc = col.get(target)
                if doc and isinstance(doc.get("specs"), list) and doc["specs"]:
                    idx = rng.randrange(len(doc["specs"]))
                    col.update(target, {f"specs[{idx}].color": "red"})
            elif op < 0.78:
                doc = col.get(target)
                if doc and isinstance(doc.get("specs"), list) and doc["specs"]:
                    idx = rng.randrange(len(doc["specs"]))
                    col.unset(target, [f"specs[{idx}]"])
            elif op < 0.90:
                col.delete(target)
            else:
                if col.get(target) is not None:
                    col.update(target, {"specs": []})
        except Exception:
            pass  # duplicate ids / not found: irrelevant to this property

    report = col.verify_indexes()
    assert all(v["ok"] for v in report.values()), report

    queries = [
        {"specs": {"$elemMatch": {"color": "red", "size": "XL",
                                  "stock": {"$gt": 0}}}},
        {"$some": {"specs.color": "red", "specs.size": "XL"}},
        {"specs.color": "red", "specs.size": "XL", "specs.stock": {"$gt": 0}},
        {"specs.color": {"$exists": False}},
        {"$and": [{"specs.stock": {"$gt": 2}}, {"specs.stock": {"$lt": 7}}]},
    ]
    for q in queries:
        ref, _ = col.find(q, force_scan=True)
        got, _ = col.find(q)
        assert ids(ref) == ids(got), q
