"""Storage layer: durability, recovery from a torn tail, compaction."""

import os

import pytest

from docstore import Collection, StorageCorruptionError
from docstore.storage import MAGIC, DATA_LOG


def test_documents_survive_reopen(tmp_path):
    p = str(tmp_path / "db")
    c = Collection(p)
    c.insert({"_id": 1, "x": 1})
    c.insert({"_id": 2, "x": 2})
    c.delete(1)
    del c
    c2 = Collection(p)
    assert c2.get(2) == {"_id": 2, "x": 2}
    assert c2.get(1) is None
    assert c2.count() == 1


def test_torn_tail_truncated_on_open(tmp_path):
    p = str(tmp_path / "db")
    c = Collection(p)
    c.insert({"_id": 1, "x": 1})
    c.insert({"_id": 2, "x": 2})
    log = os.path.join(p, DATA_LOG)
    size = os.path.getsize(log)
    with open(log, "ab") as f:
        f.write(b"\x00\x00\x00\x50garbage-not-a-full-record")
    c2 = Collection(p)
    # both good records recovered; torn tail dropped
    assert c2.count() == 2
    assert os.path.getsize(log) == size


def test_corrupt_middle_crc_stops_replay(tmp_path):
    p = str(tmp_path / "db")
    c = Collection(p)
    for i in range(5):
        c.insert({"_id": i, "x": i})
    log = os.path.join(p, DATA_LOG)
    data = bytearray(open(log, "rb").read())
    # flip a byte inside the third record's payload region
    data[len(MAGIC) + 40] ^= 0xFF
    open(log, "wb").write(bytes(data))
    c2 = Collection(p)
    # replay stops at the corruption; earlier records survive, later are dropped
    assert c2.count() < 5


def test_bad_magic_rejected(tmp_path):
    p = str(tmp_path / "db")
    c = Collection(p)
    c.insert({"_id": 1})
    log = os.path.join(p, DATA_LOG)
    open(log, "wb").write(b"XXXXX garbage")
    with pytest.raises(StorageCorruptionError):
        Collection(p)


def test_compaction_preserves_data_and_indexes(tmp_path):
    p = str(tmp_path / "db")
    c = Collection(p)
    for i in range(10):
        c.insert({"_id": i, "specs": [{"color": "red" if i % 2 else "blue"}]})
    for i in range(0, 10, 2):
        c.delete(i)
    c.create_index(["specs[].color"])
    before = sorted(d["_id"] for d in c.find({"specs.color": "red"})[0])
    c.storage.compact()
    c2 = Collection(p)
    after = sorted(d["_id"] for d in c2.find({"specs.color": "red"})[0])
    assert before == after


def test_manifest_records_indexes_across_reopen(tmp_path):
    p = str(tmp_path / "db")
    c = Collection(p)
    c.insert({"_id": 1, "specs": [{"color": "red"}]})
    c.create_index(["specs[].color", "specs[].size"], align="specs")
    del c
    c2 = Collection(p)
    assert "specs.color_mk_specs.size_mk" in c2.indexes
    assert c2.find({"specs.color": "red"})[0][0]["_id"] == 1


def test_non_finite_float_rejected(col):
    with pytest.raises(Exception):
        col.insert({"_id": 1, "price": float("nan")})


def test_unsupported_type_rejected(col):
    with pytest.raises(Exception):
        col.insert({"_id": 1, "when": object()})


def test_keys_must_be_strings(col):
    with pytest.raises(Exception):
        col.insert({"_id": 1, "obj": {1: "x"}})


def test_snapshot_reload_with_replay(tmp_path):
    from docstore import Collection
    p = str(tmp_path / "db")
    c = Collection(p)
    for i in range(150):
        c.insert({"_id": i, "specs": [{"color": "red" if i % 3 else "blue",
                                       "size": "XL", "stock": i % 2}]})
    c.create_index(["specs[].color", "specs[].size"], align="specs")
    c.create_index(["specs[].color"])
    c.snapshot_all()
    # mutate after the snapshot so the reload must replay records
    for i in range(40):
        c.update(i, {"specs[0].color": "green"})
    c.delete(120)
    c.insert({"_id": 500, "specs": [{"color": "red", "size": "XL", "stock": 1}]})
    del c

    c2 = Collection(p)
    assert c2.count() == 150
    q = {"specs.color": "green"}
    ref, _ = c2.find(q, force_scan=True)
    got, _ = c2.find(q)
    assert sorted(d["_id"] for d in ref) == sorted(d["_id"] for d in got)
    assert len(got) == 40
    assert all(v["ok"] for v in c2.verify_indexes().values())


def test_corrupt_snapshot_triggers_rebuild(tmp_path):
    from docstore import Collection
    p = str(tmp_path / "db")
    c = Collection(p)
    for i in range(30):
        c.insert({"_id": i, "specs": [{"color": "red", "size": "XL"}]})
    c.create_index(["specs[].color", "specs[].size"], align="specs")
    c.snapshot_all()
    snap = str(tmp_path / "db" / "idx" / "specs.color_mk_specs.size_mk.idx")
    with open(snap, "r+b") as f:
        f.seek(20)
        f.write(b"\xff\xff\xff")
    del c
    c2 = Collection(p)
    assert c2.count() == 30
    assert c2.find({"specs.color": "red"})[0]
    assert all(v["ok"] for v in c2.verify_indexes().values())


def test_compact_resnapshots(tmp_path):
    from docstore import Collection
    p = str(tmp_path / "db")
    c = Collection(p)
    for i in range(50):
        c.insert({"_id": i, "specs": [{"color": "red" if i % 2 else "blue"}]})
    c.create_index(["specs[].color"])
    for i in range(0, 50, 3):
        c.delete(i)
    c.compact()
    del c
    c2 = Collection(p)
    assert c2.count() == 50 - len(list(range(0, 50, 3))) == 33
    ref, _ = c2.find({"specs.color": "red"}, force_scan=True)
    got, _ = c2.find({"specs.color": "red"})
    assert sorted(d["_id"] for d in ref) == sorted(d["_id"] for d in got)
