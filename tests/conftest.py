import os
import tempfile

import pytest

from docstore import Collection


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "db")


@pytest.fixture()
def col(db_path):
    c = Collection(db_path)
    yield c
    c.storage.close() if hasattr(c.storage, "close") else None


def make(col, docs):
    for d in docs:
        col.insert(d)
    return col


SAMPLE = [
    {"_id": 1, "specs": [{"color": "red", "size": "XL", "stock": 3, "price": 100}]},
    {"_id": 2, "specs": [{"color": "red", "size": "S", "stock": 5, "price": 90},
                         {"color": "blue", "size": "XL", "stock": 2, "price": 110}]},
    {"_id": 3, "specs": [{"color": "red", "size": "XL", "stock": 0, "price": 120}]},
    {"_id": 4, "specs": []},
    {"_id": 5},
    {"_id": 6, "specs": "garbage"},
    {"_id": 7, "specs": [{"color": "green", "size": "M"}]},
]


def ids(docs):
    return sorted(d["_id"] for d in docs)
