import os
import sys

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, "..", "..", "python"))
sys.path.insert(0, root)

from tinyfin import const_fold_ir


def test_const_fold_ir_basic():
    ir = {
        "nodes": [
            {"id": "a", "requires_grad": 0},
            {"id": "b", "requires_grad": 0},
            {"id": "c", "requires_grad": 0},
        ],
        "edges": [
            ("c", "a"),
            ("c", "b"),
        ],
    }
    folded = const_fold_ir(ir)
    assert folded["edges"] == []
    node_c = [n for n in folded["nodes"] if n["id"] == "c"][0]
    assert node_c.get("const_folded") is True
