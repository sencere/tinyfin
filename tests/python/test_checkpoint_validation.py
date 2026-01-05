import sys, os, json, tempfile

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor
from tinyfin.utils import save_checkpoint, load_checkpoint


def test_checkpoint_shape_mismatch_raises():
    t = Tensor.new([2], requires_grad=False)
    t.numpy_view()[:] = [1.0, 2.0]
    tmp = tempfile.NamedTemporaryFile(delete=False)
    base = tmp.name
    tmp.close()
    meta = save_checkpoint(base, {"t": t})
    # Corrupt metadata to mismatch shape/data length
    with open(meta, "r", encoding="utf-8") as f:
        blob = json.load(f)
    blob["tensors"]["t"]["shape"] = [4]
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(blob, f)
    try:
        try:
            load_checkpoint(meta)
            assert False, "expected ValueError"
        except ValueError:
            pass
    finally:
        # cleanup generated files
        for suffix in [".ckpt.json", ".opt.bin", ".sched.json"]:
            p = base + suffix
            if os.path.exists(p):
                os.unlink(p)

def test_checkpoint_dtype_mismatch_raises():
    t = Tensor.new([1], requires_grad=False)
    t.numpy_view()[:] = [1.0]
    tmp = tempfile.NamedTemporaryFile(delete=False)
    base = tmp.name
    tmp.close()
    meta = save_checkpoint(base, {"t": t})
    with open(meta, "r", encoding="utf-8") as f:
        blob = json.load(f)
    blob["tensors"]["t"]["dtype"] = "int64"
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(blob, f)
    try:
        try:
            load_checkpoint(meta)
            assert False, "expected ValueError"
        except ValueError:
            pass
    finally:
        for suffix in [".ckpt.json", ".opt.bin", ".sched.json"]:
            p = base + suffix
            if os.path.exists(p):
                os.unlink(p)

def test_checkpoint_magic_mismatch_raises():
    t = Tensor.new([1], requires_grad=False)
    t.numpy_view()[:] = [1.0]
    tmp = tempfile.NamedTemporaryFile(delete=False)
    base = tmp.name
    tmp.close()
    meta = save_checkpoint(base, {"t": t})
    with open(meta, "r", encoding="utf-8") as f:
        blob = json.load(f)
    blob["magic"] = "BADMAGIC"
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(blob, f)
    try:
        try:
            load_checkpoint(meta)
            assert False, "expected ValueError"
        except ValueError:
            pass
    finally:
        for suffix in [".ckpt.json", ".opt.bin", ".sched.json"]:
            p = base + suffix
            if os.path.exists(p):
                os.unlink(p)

def test_checkpoint_version_mismatch_raises():
    t = Tensor.new([1], requires_grad=False)
    t.numpy_view()[:] = [1.0]
    tmp = tempfile.NamedTemporaryFile(delete=False)
    base = tmp.name
    tmp.close()
    meta = save_checkpoint(base, {"t": t})
    with open(meta, "r", encoding="utf-8") as f:
        blob = json.load(f)
    blob["version"] = 999
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(blob, f)
    try:
        try:
            load_checkpoint(meta)
            assert False, "expected ValueError"
        except ValueError:
            pass
    finally:
        for suffix in [".ckpt.json", ".opt.bin", ".sched.json"]:
            p = base + suffix
            if os.path.exists(p):
                os.unlink(p)

def test_checkpoint_checksum_mismatch_raises():
    t = Tensor.new([2], requires_grad=False)
    t.numpy_view()[:] = [1.0, 2.0]
    tmp = tempfile.NamedTemporaryFile(delete=False)
    base = tmp.name
    tmp.close()
    meta = save_checkpoint(base, {"t": t})
    with open(meta, "r", encoding="utf-8") as f:
        blob = json.load(f)
    # tweak data but keep shape/length to trigger checksum mismatch
    blob["tensors"]["t"]["data"][0] = 42.0
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(blob, f)
    try:
        try:
            load_checkpoint(meta)
            assert False, "expected ValueError"
        except ValueError:
            pass
    finally:
        for suffix in [".ckpt.json", ".opt.bin", ".sched.json"]:
            p = base + suffix
            if os.path.exists(p):
                os.unlink(p)


if __name__ == "__main__":
    test_checkpoint_shape_mismatch_raises()
    test_checkpoint_dtype_mismatch_raises()
    test_checkpoint_magic_mismatch_raises()
    test_checkpoint_version_mismatch_raises()
    test_checkpoint_checksum_mismatch_raises()
    print("[test_checkpoint_validation.py] PASS")
