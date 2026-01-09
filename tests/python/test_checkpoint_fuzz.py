import os
import sys
import tempfile

import numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, "..", "..", "python"))
sys.path.insert(0, root)

from tinyfin import Tensor
from tinyfin.utils import save_checkpoint, load_checkpoint


def _rand_shape(rng, max_dims=3, max_size=5):
    ndim = rng.integers(0, max_dims + 1)
    if ndim == 0:
        return []
    return [int(rng.integers(1, max_size + 1)) for _ in range(ndim)]


def test_checkpoint_fuzz_roundtrip():
    rng = np.random.default_rng(123)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(20):
            shape = _rand_shape(rng)
            if shape:
                data = rng.standard_normal(np.prod(shape), dtype=np.float32).reshape(shape)
            else:
                data = rng.standard_normal(1, dtype=np.float32).reshape([])

            t = Tensor.new(shape, requires_grad=bool(i % 2))
            t.numpy_view()[:] = data

            base = os.path.join(tmp, f"ckpt_{i}")
            meta_path = save_checkpoint(base, {"t": t}, metadata={"seed": i})
            tensors, meta = load_checkpoint(meta_path)

            assert meta.get("seed") == i
            out = tensors["t"].to_numpy()
            np.testing.assert_allclose(out, data, rtol=1e-6, atol=1e-7)

