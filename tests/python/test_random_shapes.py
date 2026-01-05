import sys, os, random, numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor


def tensor_from_numpy(arr, requires_grad=False):
    t = Tensor.new(list(arr.shape), requires_grad=requires_grad)
    t.numpy_view()[:] = arr.astype(np.float32)
    return t


def test_broadcast_add_matches_numpy():
    random.seed(0)
    np.random.seed(0)
    for _ in range(20):
        # generate up to 3 dims with sizes 1-3
        shape_a = [random.randint(1, 3) for _ in range(random.randint(1, 3))]
        shape_b = [random.randint(1, 3) for _ in range(random.randint(1, 3))]
        a_np = np.random.randn(*shape_a).astype(np.float32)
        b_np = np.random.randn(*shape_b).astype(np.float32)
        try:
            c_np = a_np + b_np
        except ValueError:
            # numpy broadcasting failed; skip this pair
            continue
        a = tensor_from_numpy(a_np)
        b = tensor_from_numpy(b_np)
        c = a + b
        assert c.to_numpy().shape == c_np.shape
        assert np.allclose(c.to_numpy(), c_np, atol=1e-4)


def test_broadcast_sub_matches_numpy():
    random.seed(4)
    np.random.seed(4)
    for _ in range(20):
        shape_a = [random.randint(1, 3) for _ in range(random.randint(1, 3))]
        shape_b = [random.randint(1, 3) for _ in range(random.randint(1, 3))]
        a_np = np.random.randn(*shape_a).astype(np.float32)
        b_np = np.random.randn(*shape_b).astype(np.float32)
        try:
            c_np = a_np - b_np
        except ValueError:
            continue
        a = tensor_from_numpy(a_np)
        b = tensor_from_numpy(b_np)
        c = a - b
        assert c.to_numpy().shape == c_np.shape
        assert np.allclose(c.to_numpy(), c_np, atol=1e-4)


def test_broadcast_mul_matches_numpy():
    random.seed(2)
    np.random.seed(2)
    for _ in range(20):
        shape_a = [random.randint(1, 3) for _ in range(random.randint(1, 3))]
        shape_b = [random.randint(1, 3) for _ in range(random.randint(1, 3))]
        a_np = np.random.randn(*shape_a).astype(np.float32)
        b_np = np.random.randn(*shape_b).astype(np.float32)
        try:
            c_np = a_np * b_np
        except ValueError:
            continue
        a = tensor_from_numpy(a_np)
        b = tensor_from_numpy(b_np)
        c = a * b
        assert c.to_numpy().shape == c_np.shape
        assert np.allclose(c.to_numpy(), c_np, atol=1e-4)


def test_matmul_matches_numpy():
    random.seed(1)
    np.random.seed(1)
    for _ in range(10):
        m = random.randint(1, 4)
        k = random.randint(1, 4)
        n = random.randint(1, 4)
        a_np = np.random.randn(m, k).astype(np.float32)
        b_np = np.random.randn(k, n).astype(np.float32)
        c_np = a_np @ b_np
        a = tensor_from_numpy(a_np)
        b = tensor_from_numpy(b_np)
        c = a.matmul(b)
        assert c.to_numpy().shape == c_np.shape
        assert np.allclose(c.to_numpy(), c_np, atol=1e-4)


def test_div_matches_numpy_with_broadcast():
    random.seed(3)
    np.random.seed(3)
    for _ in range(20):
        shape_a = [random.randint(1, 3) for _ in range(random.randint(1, 3))]
        shape_b = [random.randint(1, 3) for _ in range(random.randint(1, 3))]
        a_np = np.random.randn(*shape_a).astype(np.float32)
        b_np = np.random.randn(*shape_b).astype(np.float32) + 1e-3  # avoid zeros
        try:
            c_np = a_np / b_np
        except ValueError:
            continue
        a = tensor_from_numpy(a_np)
        b = tensor_from_numpy(b_np)
        c = a / b
        assert c.to_numpy().shape == c_np.shape
        assert np.allclose(c.to_numpy(), c_np, atol=1e-4)


def test_exp_matches_numpy():
    random.seed(5)
    np.random.seed(5)
    for _ in range(10):
        shape = [random.randint(1, 3) for _ in range(random.randint(1, 3))]
        a_np = np.random.randn(*shape).astype(np.float32)
        a = tensor_from_numpy(a_np)
        c = a.exp()
        assert np.allclose(c.to_numpy(), np.exp(a_np), atol=1e-4, rtol=1e-4)


def test_log_matches_numpy_positive_inputs():
    random.seed(6)
    np.random.seed(6)
    for _ in range(10):
        shape = [random.randint(1, 3) for _ in range(random.randint(1, 3))]
        a_np = np.random.rand(*shape).astype(np.float32) + 1e-3  # strictly positive
        a = tensor_from_numpy(a_np)
        c = a.log()
        assert np.allclose(c.to_numpy(), np.log(a_np), atol=1e-4, rtol=1e-4)


def test_softmax_matches_numpy():
    random.seed(7)
    np.random.seed(7)
    for _ in range(10):
        shape = [random.randint(1, 3), random.randint(1, 4)]
        a_np = np.random.randn(*shape).astype(np.float32)
        a = tensor_from_numpy(a_np)
        c = a.softmax()
        # numpy softmax along last dim
        exp = np.exp(a_np - np.max(a_np, axis=-1, keepdims=True))
        softmax_np = exp / exp.sum(axis=-1, keepdims=True)
        assert np.allclose(c.to_numpy(), softmax_np, atol=1e-4, rtol=1e-4)


def test_log_softmax_matches_numpy():
    random.seed(8)
    np.random.seed(8)
    for _ in range(10):
        shape = [random.randint(1, 3), random.randint(1, 4)]
        a_np = np.random.randn(*shape).astype(np.float32)
        a = tensor_from_numpy(a_np)
        c = a.log_softmax()
        exp = np.exp(a_np - np.max(a_np, axis=-1, keepdims=True))
        softmax_np = exp / exp.sum(axis=-1, keepdims=True)
        log_softmax_np = np.log(softmax_np)
        assert np.allclose(c.to_numpy(), log_softmax_np, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_broadcast_add_matches_numpy()
    test_matmul_matches_numpy()
    print("[test_random_shapes.py] PASS")
