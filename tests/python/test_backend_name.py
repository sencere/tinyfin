import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import backend_name, backend_set
import pytest


def test_backend_default_cpu():
    assert backend_name() == "cpu"


def test_backend_set_unknown_returns_false():
    assert backend_set("nonexistent") is False
    # cpu backend is always available
    assert backend_set("cpu") is True
    assert backend_name() == "cpu"


def test_backend_set_cuda_stub_when_available():
    # With stub registration (default when CUDA is disabled), setting cuda should succeed
    ok = backend_set("cuda")
    if ok:
        assert backend_name() == "cuda"
        # reset to cpu to avoid leaking state to other tests
        assert backend_set("cpu") is True
    else:
        # If cuda backend is not registered (e.g., build without stub), ensure name unchanged
        assert backend_name() == "cpu"


@pytest.mark.parametrize("name", ["opengl", "vulkan"])
def test_backend_set_gl_vk_stub(name):
    ok = backend_set(name)
    if ok:
        assert backend_name() == name
        assert backend_set("cpu") is True
    else:
        assert backend_name() == "cpu"


if __name__ == "__main__":
    test_backend_default_cpu()
    print("[test_backend_name.py] PASS")
