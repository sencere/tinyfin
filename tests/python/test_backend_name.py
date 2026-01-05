import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import backend_name, backend_set


def test_backend_default_cpu():
    assert backend_name() == "cpu"


def test_backend_set_unknown_returns_false():
    assert backend_set("nonexistent") is False
    # cpu backend is always available
    assert backend_set("cpu") is True
    assert backend_name() == "cpu"


if __name__ == "__main__":
    test_backend_default_cpu()
    print("[test_backend_name.py] PASS")
