import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import set_mixed_precision, autocast


def test_mixed_precision_toggle():
    assert set_mixed_precision(False) is False
    assert set_mixed_precision(True) is True
    assert set_mixed_precision(False) is False


def test_autocast_context_restores_state():
    set_mixed_precision(False)
    with autocast(True):
        assert set_mixed_precision(True) is True  # already enabled
    # after context, should be restored to False
    assert set_mixed_precision(False) is False


if __name__ == "__main__":
    test_mixed_precision_toggle()
    test_autocast_context_restores_state()
    print("[test_mixed_precision_stub.py] PASS")
