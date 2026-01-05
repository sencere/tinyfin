import sys
import os
# Add local python package to path
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import no_grad
from tinyfin import Tensor

# ensure the shared lib is loaded via Tensor import (it loads at import time)

def test_nested_no_grad():
    # simple nested usage should not raise and should restore previous state
    try:
        with no_grad():
            with no_grad():
                pass
    except Exception as e:
        raise AssertionError("Nested no_grad raised: %s" % e)

if __name__ == '__main__':
    test_nested_no_grad()
    print('[test_no_grad.py] PASS')
