import sys, os, math
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)
from tinyfin import Tensor

t = Tensor.new([3])
t.numpy_view()[:] = [1.0, 2.0, 3.0]
assert not t.has_nan_or_inf()

# introduce inf
t.numpy_view()[1] = math.inf
assert t.has_nan_or_inf()

# reset and introduce nan
t.numpy_view()[:] = [0.0, 0.0, 0.0]
t.numpy_view()[2] = math.nan
assert t.has_nan_or_inf()

print('[test_nan_inf.py] PASS')
