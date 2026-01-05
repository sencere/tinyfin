import sys, os, numpy as np
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)
from tinyfin import Tensor

# create with requires_grad via constructor
t = Tensor.new([3], requires_grad=True)
t.numpy_view()[:] = [1.0, 2.0, 3.0]
assert t.requires_grad()

out = t.sum()
out.backward()
grad = t.grad_numpy()
assert np.allclose(grad, np.ones_like(grad))

# disable requires_grad and ensure grad buffer cleared
t.requires_grad_(False)
assert not t.requires_grad()
assert t.grad_numpy() is None

# re-enable and recompute grad
t.requires_grad_(True)
out2 = t.sum()
out2.backward()
grad2 = t.grad_numpy()
assert np.allclose(grad2, np.ones_like(grad2))

print('[test_requires_grad.py] PASS')
