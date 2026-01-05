import sys, os, numpy as np
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)
from tinyfin import Tensor

t = Tensor.new([2], requires_grad=True)
t.numpy_view()[:] = [0.0, 1.0]

out = t.log()
vals = out.to_numpy()
assert np.isfinite(vals).all()
assert vals[0] > -1e6  # clamped

loss = out.sum()
loss.backward()
grad = t.grad_numpy()
assert np.allclose(grad, [0.0, 1.0])

print('[test_log_epsilon.py] PASS')
