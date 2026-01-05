import sys, os, numpy as np
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)
from tinyfin import Tensor

a = Tensor.new([3], requires_grad=True)
a.numpy_view()[:] = [1.0, -2.0, 3.0]
b = Tensor.new([3], requires_grad=True)
b.numpy_view()[:] = [0.0, 1e-14, -4.0]

out = a / b
vals = out.to_numpy()
assert np.isfinite(vals).all()

loss = out.sum()
loss.backward()
grad_a = a.grad_numpy()
grad_b = b.grad_numpy()

assert np.isfinite(grad_a).all()
assert np.isfinite(grad_b).all()

print('[test_div_epsilon.py] PASS')
