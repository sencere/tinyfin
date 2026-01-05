import sys, os, numpy as np
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)
from tinyfin import Tensor

x = Tensor.new([3], requires_grad=True)
x.numpy_view()[:] = [100.0, -100.0, 2.0]

out = x.exp()
vals = out.to_numpy()
assert np.isfinite(vals).all()

loss = out.sum()
loss.backward()
grad = x.grad_numpy()
assert np.isfinite(grad).all()

print('[test_exp_overflow.py] PASS')
