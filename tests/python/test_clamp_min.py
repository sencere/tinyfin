import sys, os, numpy as np
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)
from tinyfin import Tensor

base = Tensor.new([4], requires_grad=True)
base.numpy_view()[:] = [-2.0, -1.0, 0.5, 3.0]

out = base.clamp_min(0.1)
expected = np.maximum(base.to_numpy(), 0.1)
assert np.allclose(out.to_numpy(), expected)

# backprop using sum to get scalar; clamped entries should get zero grad
loss = out.sum()
loss.backward()
grad = base.grad_numpy()
assert np.allclose(grad, [0.0, 0.0, 1.0, 1.0])

print('[test_clamp_min.py] PASS')
