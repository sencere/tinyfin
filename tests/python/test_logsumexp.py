import sys, os, numpy as np
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)
from tinyfin import Tensor

t_base = Tensor.new([4])
t_base.numpy_view()[:] = [1.0, 2.0, -3.0, 4.0]
# create a requires_grad copy
t = Tensor.new_like(t_base, requires_grad=1)
t.numpy_view()[:] = t_base.to_numpy()

out = t.logsumexp()
np_out = np.log(np.exp(t.to_numpy()).sum())
assert abs(out.to_numpy().item() - np_out) < 1e-6

# Gradient should be softmax of inputs
out.backward()
grad = t.grad_numpy()
sm = np.exp(t.to_numpy()) / np.exp(t.to_numpy()).sum()
assert np.allclose(grad, sm, atol=1e-6)

print('[test_logsumexp.py] PASS')
