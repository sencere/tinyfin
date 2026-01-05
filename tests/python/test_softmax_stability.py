import sys, os, numpy as np
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)
from tinyfin import Tensor

logits = Tensor.new([2, 3], requires_grad=True)
logits.numpy_view()[:] = [[100.0, 0.0, -100.0], [50.0, 50.0, 50.0]]

probs = logits.softmax()
vals = probs.to_numpy()
assert np.isfinite(vals).all()
assert np.allclose(vals.sum(axis=1), 1.0)

loss = probs.logsumexp()  # simple scalar
loss.backward()
assert logits.grad_numpy() is not None
assert np.isfinite(logits.grad_numpy()).all()

print('[test_softmax_stability.py] PASS')
