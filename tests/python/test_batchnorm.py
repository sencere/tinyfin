import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)
from tinyfin import Tensor
import tinyfin as _tfcore
print('py_batchnorm symbol present on lib:', hasattr(_tfcore.lib, 'py_batchnorm'), bool(getattr(_tfcore.lib, 'py_batchnorm', None)))
import importlib.util
nn_path = os.path.join(root, 'nn.py')
spec = importlib.util.spec_from_file_location('tinyfin_nn', nn_path)
nn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nn)

BatchNorm = nn.BatchNorm

import numpy as np

# create input with two identical rows so per-channel mean/var are known
x = Tensor.new([2,3])
arr = x.numpy_view()
arr[:] = [[0.0, 10.0, 20.0], [0.0, 10.0, 20.0]]

bn = BatchNorm(3)
# ensure initial running stats are as constructor set
rm0 = bn.running_mean.to_numpy().copy()
rv0 = bn.running_var.to_numpy().copy()
print('initial running_mean:', rm0)
print('initial running_var :', rv0)
assert (rm0 == 0.0).all()
assert (rv0 == 1.0).all()

# training: should update running_mean towards batch mean
bn.train()
y = bn(x)
# read running stats
rm = bn.running_mean.to_numpy()
rv = bn.running_var.to_numpy()
# expected running mean = momentum * batch_mean (momentum default 0.1)
expected_rm = np.array([0.0, 1.0, 2.0], dtype=np.float32)
print('running mean after train:', rm, ' expected:', expected_rm)
assert np.allclose(rm, expected_rm, atol=1e-6)
# running var should move from 1.0 towards batch var (which is 0.0 here)
expected_rv = (1.0 - bn.momentum) * 1.0 + bn.momentum * 0.0
assert np.allclose(rv, np.array([expected_rv]*3, dtype=np.float32), atol=1e-6)

# eval: outputs should use running stats
bn.eval()
z = bn(x)
zn = z.to_numpy()
# compute expected using running stats
eps = bn.eps
inv = 1.0 / np.sqrt(rv + eps)
expected = (arr - rm) * inv
assert np.allclose(zn, expected, atol=1e-5)

print('[test_batchnorm.py] PASS')
