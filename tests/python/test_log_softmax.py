import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)
from tinyfin import Tensor
import numpy as np
import importlib.util
nn_path = os.path.join(root, 'nn.py')
spec = importlib.util.spec_from_file_location('tinyfin_nn', nn_path)
nn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nn)

# Test log_softmax vs softmax
x = Tensor.new([2,3])
arr = x.to_numpy()
arr[:] = np.array([0.1, 2.0, -1.0, 0.3, 0.2, -0.5]).reshape(2,3)
# create requires_grad tensor with same shape
x_req = Tensor.new_like(x, 1)
arr_req = x_req.to_numpy()
arr_req[:] = arr
p = x_req.log_softmax()
import numpy as np
# exp(log_softmax) == softmax
p_exp = np.exp(p.to_numpy())
s = x_req.softmax() if hasattr(Tensor, 'softmax') else None
if s is not None:
    s_np = s.to_numpy()
    assert np.allclose(p_exp, s_np, atol=1e-6)
print('[test_log_softmax.py] PASS')
