import sys
import os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor
import importlib.util
nn_path = os.path.join(root, 'nn.py')
spec = importlib.util.spec_from_file_location('tinyfin_nn', nn_path)
nn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nn)

Linear = nn.Linear

# Simple smoke test for Linear
x = Tensor.new([1,2])
arr = x.to_numpy()
arr[0,0] = 1.0
arr[0,1] = 2.0
# create linear layer
l = Linear(2,3)
y = l(x)
# shape should be (1,3)
assert tuple(y.to_numpy().shape) == (1,3)
print('[test_linear.py] PASS')
