import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)
from tinyfin import Tensor
import importlib.util
nn_path = os.path.join(root, 'nn.py')
spec = importlib.util.spec_from_file_location('tinyfin_nn', nn_path)
nn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nn)

Sequential = nn.Sequential
Linear = nn.Linear

# build a simple sequential model
m = Sequential(Linear(2,3), Linear(3,1))
# check lengths and indexing
assert len(m) == 2
assert isinstance(m[0], nn.Linear)
# forward pass
x = Tensor.new([1,2])
arr = x.to_numpy()
arr[0,0] = 1.0
arr[0,1] = 2.0
y = m(x)
print('[test_sequential.py] PASS')
