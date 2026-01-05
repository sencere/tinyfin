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

Dropout = nn.Dropout

# simple smoke test
x = Tensor.new([2,3])
arr = x.to_numpy()
arr[:] = [[1.0, 1.0, 1.0],[1.0,1.0,1.0]]
D = Dropout(p=0.5)
D.train()
y = D(x)
# during training, some elements should be zeroed (statistically)
yn = y.to_numpy()
assert yn.shape == (2,3)
D.eval()
y2 = D(x)
# during eval, output equals input
y2n = y2.to_numpy()
assert (y2n == x.to_numpy()).all()
print('[test_dropout.py] PASS')
