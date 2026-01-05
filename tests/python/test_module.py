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
Module = nn.Module
Parameter = nn.Parameter

class MyModel(Module):
    def __init__(self):
        super().__init__()
        base = Tensor.new([2,2])
        p = Tensor.new_like(base, 1)
        self.register_parameter('w', Parameter(p))


def test_parameters_and_zero_grad():
    m = MyModel()
    params = list(m.parameters())
    assert len(params) == 1
    # ensure zero_grad runs without error
    m.zero_grad()
    # train/eval toggles
    m.eval()
    assert not m._training
    m.train()
    assert m._training

if __name__ == '__main__':
    test_parameters_and_zero_grad()
    print('[test_module.py] PASS')
