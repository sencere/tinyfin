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


def test_rnn_shapes():
    rnn = nn.RNN(3, 5)
    inputs = [Tensor.rand(2, 3) for _ in range(4)]
    out, h = rnn(inputs)
    assert out.shape() == [4, 2, 5]
    assert h.shape() == [2, 5]


def test_lstm_shapes():
    lstm = nn.LSTM(3, 5)
    inputs = [Tensor.rand(2, 3) for _ in range(4)]
    out, (h, c) = lstm(inputs)
    assert out.shape() == [4, 2, 5]
    assert h.shape() == [2, 5]
    assert c.shape() == [2, 5]


def test_mingru_shapes():
    mingru = nn.MinGRU(3, 5)
    inputs = [Tensor.rand(2, 3) for _ in range(4)]
    out, h = mingru(inputs)
    assert out.shape() == [4, 2, 5]
    assert h.shape() == [2, 5]


if __name__ == '__main__':
    test_rnn_shapes()
    test_lstm_shapes()
    test_mingru_shapes()
    print('[test_rnn_layers.py] PASS')
