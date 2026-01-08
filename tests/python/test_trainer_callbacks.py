import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, SGDOpt
from tinyfin.data import TensorDataset, DataLoader
from tinyfin.training import Callback, Trainer
from tinyfin.callbacks import LoggingCallback

class Recorder(Callback):
    def __init__(self):
        self.batch_starts = 0
        self.batch_ends = 0
        self.epochs_start = 0
        self.epochs_end = 0
    def on_epoch_start(self, epoch): self.epochs_start += 1
    def on_epoch_end(self, epoch): self.epochs_end += 1
    def on_batch_start(self, step, batch): self.batch_starts += 1
    def on_batch_end(self, step, loss): self.batch_ends += 1

# simple dataset and model
x = Tensor.new([4, 1], requires_grad=True)
x.numpy_view()[:] = [[1.0], [2.0], [3.0], [4.0]]
target = Tensor.new([4, 1])
target.numpy_view()[:] = [[0.0], [0.0], [0.0], [0.0]]
ds = TensorDataset(x, target)
dl = DataLoader(ds, batch_size=2, shuffle=False)

def model(inp):
    return inp  # identity

def loss_fn(out, tgt):
    # simple L2
    diff = out - tgt
    return (diff * diff).sum()

rec = Recorder()
opt = SGDOpt([x], lr=0.0)  # lr not used since model is identity; just to satisfy API
logs = []
trainer = Trainer(model, loss_fn, opt, callbacks=[rec, LoggingCallback(log_every=1, sink=logs.append)])
trainer.train_epoch(dl)

assert rec.batch_starts == len(dl)
assert rec.batch_ends == len(dl)
assert rec.epochs_start == 1
assert rec.epochs_end == 1
assert len(logs) == len(dl)

print("[test_trainer_callbacks.py] PASS")
