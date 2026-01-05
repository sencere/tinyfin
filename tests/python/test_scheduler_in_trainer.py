import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, SGDOpt, StepLR
from tinyfin.data import TensorDataset, DataLoader
from tinyfin.training import Trainer

x = Tensor.new([2, 1], requires_grad=True)
x.numpy_view()[:] = [[1.0], [2.0]]
target = Tensor.new([2, 1])
target.numpy_view()[:] = [[0.0], [0.0]]

ds = TensorDataset(x, target)
dl = DataLoader(ds, batch_size=1, shuffle=False)

def model(inp):
    return inp

def loss_fn(out, tgt):
    diff = out - tgt
    return (diff * diff).sum()

opt = SGDOpt([x], lr=0.1)
sched = StepLR(opt, step_size=2, gamma=0.5)
trainer = Trainer(model, loss_fn, opt, scheduler=sched)

trainer.train_epoch(dl)
assert abs(opt.get_lr() - 0.05) < 1e-6

print("[test_scheduler_in_trainer.py] PASS")
