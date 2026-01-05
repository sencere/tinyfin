import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, SGDOpt
from tinyfin.training import Trainer


def simple_model(scale):
    def fn(x):
        s = Tensor.new([1], requires_grad=False)
        s.numpy_view()[:] = [scale]
        return x * s
    return fn


def make_batch(val):
    x = Tensor.new([1], requires_grad=True)
    x.numpy_view()[:] = [val]
    target = Tensor.new([1], requires_grad=False)
    target.numpy_view()[:] = [0.0]
    return x, target


def l2_loss(out, target):
    diff = out - target
    return (diff * diff).sum()


def test_trainer_grad_accumulation():
    # Two batches; accumulate over both then step once
    batches = [make_batch(1.0), make_batch(2.0)]
    model = simple_model(scale=1.0)
    params = [b[0] for b in batches]  # treat inputs as params for simplicity
    opt = SGDOpt(params, lr=0.1)
    trainer = Trainer(model, l2_loss, opt, accumulate_steps=2)
    loss = trainer.train_epoch(batches)
    # Only one step executed after two backward passes: grad for each param is 2*x; lr=0.1 -> delta=0.2, 0.4 respectively
    assert abs(params[0].to_numpy()[0] - 0.8) < 1e-4
    assert abs(params[1].to_numpy()[0] - 1.6) < 1e-4
    # loss should be last batch loss
    assert loss > 0


if __name__ == "__main__":
    test_trainer_grad_accumulation()
    print("[test_trainer_grad_accum.py] PASS")
