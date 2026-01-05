import sys, os, tempfile

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, SGDOpt
from tinyfin.scheduler import StepLR, ExponentialLR, LinearWarmupLR


def make_param(v):
    t = Tensor.new([1], requires_grad=True)
    t.numpy_view()[:] = [v]
    return t


def test_step_lr_state_roundtrip():
    p = make_param(1.0)
    opt = SGDOpt([p], lr=0.1)
    sch = StepLR(opt, step_size=2, gamma=0.5)
    sch.step()  # step 1, lr still 0.1
    sch.step()  # step 2, lr should decay to 0.05
    sch.step()  # step 3, lr stays 0.05

    tmp = tempfile.NamedTemporaryFile(delete=False)
    path = tmp.name
    tmp.close()
    sch.save_state(path)

    p2 = make_param(1.0)
    opt2 = SGDOpt([p2], lr=1.0)  # will be overwritten by scheduler load
    sch2 = StepLR(opt2, step_size=2, gamma=0.5)
    sch2.load_state(path)
    os.unlink(path)

    sch2.step()  # this is step 4 overall, should decay again
    assert abs(opt2.get_lr() - 0.025) < 1e-6


def test_exponential_lr_state_roundtrip():
    p = make_param(1.0)
    opt = SGDOpt([p], lr=0.2)
    sch = ExponentialLR(opt, gamma=0.9)
    sch.step(); sch.step()  # lr = 0.2 * 0.9 * 0.9 = 0.162
    tmp = tempfile.NamedTemporaryFile(delete=False)
    path = tmp.name
    tmp.close()
    sch.save_state(path)

    p2 = make_param(1.0)
    opt2 = SGDOpt([p2], lr=1.0)
    sch2 = ExponentialLR(opt2, gamma=0.5)  # gamma will be overwritten
    sch2.load_state(path)
    os.unlink(path)
    sch2.step()
    assert abs(opt2.get_lr() - (0.162 * 0.9)) < 1e-6


def test_linear_warmup_state_roundtrip():
    p = make_param(1.0)
    opt = SGDOpt([p], lr=0.0)
    sch = LinearWarmupLR(opt, warmup_steps=4, target_lr=0.2)
    sch.step(); sch.step()  # step_idx=2, lr=0.1
    tmp = tempfile.NamedTemporaryFile(delete=False)
    path = tmp.name
    tmp.close()
    sch.save_state(path)

    p2 = make_param(1.0)
    opt2 = SGDOpt([p2], lr=1.0)
    sch2 = LinearWarmupLR(opt2, warmup_steps=5, target_lr=0.5)  # overwritten
    sch2.load_state(path)
    os.unlink(path)
    sch2.step()
    # After load, step_idx=3 -> lr should be 0.15 with target 0.2 and warmup_steps=4
    assert abs(opt2.get_lr() - 0.15) < 1e-6


if __name__ == "__main__":
    test_step_lr_state_roundtrip()
    test_exponential_lr_state_roundtrip()
    test_linear_warmup_state_roundtrip()
    print("[test_scheduler_state.py] PASS")
