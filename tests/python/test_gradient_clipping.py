import sys, os, numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, SGDOpt, AdamOpt


def make_param(val):
    t = Tensor.new([1], requires_grad=True)
    t.numpy_view()[:] = [val]
    return t


def run_step(opt, param, clip_norm):
    loss = (param * param).sum()
    loss.backward()
    opt.step(clip_norm=clip_norm)
    return param.to_numpy()[0]


def test_sgd_clip_scales_grad():
    p = make_param(5.0)
    opt = SGDOpt([p], lr=1.0)
    out = run_step(opt, p, clip_norm=1.0)
    # grad is 10; clipped to norm 1 -> effective grad 1 -> new value 4
    assert abs(out - 4.0) < 1e-4


def test_adam_clip_scales_grad():
    p = make_param(5.0)
    opt = AdamOpt([p], lr=1.0, beta1=0.0, beta2=0.0, eps=1e-8)
    out = run_step(opt, p, clip_norm=1.0)
    # With beta1/beta2 zero, Adam behaves like SGD with eps; expect same clipped step
    assert abs(out - 4.0) < 1e-3


if __name__ == "__main__":
    test_sgd_clip_scales_grad()
    test_adam_clip_scales_grad()
    print("[test_gradient_clipping.py] PASS")
