import sys, os, tempfile, numpy as np

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, AdamOpt, RMSPropOpt


def make_params():
    p1 = Tensor.new([2], requires_grad=True); p1.numpy_view()[:] = [1.0, -1.0]
    p2 = Tensor.new([2], requires_grad=True); p2.numpy_view()[:] = [0.5, 0.5]
    return p1, p2


def run_step(opt, params):
    for p in params:
        p.zero_grad()
    loss = (params[0] + params[1]).sum()
    loss.backward()
    opt.step()
    return [p.to_numpy().copy() for p in params]


def test_adam_state_roundtrip():
    p1, p2 = make_params()
    opt = AdamOpt([p1, p2], lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)
    state_after_first = run_step(opt, (p1, p2))

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp.name
    tmp.close()
    opt.save_state(tmp_path)

    # fresh params resumed from saved weights/state
    q1, q2 = make_params()
    q1.numpy_view()[:] = state_after_first[0]
    q2.numpy_view()[:] = state_after_first[1]
    opt2 = AdamOpt([q1, q2], lr=0.5, beta1=0.5, beta2=0.5, eps=1e-4)  # values should be overwritten by load
    opt2.load_state(tmp_path)
    os.unlink(tmp_path)

    next_state_orig = run_step(opt, (p1, p2))
    next_state_loaded = run_step(opt2, (q1, q2))

    assert abs(opt2.get_lr() - 0.01) < 1e-6
    assert np.allclose(next_state_orig[0], next_state_loaded[0])
    assert np.allclose(next_state_orig[1], next_state_loaded[1])


def test_rmsprop_state_roundtrip():
    p1, p2 = make_params()
    opt = RMSPropOpt([p1, p2], lr=0.01, alpha=0.9, eps=1e-6, weight_decay=0.01)
    state_after_first = run_step(opt, (p1, p2))

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp.name
    tmp.close()
    opt.save_state(tmp_path)

    q1, q2 = make_params()
    q1.numpy_view()[:] = state_after_first[0]
    q2.numpy_view()[:] = state_after_first[1]
    opt2 = RMSPropOpt([q1, q2], lr=0.5, alpha=0.5, eps=1e-4, weight_decay=0.5)
    opt2.load_state(tmp_path)
    os.unlink(tmp_path)

    next_state_orig = run_step(opt, (p1, p2))
    next_state_loaded = run_step(opt2, (q1, q2))

    assert abs(opt2.get_lr() - 0.01) < 1e-6
    assert np.allclose(next_state_orig[0], next_state_loaded[0])
    assert np.allclose(next_state_orig[1], next_state_loaded[1])


if __name__ == "__main__":
    test_adam_state_roundtrip()
    test_rmsprop_state_roundtrip()
    print("[test_optimizer_state_adam_rmsprop.py] PASS")
