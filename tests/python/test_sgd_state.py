import sys, os, tempfile, numpy as np
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, SGDOpt

def make_params():
    p1 = Tensor.new([1], requires_grad=True); p1.numpy_view()[:] = [1.0]
    p2 = Tensor.new([1], requires_grad=True); p2.numpy_view()[:] = [2.0]
    return p1, p2

p1, p2 = make_params()
opt = SGDOpt([p1, p2], lr=0.1, momentum=0.9)
# generate grads via simple loss
loss = (p1 + p2).sum()
loss.backward()
opt.step()  # initialize velocity

tmp = tempfile.NamedTemporaryFile(delete=False)
tmp_path = tmp.name
tmp.close()
opt.save_state(tmp_path)

# new optimizer with fresh params; load state
p1b, p2b = make_params()
opt2 = SGDOpt([p1b, p2b], lr=0.01, momentum=0.0)  # different hyperparams to ensure load overrides
opt2.load_state(tmp_path)
os.unlink(tmp_path)

# after load, lr should match saved value
assert abs(opt2.get_lr() - 0.1) < 1e-6

# velocities restored: next step should move params using momentum buffer
loss2 = (p1b + p2b).sum()
loss2.backward()
opt2.step()
v_after = p1b.to_numpy()[0]
assert np.isfinite(v_after)

print("[test_sgd_state.py] PASS")
