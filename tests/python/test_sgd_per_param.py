import sys, os, math
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, SGDOpt

# single param with explicit lr
a = Tensor.new([2], requires_grad=True)
a.numpy_view()[:] = [1.0, 1.0]
opt = SGDOpt([a], lr=0.1, weight_decay=0.0)
# build loss to generate grads = [1,1]
loss = a.sum()
loss.backward()
opt.set_lr(0.2)
opt.step()
assert abs(a.to_numpy()[0] - (1.0 - 0.2)) < 1e-6

# per-param override (two params) grads = 1
p1 = Tensor.new([1], requires_grad=True); p1.numpy_view()[:] = [1.0]
p2 = Tensor.new([1], requires_grad=True); p2.numpy_view()[:] = [1.0]
loss2 = (p1 + p2).sum()
loss2.backward()
opt2 = SGDOpt([p1, p2], lr=0.1, weight_decay=0.0, lr_params=[0.1, 0.2], weight_decay_params=[0.0, 0.1])
opt2.step()
v1, v2 = p1.to_numpy()[0], p2.to_numpy()[0]
assert abs(v1 - 0.9) < 1e-6  # lr 0.1
assert v2 < v1  # larger lr + wd

print("[test_sgd_per_param.py] PASS")
