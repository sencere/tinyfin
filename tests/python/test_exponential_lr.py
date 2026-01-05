import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, SGDOpt, ExponentialLR

t = Tensor.new([1], requires_grad=True)
opt = SGDOpt([t], lr=0.1)
sched = ExponentialLR(opt, gamma=0.5)

sched.step()
assert abs(opt.get_lr() - 0.05) < 1e-6
sched.step()
assert abs(opt.get_lr() - 0.025) < 1e-6

print("[test_exponential_lr.py] PASS")
