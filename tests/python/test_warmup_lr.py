import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, SGDOpt, LinearWarmupLR

t = Tensor.new([1], requires_grad=True)
opt = SGDOpt([t], lr=0.0)
sched = LinearWarmupLR(opt, warmup_steps=2, target_lr=0.1)

sched.step()
assert abs(opt.get_lr() - 0.05) < 1e-6
sched.step()
assert abs(opt.get_lr() - 0.1) < 1e-6

print("[test_warmup_lr.py] PASS")
