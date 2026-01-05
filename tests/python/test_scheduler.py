import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, SGDOpt, StepLR

t = Tensor.new([1], requires_grad=True)
opt = SGDOpt([t], lr=0.1)
sched = StepLR(opt, step_size=2, gamma=0.5)

assert abs(sched.get_lr() - 0.1) < 1e-6
sched.step()  # step 1, no decay
assert abs(sched.get_lr() - 0.1) < 1e-6
sched.step()  # step 2, decay
assert abs(sched.get_lr() - 0.05) < 1e-6

print("[test_scheduler.py] PASS")
