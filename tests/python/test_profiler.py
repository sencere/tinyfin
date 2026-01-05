import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)
from tinyfin import Tensor, Profiler

# simple profiler usage smoke test
x = Tensor.new([100,100])
with Profiler() as p:
    y = x * x
    p.mark('mul')

s = p.summary()
print('[test_profiler.py] summary:\n', s)
print('[test_profiler.py] PASS')
