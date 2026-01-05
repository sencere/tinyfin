import sys, os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)
from tinyfin import Tensor
import tinyfin

# run a matmul to record some C-side profiling
A = Tensor.new([64,64])
B = Tensor.new([64,64])
A.numpy_view()[:] = 1.0
B.numpy_view()[:] = 2.0
C = A.matmul(B)

s = tinyfin.cprofiler_summary()
print('[test_cprofiler.py] profiler summary:\n', s)
print('[test_cprofiler.py] PASS')
