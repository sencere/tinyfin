import sys
import os
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

import tinyfin
Tensor = tinyfin.Tensor
lib = tinyfin.lib
DEVICE_CPU = 0
DEVICE_GPU = 1

# create tensor and set values via the C data pointer
t = Tensor.new([2,2])
data_ptr = lib.py_tensor_data_ptr(t._ptr)
import numpy as _np
arr = _np.ctypeslib.as_array(data_ptr, shape=(t.shape()[0]*t.shape()[1],))
for i in range(arr.size):
    arr[i] = float(i+1)

# move to GPU (logical stub)
tg = t.to_device(DEVICE_GPU)
assert tg.get_device() == DEVICE_GPU
# values should be preserved
npv = tg.to_numpy()
for i in range(npv.size):
    assert npv.flat[i] == float(i+1)
print('[test_device_move.py] PASS')
