import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from python.tinyfin import lib, Tensor
import numpy as np

shape = [2,3]
# create a base tensor to derive shape
base = Tensor.new(shape)
# create tensors that require grad
a = Tensor.new_like(base, requires_grad=1)
b = Tensor.new_like(base, requires_grad=1)

# fill data
def fill_tensor(t, value):
    ndim = lib.py_tensor_ndim(t._ptr)
    shape = [lib.py_tensor_shape_get(t._ptr, i) for i in range(ndim)]
    size = 1
    for d in shape: size *= d
    data_ptr = lib.py_tensor_data_ptr(t._ptr)
    arr = np.ctypeslib.as_array(data_ptr, shape=(size,))
    arr[:] = value

fill_tensor(a, 6.0)
fill_tensor(b, 2.0)

# forward div
c = a / b
# sum and backward
s = c.sum()
# zero grads
lib.py_zero_grad(a._ptr)
lib.py_zero_grad(b._ptr)
lib.py_backward(s._ptr)

ga = a.grad_numpy()
gb = b.grad_numpy()

print('a.grad sample:', ga.flatten()[:3])
print('b.grad sample:', gb.flatten()[:3])

assert np.allclose(ga, 0.5)
assert np.allclose(gb, -1.5)

# test exp
# clear grads
lib.py_zero_grad(a._ptr)

e = a.exp()
se = e.sum()
lib.py_backward(se._ptr)
ga2 = a.grad_numpy()
print('a.grad after exp sample:', ga2.flatten()[:3])
assert np.allclose(ga2, np.exp(6.0))

print('[python test_div_exp_py] PASS')
