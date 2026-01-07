import importlib.util
import os
import ctypes
_tf_path = os.path.join(os.path.dirname(__file__), 'tinyfin.py')
_spec = importlib.util.spec_from_file_location('tinyfin_core', _tf_path)
_tf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tf)
Tensor = _tf.Tensor


class Parameter:
    """Lightweight wrapper for a `Tensor` that represents a trainable parameter."""
    def __init__(self, tensor):
        # Expect `tensor` to be a tinyfin.Tensor instance (possibly created with requires_grad=1)
        self.tensor = tensor

    def zero_grad(self):
        self.tensor.zero_grad()

    def to_device(self, device):
        return Parameter(self.tensor.to_device(device))


class Module:
    def __init__(self):
        self._training = True

    def parameters(self):
        """Recursively yield Parameter instances registered as attributes."""
        for name, val in self.__dict__.items():
            if isinstance(val, Parameter):
                yield val
            elif isinstance(val, Module):
                for p in val.parameters():
                    yield p

    def modules(self):
        for name, val in self.__dict__.items():
            if isinstance(val, Module):
                yield val

    def register_parameter(self, name, param):
        setattr(self, name, param)

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self):
        self._training = True
        for m in self.modules():
            m.train()

    def eval(self):
        self._training = False
        for m in self.modules():
            m.eval()


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.layers = list(modules)
        for i, m in enumerate(self.layers):
            self.register_parameter(f"_{i}", m) if isinstance(m, Parameter) else setattr(self, f"m{i}", m)

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]

    def __call__(self, x):
        out = x
        for layer in self.layers:
            # layers are Modules (or callables) that accept a Tensor
            out = layer(out)
        return out


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # create parameter tensors that require grad
        w_base = Tensor.new([in_features, out_features])
        b_base = Tensor.new([out_features])
        w = Tensor.new_like(w_base, 1)
        b = Tensor.new_like(b_base, 1)
        self.register_parameter('weight', Parameter(w))
        self.register_parameter('bias', Parameter(b))

    def forward(self, x):
        # x @ W + b  (W shape: IN x OUT)
        W = self.weight.tensor
        b = self.bias.tensor
        x_shape = x.shape()
        w_shape = W.shape()
        if len(x_shape) < 1:
            raise ValueError(f"linear expects input with last-dim features, got shape {x_shape}")
        if x_shape[-1] != w_shape[0]:
            raise ValueError(f"linear in_features mismatch: expected {w_shape[0]}, got {x_shape[-1]}")
        x_dev = x.get_device() if hasattr(x, "get_device") else 0
        for t in (W, b):
            if hasattr(t, "get_device") and t.get_device() != x_dev:
                raise ValueError("linear parameter device mismatch with input")
        return x.matmul(W) + b

    def __call__(self, x):
        return self.forward(x)


class ReLU(Module):
    def __call__(self, x):
        return x.relu()


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self._training or self.p <= 0.0:
            return x
        import numpy as _np
        # create mask tensor like x
        mask = Tensor.new_like(x)
        arr = mask.to_numpy()
        # random mask: 1 with prob (1-p), scaled by 1/(1-p)
        rand = _np.random.rand(*arr.shape)
        m = (rand >= self.p).astype(_np.float32) / (1.0 - self.p)
        arr[:] = m
        return x * mask

    def __call__(self, x):
        return self.forward(x)


class BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.num_features = int(num_features)
        # parameters
        if self.affine:
            # create param tensors that require grad and initialize
            g = Tensor.new([num_features])
            b = Tensor.new([num_features])
            # initialize underlying buffers via numpy_view (no-copy)
            g.numpy_view()[:] = 1.0
            b.numpy_view()[:] = 0.0
            self.register_parameter('gamma', Parameter(Tensor.new_like(g, 1)))
            self.register_parameter('beta', Parameter(Tensor.new_like(b, 1)))
            # copy initialized values into params
            self.gamma.tensor.numpy_view()[:] = g.numpy_view()
            self.beta.tensor.numpy_view()[:] = b.numpy_view()
        else:
            self.gamma = None
            self.beta = None

        # running stats (buffers)
        self.running_mean = Tensor.new([num_features])
        self.running_var = Tensor.new([num_features])
        # initialize running buffers with no-copy view
        self.running_mean.numpy_view()[:] = 0.0
        self.running_var.numpy_view()[:] = 1.0

    def forward(self, x):
        x_shape = x.shape()
        if len(x_shape) < 2:
            raise ValueError(f"batchnorm expects input with channel dimension, got shape {x_shape}")
        if x_shape[1] != self.num_features:
            raise ValueError(f"batchnorm channel mismatch: expected {self.num_features}, got {x_shape[1]}")
        x_dev = x.get_device() if hasattr(x, "get_device") else 0
        for t in (self.running_mean, self.running_var):
            if hasattr(t, "get_device") and t.get_device() != x_dev:
                raise ValueError("batchnorm running stats device mismatch with input")
        if self.affine:
            for p in (self.gamma.tensor, self.beta.tensor):
                if hasattr(p, "get_device") and p.get_device() != x_dev:
                    raise ValueError("batchnorm affine params device mismatch with input")
        # call C batchnorm op if available
        tfcore = __import__('tinyfin')
        # Try the loaded library first; if the python wrapper symbol is unavailable
        # (sometimes ctypes exposes it as None), try reopening the shared object to
        # obtain a callable function pointer.
        lib = tfcore.lib
        fn = getattr(lib, 'py_batchnorm', None)
        if not callable(fn):
            # reopen library to get fresh symbol
            try:
                lib2 = ctypes.CDLL(lib._name)
                fn = getattr(lib2, 'py_batchnorm', None)
                if fn:
                    fn.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float)
                    fn.restype = ctypes.c_void_p
            except Exception:
                fn = None

        if callable(fn):
            gptr = self.gamma.tensor._ptr if self.affine else ctypes.c_void_p(0)
            bptr = self.beta.tensor._ptr if self.affine else ctypes.c_void_p(0)
            ptr = fn(x._ptr, gptr, bptr, ctypes.c_float(self.eps), int(bool(self._training)), self.running_mean._ptr, self.running_var._ptr, ctypes.c_float(self.momentum))
            return Tensor(ptr)
        # fallback: use naive Python implementation
        # (not implemented here)
        raise RuntimeError('batchnorm C op not available')

    def __call__(self, x):
        return self.forward(x)
