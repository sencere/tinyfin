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


def _device_of(t):
    return t.get_device() if hasattr(t, "get_device") else 0


def _const_like(value, like):
    t = Tensor.new([1])
    if hasattr(like, "get_device") and t.get_device() != like.get_device():
        t.set_device(like.get_device())
    t.numpy_view()[:] = float(value)
    return t


def _zeros_like_shape(shape, like):
    t = Tensor.new(list(shape))
    if hasattr(like, "get_device") and t.get_device() != like.get_device():
        t.set_device(like.get_device())
    t.numpy_view()[:] = 0.0
    return t


def _sigmoid(x):
    one = _const_like(1.0, x)
    return one / (one + x.neg().exp())


def _tanh(x):
    ex = x.exp()
    enx = x.neg().exp()
    return (ex - enx) / (ex + enx)


def _stack_sequence(seq):
    if not seq:
        raise ValueError("sequence is empty")
    out = seq[0].unsqueeze(0)
    for t in seq[1:]:
        out = out.concat(t.unsqueeze(0), axis=0)
    return out


def _sequence_from_tensor(x):
    shape = x.shape()
    if len(shape) == 2:
        return [x]
    if len(shape) == 3:
        steps = shape[0]
        return [x.slice(0, i, i + 1).squeeze(0) for i in range(steps)]
    raise ValueError(f"expected [B,I] or [T,B,I] input, got shape {shape}")


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, nonlinearity="tanh", bias=True):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.nonlinearity = str(nonlinearity).lower()
        self.bias = bool(bias)

        w_ih = Tensor.xavier_uniform([self.input_size, self.hidden_size], requires_grad=True)
        w_hh = Tensor.xavier_uniform([self.hidden_size, self.hidden_size], requires_grad=True)
        self.register_parameter("weight_ih", Parameter(w_ih))
        self.register_parameter("weight_hh", Parameter(w_hh))

        if self.bias:
            b_ih = Tensor.new([self.hidden_size], requires_grad=True)
            b_hh = Tensor.new([self.hidden_size], requires_grad=True)
            b_ih.numpy_view()[:] = 0.0
            b_hh.numpy_view()[:] = 0.0
            self.register_parameter("bias_ih", Parameter(b_ih))
            self.register_parameter("bias_hh", Parameter(b_hh))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x, h):
        x_shape = x.shape()
        h_shape = h.shape()
        if len(x_shape) != 2:
            raise ValueError(f"RNNCell expects input [B,I], got {x_shape}")
        if x_shape[1] != self.input_size:
            raise ValueError(f"RNNCell input_size mismatch: expected {self.input_size}, got {x_shape[1]}")
        if len(h_shape) != 2 or h_shape[0] != x_shape[0] or h_shape[1] != self.hidden_size:
            raise ValueError(f"RNNCell hidden shape mismatch: expected [{x_shape[0]},{self.hidden_size}], got {h_shape}")

        x_dev = _device_of(x)
        if _device_of(self.weight_ih.tensor) != x_dev or _device_of(self.weight_hh.tensor) != x_dev:
            raise ValueError("RNNCell parameter device mismatch with input")
        if self.bias:
            if _device_of(self.bias_ih.tensor) != x_dev or _device_of(self.bias_hh.tensor) != x_dev:
                raise ValueError("RNNCell bias device mismatch with input")

        pre = x.matmul(self.weight_ih.tensor) + h.matmul(self.weight_hh.tensor)
        if self.bias:
            pre = pre + self.bias_ih.tensor + self.bias_hh.tensor

        if self.nonlinearity == "tanh":
            return _tanh(pre)
        if self.nonlinearity == "relu":
            return pre.relu()
        raise ValueError(f"unsupported nonlinearity: {self.nonlinearity}")

    def __call__(self, x, h):
        return self.forward(x, h)


class RNN(Module):
    def __init__(self, input_size, hidden_size, nonlinearity="tanh", bias=True):
        super().__init__()
        self.cell = RNNCell(input_size, hidden_size, nonlinearity=nonlinearity, bias=bias)
        self.input_size = self.cell.input_size
        self.hidden_size = self.cell.hidden_size

    def forward(self, inputs, h0=None, return_sequences=True):
        if isinstance(inputs, Tensor):
            inputs = _sequence_from_tensor(inputs)
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("RNN inputs must be a list/tuple of tensors")
        if not inputs:
            raise ValueError("RNN inputs cannot be empty")

        batch = inputs[0].shape()[0]
        if h0 is None:
            h = _zeros_like_shape([batch, self.hidden_size], inputs[0])
        else:
            h = h0

        outputs = []
        for x in inputs:
            h = self.cell(x, h)
            outputs.append(h)

        out = _stack_sequence(outputs) if return_sequences else outputs[-1]
        return out, h

    def __call__(self, inputs, h0=None, return_sequences=True):
        return self.forward(inputs, h0=h0, return_sequences=return_sequences)


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.bias = bool(bias)

        self.register_parameter("weight_ii", Parameter(Tensor.xavier_uniform([self.input_size, self.hidden_size], requires_grad=True)))
        self.register_parameter("weight_if", Parameter(Tensor.xavier_uniform([self.input_size, self.hidden_size], requires_grad=True)))
        self.register_parameter("weight_ig", Parameter(Tensor.xavier_uniform([self.input_size, self.hidden_size], requires_grad=True)))
        self.register_parameter("weight_io", Parameter(Tensor.xavier_uniform([self.input_size, self.hidden_size], requires_grad=True)))

        self.register_parameter("weight_hi", Parameter(Tensor.xavier_uniform([self.hidden_size, self.hidden_size], requires_grad=True)))
        self.register_parameter("weight_hf", Parameter(Tensor.xavier_uniform([self.hidden_size, self.hidden_size], requires_grad=True)))
        self.register_parameter("weight_hg", Parameter(Tensor.xavier_uniform([self.hidden_size, self.hidden_size], requires_grad=True)))
        self.register_parameter("weight_ho", Parameter(Tensor.xavier_uniform([self.hidden_size, self.hidden_size], requires_grad=True)))

        if self.bias:
            self.register_parameter("bias_i", Parameter(Tensor.new([self.hidden_size], requires_grad=True)))
            self.register_parameter("bias_f", Parameter(Tensor.new([self.hidden_size], requires_grad=True)))
            self.register_parameter("bias_g", Parameter(Tensor.new([self.hidden_size], requires_grad=True)))
            self.register_parameter("bias_o", Parameter(Tensor.new([self.hidden_size], requires_grad=True)))
            for name in ("bias_i", "bias_f", "bias_g", "bias_o"):
                getattr(self, name).tensor.numpy_view()[:] = 0.0
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_g = None
            self.bias_o = None

    def forward(self, x, h, c):
        x_shape = x.shape()
        h_shape = h.shape()
        c_shape = c.shape()
        if len(x_shape) != 2:
            raise ValueError(f"LSTMCell expects input [B,I], got {x_shape}")
        if x_shape[1] != self.input_size:
            raise ValueError(f"LSTMCell input_size mismatch: expected {self.input_size}, got {x_shape[1]}")
        if len(h_shape) != 2 or h_shape[0] != x_shape[0] or h_shape[1] != self.hidden_size:
            raise ValueError(f"LSTMCell hidden shape mismatch: expected [{x_shape[0]},{self.hidden_size}], got {h_shape}")
        if c_shape != h_shape:
            raise ValueError(f"LSTMCell cell state shape mismatch: expected {h_shape}, got {c_shape}")

        x_dev = _device_of(x)
        params = [
            self.weight_ii.tensor, self.weight_if.tensor, self.weight_ig.tensor, self.weight_io.tensor,
            self.weight_hi.tensor, self.weight_hf.tensor, self.weight_hg.tensor, self.weight_ho.tensor,
        ]
        for p in params:
            if _device_of(p) != x_dev:
                raise ValueError("LSTMCell parameter device mismatch with input")
        if self.bias:
            for p in (self.bias_i.tensor, self.bias_f.tensor, self.bias_g.tensor, self.bias_o.tensor):
                if _device_of(p) != x_dev:
                    raise ValueError("LSTMCell bias device mismatch with input")

        i = x.matmul(self.weight_ii.tensor) + h.matmul(self.weight_hi.tensor)
        f = x.matmul(self.weight_if.tensor) + h.matmul(self.weight_hf.tensor)
        g = x.matmul(self.weight_ig.tensor) + h.matmul(self.weight_hg.tensor)
        o = x.matmul(self.weight_io.tensor) + h.matmul(self.weight_ho.tensor)

        if self.bias:
            i = i + self.bias_i.tensor
            f = f + self.bias_f.tensor
            g = g + self.bias_g.tensor
            o = o + self.bias_o.tensor

        i = _sigmoid(i)
        f = _sigmoid(f)
        g = _tanh(g)
        o = _sigmoid(o)

        c_next = f * c + i * g
        h_next = o * _tanh(c_next)
        return h_next, c_next

    def __call__(self, x, h, c):
        return self.forward(x, h, c)


class LSTM(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size, bias=bias)
        self.input_size = self.cell.input_size
        self.hidden_size = self.cell.hidden_size

    def forward(self, inputs, h0=None, c0=None, return_sequences=True):
        if isinstance(inputs, Tensor):
            inputs = _sequence_from_tensor(inputs)
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("LSTM inputs must be a list/tuple of tensors")
        if not inputs:
            raise ValueError("LSTM inputs cannot be empty")

        batch = inputs[0].shape()[0]
        if h0 is None:
            h = _zeros_like_shape([batch, self.hidden_size], inputs[0])
        else:
            h = h0
        if c0 is None:
            c = _zeros_like_shape([batch, self.hidden_size], inputs[0])
        else:
            c = c0

        outputs = []
        for x in inputs:
            h, c = self.cell(x, h, c)
            outputs.append(h)

        out = _stack_sequence(outputs) if return_sequences else outputs[-1]
        return out, (h, c)

    def __call__(self, inputs, h0=None, c0=None, return_sequences=True):
        return self.forward(inputs, h0=h0, c0=c0, return_sequences=return_sequences)


class MinGRUCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.bias = bool(bias)

        self.register_parameter("weight_xz", Parameter(Tensor.xavier_uniform([self.input_size, self.hidden_size], requires_grad=True)))
        self.register_parameter("weight_hz", Parameter(Tensor.xavier_uniform([self.hidden_size, self.hidden_size], requires_grad=True)))
        self.register_parameter("weight_xh", Parameter(Tensor.xavier_uniform([self.input_size, self.hidden_size], requires_grad=True)))
        self.register_parameter("weight_hh", Parameter(Tensor.xavier_uniform([self.hidden_size, self.hidden_size], requires_grad=True)))

        if self.bias:
            self.register_parameter("bias_z", Parameter(Tensor.new([self.hidden_size], requires_grad=True)))
            self.register_parameter("bias_h", Parameter(Tensor.new([self.hidden_size], requires_grad=True)))
            self.bias_z.tensor.numpy_view()[:] = 0.0
            self.bias_h.tensor.numpy_view()[:] = 0.0
        else:
            self.bias_z = None
            self.bias_h = None

    def forward(self, x, h):
        x_shape = x.shape()
        h_shape = h.shape()
        if len(x_shape) != 2:
            raise ValueError(f"MinGRUCell expects input [B,I], got {x_shape}")
        if x_shape[1] != self.input_size:
            raise ValueError(f"MinGRUCell input_size mismatch: expected {self.input_size}, got {x_shape[1]}")
        if len(h_shape) != 2 or h_shape[0] != x_shape[0] or h_shape[1] != self.hidden_size:
            raise ValueError(f"MinGRUCell hidden shape mismatch: expected [{x_shape[0]},{self.hidden_size}], got {h_shape}")

        x_dev = _device_of(x)
        params = [self.weight_xz.tensor, self.weight_hz.tensor, self.weight_xh.tensor, self.weight_hh.tensor]
        for p in params:
            if _device_of(p) != x_dev:
                raise ValueError("MinGRUCell parameter device mismatch with input")
        if self.bias:
            if _device_of(self.bias_z.tensor) != x_dev or _device_of(self.bias_h.tensor) != x_dev:
                raise ValueError("MinGRUCell bias device mismatch with input")

        z = x.matmul(self.weight_xz.tensor) + h.matmul(self.weight_hz.tensor)
        n = x.matmul(self.weight_xh.tensor) + h.matmul(self.weight_hh.tensor)
        if self.bias:
            z = z + self.bias_z.tensor
            n = n + self.bias_h.tensor

        z = _sigmoid(z)
        n = _tanh(n)
        one = _const_like(1.0, z)
        h_next = (one - z) * h + z * n
        return h_next

    def __call__(self, x, h):
        return self.forward(x, h)


class MinGRU(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.cell = MinGRUCell(input_size, hidden_size, bias=bias)
        self.input_size = self.cell.input_size
        self.hidden_size = self.cell.hidden_size

    def forward(self, inputs, h0=None, return_sequences=True):
        if isinstance(inputs, Tensor):
            inputs = _sequence_from_tensor(inputs)
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("MinGRU inputs must be a list/tuple of tensors")
        if not inputs:
            raise ValueError("MinGRU inputs cannot be empty")

        batch = inputs[0].shape()[0]
        if h0 is None:
            h = _zeros_like_shape([batch, self.hidden_size], inputs[0])
        else:
            h = h0

        outputs = []
        for x in inputs:
            h = self.cell(x, h)
            outputs.append(h)

        out = _stack_sequence(outputs) if return_sequences else outputs[-1]
        return out, h

    def __call__(self, inputs, h0=None, return_sequences=True):
        return self.forward(inputs, h0=h0, return_sequences=return_sequences)
