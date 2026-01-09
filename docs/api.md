# tinyfin Python API (stub)

See `docs/versioning.md` for versioning and deprecation policy.
See `docs/support_matrix.md` for platform and Python support.

Stable namespaces:
- `tinyfin.tensor`: `Tensor`, `no_grad`, `bce_loss`, `cross_entropy_logits`, `assert_finite` (Tensor helpers include `Tensor.from_numpy`)
- `tinyfin.nn`: `Module`, `Parameter`, `Sequential`, `Linear`, `Embedding`, `Flatten`, `MaxPool2d`, `Conv2d`, `BatchNorm`, `Dropout`, `MLP`, `CrossEntropyLoss`
- `tinyfin.optim`: `SGDOpt`, `AdamOpt`, `RMSPropOpt`
- `tinyfin.utils`: `Profiler`, `assert_finite`, `save_checkpoint`, `load_checkpoint`
- `tinyfin.data`: `Dataset`, `Transform`, `Compose`, `TensorDataset` (including `from_numpy`), `DataLoader` (including `from_numpy`)
- Built-in loaders: `tinyfin.data.load_mnist()` / `load_cifar10()` (pure Python download/unpack; fall back paths exist)
- `tinyfin.training`: `Callback`, `CallbackList`, `Trainer` (includes `fit`)
- `tinyfin.scheduler`: `StepLR`, `ExponentialLR`, `LinearWarmupLR`
- `tinyfin.callbacks`: `LoggingCallback`, `CheckpointCallback`
- Autograd toggles: `no_grad`, `set_retain_graph_default(retain)`, `get_retain_graph_default()`
- Graph helpers: `export_graph(tensor, path)`, `export_graph_ir(tensor)` (nodes/edges), `const_fold_ir(ir)`, `graph_cache_key(tensor)`, `graph_cache_run(fn, *args, **kwargs)`
- Experimental helpers (gated by `TINYFIN_ENABLE_HIGHER_ORDER=1`): `tinyfin.vjp` (vector-Jacobian product; returns `y` and `grad_x` numpy copy), `tinyfin.jvp` (finite-difference JVP), and `tinyfin.hvp` (finite-difference Hessian-vector product).
- Checkpointing: `tinyfin.utils.save_checkpoint(path, tensors, optimizer=None, scheduler=None, metadata=None)` and `load_checkpoint(path, optimizer=None, scheduler=None, strict=True|False)` with magic/version + checksum validation.

Env toggles:
- `TINYFIN_BACKEND`: select registered backend (e.g., `cuda` when built).
- `TINYFIN_THREADS`: OpenMP matmul threads (if compiled with OpenMP).
- `TINYFIN_SCRATCH_BYTES`: scratch arena size (default 1MB).
- `TINYFIN_ENABLE_HIGHER_ORDER`: enable experimental higher-order helpers (`vjp/jvp/hvp`).
- `TINYFIN_RETAIN_GRAPH` / `TINYFIN_PERSISTENT_TAPE`: default retain-graph behavior for backward (higher-order experiments).
- Tensor helpers: `reshape`, `squeeze`, `unsqueeze`, `maxpool2d`, `avgpool2d` (with shape validation)
- New tensor ops: `transpose` (2D), `permute` (custom order), `concat`, `stack`, `pad2d`, and Python-level `split`.
- Backend helpers: `backend_name()` (defaults to `cpu`), `backend_set(name)` (CPU default; optional CUDA backend when built with `ENABLE_CUDA=1` and selected via env or API)
- Env toggles: `TINYFIN_BACKEND` to pick a registered backend (`cuda` when built with CUDA); `TINYFIN_THREADS` to enable OpenMP parallel matmul (if compiled with OpenMP).
- Higher-order scope: `vjp` uses a single backward pass with `retain_graph`; `jvp`/`hvp` are finite-difference approximations that allocate extra tensors and can be slow or numerically sensitive. Use them for experiments and validation, not production training loops.

Common patterns:
```python
import tinyfin
import numpy as np
from tinyfin.tensor import Tensor, no_grad, assert_finite
from tinyfin.nn import Sequential, Linear
from tinyfin.optim import SGDOpt, AdamOpt
from tinyfin.utils import save_checkpoint, load_checkpoint

x = Tensor.new([32, 128], requires_grad=True)
model = Sequential(Linear(128, 64), Linear(64, 10))
opt = SGDOpt([p.tensor for p in model.parameters()], lr=1e-2, momentum=0.9, weight_decay=1e-4)

out = model(x)
loss = tinyfin.cross_entropy_logits(out, target)
loss.backward()
assert_finite([p.tensor for p in model.parameters()])
opt.step(); opt.zero_grad()

# Swap optimizers easily and reload saved state
opt = AdamOpt([p.tensor for p in model.parameters()], lr=1e-3)
opt.save_state("/tmp/adam_state.bin")
opt.load_state("/tmp/adam_state.bin")

# Save a small checkpoint bundle (tensors + optimizer + scheduler)
ckpt_path = save_checkpoint("/tmp/model", {"w1": x}, optimizer=opt, scheduler=StepLR(opt, step_size=10, gamma=0.5))
loaded, meta = load_checkpoint(ckpt_path, optimizer=opt)

# Tensor helpers
y = Tensor.new([2, 3, 4, 4], requires_grad=False)
pooled = y.maxpool2d(2)  # shape -> [2, 3, 2, 2]
reshaped = pooled.reshape([2, 12])
sq = reshaped.unsqueeze(0).squeeze(0)

# NumPy bridge
arr = np.random.randn(8, 16).astype(np.float32)
t = Tensor.from_numpy(arr, requires_grad=True)
```

Device/dtype helpers:
- `Tensor.set_device(device)`, `Tensor.to_device(device)`
- `Tensor.set_dtype(dtype)`, `Tensor.get_device()`

Stability helpers:
- `Tensor.has_nan_or_inf()`
- Epsilon guards in `log`, `div`, `exp`, stable `softmax/log_softmax`, `clamp_min`.
