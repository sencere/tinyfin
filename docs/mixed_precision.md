# Mixed precision (stub) roadmap

Mixed precision is gated via `tinyfin.set_mixed_precision(True|False)` or `with tinyfin.autocast(): ...`. The current implementation is a no-op placeholder; use it to wire tests/examples while real fp16/bfloat16 support is built.

Example:
```python
import tinyfin
from tinyfin import Tensor

x = Tensor.randn(128, 128, requires_grad=True)
with tinyfin.autocast():
    y = x.matmul(x)
```

Planned work:
- Backend-specific autocast: enable fp16/bfloat16 where supported (CUDA first; OpenGL/Vulkan later), with safe fallbacks.
- Clear docs on numerics, supported ops, and how to disable per-op.
- Tests to assert correctness vs. full precision on small models.

Current behavior:
- `autocast` does not change dtype yet.
- `set_mixed_precision(True)` only toggles an internal flag for future backends.
