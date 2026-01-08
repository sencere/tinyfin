# tinyfin how-to

Quick recipes for common tinyfin tasks.

## Select a backend
```python
from tinyfin import backend_set, backend_name

backend_set("cpu")
print("backend:", backend_name())
```

## Build a tiny model and train
```python
import numpy as np
from tinyfin import Tensor
from tinyfin.nn import MLP, CrossEntropyLoss
from tinyfin.optim import SGDOpt
from tinyfin.training import Trainer
from tinyfin.data import DataLoader

x = np.random.randn(256, 32).astype(np.float32)
y = np.random.randint(0, 10, size=(256,), dtype=np.int64).astype(np.float32)
loader = DataLoader.from_numpy(x, y, batch_size=32, shuffle=True)

model = MLP(32, [64], 10)
loss_fn = CrossEntropyLoss()
opt = SGDOpt(model.parameters(), lr=1e-2)
trainer = Trainer(model, loss_fn=loss_fn, optimizer=opt)
trainer.fit(loader, epochs=1)
```

## Export a small autograd graph
```python
import tinyfin
from tinyfin import Tensor

x = Tensor.new([2], requires_grad=True)
y = Tensor.new([2], requires_grad=True)
z = (x * y).sum()
tinyfin.export_graph(z, "graph.dot")
```

## Run the profiler
```bash
python examples/perf_profile.py cpu matmul 512 512 512 20
python examples/perf_profile.py cpu conv2d 16 3 32 32 8 3 20
python examples/perf_profile.py cpu elem 1048576 50
```
