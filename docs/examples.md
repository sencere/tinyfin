# tinyfin examples roadmap

This note tracks the example suite we want to ship and how to run them across backends.

## Core demos (CPU/CUDA/OpenGL/Vulkan where available)
- MNIST/CIFAR CNN: minimal training/eval loops with DataLoader and checkpoints; flag to select backend (`TINYFIN_BACKEND=cpu|cuda|opengl|vulkan`).
- Transformer/BERT-style: small transformer block demo (embedding + FFN with residual) with synthetic data; export traces for profiling (`examples/transformer_block_tiny.py`).
- Text generation: tiny character-level model for quick smoke tests (`examples/text_gen_tiny.py`).
- Perf profiler: benchmark matmul/conv/elementwise on each backend and emit a simple report.

## Testing and reproducibility
- Deterministic seeds + assertions for loss decrease on small synthetic datasets.
- Backend coverage: examples should run on CPU by default, and conditionally on CUDA/OpenGL/Vulkan when available, falling back cleanly otherwise.
- Backend toggles: use env `TINYFIN_BACKEND=cpu|cuda|opengl|vulkan` and `tinyfin.backend_set` inside examples; device mapping currently uses 0=CPU, 1=GPU placeholder.

## Ergonomics guideline
- Prefer high-level helpers (e.g., `Tensor.from_numpy`) to keep examples short and consistent.
- Prefer `Embedding`, `MLP`, `Flatten`, `MaxPool2d`, and `CrossEntropyLoss` to minimize manual tensor ops.
- Use the shared training log format: `[train] epoch=... step=... loss=...` with optional `acc=` or `backend=`.

## Next steps
- Implement the MNIST/CIFAR CNN script first, then a lightweight transformer demo (backend-aware toggle added to transformer_tiny.py).
- Add backend-aware data/device helpers so examples move tensors to the selected backend cleanly.
- Add README snippets showing how to toggle backends and run each example.
 - Keep training loops minimal by using `Trainer.fit` where possible.
