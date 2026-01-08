Use the examples to see tinyfin without external frameworks. Run from repo root after `make libtinyfin.so`.

- `mnist_mlp.py` — synthetic MNIST-like MLP (CPU, numpy only).  
  `python examples/mnist_mlp.py`

- `cifar_cnn.py` — synthetic CIFAR-like conv + pool + linear classifier (CPU).  
  `python examples/cifar_cnn.py`

- `transformer_tiny.py` — transformer-inspired token-mixing/FFN block on random data (no external deps).  
  `python examples/transformer_tiny.py`

- `transformer_block_tiny.py` — tiny transformer-style block (embedding + FFN + residual).  
  `python examples/transformer_block_tiny.py`

- `text_gen_tiny.py` — tiny character-level text generation on a short string.  
  `python examples/text_gen_tiny.py`

- `rnn_tiny.py` — minimal LSTM sequence example with a tiny training loop.  
  `python examples/rnn_tiny.py`

- `cuda_matmul.py` — CUDA matmul demo if built with `ENABLE_CUDA=1` (falls back to CPU).  
  `TINYFIN_BACKEND=cuda python examples/cuda_matmul.py`

- `perf_profile.py` — throughput/GFLOP/s profiler (matmul/conv2d/elem).  
  `python examples/perf_profile.py cpu matmul 512 512 512 20`  
  `python examples/perf_profile.py cpu conv2d 16 3 32 32 8 3 20`  
  `python examples/perf_profile.py cpu elem 1048576 50`

- `graph_export.py` — exports a tiny autograd graph to DOT.  
  `python examples/graph_export.py`

- `save_load.py` — saves and reloads a tiny linear model + optimizer state.  
  `python examples/save_load.py`

- `backend_mnist_cnn.py` — synthetic MNIST-like CNN with backend toggle.  
  `TINYFIN_BACKEND=cpu|cuda|opengl|vulkan python examples/backend_mnist_cnn.py`

Notes:
- For CUDA, build with `ENABLE_CUDA=1` and set `TINYFIN_BACKEND=cuda`.
- For BLAS matmul backend, build with `ENABLE_BLAS=1` and set `TINYFIN_BACKEND=blas`.
