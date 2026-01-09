# tinyfin backend support matrix

Legend:
- **Y**: implemented
- **F**: falls back to CPU implementation
- **S**: stub backend (logs + falls back)

| Op / Backend | cpu | cuda | blas | opengl | vulkan |
| --- | --- | --- | --- | --- | --- |
| add | Y | Y (float32) | F | S | S |
| mul | Y | Y (float32) | F | S | S |
| matmul | Y | Y (float32) | Y (float32) | S | S |
| conv2d (fwd) | Y | Y (float32) | Y (float32) | S | S |
| conv2d (bwd) | Y | Y (float32, host-resident) | F | S | S |

Notes:
- CUDA paths require `ENABLE_CUDA=1` and `TINYFIN_BACKEND=cuda`.
- BLAS paths require `ENABLE_BLAS=1` and `TINYFIN_BACKEND=blas`.
- OpenGL/Vulkan are registered stubs only (no GPU kernels yet).
