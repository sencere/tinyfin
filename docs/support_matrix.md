# tinyfin platform and Python support matrix

## Supported platforms
- Linux x86_64 (primary)
- macOS x86_64/arm64 (tested via CI when available)
- Windows x86_64 (best-effort; community supported)

## Python versions
- 3.9, 3.10, 3.11 (primary)
- 3.12 (experimental; report issues)

## Build notes
- CPU-only builds are supported on all platforms.
- CUDA support requires a compatible NVIDIA toolchain and `ENABLE_CUDA=1`.
- OpenGL/Vulkan backends are stubs only; real GPU kernels are not yet available.

## Policy
- Primary platforms receive regular CI coverage and regression fixes.
- Best-effort platforms may lag; issues and patches are welcome.
