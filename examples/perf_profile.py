"""
Throughput profiler for matmul/conv2d/elementwise on CPU / BLAS / CUDA.

Usage:
    python perf_profile.py cpu 512 512 512 20
    python perf_profile.py cuda 512 512 512 20
    python perf_profile.py all 512 512 512 20   # runs cpu, blas (if available), cuda (if available)
    python perf_profile.py cpu matmul 512 512 512 20
    python perf_profile.py cpu conv2d 16 3 32 32 8 3 20   # N C H W O K iters
    python perf_profile.py cpu elem 1048576 50            # size iters
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
from tinyfin import Tensor, backend_set, backend_name


def flops_matmul(m, k, n):
    return 2 * m * k * n


def flops_conv2d(n, c, h, w, o, k):
    out_h = h - k + 1
    out_w = w - k + 1
    return 2 * n * o * out_h * out_w * c * k * k


def run(device, op, dims, iters):
    if device == "cuda":
        os.environ["TINYFIN_BACKEND"] = "cuda"
        ok = backend_set("cuda")
    elif device == "blas":
        os.environ["TINYFIN_BACKEND"] = "blas"
        ok = backend_set("blas")
    elif device in ("opengl", "vulkan"):
        os.environ["TINYFIN_BACKEND"] = device
        ok = backend_set(device)
    else:
        os.environ["TINYFIN_BACKEND"] = "cpu"
        ok = backend_set("cpu")

    if not ok:
        raise RuntimeError(f"backend '{device}' not available")

    print(f"backend: {backend_name()}, device: {device}, op: {op}")
    device_id = 1 if backend_name() == "cuda" else 0

    if op == "matmul":
        m, k, n = dims
        a = Tensor.from_numpy(np.random.randn(m, k).astype(np.float32), requires_grad=False, device=device_id)
        b = Tensor.from_numpy(np.random.randn(k, n).astype(np.float32), requires_grad=False, device=device_id)
        _ = a.matmul(b)
        start = time.time()
        for _ in range(iters):
            _ = a.matmul(b)
        dur = time.time() - start
        gflops = (flops_matmul(m, k, n) * iters) / dur / 1e9
    elif op == "conv2d":
        n, c, h, w, o, k = dims
        x = Tensor.from_numpy(np.random.randn(n, c, h, w).astype(np.float32), requires_grad=False, device=device_id)
        w_t = Tensor.from_numpy(np.random.randn(o, c, k, k).astype(np.float32), requires_grad=False, device=device_id)
        _ = x.conv2d(w_t)
        start = time.time()
        for _ in range(iters):
            _ = x.conv2d(w_t)
        dur = time.time() - start
        gflops = (flops_conv2d(n, c, h, w, o, k) * iters) / dur / 1e9
    elif op == "elem":
        size = dims[0]
        a = Tensor.from_numpy(np.random.randn(size).astype(np.float32), requires_grad=False, device=device_id)
        b = Tensor.from_numpy(np.random.randn(size).astype(np.float32), requires_grad=False, device=device_id)
        _ = a * b + a
        start = time.time()
        for _ in range(iters):
            _ = a * b + a
        dur = time.time() - start
        gflops = (size * iters) / dur / 1e9
    else:
        raise ValueError(f"unknown op: {op}")

    sps = iters / dur
    print(f"{iters} iters in {dur:.3f}s -> {sps:.2f} it/s, {gflops:.2f} GFLOP/s")
    return {"backend": backend_name(), "device": device, "op": op, "iters": iters, "seconds": dur, "it_per_s": sps, "gflops": gflops}


def _is_int(val):
    try:
        int(val)
        return True
    except Exception:
        return False


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    arg2 = sys.argv[2] if len(sys.argv) > 2 else None
    if arg2 and not _is_int(arg2):
        op = arg2
        args = sys.argv[3:]
    else:
        op = "matmul"
        args = sys.argv[2:]

    if op == "matmul":
        m = int(args[0]) if len(args) > 0 else 256
        k = int(args[1]) if len(args) > 1 else 256
        n = int(args[2]) if len(args) > 2 else 256
        iters = int(args[3]) if len(args) > 3 else 10
        dims = (m, k, n)
    elif op == "conv2d":
        n = int(args[0]) if len(args) > 0 else 16
        c = int(args[1]) if len(args) > 1 else 3
        h = int(args[2]) if len(args) > 2 else 32
        w = int(args[3]) if len(args) > 3 else 32
        o = int(args[4]) if len(args) > 4 else 8
        k = int(args[5]) if len(args) > 5 else 3
        iters = int(args[6]) if len(args) > 6 else 10
        dims = (n, c, h, w, o, k)
    elif op == "elem":
        size = int(args[0]) if len(args) > 0 else 1024 * 1024
        iters = int(args[1]) if len(args) > 1 else 50
        dims = (size,)
    else:
        raise ValueError(f"unknown op: {op}")

    if device == "all":
        results = []
        for dev in ("cpu", "blas", "cuda", "opengl", "vulkan"):
            try:
                res = run(dev, op, dims, iters)
                results.append(res)
            except Exception as e:
                print(f"[{dev}] skipped: {e}")
        if results:
            print("\nSummary:")
            for r in results:
                print(f"{r['device']:>6} | {r['op']:>6} | {r['it_per_s']:8.2f} it/s | {r['gflops']:8.2f} GFLOP/s | {r['seconds']:.3f}s total")
    else:
        run(device, op, dims, iters)


if __name__ == "__main__":
    main()
