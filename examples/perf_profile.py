"""
Throughput profiler for matmul on CPU / BLAS / CUDA.

Usage:
    python perf_profile.py cpu 512 512 512 20
    python perf_profile.py cuda 512 512 512 20
    python perf_profile.py all 512 512 512 20   # runs cpu, blas (if available), cuda (if available)
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
from tinyfin import Tensor, backend_set, backend_name


def flops_matmul(m, k, n):
    return 2 * m * k * n


def run(device, m, k, n, iters):
    if device == "cuda":
        os.environ["TINYFIN_BACKEND"] = "cuda"
        backend_set("cuda")
    elif device == "blas":
        os.environ["TINYFIN_BACKEND"] = "blas"
        backend_set("blas")
    else:
        os.environ["TINYFIN_BACKEND"] = "cpu"
        backend_set("cpu")

    print(f"backend: {backend_name()}, device: {device}")
    a = Tensor.new([m, k], requires_grad=False)
    b = Tensor.new([k, n], requires_grad=False)
    a.numpy_view()[:] = np.random.randn(m, k).astype(np.float32)
    b.numpy_view()[:] = np.random.randn(k, n).astype(np.float32)
    if device == "cuda":
        a.set_device(1)
        b.set_device(1)

    # warmup
    _ = a.matmul(b)
    start = time.time()
    for _ in range(iters):
        c = a.matmul(b)
    dur = time.time() - start
    sps = iters / dur
    gflops = (flops_matmul(m, k, n) * iters) / dur / 1e9
    print(f"{iters} iters in {dur:.3f}s -> {sps:.2f} it/s, {gflops:.2f} GFLOP/s")
    return {"backend": backend_name(), "device": device, "iters": iters, "seconds": dur, "it_per_s": sps, "gflops": gflops}


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    iters = int(sys.argv[5]) if len(sys.argv) > 5 else 10

    if device == "all":
        results = []
        for dev in ("cpu", "blas", "cuda"):
            try:
                res = run(dev, m, k, n, iters)
                results.append(res)
            except Exception as e:
                print(f"[{dev}] skipped: {e}")
        if results:
            print("\nSummary:")
            for r in results:
                print(f"{r['device']:>6} | {r['it_per_s']:8.2f} it/s | {r['gflops']:8.2f} GFLOP/s | {r['seconds']:.3f}s total")
    else:
        run(device, m, k, n, iters)


if __name__ == "__main__":
    main()
