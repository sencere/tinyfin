"""
Tiny Transformer-inspired feedforward demo on random data using tinyfin only.
No external deps, no true attention (since transpose/head ops aren't exposed yet);
this illustrates multi-layer MLP over sequence tokens.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
from tinyfin import Tensor, cross_entropy_logits, SGDOpt, backend_set


def relu(x: Tensor) -> Tensor:
    return x.clamp_min(0.0)


def main():
    backend = os.environ.get("TINYFIN_BACKEND", "cpu")
    backend_set(backend)
    batch, seq, dim, hidden = 8, 16, 32, 64
    rng = np.random.default_rng(0)

    def param(shape, scale=0.02):
        t = Tensor.new(list(shape), requires_grad=True)
        t.numpy_view()[:] = rng.standard_normal(shape, dtype=np.float32) * scale
        return t

    # Token mixing + feedforward
    w_m1 = param((dim, dim))
    b_m1 = param((dim,))
    w_m2 = param((dim, dim))
    b_m2 = param((dim,))
    w_ff1 = param((dim, hidden))
    b_ff1 = param((hidden,))
    w_ff2 = param((hidden, dim))
    b_ff2 = param((dim,))

    params = [w_m1, b_m1, w_m2, b_m2, w_ff1, b_ff1, w_ff2, b_ff2]
    opt = SGDOpt(params, lr=0.05)

    for step in range(50):
        opt.zero_grad()
        x = Tensor.new([batch * seq, dim], requires_grad=True)
        x.numpy_view()[:] = rng.standard_normal((batch * seq, dim), dtype=np.float32)

        # token mixing (two linear layers with residual)
        h = relu(x.matmul(w_m1) + b_m1)
        x_mixed = h.matmul(w_m2) + b_m2 + x

        # feedforward
        ff = relu(x_mixed.matmul(w_ff1) + b_ff1)
        out = ff.matmul(w_ff2) + b_ff2

        # dummy targets (same shape)
        targets = Tensor.new([batch * seq, dim], requires_grad=False)
        targets.numpy_view()[:] = rng.standard_normal((batch * seq, dim), dtype=np.float32)
        loss = cross_entropy_logits(out, targets)
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"[backend={backend}] step {step} loss={loss.to_numpy().mean():.4f}")


if __name__ == "__main__":
    main()
