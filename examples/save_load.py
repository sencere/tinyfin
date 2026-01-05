"""
Example showing how to save and load a tinyfin model + optimizer state.
Uses the checkpoint helpers in tinyfin.utils.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
from tinyfin import Tensor, SGDOpt
from tinyfin.utils import save_checkpoint, load_checkpoint


def main():
    rng = np.random.default_rng(0)
    # simple linear model y = xW + b
    W = Tensor.new([4, 2], requires_grad=True)
    b = Tensor.new([2], requires_grad=True)
    W.numpy_view()[:] = rng.standard_normal((4, 2), dtype=np.float32) * 0.1
    b.numpy_view()[:] = 0.0

    params = [W, b]
    opt = SGDOpt(params, lr=0.1)

    # fake training step
    x = Tensor.new([3, 4], requires_grad=True); x.numpy_view()[:] = rng.standard_normal((3, 4), dtype=np.float32)
    target = Tensor.new([3, 2], requires_grad=False); target.numpy_view()[:] = rng.standard_normal((3, 2), dtype=np.float32)
    pred = x.matmul(W) + b
    loss = ((pred - target) * (pred - target)).sum()
    loss.backward()
    opt.step()

    # save checkpoint
    ckpt_path = save_checkpoint("checkpoints/linear", {"W": W, "b": b}, optimizer=opt)
    print(f"Saved checkpoint to {ckpt_path}")

    # zero params and reload
    for p in params:
        p.numpy_view()[:] = 0
    opt = SGDOpt(params, lr=0.1)
    tensors, meta = load_checkpoint(ckpt_path, optimizer=opt, strict=False)
    print("Loaded keys:", list(tensors.keys()))
    print("W[0,0] after load:", tensors["W"].to_numpy()[0, 0])
    print("Optimizer state reloaded:", meta.get("optimizer_loaded", False))


if __name__ == "__main__":
    main()
