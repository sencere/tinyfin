"""
Export a tiny autograd graph to DOT.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
from tinyfin import Tensor, export_graph


def main():
    x = Tensor.new([2], requires_grad=True)
    y = Tensor.new([2], requires_grad=True)
    x.numpy_view()[:] = np.array([1.0, 2.0], dtype=np.float32)
    y.numpy_view()[:] = np.array([3.0, 4.0], dtype=np.float32)

    z = (x * y).sum()

    out_path = "graph.dot"
    export_graph(z, out_path)
    print(f"Graph exported to {out_path}. Render with: dot -Tpng graph.dot -o graph.png")


if __name__ == "__main__":
    main()
