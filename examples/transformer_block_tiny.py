"""
Tiny transformer-style block: token embeddings + FFN with residual.
Uses synthetic token sequences and trains a next-token classifier.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
from tinyfin import Tensor, backend_set
from tinyfin import nn
from tinyfin.nn import CrossEntropyLoss
from tinyfin.optim import SGDOpt
from tinyfin.training import Trainer


class TinyBlock(nn.Module):
    def __init__(self, vocab, dim, hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.ffn = nn.MLP(dim, [hidden], dim)
        self.out = nn.Linear(dim, vocab)

    def forward(self, token_ids):
        # token_ids: [N] -> emb: [N, dim]
        emb = self.embed(token_ids)
        h = self.ffn(emb) + emb
        return self.out(h)


def main():
    backend = os.environ.get("TINYFIN_BACKEND", "cpu")
    backend_set(backend)
    rng = np.random.default_rng(0)
    vocab = 32
    seq = 16
    batch = 8
    dim = 32
    hidden = 64

    model = TinyBlock(vocab, dim, hidden)
    loss_fn = CrossEntropyLoss()
    opt = SGDOpt(model.parameters(), lr=0.1)
    trainer = Trainer(model, loss_fn=loss_fn, optimizer=opt)

    def make_batch():
        x_ids = rng.integers(0, vocab, size=(batch, seq), dtype=np.int64)
        y_ids = np.roll(x_ids, shift=-1, axis=1)
        x_flat = x_ids.reshape(-1)
        y_flat = y_ids.reshape(-1).astype(np.float32)
        x = Tensor.from_numpy(x_flat, requires_grad=False)
        y = Tensor.from_numpy(y_flat, requires_grad=False)
        return x, y

    class SimpleLoader:
        def __iter__(self):
            for _ in range(50):
                yield make_batch()

    trainer.fit(SimpleLoader(), epochs=1)
    print(f"[transformer_block_tiny] done on backend={backend}")


if __name__ == "__main__":
    main()
