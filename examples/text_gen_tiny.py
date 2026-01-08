"""
Tiny character-level text generation (bigram-style MLP).
Trains on a short string and samples from the learned next-char distribution.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
from tinyfin import Tensor, backend_set
from tinyfin import nn
from tinyfin.nn import CrossEntropyLoss
from tinyfin.optim import SGDOpt
from tinyfin.training import Trainer
from tinyfin.data import DataLoader


def build_dataset(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    x_ids = ids[:-1]
    y_ids = ids[1:]
    return x_ids, y_ids.astype(np.float32), stoi, itos


def sample(model, stoi, itos, start_char, steps=40):
    vocab = len(itos)
    ch = start_char
    out = [ch]
    for _ in range(steps):
        x = Tensor.from_numpy(np.array([stoi.get(ch, 0)], dtype=np.int64), requires_grad=False)
        logits = model(x)
        probs = logits.to_numpy().reshape(-1)
        probs = np.exp(probs - probs.max())
        probs = probs / probs.sum()
        idx = int(np.random.choice(vocab, p=probs))
        ch = itos[idx]
        out.append(ch)
    return "".join(out)


def main():
    backend = os.environ.get("TINYFIN_BACKEND", "cpu")
    backend_set(backend)

    text = "hello tinyfin\n" * 20
    x_ids, y_ids, stoi, itos = build_dataset(text)
    loader = DataLoader.from_numpy(
        x_ids.astype(np.int64),
        y_ids,
        batch_size=32,
        shuffle=True,
        requires_grad=[False, False],
    )

    vocab = len(itos)
    model = nn.Sequential(
        nn.Embedding(vocab, 32),
        nn.MLP(32, [64], vocab),
    )
    loss_fn = CrossEntropyLoss()
    opt = SGDOpt(model.parameters(), lr=0.2)
    trainer = Trainer(model, loss_fn=loss_fn, optimizer=opt)
    trainer.fit(loader, epochs=2)

    seed = "h"
    print(f"[text_gen] seed={seed!r} -> {sample(model, stoi, itos, seed, steps=40)}")


if __name__ == "__main__":
    main()
