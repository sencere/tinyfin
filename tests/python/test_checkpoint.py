import sys, os, tempfile

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)

from tinyfin import Tensor, SGDOpt
from tinyfin.scheduler import StepLR
from tinyfin.utils import save_checkpoint, load_checkpoint


def make_param(val):
    t = Tensor.new([2], requires_grad=True)
    t.numpy_view()[:] = [val, -val]
    return t


def test_checkpoint_roundtrip():
    p = make_param(1.0)
    opt = SGDOpt([p], lr=0.1, momentum=0.0)
    sch = StepLR(opt, step_size=1, gamma=0.5)

    # Do a single step to change both param data and optimizer/scheduler state.
    loss = (p * p).sum()
    loss.backward()
    opt.step()
    sch.step()

    tmp = tempfile.NamedTemporaryFile(delete=False)
    base = tmp.name
    tmp.close()
    meta_path = save_checkpoint(base, {"p": p}, optimizer=opt, scheduler=sch, metadata={"tag": "unit"})

    # Load into fresh objects.
    p2 = make_param(0.0)
    opt2 = SGDOpt([p2], lr=1.0)
    sch2 = StepLR(opt2, step_size=3, gamma=0.1)
    tensors, meta = load_checkpoint(meta_path, optimizer=opt2, scheduler=sch2)

    # Param data restored
    assert abs(tensors["p"].to_numpy()[0] - p.to_numpy()[0]) < 1e-6
    # Optimizer lr restored via scheduler state and optimizer state
    assert abs(opt2.get_lr() - opt.get_lr()) < 1e-6
    # Metadata round-trip
    assert meta["tag"] == "unit"

    # Follow-up step should keep continuity of scheduler decay
    sch2.step()
    assert opt2.get_lr() < opt.get_lr()


if __name__ == "__main__":
    test_checkpoint_roundtrip()
    print("[test_checkpoint.py] PASS")
