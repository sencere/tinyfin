import sys, os, math, numpy as np
here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, '..', '..', 'python'))
sys.path.insert(0, root)
from tinyfin import Tensor, bce_loss, cross_entropy_logits


def _assert_allclose(a, b, eps=1e-6):
    assert np.allclose(a, b, atol=eps)


def _bce_elem(p, t):
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(t * math.log(p) + (1.0 - t) * math.log(1.0 - p))


# simple BCE reduction tests (unweighted)
pred = Tensor.new([4])
pred.numpy_view()[:] = [0.1, 0.9, 0.2, 0.8]
target = Tensor.new([4])
target.numpy_view()[:] = [0.0, 1.0, 0.0, 1.0]
arr = pred.to_numpy()
tarr = target.to_numpy()

loss_sum = bce_loss(pred, target, logits=False, reduction='sum').to_numpy().item()
loss_mean = bce_loss(pred, target, logits=False, reduction='mean').to_numpy().item()
loss_none = bce_loss(pred, target, logits=False, reduction='none').to_numpy()

expected_none = np.array([_bce_elem(p, t) for p, t in zip(arr, tarr)])
_assert_allclose(loss_none, expected_none)
assert abs(loss_sum / 4.0 - loss_mean) < 1e-6

# BCE with per-element weights
pred_w = Tensor.new([2])
pred_w.numpy_view()[:] = [0.2, 0.6]
target_w = Tensor.new([2])
target_w.numpy_view()[:] = [0.0, 1.0]
weight = Tensor.new([2])
weight.numpy_view()[:] = [1.0, 3.0]

weighted_losses = np.array([_bce_elem(0.2, 0.0) * 1.0, _bce_elem(0.6, 1.0) * 3.0])
wsum = weight.to_numpy().sum()

loss_sum_w = bce_loss(pred_w, target_w, logits=False, weight=weight, reduction='sum').to_numpy().item()
loss_mean_w = bce_loss(pred_w, target_w, logits=False, weight=weight, reduction='mean').to_numpy().item()
loss_none_w = bce_loss(pred_w, target_w, logits=False, weight=weight, reduction='none').to_numpy()

assert abs(loss_sum_w - weighted_losses.sum()) < 1e-6
assert abs(loss_mean_w - weighted_losses.sum() / wsum) < 1e-6
_assert_allclose(loss_none_w, weighted_losses)

# Cross-entropy with class weights and reduction modes
logits = Tensor.new([2, 3])
logits.numpy_view()[:] = [[2.0, 0.0, 1.0], [1.0, 3.0, 0.0]]
target_ce = Tensor.new([2])
target_ce.numpy_view()[:] = [0, 2]
class_weight = Tensor.new([3])
class_weight.numpy_view()[:] = [1.0, 2.0, 3.0]

logits_np = logits.to_numpy()
target_np = target_ce.to_numpy().astype(int)
weight_np = class_weight.to_numpy()

def _ce_elem(row, idx):
    maxv = float(np.max(row))
    logsum = maxv + math.log(float(np.exp(row - maxv).sum()))
    return logsum - float(row[idx])

weighted_ce = []
for i in range(logits_np.shape[0]):
    l = _ce_elem(logits_np[i], int(target_np[i]))
    w = weight_np[int(target_np[i])]
    weighted_ce.append(l * w)
weighted_ce = np.array(weighted_ce)
wsum_ce = weight_np[target_np].sum()

loss_sum_ce = cross_entropy_logits(logits, target_ce, weight=class_weight, reduction='sum').to_numpy().item()
loss_mean_ce = cross_entropy_logits(logits, target_ce, weight=class_weight, reduction='mean').to_numpy().item()
loss_none_ce = cross_entropy_logits(logits, target_ce, weight=class_weight, reduction='none').to_numpy()

assert abs(loss_sum_ce - weighted_ce.sum()) < 1e-6
assert abs(loss_mean_ce - weighted_ce.sum() / wsum_ce) < 1e-6
_assert_allclose(loss_none_ce, weighted_ce)

print('[test_loss_reduction.py] PASS')
