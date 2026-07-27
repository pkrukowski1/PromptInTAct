## Why `GradientCosineTracker` produces all-NaN for `interval_drift__align`

### Summary

The cosine similarity between `interval_drift_loss` and `align_repr_loss` gradients is NaN for virtually every training step because the `align_repr_loss` gradient is zero on all shared parameters. This is not a bug in the tracker — it is a structural consequence of the align loss design.

### Verification

A standalone experiment (`experiments/simple_mlp/split_mnist_mlp.py`) was run with an MLP + IntervalActivation on Split MNIST. The `GradientCosineTracker` was verified to produce valid cosine values (-0.43 mean) when both losses are forced non-zero via artificial bounds. When using real cumulative bounds, 0 out of 1150 steps had a non-NaN cosine.

### Root cause: `align_repr_loss` condition is too strict

The align loss checks per-dimension non-overlap (`interval_regularization.py:377`):

```
non_overlap_mask = (new_lb > ub) | (new_ub < lb)
```

Translation: a dimension `i` is flagged only if the **entire batch** lies completely above the cumulative maximum OR completely below the cumulative minimum on that dimension:

| Condition | Meaning |
|---|---|
| `batch_min[i] > cum_max[i]` | Every sample in the batch is above the old hypercube on dim `i` |
| `batch_max[i] < cum_min[i]` | Every sample in the batch is below the old hypercube on dim `i` |

A batch of 256 samples on related tasks almost always spans across the cumulative bounds:

```
cum bounds:   [ -0.5  =================  +0.5 ]
batch range:  [ -1.5 ========================== +1.2 ]
                            ↑                       ↑
                     batch min < cum min     batch max > cum max
                     AND                     AND
                     batch max > cum min     batch min < cum max
```

Both `batch_min[i]` is **below** cum_min AND `batch_max[i]` is **above** cum_max for the same dimension. Neither `above` nor `below` triggers. This pattern holds for every dimension on every batch.

Empirical measurement on Split MNIST (digits 0-4 → 5-9, 128-dim hidden, 115 batches):
- Steps where `above > 0` on any dimension: **0**
- Steps where `below > 0` on any dimension: **0**

### Why it gets worse over tasks

After each task, cumulative bounds expand via `reset_range()` (`interval_activation.py:89-90`):

```
self.min = torch.minimum(self.min, min_vals)
self.max = torch.maximum(self.max, max_vals)
```

Bounds only grow, never shrink. By task 5+, the hypercube covers most activation patterns seen so far. The probability of any new batch being **entirely** outside on a dimension approaches zero.

### Consequence for the tracker

1. `align_repr_loss.backward()` produces `grad = 0` on all model parameters
2. `GradientCosineTracker.record()` captures `align_grads[p]` as a zero tensor
3. After the fix (filtering to non-zero grads, `gradient_cosine.py:41-44`), `shared = []` because no param has non-zero align grad
4. Cosine is correctly recorded as `NaN`

### What is affected

| Loss | Gradient non-zero every step? |
|---|---|
| `var_loss` | Yes — `acts_flat.var()` always has gradient |
| `output_reg_loss` | Yes — parameter drift after task 1 |
| `interval_drift_loss` | Partially — `mask` may be empty on dims where no activations fall within bounds |
| `align_repr_loss` | **Almost never** — strict non-overlap condition |

### Options

1. **Track a different loss pair.** `interval_drift__output_reg` or `interval_drift__var` will produce valid cosines at every step.

2. **Relax the align loss** to fire on **any** dimension where `batch_min[i] < cum_min[i]` OR `batch_max[i] > cum_max[i]` (removing the "entire batch" requirement). This would make it a per-sample or per-center drift measure.

3. **Accept the sparsity.** The tracker is correct to output NaN when there is no shared gradient signal. On the full ViT/ImageNet-R setup, violations may occur occasionally (especially early tasks with disparate classes), and those steps will produce valid cosines.
