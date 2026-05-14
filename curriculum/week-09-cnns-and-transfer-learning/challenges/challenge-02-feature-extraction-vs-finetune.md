# Challenge 2 — Feature Extraction vs. Full Fine-Tune

**Time estimate:** 2-3 hours.

## Problem statement

Quantify the accuracy / compute trade-off between **feature extraction** (freeze the backbone, train only the head) and **full fine-tuning** (train the whole network end-to-end with differential learning rates) on CIFAR-10, using pretrained ResNet-18.

The point: feature extraction is fast and robust; fine-tuning is slow but reaches higher accuracy. The numbers from your own machine, plotted on a single figure, are the most honest answer you will ever produce to the interview question "when should I fine-tune and when should I freeze?"

This is the empirical companion to Lecture 3 Sections 4-5.

References for this challenge:

- Transfer learning tutorial: <https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>
- ResNet18 weights: <https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html>
- The discussion in Yosinski et al. 2014 ("How transferable are features in deep neural networks?"): <https://arxiv.org/abs/1411.1792>

## What you will produce

A single script `feature_vs_finetune.py` plus a one-page write-up `notes.md` that:

1. Loads CIFAR-10 via `torchvision.datasets.CIFAR10` with the 224×224 resize and ImageNet normalization from Lecture 3 Section 7.
2. Builds **three** ResNet-18 variants:
   - **A: feature extractor.** Pretrained ResNet-18, backbone frozen, new 10-class `Linear(512, 10)` head.
   - **B: partial fine-tune.** Pretrained ResNet-18, only `layer4` and `fc` unfrozen.
   - **C: full fine-tune.** Pretrained ResNet-18, all parameters trainable with differential learning rates (backbone at `1e-4`, head at `1e-3`).
3. Trains all three with otherwise-identical hyperparameters: `batch_size=64`, `n_epochs=10`, `torch.manual_seed(42)`, `nn.CrossEntropyLoss`, the eight-line training loop from Week 8.
4. For each variant, records:
   - The number of trainable parameters.
   - The wall-clock time per epoch.
   - The peak GPU memory (or RSS for CPU runs) during training.
   - The test accuracy per epoch.
5. Produces a 2-panel figure `feature_vs_finetune.png`:
   - Left panel: test accuracy per epoch, three curves, with a legend that shows the final accuracy and trainable parameter count.
   - Right panel: total compute (cumulative wall-clock time) on x-axis, test accuracy on y-axis, three curves. This is the "accuracy-vs-compute" view that makes the trade-off concrete.
6. Writes `notes.md` (~300 words) that reports the final numbers and answers four questions:
   - What is the final test accuracy of each variant?
   - At what *epoch* do A, B, C cross 85%? 88%? 90%?
   - At what *wall-clock time* do A, B, C cross those same thresholds?
   - Given a 10-minute budget on a Colab T4, which variant should a practitioner pick?

## Acceptance criteria

- [ ] `feature_vs_finetune.py` runs end-to-end with `python feature_vs_finetune.py` and produces `feature_vs_finetune.png` plus `notes.md`.
- [ ] `python -m py_compile feature_vs_finetune.py` succeeds.
- [ ] All three variants reach at least the following final accuracies:
  - A (frozen): **≥85%** in 10 epochs.
  - B (partial fine-tune): **≥89%** in 10 epochs.
  - C (full fine-tune): **≥92%** in 10 epochs.
- [ ] The figure shows three clearly-labeled curves on each panel with a legend.
- [ ] `notes.md` answers all four questions and is honest about runtime (include your hardware in the report).
- [ ] The training script uses `torch.manual_seed(42)` at the top.
- [ ] No emojis, no exclamation marks, type hints on every Python function.

## Reference scaffold

```python
"""
Challenge 2 -- feature extraction vs. full fine-tune on CIFAR-10.

Usage: python feature_vs_finetune.py
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch
from torch import nn


RANDOM_STATE: int = 42
N_EPOCHS: int = 10
BATCH_SIZE: int = 64
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders() -> Tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader"]:
    # Same pipeline as Exercise 3 Part C.
    ...


def build_feature_extractor(n_classes: int = 10) -> nn.Module:
    # Frozen backbone, new head.
    ...


def build_partial_finetune(n_classes: int = 10) -> nn.Module:
    # Freeze everything except layer4 and the new head.
    ...


def build_full_finetune(n_classes: int = 10) -> nn.Module:
    # No freezing.
    ...


def make_optimizer(model: nn.Module, variant: str) -> torch.optim.Optimizer:
    """A and B use a single param group at lr=1e-3 (only trainable params).
    C uses two groups: backbone at 1e-4, head at 1e-3."""
    ...


def train_variant(
    variant_name: str,
    model_builder,
    n_epochs: int = N_EPOCHS,
) -> Dict[str, List[float]]:
    """Returns {"test_acc": [...], "epoch_time": [...]}."""
    ...


def main() -> None:
    torch.manual_seed(RANDOM_STATE)
    results = {
        "A_frozen":     train_variant("A_frozen",      build_feature_extractor),
        "B_partial":    train_variant("B_partial",     build_partial_finetune),
        "C_full":       train_variant("C_full",        build_full_finetune),
    }
    # ... save the figure and the notes.


if __name__ == "__main__":
    main()
```

## Hints

- **The frozen variant is the fastest per epoch** — only the head is updated; the backward pass stops at the frozen boundary. Expect ~30% less wall-clock time than the full fine-tune at the same batch size.
- **The full fine-tune has the highest accuracy ceiling** but takes the longest to reach it. With the differential LR (`1e-4` backbone, `1e-3` head), it should cross 90% by epoch 3-4 and stabilize around 92-93% by epoch 10.
- **The partial fine-tune is the practical sweet spot.** Half the trainable params, 70% of the compute, and accuracy within 1-2 percentage points of the full fine-tune. This is what production teams reach for when they want to fine-tune a large model on a small dataset and they care about iteration time.
- **Measure wall-clock time around the inner loop, not the data download.** Cache the dataset by running the script once before the timed experiment. The download is a one-time cost that should not appear in the comparison.
- **Memory measurement.** Use `torch.cuda.max_memory_allocated()` (on CUDA) reset before each variant with `torch.cuda.reset_peak_memory_stats()`. On CPU, use `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` (Linux) or fall back to reporting "CPU, not measured" — that is honest and acceptable for this challenge.

## The expected result (rough)

On a Colab T4, 10 epochs each:

| Variant          | Trainable params | Epoch time | Final test acc | Time to 90% acc |
|------------------|------------------|-----------:|---------------:|----------------:|
| A: frozen        | ~5k              | ~30 s      | ~87.5%         | does not reach  |
| B: partial       | ~4.7M            | ~40 s      | ~90.5%         | epoch 6         |
| C: full          | ~11.2M           | ~45 s      | ~92.8%         | epoch 3         |

Your numbers will differ; the *ranking* should hold. If your full fine-tune is slower to converge than your frozen variant, you have a bug in the differential learning rates (most likely the backbone LR is too small — try `5e-4` instead of `1e-4`).

## Why this matters

Three things to walk away from this challenge with:

1. **Feature extraction is the right first move.** It is fast, the head is a single Linear, and on most natural-image classification tasks you reach within 5 percentage points of the best possible result. The C5 conviction is to ALWAYS run feature extraction first before considering fine-tuning.
2. **Full fine-tune is the right second move when accuracy matters.** The 2-5 percentage point improvement over feature extraction is real and is the difference between "I shipped a working model" and "I shipped a model that beats the baseline." Use it when the application can afford the compute.
3. **Partial fine-tune is what real production teams do.** Mid-stack unfreezing is the most-common variant when fine-tuning large models (think: foundation model on a domain-specific dataset). You should know it exists and have the muscle memory to construct it.

## Stretch goals

- Add a **fourth variant**: full fine-tune with `backbone_lr=1e-3, head_lr=1e-3` (same LR everywhere). This is what students do by default; the result is usually *worse* than feature extraction. Show empirically why differential learning rates are not optional.
- Try **gradual unfreezing**: train for 3 epochs with the backbone frozen, then unfreeze `layer4` and train for 3 more, then unfreeze `layer3` and train for 4 more. This is the recipe in Howard and Ruder 2018 ("Universal Language Model Fine-tuning"; <https://arxiv.org/abs/1801.06146>) and it sometimes reaches higher accuracy than the all-at-once fine-tune for the same epoch budget.
- Repeat the experiment on a much smaller dataset — sample 5,000 CIFAR-10 training images instead of the full 50,000. The pattern should flip: with little data, feature extraction wins; with lots of data, full fine-tune wins. This is the empirical version of the "transfer learning works because the small dataset cannot specify a 11M-parameter model from scratch" argument.
