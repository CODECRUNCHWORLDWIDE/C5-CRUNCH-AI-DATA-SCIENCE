# Challenge 1 — CIFAR-10: MLP vs. Tiny CNN Baseline

**Time estimate:** 2 hours.

## Problem statement

Quantify the MLP-vs-CNN gap on CIFAR-10. Train two PyTorch models, end to end, with **identical hyperparameters everywhere except the architecture**, and report the test-accuracy gap. The result motivates Week 9 (convolutional networks): the gap should be at least ten percentage points, and that gap is the entire reason CNNs replaced MLPs for vision tasks in the 2010s.

The point is not to chase a specific CIFAR-10 number. The point is to *demonstrate*, on your own machine, that a hand-rolled tiny CNN beats the same-parameter-count MLP on the same data. That single experiment is a more honest motivation for Week 9 than any tutorial that asserts the result.

PyTorch / torchvision references for this challenge:

- CIFAR-10 dataset: <https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html>
- `nn.Conv2d`: <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>
- `nn.MaxPool2d`: <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>
- Krizhevsky 2009 (the CIFAR-10 paper): <https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf>

## What you will produce

A single script `cifar10_mlp_vs_cnn.py` (or notebook) and a one-page markdown writeup `notes.md` that:

1. Loads CIFAR-10 via `torchvision.datasets.CIFAR10(root="./data", train=..., download=True, transform=...)` with the standard CIFAR-10 normalization (mean `(0.4914, 0.4822, 0.4465)`, std `(0.2470, 0.2435, 0.2616)`).
2. Trains an **MLP baseline**: `Flatten -> Linear(3072, h) -> ReLU -> Linear(h, h) -> ReLU -> Linear(h, 10)` with `h = 256`. Roughly 850k parameters.
3. Trains a **tiny CNN**: two `Conv2d -> ReLU -> MaxPool` blocks followed by a `Flatten -> Linear -> Linear` head, sized so the total parameter count is within 20% of the MLP. The C5 reference CNN: `Conv2d(3, 32, 3, padding=1) -> ReLU -> MaxPool(2) -> Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool(2) -> Flatten -> Linear(64*8*8, 128) -> ReLU -> Linear(128, 10)`. Roughly 600k parameters.
4. Trains both with **identical hyperparameters**: `optim.Adam(lr=1e-3)`, `batch_size=128`, `n_epochs=15`, `torch.manual_seed(42)`, the eight-line training loop from Lecture 2.
5. Records the train loss per epoch, the test accuracy per epoch, the parameter count, and the wall-clock time for each model.
6. Saves a 2-panel figure `mlp_vs_cnn.png`: left panel, test accuracy per epoch with both models; right panel, train loss per epoch with both models (both panels use the same x-axis).
7. Writes `notes.md` (~250 words) that reports the final numbers and answers two questions: (a) what is the test-accuracy gap, in percentage points? (b) the CNN has fewer parameters than the MLP — why does it still win?

## Acceptance criteria

- [ ] `cifar10_mlp_vs_cnn.py` runs end-to-end with `python cifar10_mlp_vs_cnn.py` on a CPU. Expected runtime: ~30-60 min on a laptop CPU; ~5-10 min on a recent GPU. Document your runtime in `notes.md`.
- [ ] `python -m py_compile cifar10_mlp_vs_cnn.py` succeeds.
- [ ] The MLP reaches between **48% and 55% test accuracy** after 15 epochs. Lower is a bug; higher means you over-engineered.
- [ ] The CNN reaches at least **65% test accuracy** after 15 epochs. The C5 expectation is 68-72%; reaching higher than 73% with this exact architecture is unusual without data augmentation, so document the run if you do.
- [ ] The test-accuracy gap (CNN − MLP) is at least **10 percentage points**.
- [ ] `mlp_vs_cnn.png` shows both curves clearly with a legend that includes the final test accuracy and the parameter count for each model.
- [ ] `notes.md` is honest about *why* the CNN wins — see hints below.
- [ ] The training script uses `torch.manual_seed(42)` and prints the final test accuracy for both models.

## Hints

<details>
<summary>Why the CNN wins (the short version, for your notes.md)</summary>

The MLP treats each of the 3072 input pixels as an independent feature; it does not know that pixel 0 and pixel 1 are neighbors in the image. The CNN's `Conv2d` layer is a *parameter-shared local-receptive-field* feature extractor: the same 3×3 filter slides over the whole image. Two consequences:

1. **Translation invariance.** A car in the top-left and a car in the bottom-right produce similar feature responses through the same filter. The MLP has to learn this invariance from data alone, which on 50k training images is barely enough.
2. **Parameter efficiency.** A 3×3 conv with 32 output channels has `3*3*3*32 = 864 + 32 = 896` parameters. A fully-connected layer that would see the same input region (a 3×3 patch with 3 channels = 27 inputs to 32 outputs) has `27*32 + 32 = 896` parameters too — but the FC layer only learns that one patch; the conv layer applies the same 896 parameters to every patch in the image.

The C5 sentence: *the CNN's prior — "translation invariance, local feature extraction" — is a closer match to the structure of natural images than the MLP's prior — "every pixel is independent" — and the data-efficiency gap is the result.*

</details>

<details>
<summary>The reference CNN architecture, exactly</summary>

```python
class TinyCNN(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
```

Parameter count: `3*3*3*32 + 32 + 3*3*32*64 + 64 + 64*8*8*128 + 128 + 128*10 + 10 = 880 + 18,496 + 524,416 + 1,290 = 545,082`. About 545k.

The MLP at `h=256` has `3*32*32*256 + 256 + 256*256 + 256 + 256*10 + 10 = 786,432 + 256 + 65,792 + 2,570 = 855,050`. About 855k.

The CNN has fewer parameters and wins by ten percentage points. That is the headline of this challenge.

</details>

<details>
<summary>Why 15 epochs and not 50</summary>

You can push both models further with more epochs, dropout, data augmentation (`RandomCrop`, `RandomHorizontalFlip` from `torchvision.transforms.v2`), and a learning-rate schedule. The C5 default of 15 epochs is the minimum that produces a stable gap; the gap does not narrow with more epochs (both models plateau without augmentation), but the per-model absolute numbers may rise. If you have a GPU and 30 minutes to spare, redo the experiment at 50 epochs and add a paragraph to `notes.md` about what changes.

</details>

<details>
<summary>If you want a stretch: data augmentation</summary>

Replace the train transform with:

```python
train_transform = v2.Compose([
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.ToTensor(),
    v2.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
```

The test transform should stay the basic `ToTensor + Normalize` (no augmentation at test time). With augmentation, the same CNN reaches 75-80% in 15 epochs. The MLP gains very little from augmentation — another data point in favor of "the CNN's prior is the right one." Reference: <https://pytorch.org/vision/stable/transforms.html>.

</details>

## Submission checklist

- [ ] `cifar10_mlp_vs_cnn.py` in your week-08 portfolio directory
- [ ] `mlp_vs_cnn.png` saved alongside
- [ ] `notes.md` with final numbers, the gap, the "why" answer, and your runtime
- [ ] `python -m py_compile cifar10_mlp_vs_cnn.py` exits cleanly
- [ ] The script is reproducible: `torch.manual_seed(42)` set; the same run twice produces the same numbers (within floating-point noise)

When you have completed this, you have already done the warm-up for Week 9. Push, then move on.
