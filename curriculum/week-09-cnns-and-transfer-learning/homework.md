# Week 9 — Homework

Six problems, about six hours total. Commit each in your Week 9 repo under a `homework/` directory. Each problem expects either a small `.py` file or a short Markdown write-up; specifics below.

---

## Problem 1 — Compute output shapes by hand (45 minutes)

For each of the following layer configurations, compute the output tensor shape on paper using the formula from Lecture 1 Section 4, then verify in PyTorch with the corresponding `nn.Conv2d` or `nn.MaxPool2d` call. The input batch is `(8, 3, 224, 224)` unless otherwise stated.

Configurations:

1. `Conv2d(3, 64, kernel_size=7, stride=2, padding=3)` on `(8, 3, 224, 224)`.
2. `MaxPool2d(kernel_size=3, stride=2, padding=1)` on `(8, 64, 112, 112)`.
3. `Conv2d(64, 128, kernel_size=3, stride=2, padding=1)` on `(8, 64, 56, 56)`.
4. `Conv2d(128, 128, kernel_size=3, stride=1, padding=1)` on `(8, 128, 28, 28)`.
5. `Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=2)` on `(8, 256, 14, 14)`.
6. `AdaptiveAvgPool2d((1, 1))` on `(8, 512, 7, 7)`.

For each, write the predicted output shape and the formula computation that produced it (one line each). Then run the corresponding `nn.Module` in Python and verify.

Save the work as `homework/01-output-shapes.md` (the table of predictions and verifications) and `homework/01-verify.py` (the verification script).

Acceptance: every predicted shape matches the PyTorch output exactly. `python -m py_compile homework/01-verify.py` succeeds.

References:
- `Conv2d`: <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>
- `MaxPool2d`: <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>
- `AdaptiveAvgPool2d`: <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html>

---

## Problem 2 — Visualize the first-layer filters of pretrained ResNet-18 (1 hour)

Load `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`. Inspect `model.conv1`. It is `Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)` — so the weight tensor has shape `(64, 3, 7, 7)`. Each of those 64 filters is a 7×7 RGB image you can plot.

Produce `homework/02-filters.png`: an 8×8 grid of subplots, each showing one of the 64 filters as a 7×7 RGB image (rescale each filter independently so its min maps to 0 and its max maps to 1; matplotlib's `imshow` handles RGB).

Acceptance: the figure is readable, the filters show a mix of orientation-selective edges (you should see at least a dozen filters that look like oriented Gabor-like edges), color-opponent blobs, and a few that look like center-surround detectors. This is the standard observation that pretrained CNN first layers learn similar features regardless of training task. Write `homework/02-filters.md` (~100 words) describing what you see.

References:
- ResNet-18 in torchvision: <https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html>
- The first-layer visualization is a classical result; see <https://distill.pub/2017/feature-visualization/> for context.

---

## Problem 3 — Activation maps on a single image (1 hour)

Pick one CIFAR-10 test image. Forward it through `torchvision.models.resnet18(weights=...)` and capture the output of `model.layer1`, `model.layer2`, `model.layer3`, and `model.layer4` using PyTorch forward hooks (see <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook>).

For each of the four captured tensors, take channel 0 (just to pick one), upsample it back to the input resolution with `torch.nn.functional.interpolate(mode="bilinear")`, and overlay it on the original image as a heatmap. Produce a 1×5 figure: the original image, then four heatmaps (one per layer).

Acceptance: `homework/03-activations.png`. Visually, the heatmaps should become coarser and more semantic with depth: layer1 highlights edges, layer4 highlights large parts of objects. Write `homework/03-activations.md` (~100 words) describing the trend you see.

References:
- Forward hooks: <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook>
- `F.interpolate`: <https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html>

---

## Problem 4 — Replace the head correctly across three architectures (45 minutes)

Write a single Python file `homework/04-head-swap.py` containing three functions:

```python
def swap_resnet_head(model: nn.Module, n_classes: int) -> nn.Module: ...
def swap_vgg_head(model: nn.Module, n_classes: int) -> nn.Module: ...
def swap_efficientnet_head(model: nn.Module, n_classes: int) -> nn.Module: ...
```

Each takes a pretrained model from torchvision and returns the same model with the final classification layer replaced by an `nn.Linear` of the right input dimension and `n_classes` outputs. For ResNet, the head is `model.fc`. For VGG, the head is `model.classifier[-1]` (the last layer in the FC stack). For EfficientNet, the head is `model.classifier[-1]`. Each function must work on the corresponding pretrained model and produce a model whose forward pass returns `(N, n_classes)` logits.

Acceptance: `python -m py_compile homework/04-head-swap.py` succeeds; three pytest functions (which you write) verify each swap produces the right output shape on a `(1, 3, 224, 224)` input.

References:
- `torchvision.models`: <https://pytorch.org/vision/stable/models.html>
- ResNet-18: <https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html>
- VGG-16: <https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html>
- EfficientNet-B0: <https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html>

---

## Problem 5 — Augmentation ablation on CIFAR-10 (1.5 hours)

Train the `TinyVGG` from Exercise 2 on CIFAR-10 for 15 epochs with `Adam(lr=1e-3)`, in three augmentation conditions:

1. **No augmentation** — just `ToImage -> ToDtype -> Normalize`.
2. **Flip only** — add `RandomHorizontalFlip()`.
3. **Crop + flip** — add `RandomCrop(32, padding=4)` and `RandomHorizontalFlip()` (the C5 default).

For each condition, record train accuracy and test accuracy per epoch. Produce `homework/05-augmentation.png` with three pairs of curves (3 train, 3 test) on the same axes; the train-curve fanout shows the difference in fitting capacity, the test-curve gap shows the regularization benefit.

In `homework/05-augmentation.md` (~200 words):

1. Report the final test accuracy for each condition.
2. The honest C5 reading: the crop-and-flip configuration should add roughly 5-8 percentage points of test accuracy over no augmentation. With no augmentation, train accuracy crosses 95% around epoch 8-10 while test accuracy plateaus around 65% — classic overfitting.
3. Confirm or refute in your data, and discuss what your numbers tell you about the regularization role of augmentation.

References:
- `RandomCrop`: <https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomCrop.html>
- `RandomHorizontalFlip`: <https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomHorizontalFlip.html>

---

## Problem 6 — Read and write up the ResNet paper Section 3 (1 hour)

Read **Section 3 ("Deep Residual Learning") and Section 4.1 ("ImageNet Classification")** of He, Zhang, Ren, Sun 2015 (<https://arxiv.org/abs/1512.03385>). Pages 2-6.

Write `homework/06-resnet-paper.md` (~400 words) covering:

1. **The degradation problem.** What did the authors observe about plain (non-residual) networks of 34 and 56 layers? Reference the Figure 1 plot.
2. **The residual learning formulation.** State `H(x) = F(x) + x` and explain why the authors argue this is easier to optimize than learning `H(x)` directly. Reference Section 3.1.
3. **The shortcut design choices.** When the input and output dimensions of a block match, the shortcut is identity; when they do not, the paper considers three options (A: zero-padding the channels; B: 1×1 projection shortcut; C: project all shortcuts). Which one does the C5 ResNet-18 implementation use, and why?
4. **The empirical result.** From Table 2, what is the gap between plain-34 and ResNet-34 on ImageNet top-1 error?

Write in your own words; do not paraphrase the paper. Include the equation numbers and table references you used. The C5 conviction is that you should be able to summarize Section 3 of this paper to a colleague in five minutes.

References:
- ResNet paper: <https://arxiv.org/abs/1512.03385>
- The C5 reading guide for the paper is at <../resources.md>

---

## Submission checklist

- [ ] All six problems' artifacts are in `homework/` of your Week 9 repo.
- [ ] Every `.py` file passes `python -m py_compile`.
- [ ] Every figure is reproducible by re-running the corresponding script.
- [ ] All Markdown write-ups stay within the word counts noted in each problem.
- [ ] You set `torch.manual_seed(42)` at the top of every training script.
- [ ] No emojis in any file; type hints on every Python function; the writing voice is plain and declarative.
