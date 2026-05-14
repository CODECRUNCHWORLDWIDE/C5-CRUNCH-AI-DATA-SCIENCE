# Week 9 — Exercise Solutions

Reference solutions with commentary. Do not read these before attempting the exercises yourself. The discomfort of staring at a half-finished `naive_conv2d` function is the discomfort that builds the mental model; reading the solution first skips it.

The solutions here are **one** correct implementation. There are usually several. If your code passes the tests and is readable, it is correct, even if it does not match the form here.

---

## Exercise 1 — Convolutions by hand

### Part A — `conv_output_size`

```python
def conv_output_size(
    in_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
```

Integer floor division is Python's `//`. The formula maps one-to-one onto the PyTorch documentation at <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>. Verify by hand on the three examples in the docstring: `(32 + 2 - 2 - 1) // 1 + 1 = 32`, `(32 + 2 - 2 - 1) // 2 + 1 = 16`, `(224 + 6 - 6 - 1) // 2 + 1 = 112`.

### Part B — `naive_conv2d`

```python
def naive_conv2d(
    x: NDArray[np.float32],
    weight: NDArray[np.float32],
    bias: NDArray[np.float32],
    stride: int = 1,
    padding: int = 0,
) -> NDArray[np.float32]:
    n_batch, c_in, h_in, w_in = x.shape
    c_out, _, k_h, k_w = weight.shape
    h_out = conv_output_size(h_in, k_h, stride=stride, padding=padding)
    w_out = conv_output_size(w_in, k_w, stride=stride, padding=padding)

    if padding > 0:
        x_padded = np.pad(
            x,
            pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        x_padded = x

    out = np.zeros((n_batch, c_out, h_out, w_out), dtype=np.float32)
    for n in range(n_batch):
        for co in range(c_out):
            for i in range(h_out):
                for j in range(w_out):
                    i0 = i * stride
                    j0 = j * stride
                    patch = x_padded[n, :, i0:i0 + k_h, j0:j0 + k_w]
                    out[n, co, i, j] = float((weight[co] * patch).sum()) + float(bias[co])
    return out
```

Commentary:

- **Four explicit loops, vectorized inner product.** The inner sum over `(c_in, ki, kj)` becomes a single `np.sum` of an elementwise product. Three nested loops in NumPy would be ~100x slower in pure Python; the elementwise product is a single C call.
- **`np.pad`** is the right tool for the padding. `mode="constant"` defaults to zero padding, which is what `Conv2d` does by default.
- **Why this is called "cross-correlation," not "convolution."** Mathematical convolution would flip the kernel before applying it; PyTorch's "convolution" does not. The flip is irrelevant for learned weights (the optimizer learns the un-flipped form), so the field has settled on calling cross-correlation "convolution" by abuse of notation. The C5 reading: when someone asks you to implement convolution in an interview, ask "PyTorch convention or signal-processing convention?" — the difference is the kernel flip.
- **Speed.** This is `O(N * C_out * H_out * W_out * C_in * K * K)`. For a 1×3×32×32 input and a 32-channel 3×3 conv, that is `1 * 32 * 32 * 32 * 3 * 9 = 884k` multiply-adds. The pure-Python overhead of four nested loops is dominant; the actual arithmetic is trivial. PyTorch's `cuDNN` does the same calculation 100-10000x faster on GPU using `im2col` + GEMM (Lecture 1 Section 9).

### Part C — Sobel-Y

```python
def sobel_y_kernel() -> NDArray[np.float32]:
    return np.array(
        [[[[-1.0, -2.0, -1.0],
           [ 0.0,  0.0,  0.0],
           [ 1.0,  2.0,  1.0]]]],
        dtype=np.float32,
    )


def horizontal_step_image(height: int = 8, width: int = 8) -> NDArray[np.float32]:
    img = np.zeros((1, 1, height, width), dtype=np.float32)
    img[0, 0, height // 2:, :] = 1.0
    return img
```

Commentary:

- The Sobel-Y kernel has positive values below the center row, negative above. Applied with stride 1 and padding 1 to a step-edge image, it produces a strong positive response on the row just above the edge (the kernel's center sits on the row where the gradient is steepest). The test verifies this empirically.
- The kernel sums to zero, so it produces zero response on constant regions. Verify by mentally convolving the kernel with the all-zero region above the step: every kernel value multiplies a zero, the sum is zero. The same logic for the all-ones region below the step: each kernel value multiplies a one, but the kernel sums to zero, so the response is still zero. Only the transition rows produce a non-zero response.

### Tests pass commentary

The `test_translation_equivariance` test is the experiment from Lecture 1 Section 5 made into a unit test. The roll-then-conv vs. conv-then-roll equivalence holds exactly (to floating-point precision) for `Conv2d` with `padding_mode="zeros"`, away from boundaries. The test only checks the interior to avoid boundary artifacts.

---

## Exercise 2 — TinyVGG on CIFAR-10

### Part B — `build_tiny_vgg`

```python
def build_tiny_vgg(n_classes: int = 10) -> "object":
    import torch
    from torch import nn

    class TinyVGG(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # Block 2
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # Block 3
                nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(256, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = self.features(x)
            h = self.global_pool(h)
            h = torch.flatten(h, start_dim=1)
            return self.classifier(h)

    return TinyVGG()
```

Commentary:

- **`bias=False` on every conv** because BN has its own bias. Saves a tiny amount of memory and is the convention.
- **`inplace=True` on ReLU.** Memory optimization; safe here because BN's output is not used after ReLU. Skip if you ever hit an "a variable needed for gradient computation has been modified" error.
- **`nn.Sequential` for the body.** Cleaner than defining each layer in `__init__` and wiring them in `forward`. The `nn.Sequential` form is the right choice when the data flows linearly through every layer; switch to a custom `forward` only when you need skip connections (which is what ResNet does in Lecture 2 Section 4).
- **Parameter count.** Run `sum(p.numel() for p in model.parameters() if p.requires_grad)` after building; you should see ~1.15M.

### Part C — `build_cifar10_loaders`

```python
def build_cifar10_loaders(
    root: str = DATA_ROOT,
    batch_size: int = 128,
    num_workers: int = 2,
) -> "Tuple[object, object]":
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2

    train_tf = v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    test_tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_tf,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_tf,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
```

Commentary:

- **Augmentation only on the training loader.** `RandomCrop` and `RandomHorizontalFlip` are training-time data augmentation; the test loader must show the actual test images.
- **`RandomCrop(32, padding=4)` is the canonical CIFAR-10 augmentation.** It pads the 32×32 image to 40×40 and then crops a random 32×32 region. The model sees slightly-shifted versions of every training image, which acts as a regularizer.
- **`v2.ToImage` then `v2.ToDtype(float32, scale=True)`** replaces the deprecated `v2.ToTensor()`. Same behavior: PIL → tensor → divide by 255.
- **`pin_memory=torch.cuda.is_available()`** only enables pinned memory when there is a GPU to transfer to. On CPU-only setups, pinned memory is wasted overhead.

### Part D — training loop

```python
def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    import torch

    model.train()
    total_loss = 0.0
    n_seen = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n_seen += X.size(0)
    return total_loss / n_seen


def evaluate(model, loader, device) -> float:
    import torch

    model.eval()
    n_correct = 0
    n_seen = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X).argmax(dim=1)
            n_correct += (pred == y).sum().item()
            n_seen += X.size(0)
    return n_correct / n_seen
```

Commentary:

- **`with torch.no_grad():` in `evaluate`.** Halves memory and roughly doubles speed for the forward-only pass. The decorator form `@torch.no_grad()` (Lecture 3) is equivalent.
- **`loss.item() * X.size(0)` for the running mean.** `loss.item()` is the *mean* loss over the batch; multiplying by the batch size recovers the *sum* of per-example losses; dividing by `n_seen` at the end gives the correct dataset-wide mean. Forgetting the `* X.size(0)` is a common subtle bug when batch sizes vary at the end of an epoch.

---

## Exercise 3 — Transfer Learning

### Part B — `build_feature_extractor`

```python
def build_feature_extractor(n_classes: int = 10) -> "object":
    import torch
    from torch import nn
    from torchvision.models import resnet18, ResNet18_Weights

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    # Step 1: freeze every parameter.
    for p in model.parameters():
        p.requires_grad = False

    # Step 2: replace the head. The new Linear's parameters default to
    # requires_grad=True, which is what we want.
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def count_trainable_parameters(model: "object") -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Commentary:

- **Freeze before swap.** Order matters. The opposite order (swap then iterate-and-freeze) freezes the new head too; nothing trains.
- **Trainable parameter count = 5130.** The new `Linear(512, 10)` has `512 * 10 = 5120` weights and `10` biases. The test checks this exactly.
- **The frozen backbone's parameters still participate in the forward pass.** They just are not updated by `optimizer.step()`. Their gradients are not computed (PyTorch's autograd notices that `requires_grad=False` on a leaf and short-circuits), which makes the backward pass faster.

### Part C — `build_cifar10_loaders_224`

```python
def build_cifar10_loaders_224(
    root: str = DATA_ROOT,
    batch_size: int = 64,
    num_workers: int = 2,
) -> "Tuple[object, object]":
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2

    train_tf = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_tf = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_tf,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_tf,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
```

Commentary:

- **`v2.Resize((224, 224))` first.** The order matters: PIL-image-time transforms (resize, crop, flip) before the `ToImage` cast.
- **ImageNet normalization, not CIFAR-10's.** This is the subtle bug from Lecture 3 Section 7. The pretrained backbone was trained with `mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)`; using CIFAR-10's per-channel stats here costs you 2-3 percentage points of test accuracy.
- **Smaller batch size (64) than the from-scratch CNN (128).** At 224×224 instead of 32×32, each batch is 49x larger in memory; halving the batch size keeps memory usage manageable on smaller GPUs.

### Part E — Fine-tune optimizer

```python
def build_finetune(n_classes: int = 10) -> "object":
    import torch
    from torch import nn
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    # No freezing.
    return model


def make_finetune_optimizer(
    model: "object",
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
) -> "object":
    import torch

    head_params = list(model.fc.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    return torch.optim.Adam(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
    )
```

Commentary:

- **`id(p)` for identity comparison.** `==` on tensors is element-wise; you want to know if two references point to the same object. `id()` returns the Python object identity (which is fine because each parameter is a distinct `nn.Parameter` instance).
- **The training loop does not change.** Per-parameter-group learning rates are handled inside `Optimizer.step()`. The C5 conviction is that PyTorch's optimizer API has been remarkably stable since 2018; this is one of the reasons.

---

## Common pitfalls (read this list at least once)

1. **Forgetting `model.train()` / `model.eval()`.** BatchNorm uses batch statistics in train mode and running statistics in eval mode. If you evaluate in train mode, your accuracy will fluctuate batch-to-batch; if you train in eval mode, BN's running statistics will not be updated and training will be unstable.
2. **Forgetting `optimizer.zero_grad()`.** Gradients accumulate across `backward()` calls by default. If you forget to zero, after 10 batches your gradients are 10x what they should be and your loss explodes.
3. **Putting `optimizer.zero_grad()` AFTER `optimizer.step()`.** Common in beginner code; mostly harmless because the next iteration still zeros before backward, but conceptually wrong. The canonical order is `zero -> forward -> backward -> step`.
4. **Using CIFAR-10 normalization with a pretrained ImageNet model (Exercise 3).** 2-3 percentage points of accuracy left on the table. Use `IMAGENET_MEAN` and `IMAGENET_STD` when feeding a pretrained backbone.
5. **Forgetting to swap `model.fc`.** A pretrained ResNet-18 has a 1000-class head; if you train it on CIFAR-10 with `CrossEntropyLoss` and 10-class labels without swapping, you will see a runtime error about target indices being out of bounds. The error message is unambiguous; the fix is one line.
6. **Swapping `model.fc` then freezing all parameters.** Freezes the new head too. Always freeze first, swap second.
7. **Using `pretrained=True` instead of the modern `weights=` API.** The legacy form still works but prints a deprecation warning and may stop working in a future PyTorch release. Always use `weights=ResNet18_Weights.IMAGENET1K_V1` or the convenience string `"IMAGENET1K_V1"`.
8. **Not setting `torch.manual_seed(...)`.** Random-init runs are not reproducible; the head's initial weights vary, and so do the test accuracies. The C5 default is `torch.manual_seed(42)` at the top of every training script.

---

## A note on the `RUN_SLOW=1` gate

Exercises 2 and 3 each have one "slow" test that actually trains the model. They are gated behind the `RUN_SLOW=1` environment variable so that `pytest` in a CI environment without GPU does not take an hour. To run the slow tests locally:

```bash
RUN_SLOW=1 pytest exercise-02-build-a-cnn.py::test_tiny_vgg_reaches_seventy_percent_on_cifar10_slow
RUN_SLOW=1 pytest exercise-03-transfer-learning.py::test_feature_extractor_reaches_eighty_five_percent_slow
```

The fast tests (forward-pass shape checks, parameter counts, freeze verification) run in under five seconds and should be passing before you start a long training run.
