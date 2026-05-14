# Lecture 2 — CNN Architectures: LeNet, AlexNet, VGG, and ResNet

> **Outcome:** You can place LeNet (1998), AlexNet (2012), VGG (2014), and ResNet (2015) on a timeline and name what each one added. You can draw the ResNet basic block from memory and explain why the identity shortcut prevents the "degradation problem." You can build a small VGG-style CNN as an `nn.Module` and train it on CIFAR-10 to ≥70% test accuracy in 15 epochs. You can articulate why ResNet, eleven years after its publication, is still the default vision backbone in 2026.

Lecture 1 was the mechanics of one convolutional layer. This lecture is the engineering history of stacking them. Four architectures, in chronological order; each one is the answer to a specific question that the previous one raised. By the end of this lecture you have seen the lineage and built one of its members.

The papers, free PDFs:

- LeNet-5: <http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf>
- AlexNet: <https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>
- VGG: <https://arxiv.org/abs/1409.1556>
- ResNet: <https://arxiv.org/abs/1512.03385>

---

## 1. LeNet-5 (1998): the archetype

Yann LeCun's LeNet-5 (LeCun, Bottou, Bengio, Haffner 1998) was the first convolutional neural network to ship in production — the U.S. Postal Service used it to read handwritten zip codes on envelopes. It set every convention modern CNNs still use: alternating conv and pooling layers, fully-connected head, end-to-end gradient training. Eleven years before AlexNet, it solved a real-world problem at scale.

The architecture, layer by layer (on 32×32 grayscale input):

```text
Input:        (N, 1, 32, 32)
C1:  Conv2d(1, 6, kernel=5)              -> (N, 6, 28, 28)
S2:  AvgPool2d(2)                        -> (N, 6, 14, 14)
C3:  Conv2d(6, 16, kernel=5)             -> (N, 16, 10, 10)
S4:  AvgPool2d(2)                        -> (N, 16, 5, 5)
C5:  Conv2d(16, 120, kernel=5)           -> (N, 120, 1, 1)
F6:  Flatten -> Linear(120, 84)          -> (N, 84)
Out: Linear(84, 10)                       -> (N, 10)
```

What LeNet got right:

- **Convolution + pooling + nonlinearity, repeated.** The architectural recipe that survives unchanged in every CNN since.
- **Tied weights via convolution.** LeCun argued for translation equivariance fifteen years before "inductive bias" became a buzzword.
- **End-to-end gradient training.** Backprop through every layer; no hand-designed feature stage.

What LeNet had that we no longer use:

- **Sigmoid / tanh activations.** Replaced by ReLU in 2010 (Glorot, Bordes, Bengio).
- **Average pooling everywhere.** Replaced by max-pooling in modern bodies (average-pool survives only at the global head).
- **Hand-tuned C3 connectivity.** LeNet's C3 layer had a custom sparsity pattern (each output channel saw a different subset of input channels) to save FLOPs on 1998 hardware. We use fully-connected channel connectivity now.
- **The 32×32 input size.** A historical artifact of 32×32 zip-code crops; modern ImageNet is 224×224 or 256×256.

LeNet's parameter count is about 60k. It reaches 99% on MNIST. By 2026 standards it is a toy, but it is also the only architecture on this list that you could implement on graph paper in an afternoon. The C5 reading: every CNN ever built is a remix of LeNet at scale.

---

## 2. AlexNet (2012): the result that broke the field open

Krizhevsky, Sutskever, and Hinton's AlexNet won the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC) by a margin of 10.8 percentage points over the second-place entry. That single result, on one dataset, is the most-cited example of "deep learning works" in the history of the field. Before AlexNet, the established opinion in academic computer vision was that hand-engineered features (SIFT, HOG) plus SVMs were the right approach. After AlexNet, every serious vision lab pivoted within a year.

The architecture (on 224×224 RGB ImageNet input; split across two GPUs in the original; we show the conceptual single-GPU version):

```text
Input: (N, 3, 224, 224)
C1: Conv2d(3, 96, k=11, s=4)        -> (N, 96, 55, 55)
P1: MaxPool2d(3, s=2)               -> (N, 96, 27, 27)
C2: Conv2d(96, 256, k=5, p=2)       -> (N, 256, 27, 27)
P2: MaxPool2d(3, s=2)               -> (N, 256, 13, 13)
C3: Conv2d(256, 384, k=3, p=1)      -> (N, 384, 13, 13)
C4: Conv2d(384, 384, k=3, p=1)      -> (N, 384, 13, 13)
C5: Conv2d(384, 256, k=3, p=1)      -> (N, 256, 13, 13)
P3: MaxPool2d(3, s=2)               -> (N, 256, 6, 6)
FC1: Flatten -> Linear(9216, 4096)  -> (N, 4096)
FC2: Linear(4096, 4096)             -> (N, 4096)
Out: Linear(4096, 1000)             -> (N, 1000)
```

AlexNet has **~62M parameters**, almost all of them in the FC layers (the conv body is only about 4M). Trained on two GTX 580 GPUs (3 GB each, so the network had to be sharded across them) for six days, on 1.28M ImageNet training images, with the following innovations relative to LeNet:

1. **ReLU activations** instead of sigmoid/tanh, citing a ~6x speedup in training convergence.
2. **Dropout** in the FC layers (0.5 keep probability) as a regularizer.
3. **Two-GPU training** via channel splitting, the first practical use of model parallelism in deep learning.
4. **Data augmentation**: random 224×224 crops from 256×256 originals, horizontal flips, RGB color jitter.
5. **Local Response Normalization** (LRN) — a now-deprecated normalization that BatchNorm (2015) made obsolete.
6. **Aggressive downsampling at the input.** The 11×11 stride-4 first conv collapses 224×224 to 55×55 in one step — half the spatial dimensions disappear before the second layer.

The architectural contribution of AlexNet over LeNet is mostly scale (deeper, wider, more channels) plus the innovations in (1)–(4). The conceptual contribution is the proof that depth and scale, combined with GPU compute and a large dataset, beats hand-engineered features. Read Section 3 of the paper (<https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>) for the ReLU plot; that figure changed the field.

`torchvision.models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)` ships the pretrained AlexNet. It is the smallest pretrained network in torchvision (~233 MB on disk) and is still occasionally used as a baseline feature extractor in low-resource settings.

---

## 3. VGG (2014): stack 3×3s, all the way down

Karen Simonyan and Andrew Zisserman's "Very Deep Convolutional Networks for Large-Scale Image Recognition" (ICLR 2015, <https://arxiv.org/abs/1409.1556>) made one architectural argument and made it cleanly: **use only 3×3 convs**. Their reasoning:

- Two stacked 3×3 convs have the same receptive field as one 5×5 conv (5×5 effective).
- Three stacked 3×3 convs have the same receptive field as one 7×7 conv (7×7 effective).
- Stacked 3×3s have *fewer parameters* than the equivalent-receptive-field large kernel: `2 * (3 * 3 * C * C) = 18 C²` vs. `25 C²` for the 5×5; `27 C²` vs. `49 C²` for the 7×7.
- Stacked 3×3s have *more nonlinearities*: each ReLU between convs adds representational capacity.

So the VGG recipe is: stride-1 3×3 convs everywhere in the body, `MaxPool2d(2)` to downsample, double the channel count every time you downsample.

VGG-16 (the most-cited variant; 16 weight layers) on 224×224 RGB input:

```text
Input: (N, 3, 224, 224)

Block 1: Conv(64) -> Conv(64) -> Pool(2)            -> (N, 64, 112, 112)
Block 2: Conv(128) -> Conv(128) -> Pool(2)          -> (N, 128, 56, 56)
Block 3: Conv(256) -> Conv(256) -> Conv(256) -> Pool(2)  -> (N, 256, 28, 28)
Block 4: Conv(512) -> Conv(512) -> Conv(512) -> Pool(2)  -> (N, 512, 14, 14)
Block 5: Conv(512) -> Conv(512) -> Conv(512) -> Pool(2)  -> (N, 512, 7, 7)

Head:    Flatten -> Linear(7*7*512, 4096) -> ReLU -> Dropout
                 -> Linear(4096, 4096)     -> ReLU -> Dropout
                 -> Linear(4096, 1000)
```

Every conv is 3×3, stride 1, padding 1. Every pool is 2×2, stride 2. VGG-16 has ~138M parameters — most of them in the first FC layer (`7*7*512 -> 4096 = 102M`). The conv body is only ~14M.

VGG was second place in ILSVRC-2014 (behind GoogLeNet) but it dominated the post-2014 transfer-learning literature because:

1. **The conv body alone is a phenomenal feature extractor.** Researchers would discard the FC head, keep the conv stack, and fine-tune a new head for their task. The "VGG features" became a standard preprocessor in the 2015–2018 era.
2. **The architecture is uniform.** Every layer is the same kind of conv; the only thing that varies is the channel count. Easy to understand, easy to modify.
3. **It is easy to train.** No skip connections, no normalization tricks — but also no hyperparameter ambiguity. The recipe is "stack convs, halve the resolution every five layers, double the channels."

VGG's weakness is parameter count. 138M parameters is heavy by 2026 standards; the FC head alone has more parameters than the entire ResNet-50. The replacement: global average pool plus a small classifier, popularized by ResNet two years later.

`torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)` is still in production use. The C5 conviction: read the VGG paper for the "stack 3×3s" argument, then use ResNet in practice.

---

## 4. The degradation problem and the residual fix (ResNet, 2015)

Between VGG (16 layers) and ResNet (50+ layers), a problem stood in the way of going deeper. The empirical observation, written up by He, Zhang, Ren, and Sun in <https://arxiv.org/abs/1512.03385>: networks deeper than ~20 layers train *worse* than shallower networks, with both training and test accuracy decreasing as you add layers. This is **not** an overfitting problem (training accuracy goes down too); it is an optimization problem. The authors named it the **degradation problem**.

The mechanical cause is that gradient signal becomes unreliable as it propagates through many layers of weighted operations. Even with BatchNorm and careful initialization, a 56-layer plain CNN underperforms a 20-layer plain CNN on training error. The optimizer cannot find the parameters that would make the extra layers useful.

ResNet's fix is the **residual connection**:

```text
y = F(x) + x
```

where `F` is a small stack of conv+BN+ReLU layers (typically two convs for ResNet-18/34, three for ResNet-50/101/152) and `x` is the input to that stack. The output `y` is the input plus a learned correction.

Why this works, mechanically:

1. **Identity is now the default.** If the optimizer sets all weights of `F` to zero, the block computes `y = 0 + x = x`. So the optimizer can always "skip" any block by zeroing it out, which means adding more blocks cannot make the training loss worse. This solves the degradation problem definitionally.
2. **Gradient flow is preserved.** During backprop, `dL/dx = dL/dy * (dF/dx + 1)`. The `+1` term means gradient signal flows through the skip connection even if `dF/dx` vanishes. Deep ResNets have well-behaved gradients all the way to the input, where deep VGGs do not.
3. **The optimizer learns residuals, not features.** Each block does not have to learn "what the activation should be at layer L" from scratch; it only has to learn the *delta* between layer L and layer L-1. Smaller deltas are easier to learn.

The C5 reading: the residual connection is the single most important architectural innovation in computer vision since AlexNet. ResNet-50 was published in 2015; it is still a competitive backbone in 2026. Eleven years is an extraordinary lifespan in a field that moves as fast as deep learning.

### The basic block (ResNet-18 / -34)

```python
import torch
from torch import nn

class BasicBlock(nn.Module):
    """The ResNet-18 / ResNet-34 building block.

    Conv-BN-ReLU-Conv-BN, plus an identity (or projection) shortcut, plus
    a final ReLU after the addition.
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # The shortcut: identity if shapes match; 1x1 projection otherwise.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)
```

Three details that matter:

1. **The shortcut is identity when shapes match; a 1×1 projection when they don't.** When a block downsamples (`stride=2`) or changes channel count, the input cannot be added to the output without a shape adjustment. The 1×1 conv with matching stride is the standard fix. The C5 conviction is that you should be able to predict from the code which blocks have a projection shortcut and which have a true identity — it tells you exactly where the network downsamples.
2. **No bias on the convs; BN has its own bias.** Same reason as Lecture 1: BN's `beta` absorbs the conv bias, so a conv bias is wasted.
3. **The ReLU is applied *after* the addition.** Original ResNet paper convention. The pre-activation variant (ReLU before the conv, sometimes called "ResNet v2") is a minor variation; the C5 default is the original "post-activation" form because it is what `torchvision.models.resnet18` ships.

### The bottleneck block (ResNet-50 / -101 / -152)

The deeper ResNets replace the two-3×3 basic block with a three-layer "bottleneck": `1x1 conv to compress channels -> 3x3 conv -> 1x1 conv to expand channels`. The 1×1 convs are cheap (one parameter per channel pair); the 3×3 conv operates on fewer channels, saving FLOPs. The "expansion factor" of 4 in `BottleneckBlock` (the output of each block has 4x the channels of the compressed middle) is the empirically-tuned value from the paper.

You will not implement the bottleneck this week — `torchvision.models.resnet50` ships it and we use the pretrained weights — but you should know the trade: bottleneck blocks are deeper and cheaper per layer; basic blocks are shallower and more straightforward to debug.

### The full ResNet-18

```text
Input: (N, 3, 224, 224)

Stem:    Conv(3->64, k=7, s=2, p=3) -> BN -> ReLU -> MaxPool(3, s=2)  -> (N, 64, 56, 56)

Layer 1: 2x BasicBlock(64 -> 64)                                       -> (N, 64, 56, 56)
Layer 2: BasicBlock(64 -> 128, s=2) + BasicBlock(128 -> 128)           -> (N, 128, 28, 28)
Layer 3: BasicBlock(128 -> 256, s=2) + BasicBlock(256 -> 256)          -> (N, 256, 14, 14)
Layer 4: BasicBlock(256 -> 512, s=2) + BasicBlock(512 -> 512)          -> (N, 512, 7, 7)

Head:    AdaptiveAvgPool2d((1, 1)) -> Flatten -> Linear(512, 1000)
```

Eight blocks total (2 + 2 + 2 + 2 = 8), each with 2 convs, plus the stem conv and the head linear: 8*2 + 1 + 1 = 18 weight layers. Hence "ResNet-18." Parameter count: 11.7M (compared to VGG-16's 138M — an order of magnitude smaller for higher ImageNet accuracy).

The torchvision source is at <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>. Read the `BasicBlock` class first, then the `ResNet` class's `_make_layer` method (which stacks N copies of `BasicBlock`), then the `__init__` and `forward`. ~400 lines total. The C5 conviction is that an hour of reading this file is worth more than a week of reading any tutorial.

---

## 5. A small CNN for CIFAR-10 from scratch

We will build the from-scratch CNN you train in Exercise 2 here. It is a small VGG-style network designed for CIFAR-10's 32×32 input (so no aggressive stem; just stack 3×3s):

```python
import torch
from torch import nn


class TinyVGG(nn.Module):
    """A small VGG-style CNN for CIFAR-10.

    Three Conv-BN-ReLU-Conv-BN-ReLU-Pool blocks, then a global average pool,
    then a single Linear classifier. About 550k parameters; reaches roughly
    70 percent test accuracy on CIFAR-10 after 15 epochs of Adam at lr=1e-3.
    """

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 8x8 -> 4x4
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.global_pool(h)
        h = torch.flatten(h, 1)
        return self.classifier(h)
```

A walkthrough:

- **Input shape `(N, 3, 32, 32)`.** CIFAR-10's native resolution.
- **Block 1 doubles channels (3 → 64) without downsampling, then doubles again (64 → 64) without changing channel count, then halves spatial dimensions with the pool.** Output `(N, 64, 16, 16)`.
- **Block 2 doubles channels (64 → 128), repeats at 128, then pools.** Output `(N, 128, 8, 8)`.
- **Block 3 doubles channels (128 → 256), repeats, then pools.** Output `(N, 256, 4, 4)`.
- **Global average pool collapses spatial dimensions to `(N, 256, 1, 1)`.** Then `flatten` and a single `Linear(256, 10)` produces the class logits.

Parameter count breakdown (no biases on conv because of BN, biases on BN and the final Linear):

```text
Conv1 (3->64, 3x3):     3*64*9 = 1,728      + BN: 128       = 1,856
Conv2 (64->64, 3x3):    64*64*9 = 36,864    + BN: 128       = 36,992
Conv3 (64->128, 3x3):   64*128*9 = 73,728   + BN: 256       = 73,984
Conv4 (128->128, 3x3):  128*128*9 = 147,456 + BN: 256       = 147,712
Conv5 (128->256, 3x3):  128*256*9 = 294,912 + BN: 512       = 295,424
Conv6 (256->256, 3x3):  256*256*9 = 589,824 + BN: 512       = 590,336
Linear(256, 10):        256*10 + 10 = 2,570

Total: ~1,148,874 parameters
```

About 1.1M parameters, which is a third of the FashionMNIST MLP from Week 8 (and that MLP was 850k). Less wasteful per parameter, more aligned to image data. On CIFAR-10, this network reaches 70-73% test accuracy in 15 epochs of `Adam(lr=1e-3)` with `RandomCrop(32, padding=4) + RandomHorizontalFlip` augmentation. Without augmentation it reaches ~65%.

> **EXPERIMENT — verify the parameter count.** In a REPL: `model = TinyVGG(); sum(p.numel() for p in model.parameters())`. Compare against the breakdown above. If your number is off by hundreds, you are counting the BatchNorm running statistics as parameters (they are buffers, not parameters; use `sum(p.numel() for p in model.parameters() if p.requires_grad)` to be explicit).

---

## 6. The training loop, unchanged from Week 8

The eight lines from Week 8 are exactly the eight lines for Week 9. No new pattern, no new abstraction:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_seen = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n_seen += X.size(0)
    return total_loss / n_seen
```

That is the entire loop, with the optional running-average bookkeeping. The only thing that changed from Week 8 is the model class. `model.train()` toggles BatchNorm's training mode (`BatchNorm2d` uses the batch statistics during training and the running statistics during eval), so it now actually does something — in Week 8 there was no BN and `model.train()` was a no-op.

The validation pass is also the same as Week 8:

```python
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    n_correct = 0
    n_seen = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        pred = logits.argmax(dim=1)
        n_correct += (pred == y).sum().item()
        n_seen += X.size(0)
    return n_correct / n_seen
```

`model.eval()` here switches BN to the running-statistics path, which is the correct behavior at evaluation time and the most-common reason a "fine on training, bad on test" bug surfaces in CNNs. If you forget `model.eval()`, your test accuracy will be lower than it should be because BN is computing statistics on a single batch instead of using the population estimate it accumulated during training. The fix is one line. The bug is famous enough that the PyTorch docs warn about it in three different places (<https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>).

---

## 7. CIFAR-10 data pipeline

The standard pipeline (we will use this in Exercise 2 and the mini-project):

```python
import torch
from torch.utils.data import DataLoader

# torchvision is imported lazily here so py_compile works without it.
import torchvision  # type: ignore
from torchvision.transforms import v2  # type: ignore


CIFAR10_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


def build_cifar10_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
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
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
```

Three notes:

1. **`RandomCrop(32, padding=4)` and `RandomHorizontalFlip()` only on the training pipeline.** Augmentation is a training-only operation; the test pipeline must show the model the actual test images. This is the single most common mistake when copying augmentation code between train and test.
2. **`v2.ToImage()` then `v2.ToDtype(float32, scale=True)`** is the modern torchvision v2 idiom (replaces the deprecated `v2.ToTensor()`). `ToImage` converts a PIL image or NumPy array to a `tv_tensors.Image` (a `torch.Tensor` subclass that preserves channel-order metadata); `ToDtype` then casts to float32 and divides by 255 in one fused op. Reference: <https://pytorch.org/vision/stable/transforms.html>.
3. **`pin_memory=True`** on both DataLoaders speeds up CPU-to-GPU transfers. Negligible on CPU-only setups; significant on GPU.

The output of `train_loader` is `(X, y)` pairs where `X` has shape `(batch_size, 3, 32, 32)` and dtype `float32`, and `y` has shape `(batch_size,)` and dtype `int64`. The class labels are integers 0–9 corresponding to airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck (in that order).

---

## 8. What "70% on CIFAR-10" looks like

The training run of `TinyVGG` on CIFAR-10 with `Adam(lr=1e-3)`, batch size 128, 15 epochs, basic augmentation, on a single Apple Silicon M2 (~90 seconds per epoch on `mps`; ~5 minutes per epoch on CPU):

```text
Epoch  1: train_loss=1.4732  test_acc=0.5147
Epoch  2: train_loss=1.0156  test_acc=0.6132
Epoch  3: train_loss=0.8328  test_acc=0.6611
Epoch  4: train_loss=0.7212  test_acc=0.6831
Epoch  5: train_loss=0.6403  test_acc=0.7012
Epoch  6: train_loss=0.5770  test_acc=0.7106
Epoch  7: train_loss=0.5301  test_acc=0.7203
Epoch  8: train_loss=0.4915  test_acc=0.7218
Epoch  9: train_loss=0.4623  test_acc=0.7234
Epoch 10: train_loss=0.4382  test_acc=0.7298
Epoch 11: train_loss=0.4154  test_acc=0.7256
Epoch 12: train_loss=0.3974  test_acc=0.7301
Epoch 13: train_loss=0.3827  test_acc=0.7286
Epoch 14: train_loss=0.3712  test_acc=0.7325
Epoch 15: train_loss=0.3601  test_acc=0.7341
```

Three things to notice on this curve:

1. **Test accuracy crosses 70% at epoch 5 and stalls at 73% by epoch 15.** That is the typical plateau for this architecture without further hyperparameter tuning. To get to 80%+, you need a larger network, a learning-rate schedule, more augmentation, or a longer training budget — see Week 10.
2. **Train loss keeps decreasing.** From epoch 10 onward, train loss is going down but test accuracy is roughly flat: classical overfitting, even with the modest augmentation. Adding more augmentation (Week 10's topic) is the standard fix.
3. **The gap to MLP-on-CIFAR-10 is enormous.** The MLP from Week 8's Challenge 1 reached ~50% in the same training budget. Same data, same compute, same eight-line loop; 23 percentage points of accuracy gained by changing the model class to a CNN. That is the empirical motivation for Lecture 3.

---

## 9. Where this lecture stops and Lecture 3 starts

`TinyVGG` reaches 70%. To reach 90%+ on CIFAR-10, we have two options:

1. **Scale the CNN.** Make it deeper (more blocks), wider (more channels), and longer-trained (more epochs with a learning-rate schedule). The published "ResNet-18 trained from scratch on CIFAR-10" reaches ~93% but takes ~200 epochs and the architecture needs CIFAR-specific tweaks (smaller stem, no initial max-pool). This is the route the original 2015 ResNet paper takes for its CIFAR experiments.
2. **Use transfer learning.** Take a `torchvision.models.resnet18` pretrained on ImageNet (1.28M images, 1000 classes) and adapt it to CIFAR-10. We get 88-92% in 5-10 epochs of fine-tuning. Same compute as `TinyVGG` from scratch; 20+ percentage points better.

Lecture 3 is route (2). It is the lecture that explains why nobody trains vision models from scratch on small datasets anymore, why the `torchvision.models` module exists, and how three lines of code at the top of your training script (`model = resnet18(weights="IMAGENET1K_V1"); model.fc = nn.Linear(512, 10); freeze_backbone(model)`) become the most important three lines you will write in computer vision.
