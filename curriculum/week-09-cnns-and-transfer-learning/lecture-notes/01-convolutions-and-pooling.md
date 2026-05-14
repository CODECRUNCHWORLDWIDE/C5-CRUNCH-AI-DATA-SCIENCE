# Lecture 1 — Convolutions and Pooling

> **Outcome:** You can describe the 2D convolution operation mechanically and explain why it makes sense for images. You can compute the output spatial shape of a `Conv2d` layer from its hyperparameters without consulting the docs. You can argue why weight sharing, locality, and translation equivariance are appropriate inductive biases for vision and not for, for example, tabular data. You can place `MaxPool2d`, `AvgPool2d`, and `AdaptiveAvgPool2d` in a CNN and name what each one is for. By the end of this lecture, "convolution" stops being a black box marked `nn.Conv2d` and starts being an operation you could implement from scratch in NumPy in under fifty lines.

Week 8 left you with a 2-layer MLP that reached 88% on FashionMNIST. The same architecture on CIFAR-10 stalls at 50%. That gap is the entire reason we are here. Fully-connected layers throw away the most important structural fact about images: nearby pixels are correlated, and the *position* of a feature in the image rarely matters as much as its *presence*. A convolutional layer respects both facts. This lecture is the mechanics.

We target **PyTorch 2.x**; the primary references are `torch.nn.Conv2d` (<https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>) and `torch.nn.functional.conv2d` (<https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html>). The visual companion is the distill.pub receptive-field essay (<https://distill.pub/2019/computing-receptive-fields/>). Open both.

---

## 1. The MLP-on-images problem

A 32×32 RGB CIFAR-10 image has `3 * 32 * 32 = 3072` pixel intensities. The Week 8 MLP flattened that to a `(3072,)` vector and fed it to `nn.Linear(3072, h)`. With `h = 256`, the first layer has `3072 * 256 + 256 = 786,688` parameters. The model has roughly **850k parameters total**, learns a separate weight from every pixel to every hidden unit, and reaches 50% test accuracy. That is the baseline.

Three things are wrong with this picture, and they are wrong for the same reason:

1. **The MLP has no notion of spatial proximity.** Pixel (0, 0) and pixel (0, 1) are physically adjacent in the image, but the MLP sees them as positions 0 and 1 in a flat vector. Swap any two pixel columns globally and the MLP, after retraining, would learn the same accuracy on the permuted dataset. That is a damning invariance: the model is throwing away the entire 2D structure of the image.
2. **The MLP has no notion of weight sharing.** A "vertical edge" feature that lives in the top-left corner of one image and the bottom-right corner of another is, to the MLP, two completely unrelated phenomena. The MLP must learn the same edge detector independently at every location it could appear. With only 50k CIFAR-10 training images, there is not enough data to do that.
3. **The MLP has too many parameters for the data.** 850k parameters fit to 50k images is a 17:1 parameter-to-example ratio. Even with regularization, this overfits in a few epochs. The Week 8 challenge's training curves showed exactly that: training accuracy climbs past 70% while test accuracy stalls at 50%.

A convolutional layer fixes all three. Weight sharing across spatial positions reduces parameter count by orders of magnitude. Locality of the receptive field encodes the "nearby pixels are correlated" prior. Translation equivariance encodes the "the position of a feature does not matter as much as its presence" prior. Three properties, one operation, and the inductive biases match the data.

---

## 2. The 2D convolution operation, mechanically

A 2D convolution layer takes an input tensor of shape `(N, C_in, H_in, W_in)` and produces an output of shape `(N, C_out, H_out, W_out)`. The parameters of the layer are a **weight tensor** of shape `(C_out, C_in, K, K)` (where `K` is the kernel size) and a **bias** of shape `(C_out,)`.

The output activation at batch `n`, output channel `c_out`, output spatial position `(i, j)` is:

```text
output[n, c_out, i, j] = bias[c_out] + sum over c_in, ki, kj of:
    weight[c_out, c_in, ki, kj] * input[n, c_in, i*stride + ki - padding, j*stride + kj - padding]
```

That is the entire operation. Three loops over the kernel position (`c_in`, `ki`, `kj`), one multiplication and one addition per loop iteration. Repeat for every batch element, output channel, and output spatial position. PyTorch's `torch.nn.functional.conv2d` is exactly this sum, dispatched to a hand-tuned cuDNN or oneDNN implementation that uses Winograd, FFT, or implicit-GEMM tricks under the hood.

The reference is at <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>. The full method signature is:

```python
torch.nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode="zeros",
)
```

We use the first five arguments routinely; the rest are specialist (we will hit `groups` again in Week 11 when we discuss depthwise separable convolutions).

> **EXPERIMENT — write a 3x3 edge detector by hand.** In a Python REPL (with `torch` installed): construct a 1×1×5×5 input that is the identity image (a `5x5` array with a single bright pixel in the center). Construct a 1×1×3×3 kernel that is the Sobel-Y edge detector: `[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]`. Call `torch.nn.functional.conv2d(input, kernel.unsqueeze(0).unsqueeze(0), padding=1)`. Inspect the output. The center value should be 0 (the kernel sums to zero on a constant region); the values above and below should mirror each other. You have just convolved a hand-designed edge detector across an image. A learned `Conv2d` layer learns ~64 such kernels per layer, automatically.

---

## 3. The four hyperparameters: kernel, stride, padding, dilation

### Kernel size

The spatial extent of the kernel. In modern CNNs, `kernel_size=3` is the overwhelming default (VGG, ResNet, EfficientNet). Two 3×3 convs back-to-back have a 5×5 effective receptive field with `2 * 9 = 18` parameters per `(in_channel, out_channel)` pair — fewer parameters than a single 5×5 conv at 25 parameters per pair, and (because of the intermediate nonlinearity) more expressive. This argument is the entire point of the VGG paper (Simonyan and Zisserman 2014, <https://arxiv.org/abs/1409.1556>): stack `3x3`s, never use a `5x5` or `7x7` except at the input stem.

Two exceptions in 2026:

1. **The ResNet stem.** ResNet uses a single `7x7` conv with stride 2 as its first layer, immediately followed by a `3x3` max pool. This is the only place modern CNNs use a large kernel. The reason is that on a 224×224 ImageNet input, the first layer needs a large receptive field to compress the spatial resolution quickly; subsequent layers use `3x3` exclusively. CIFAR-10 variants of ResNet (the `cifar_resnet` family) replace this with a single `3x3` stride-1 conv because CIFAR images are 32×32 and there is no need to downsample aggressively at the input.
2. **Depthwise separable convolutions.** MobileNet and EfficientNet use `3x3` depthwise convs followed by `1x1` pointwise convs. The `1x1` here is a true channel-mixing operation, not a spatial one. We will see this in Week 11.

### Stride

How far the kernel moves between applications. Stride 1 is the default; the output is the same spatial size as the input (up to padding adjustments). Stride 2 halves the output spatial size and is the canonical way to downsample inside a CNN. Strides of 3 or higher are rare; they discard too much information.

Stride 2 is the alternative to `MaxPool2d(2)`. The C5 conviction is: use stride-2 convs in the body of the network (this is what ResNet does), use `MaxPool2d(2)` if you want a parameter-free downsample (this is what VGG does). Both work; ResNet's strided-conv variant is slightly more flexible because the downsampling step also learns features.

### Padding

How many zeros are added around the input before the convolution is applied. The most common choices:

- **`padding=0`** ("valid" padding): the kernel only sees inside the image, so the output shrinks by `K - 1` pixels on each spatial dimension. Used in LeNet and original AlexNet but rare in modern nets because the spatial size decays too fast with depth.
- **`padding=(K-1)//2`** ("same" padding, for odd K): the output is the same spatial size as the input, at stride 1. For `K=3`, this is `padding=1`. For `K=5`, this is `padding=2`. For `K=7`, this is `padding=3`. The C5 default for every conv in the body of the network.
- **Larger padding**: used occasionally for boundary-aware applications; rare.

### Dilation

The spacing between kernel elements. `dilation=1` is the default (consecutive pixels). `dilation=2` skips every other pixel, so a 3×3 kernel covers a 5×5 area with 9 weights. Dilated convolutions are mainly used in semantic segmentation (DeepLab, U-Net variants) and in WaveNet (1-D audio). We will not use them in Week 9 but you should know they exist.

---

## 4. The output-size formula

Given an input spatial dimension `in_size`, kernel size `K`, stride `S`, padding `P`, and dilation `D`, the output spatial size is:

```text
out_size = floor((in_size + 2*P - D*(K-1) - 1) / S) + 1
```

For the common case `D=1`, this simplifies to:

```text
out_size = floor((in_size + 2*P - K) / S) + 1
```

You will run this formula a hundred times this week. Memorize it. Three worked examples:

**Example 1.** `Conv2d(3, 64, kernel_size=3, stride=1, padding=1)` on a 32×32 input.

```text
out_size = floor((32 + 2*1 - 3) / 1) + 1 = floor(31) + 1 = 32
```

Same-size padding with stride 1; the output is 32×32. The output channel count is `out_channels=64`. So the output tensor is `(N, 64, 32, 32)`.

**Example 2.** `Conv2d(64, 128, kernel_size=3, stride=2, padding=1)` on a 32×32 input.

```text
out_size = floor((32 + 2*1 - 3) / 2) + 1 = floor(31/2) + 1 = 15 + 1 = 16
```

Stride-2 downsample; the output is 16×16. This is the standard "halve the spatial dimensions, double the channels" block. The output tensor is `(N, 128, 16, 16)`.

**Example 3.** `Conv2d(3, 64, kernel_size=7, stride=2, padding=3)` on a 224×224 input (the ResNet stem).

```text
out_size = floor((224 + 2*3 - 7) / 2) + 1 = floor(223/2) + 1 = 111 + 1 = 112
```

The ResNet stem turns a 224×224 input into a 112×112 feature map. The C5 convention is to compute these by hand on every conv block of every architecture you read; after Week 9 you will do this on sight.

> **EXPERIMENT — verify the formula.** In a REPL, create `x = torch.zeros(1, 3, 32, 32)`. Apply each of the three examples above with `conv = nn.Conv2d(...); print(conv(x).shape)`. Verify the output shapes match the formula. Now break things on purpose: try `Conv2d(3, 1, kernel_size=4, stride=1, padding=1)` on the 32×32 input. The output is 31×31 (one less than the input) — even-kernel "same" padding does not exist; you have to choose between losing a row on the left or a row on the right. The convention in PyTorch is to lose it on the right. This is one of several reasons modern nets use odd kernel sizes exclusively.

---

## 5. Why convolutions work: three inductive biases

A convolution makes three architectural assumptions that fully-connected layers do not. They are the inductive biases that explain why CNNs reach 90%+ on CIFAR-10 with fewer parameters than the MLP and why MLPs do not.

### Weight sharing

The same kernel weights are applied at every spatial position. A `Conv2d(64, 128, 3)` layer has `9 * 64 * 128 + 128 = 73,856` parameters regardless of whether the input is 32×32 or 224×224. The "same" `Linear` layer between two 32×32×64 feature maps would have `(32 * 32 * 64) * (32 * 32 * 128) = 8.6 billion parameters`. Weight sharing is a four-orders-of-magnitude parameter reduction.

What weight sharing buys conceptually: the same feature detector (an edge of a particular orientation, a corner, a small color blob) is reused at every position in the image. The model learns "what to look for" once, not "what to look for at position (5, 7)" separately from "what to look for at position (12, 18)." For natural images, this is the right prior. For permuted-pixel datasets where each pixel is its own independent variable (a textbook example: MNIST with the pixel order randomly shuffled and the shuffle held fixed), this is the wrong prior, and a convolutional net would not beat an MLP. The fact that natural images are not pixel-shuffled is the entire reason CNNs work.

### Locality

Each output activation depends only on a small window of the input — the receptive field of that layer. A single `Conv2d(C_in, C_out, 3, padding=1)` has a 3×3 receptive field at the previous layer's resolution; stacking convolutions enlarges it. The receptive field grows linearly with depth in a non-strided network and roughly exponentially in a network with stride-2 downsampling.

What locality buys conceptually: most "features" in images are local. An edge is a property of a 3×3 patch; a corner is a property of a 5×5 patch; an eye is a property of a 30×30 patch. The model does not need to look at the whole image to detect any single feature, so it should not be wired to. Locality is the architectural enforcement of this prior.

### Translation equivariance

If you shift the input by `(Δi, Δj)`, the output of a convolutional layer shifts by the same `(Δi, Δj)` (up to boundary effects). This is true of `Conv2d` by construction. It is not true of `Linear` layers, where the parameters depend on the absolute pixel position.

Translation equivariance is not the same as translation invariance — the latter is what you usually *want* for image classification, and it is achieved by chaining several equivariant conv layers with a final global pool. The C5 reading: a CNN is "equivariant by layers, invariant at the head." This is the reason a CIFAR-10 cat at the top-left and a CIFAR-10 cat at the bottom-right get classified as the same class — the conv stack moves the activation pattern with the cat, and the global pool collapses position information at the end.

> **EXPERIMENT — observe translation equivariance.** Build a random `Conv2d(3, 16, 3, padding=1)` (no need to train it; the property is structural). Pass two inputs through it: an image and the same image shifted by 5 pixels to the right (use `torch.roll(image, shifts=5, dims=-1)`). Compare the output feature maps: the second one should equal the first one rolled by 5 pixels to the right (up to boundary effects). The property does *not* hold for a random `Linear` layer applied to the flattened pixels; verify that too. This experiment is the difference between "I read about translation equivariance" and "I have seen it."

---

## 6. Pooling: max, average, and adaptive

Pooling is a non-parametric (no learnable weights) operation that aggregates a spatial neighborhood into a single value. PyTorch ships three pool layers we care about:

- **`nn.MaxPool2d(kernel_size, stride=None, padding=0)`**: take the maximum value in each window. The default `stride` equals `kernel_size`, so `MaxPool2d(2)` halves both spatial dimensions. The most common pooling op in CNN bodies. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>.
- **`nn.AvgPool2d(kernel_size, stride=None, padding=0)`**: take the mean. Used occasionally in modern architectures (DenseNet, some Inception variants), rarer than max-pool. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html>.
- **`nn.AdaptiveAvgPool2d(output_size)`**: takes the mean within whatever window is necessary to produce the target output size, regardless of input size. `AdaptiveAvgPool2d((1, 1))` is the **global average pool** at the head of every modern CNN. It collapses an `(N, C, H, W)` feature map to `(N, C, 1, 1)`, which is then flattened to `(N, C)` for the final classification layer. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html>.

The output-size formula for `MaxPool2d` is the same as for `Conv2d` (with the corresponding hyperparameters); for `MaxPool2d(2)`, the output is half the input size.

### Why max-pool

Three reasons.

1. **Downsampling.** Halves the spatial dimensions, which means the next conv layer has a 2x larger effective receptive field for the same kernel size.
2. **Slight translation invariance.** If a feature shifts by one pixel inside the 2×2 window, the max-pool output is unchanged. This is local invariance, not global, but it makes the model robust to small jitter.
3. **Cheap.** No parameters, no FLOPs other than 4 comparisons per spatial position. A free downsample.

### Why global average pool

The classification head of every modern CNN looks like:

```text
... -> AdaptiveAvgPool2d((1, 1)) -> Flatten -> Linear(C, n_classes)
```

That `(1, 1)` adaptive average pool replaces the fully-connected head of AlexNet and VGG (which had ~120M parameters in the FC layers alone, more than the entire conv body). The replacement is the reason ResNet-18 has only 11.7M parameters total — the global pool is parameter-free, and the final classification layer is a tiny `Linear(512, 1000)` mapping the 512-dimensional global-pooled embedding to 1000 ImageNet classes.

The choice of *average* over *max* at the global pool is empirical: average tends to give better-calibrated logits because every spatial location contributes to the final score. The original ResNet paper used global average pool; every architecture since has copied that choice.

### Stride-2 conv vs. pooling

In 2026, most architectures use stride-2 convolutions for downsampling inside the body and global average pool at the head. Older architectures (AlexNet, VGG, LeNet) used `MaxPool2d` for downsampling and fully-connected layers at the head. Either pattern works; the stride-2-conv pattern is slightly more flexible because the downsampling step learns features.

---

## 7. Receptive field arithmetic

The **receptive field** of a hidden activation is the set of input pixels that can affect that activation's value. It grows with depth, kernel size, stride, and dilation.

The recursive formula for the receptive field after `L` layers, where each layer has kernel `k_l`, stride `s_l`, and dilation `d_l`:

```text
RF_0 = 1
J_0 = 1
RF_l = RF_{l-1} + (d_l * (k_l - 1)) * J_{l-1}
J_l = J_{l-1} * s_l
```

Where `J_l` is the **jump** (effective stride) at layer `l`. The recurrence accumulates the kernel-induced expansion at each layer, scaled by the current jump.

Worked example for a 4-layer CNN: `Conv2d(3, 64, 3, s=1, p=1) -> MaxPool2d(2) -> Conv2d(64, 128, 3, s=1, p=1) -> MaxPool2d(2)`. All convs have kernel 3, stride 1, dilation 1. All pools have kernel 2, stride 2, dilation 1.

```text
RF_0 = 1                                J_0 = 1
After Conv1: RF_1 = 1 + 2*1 = 3         J_1 = 1*1 = 1
After Pool1: RF_2 = 3 + 1*1 = 4         J_2 = 1*2 = 2
After Conv2: RF_3 = 4 + 2*2 = 8         J_3 = 2*1 = 2
After Pool2: RF_4 = 8 + 1*2 = 10        J_4 = 2*2 = 4
```

So the activations at the output of this 4-layer block have a 10×10 receptive field on the input image, with a jump (stride) of 4 between adjacent activations. For a 32×32 CIFAR-10 input, the output feature map is `8x8` at 128 channels, and each of those 64 spatial positions sees a 10×10 region of the input.

Reference for the derivation: the distill.pub essay at <https://distill.pub/2019/computing-receptive-fields/> has interactive figures. The 2017 paper "A guide to convolution arithmetic for deep learning" (Dumoulin and Visin, <https://arxiv.org/abs/1603.07285>) is the formal reference.

> **EXPERIMENT — verify the receptive field empirically.** Build the 4-layer network above (no need to train). Pick one output activation, say `output[0, 0, 4, 4]`. Backprop the gradient `∂output[0, 0, 4, 4] / ∂input` by setting all other activations to zero and calling `.backward()` (you may need `retain_grad=True` on the input). The non-zero region of the gradient on the input is the receptive field of that activation. Count the pixels; it should match the analytic 10×10. If you mistakenly pick an activation near the boundary, the receptive field will be clipped — that is fine, just confirms the formula handles boundaries.

---

## 8. The C5 reference CNN block

For Week 9 (and most of Weeks 10–11), the canonical CNN block we will reach for is:

```python
import torch
from torch import nn

class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU -> [optional MaxPool2d(2)]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))
```

Five notes:

1. **`bias=False` on the conv when followed by BatchNorm.** The BN layer has its own learnable bias (`beta`); the conv bias would be redundant and absorbed into BN's shift, so we omit it. This is standard practice. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>.
2. **BatchNorm before ReLU.** The original BN paper proposed BN-after-ReLU; the field landed on BN-before-ReLU because it consistently trains faster. Either order works; we use the modern convention.
3. **`inplace=True` on ReLU.** A small memory optimization: the ReLU overwrites its input rather than allocating a fresh tensor. Safe because the input to ReLU here is the BN output, which we do not need to keep around. Skip the `inplace=True` if you have ever been bitten by an "a variable needed for gradient computation has been modified" error; the default `inplace=False` is universally safe.
4. **`MaxPool2d(2)` as the downsample.** Halves the spatial dimensions. The Identity pass-through (when `downsample=False`) means the same block class handles both same-size and downsampling positions in the network.
5. **3×3 kernel with `padding=1`.** Same-size padding by default; the conv preserves spatial dimensions; the optional pool is the only thing that downsamples.

We will stack these `ConvBlock`s in Exercise 2 to build a small VGG-style CNN for CIFAR-10 and we will use the same structure (with residual connections added) as a teaching reference in Lecture 2.

---

## 9. Why the math is fast: the cuDNN secret

`torch.nn.Conv2d.forward` does not actually run a six-nested loop over `(N, C_out, H_out, W_out, C_in, K, K)`. That would be hopelessly slow. The actual implementation lowers the convolution to one of three optimized algorithms, picked at runtime by cuDNN (on NVIDIA GPUs) or oneDNN (on CPUs):

1. **Implicit GEMM**: rearrange the input patches into a 2D matrix (the `im2col` operation) and the kernel into a 2D matrix; the convolution becomes a single GEMM (general matrix multiplication) call, which is the most optimized routine in scientific computing. This is the default for most kernel sizes.
2. **Winograd**: a clever factorization that reduces the multiplication count for small kernels (3×3) at the cost of more additions. About 2x speedup over GEMM for the common case.
3. **FFT-based**: turn the convolution into a pointwise multiplication in the frequency domain. Only competitive for very large kernels; rare in CNNs but useful in some signal-processing contexts.

PyTorch picks one of these automatically. The selection can be biased with `torch.backends.cudnn.benchmark = True`, which runs all three on the first batch and picks the fastest. For Week 9 you do not need to think about this; the default is good enough. The C5 conviction is that you should *know* this is how it works, even if you do not have to touch it.

---

## 10. What we have done and what is next

You can now:

- State the convolution operation as a six-nested sum and identify the role of every hyperparameter (`kernel_size`, `stride`, `padding`, `dilation`).
- Compute the output spatial shape of any `Conv2d` or `MaxPool2d` layer from the formula.
- Name the three inductive biases of convolution (weight sharing, locality, translation equivariance) and argue why they are appropriate for natural images.
- Place `MaxPool2d`, `AvgPool2d`, and `AdaptiveAvgPool2d` in a CNN and name what each one is for.
- Compute the receptive field of a stack of conv-and-pool layers from the recurrence.
- Recognize the C5 reference `ConvBlock` (Conv → BN → ReLU → optional MaxPool) and reproduce it from memory.

Lecture 2 puts these blocks together into the CNN architectural lineage: LeNet → AlexNet → VGG → ResNet. The conceptual move from LeNet to AlexNet is one of scale. The move from AlexNet to VGG is one of regularity (stack 3×3s, never 5×5 or 7×7). The move from VGG to ResNet is the residual connection that makes depth work. By the end of Lecture 2 you will have built a small CNN that reaches 70% on CIFAR-10 and you will know why ResNet is the architecture you reach for when 70% is not enough.

Lecture 3 then leaves the from-scratch path and introduces transfer learning: load a pretrained ResNet-18, swap the head, and reach 90%+ on the same dataset in fewer epochs than your hand-built CNN took to reach 70%. That is the lecture where you stop fighting the architecture and start riding it.
