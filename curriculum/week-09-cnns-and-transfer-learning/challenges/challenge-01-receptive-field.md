# Challenge 1 — Compute the Receptive Field

**Time estimate:** 2 hours.

## Problem statement

Compute, on paper, the receptive field at every layer of a 6-layer CNN. Then verify your computation in PyTorch with the gradient-of-output-with-respect-to-input trick. The two results should agree exactly.

The point of this challenge is to make the recursion from Lecture 1 Section 7 muscle-memory. After two hours of computing receptive fields by hand, the question "how much of the input does this activation see?" stops being mysterious for every CNN you read for the next decade.

Reference: <https://distill.pub/2019/computing-receptive-fields/>.

## The network

A 6-layer CNN, in this exact order:

```text
Layer 1: Conv2d(in_channels=3,  out_channels=32, kernel_size=3, stride=1, padding=1)
Layer 2: MaxPool2d(kernel_size=2, stride=2)
Layer 3: Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
Layer 4: MaxPool2d(kernel_size=2, stride=2)
Layer 5: Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
Layer 6: MaxPool2d(kernel_size=2, stride=2)
```

Applied to a `(N, 3, 64, 64)` input.

## What you will produce

A single file `receptive_field.md` plus a verification script `verify_rf.py`. The markdown file has:

1. A table with one row per layer showing: layer name, kernel size, stride, padding, dilation, **receptive field at the output of this layer**, **jump (effective stride) at the output of this layer**, and the **output spatial size**.
2. A short paragraph (~150 words) on what the final-layer receptive field of 22 (or whatever you get) tells you about what the deepest activations are "seeing" of a 64×64 input.
3. The output of `python verify_rf.py`, which should print "verified: receptive field at layer 6 is 22x22 (jump = 8)."

The verify script builds the 6-layer network in PyTorch, picks one activation at a specific spatial position in the layer-6 output, and uses `.backward()` on that scalar to find which input pixels have nonzero gradient. The bounding box of the nonzero gradient is the empirical receptive field.

## The math (use Lecture 1 Section 7)

The recursion:

```text
RF_0 = 1
J_0 = 1
RF_l = RF_{l-1} + (d_l * (k_l - 1)) * J_{l-1}
J_l  = J_{l-1} * s_l
```

For each layer, compute `RF_l` and `J_l` from the previous values and the layer's `(k_l, s_l, d_l)`. The output spatial size at layer `l` follows the standard formula from Lecture 1 Section 4:

```text
out_size_l = floor((in_size_{l-1} + 2*p_l - d_l*(k_l - 1) - 1) / s_l) + 1
```

For pooling layers, treat them exactly like a conv with the same kernel and stride and `padding=0`.

## The verification trick

```python
import torch
from torch import nn


def empirical_receptive_field(
    model: nn.Module,
    input_shape: tuple[int, int, int, int],
    layer_idx: int,
    out_pos: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return (top, left, height, width) of the receptive field of one activation.

    The idea: backprop a one-hot gradient through the model and inspect
    which input pixels have nonzero gradient.
    """
    x = torch.zeros(*input_shape, requires_grad=True)
    # Forward through `layer_idx` layers (0-indexed: layer_idx=5 means six layers).
    h = x
    for k, layer in enumerate(model):
        h = layer(h)
        if k == layer_idx:
            break
    # Pick one channel and one spatial position; backprop.
    target = h[0, 0, out_pos[0], out_pos[1]]
    target.backward()
    # Find the bounding box of nonzero grad on the input.
    grad = x.grad[0].abs().sum(dim=0)  # collapse channels
    nonzero = (grad > 0).nonzero(as_tuple=False)
    if nonzero.numel() == 0:
        return 0, 0, 0, 0
    top = int(nonzero[:, 0].min())
    left = int(nonzero[:, 1].min())
    bottom = int(nonzero[:, 0].max())
    right = int(nonzero[:, 1].max())
    return top, left, bottom - top + 1, right - left + 1
```

Pick `out_pos` near the center of the layer-6 output to avoid boundary clipping. For a 6-layer network of the architecture above on a 64×64 input, the layer-6 output is 8×8, so try `out_pos=(4, 4)`.

## Acceptance criteria

- [ ] `receptive_field.md` contains the per-layer table with `RF_l`, `J_l`, and `out_size_l` for all six layers.
- [ ] The final-layer receptive field matches what the gradient-trick script reports.
- [ ] `python verify_rf.py` runs without errors and prints the verification line.
- [ ] `python -m py_compile verify_rf.py` succeeds.
- [ ] The markdown discussion answers: "the layer-6 receptive field is N pixels; the input is 64×64; what fraction of the image does each layer-6 activation see?"

## Hint (do not read until you have tried)

For the given network, the recursion proceeds as follows:

```text
Layer 0 (input):                        RF_0 = 1,   J_0 = 1
Layer 1 (Conv 3, s=1):  RF = 1 + 2*1*1 = 3,         J = 1*1 = 1
Layer 2 (Pool 2, s=2):  RF = 3 + 1*1*1 = 4,         J = 1*2 = 2
Layer 3 (Conv 3, s=1):  RF = 4 + 2*1*2 = 8,         J = 2*1 = 2
Layer 4 (Pool 2, s=2):  RF = 8 + 1*1*2 = 10,        J = 2*2 = 4
Layer 5 (Conv 5, s=1):  RF = 10 + 4*1*4 = 26,       J = 4*1 = 4
Layer 6 (Pool 2, s=2):  RF = 26 + 1*1*4 = 30,       J = 4*2 = 8
```

So the final receptive field is **30×30** on a 64×64 input, with a jump of 8 between adjacent layer-6 activations. Each layer-6 activation "sees" roughly a quarter of the input image (`30 * 30 = 900` pixels out of `64 * 64 = 4096`, about 22%). The 8×8 layer-6 grid means 64 activations cover the entire image with overlapping 30×30 windows centered on a stride-8 lattice.

The empirical verification will find a slightly smaller receptive field at boundary positions (the bounding box gets clipped by the image edge); pick a center position to avoid this.

## Why this matters

Once you can compute the receptive field of any CNN by hand, three things in vision become obvious:

1. **Why aggressive stride-2 stems work.** ResNet's `Conv2d(3, 64, 7, stride=2, padding=3)` followed by `MaxPool2d(3, stride=2)` collapses a 224×224 input to 56×56 in two layers; the receptive field at that point is large enough that subsequent 3×3 convs cover broad image regions.
2. **Why dilation matters in segmentation.** Dense-prediction tasks (semantic segmentation) need every output pixel to see large input context without losing spatial resolution. Dilated convolutions multiply the receptive field without strides; the math here is what makes that possible.
3. **Why vision transformers (Week 13) are interesting.** ViT has effectively a global receptive field from layer 1 because every token attends to every other token. The CNN had to build the receptive field up layer by layer; ViT has it for free. This is one of the architectural reasons ViT can match or exceed CNNs.

## Stretch goals

- Plot the receptive field as a function of layer depth for ResNet-18 (use `torchvision.models.resnet18` and inspect each `BasicBlock`). The receptive field of the layer-4 output of ResNet-18 on a 224×224 input is over 200×200 — almost the entire image. Plot is illuminating.
- Extend `empirical_receptive_field` to handle skip connections (the gradient trick still works through `+`; you just see contributions from both branches). Verify on a `BasicBlock` from ResNet that the residual connection makes the receptive field of a block equal to the maximum of the two paths.
- Read the original paper on the topic: Le and Borji 2017, "What are the Receptive, Effective Receptive, and Projective Fields of Neurons in Convolutional Neural Networks?" (<https://arxiv.org/abs/1705.07049>). The "effective" receptive field — the region of the input that actually carries meaningful gradient signal — is usually smaller than the theoretical receptive field. The paper measures it empirically.
