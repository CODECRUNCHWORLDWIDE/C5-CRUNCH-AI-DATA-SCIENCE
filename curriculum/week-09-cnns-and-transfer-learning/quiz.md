# Week 9 — Quiz

Ten questions. Lectures closed. Aim for 9/10. The answer key is at the bottom; do not peek.

---

**Q1.** You build `nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)` and pass it a tensor of shape `(8, 3, 32, 32)`. What is the output shape?

- A) `(8, 64, 30, 30)`
- B) `(8, 64, 32, 32)`
- C) `(8, 64, 34, 34)`
- D) `(8, 3, 32, 32)`

---

**Q2.** Which of the following is **not** an inductive bias of a `Conv2d` layer (and is therefore something the layer does *not* assume about its input)?

- A) Translation equivariance — shifting the input shifts the output the same way.
- B) Locality — each output depends only on a small spatial neighborhood of the input.
- C) Weight sharing — the same kernel is applied at every spatial position.
- D) Rotation equivariance — rotating the input by 90 degrees rotates the output by 90 degrees.

---

**Q3.** Why do modern CNN blocks set `bias=False` on the conv layer when it is followed by a `BatchNorm2d`?

- A) BatchNorm only works on layers without bias; the layer would error otherwise.
- B) The BN layer's `beta` parameter is itself a learnable bias on the same channel; the conv bias would be redundant and absorbed into BN's shift, so omitting it saves a few parameters.
- C) Conv biases are deprecated in PyTorch 2.x.
- D) Setting `bias=False` makes the convolution faster on cuDNN.

---

**Q4.** You are looking at `model.fc` on a torchvision `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`. It is `nn.Linear(512, 1000)`. To adapt the model for CIFAR-10 transfer learning, you write:

```python
model.fc = nn.Linear(model.fc.in_features, 10)
for p in model.parameters():
    p.requires_grad = False
```

You then train and observe that the loss does not decrease. What went wrong?

- A) `model.fc.in_features` returns the wrong number; it should be 1000, not 512.
- B) You froze every parameter, including the new head. The freeze loop runs *after* the head swap, so it freezes the head's freshly-initialized parameters too. The fix is to freeze first, then swap.
- C) `nn.Linear(512, 10)` should be `nn.Linear(1000, 10)` to match the previous head's output dimension.
- D) You forgot to call `model.train()` before the training loop.

---

**Q5.** A 4-layer CNN applies, in order: `Conv2d(3, 32, k=3, s=1, p=1)`, `MaxPool2d(2, s=2)`, `Conv2d(32, 64, k=3, s=1, p=1)`, `MaxPool2d(2, s=2)`. Using the receptive-field recursion from Lecture 1 Section 7, what is the receptive field at the output of the second max-pool?

- A) 6×6
- B) 10×10
- C) 12×12
- D) 16×16

---

**Q6.** Which of the following best describes the residual connection `y = F(x) + x` in ResNet's `BasicBlock`?

- A) It is a regularization technique that adds noise to the activations.
- B) It is a downsampling shortcut that lets the network learn coarser features.
- C) It makes "do nothing" the default mapping of a block — if the block's learned weights are zero, the block computes the identity. This solves the "degradation problem" where deeper plain networks train worse than shallower ones, and it lets gradient signal flow directly to earlier layers because `dL/dx = dL/dy * (dF/dx + 1)`.
- D) It is an alternative to BatchNorm that normalizes the output of each block.

---

**Q7.** You fine-tune a pretrained `resnet18` on CIFAR-10 by resizing the input to 224×224 and applying normalization. Which set of statistics should you use in `transforms.v2.Normalize(mean, std)`?

- A) CIFAR-10 statistics: `mean=(0.4914, 0.4822, 0.4465)`, `std=(0.2470, 0.2435, 0.2616)`.
- B) ImageNet statistics: `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`. The pretrained backbone was trained with these stats on ImageNet; matching its input distribution preserves the pretrained features.
- C) `mean=(0.5, 0.5, 0.5)`, `std=(0.5, 0.5, 0.5)` — the generic "scale to [-1, 1]" choice.
- D) No normalization is needed; `ToDtype(float32, scale=True)` already scales pixels to `[0, 1]`.

---

**Q8.** What does `nn.AdaptiveAvgPool2d((1, 1))` do, and where is it typically placed in a modern CNN?

- A) It is an attention layer; it lives between every conv block.
- B) It applies dropout adaptively; it lives at the end of each conv block.
- C) It averages each channel of the feature map down to a single number, producing an `(N, C, 1, 1)` output regardless of input spatial size. It typically lives between the conv body and the final `Linear` classifier — the "global average pool" head that every modern CNN uses to replace the fully-connected stack from AlexNet / VGG.
- D) It is a special variant of `MaxPool2d` for variable-sized inputs.

---

**Q9.** You set up a fine-tuning optimizer with two parameter groups: backbone at `lr=1e-4`, head at `lr=1e-3`. After 5 epochs you observe the model has *worse* test accuracy than the same model trained as a feature extractor (frozen backbone, head only). The most likely explanation is:

- A) Fine-tuning is always worse than feature extraction; you should not have tried it.
- B) The backbone learning rate is too high — at `1e-4` the pretrained features are being destroyed faster than they are being adapted. Try `1e-5` for the backbone, or freeze the backbone for the first few epochs ("warmup") before unfreezing.
- C) The head's learning rate is too low; raise it to `1e-2`.
- D) You should switch from Adam to SGD.

---

**Q10.** Which of the following correctly describes the historical sequence of CNN architectures covered in Lecture 2?

- A) LeNet (1998) → AlexNet (2012) → VGG (2014) → ResNet (2015), where each architecture added scale, depth-uniformity, and residual connections respectively.
- B) ResNet (1998) → VGG (2012) → AlexNet (2014) → LeNet (2015).
- C) LeNet (1998) → VGG (2012) → AlexNet (2014) → ResNet (2015).
- D) AlexNet (1998) → LeNet (2012) → ResNet (2014) → VGG (2015).

---

## Answer key

(Read after attempting.)

**Q1: B.** With `padding=1` and `kernel=3, stride=1`, the output spatial size equals the input spatial size. The output-channel count is `out_channels=64`, so the output tensor is `(8, 64, 32, 32)`.

**Q2: D.** Convolution is translation-equivariant but **not** rotation-equivariant. A 90-degree rotation of the input does not produce a 90-degree rotation of the output unless the kernel happens to be rotation-symmetric (which a learned kernel almost never is). This is why CNNs need rotation augmentation to be robust to rotated inputs; the layer does not bake in rotation invariance.

**Q3: B.** BatchNorm normalizes activations and then applies a learnable affine transform `beta * normalized + gamma`. The `beta` parameter is a per-channel bias, so a conv bias before BN gets cancelled by BN's normalization step and then re-added by `beta` — pure redundancy.

**Q4: B.** Freeze first, swap second. The opposite order applies the freeze to the new head, which means no parameters train at all and the loss does not move. This is the most common transfer-learning bug and the C5 lectures call it out three separate times.

**Q5: B.** Using the recursion: RF_0=1, J_0=1. After Conv(3,1,1): RF=1+2*1=3, J=1. After Pool(2,2): RF=3+1*1=4, J=2. After Conv(3,1,1): RF=4+2*2=8, J=2. After Pool(2,2): RF=8+1*2=10, J=4. So 10×10.

**Q6: C.** The residual connection's identity default explains the empirical observation that ResNet-50 trains better than VGG-50 from the same initial loss. The gradient-flow argument is the second reason; the optimizer argument (about learning residuals rather than features) is the third. All three are correct and synergistic.

**Q7: B.** Use ImageNet statistics when feeding a pretrained ImageNet backbone. Lecture 3 Section 7 covers this; the wrong choice costs 2-3 percentage points of test accuracy.

**Q8: C.** The global average pool is the architectural element that replaces the FC head of AlexNet/VGG. Combined with a small final `Linear(C, n_classes)`, it makes the network's parameter count nearly independent of input image size — ResNet-50 has ~26M params with the global-pool head; the equivalent FC head would have hundreds of millions.

**Q9: B.** Catastrophic forgetting: a too-large backbone LR overwrites the pretrained features in the first few hundred steps, before the head has had time to learn its mapping. The fix is a smaller backbone LR or a "warmup" where the backbone is frozen for the first few epochs.

**Q10: A.** LeNet 1998 (LeCun et al.), AlexNet 2012 (Krizhevsky et al.), VGG 2014/ICLR 2015 (Simonyan and Zisserman), ResNet 2015/CVPR 2016 (He et al.). The conceptual contributions: LeNet established the conv+pool+FC recipe; AlexNet added scale, ReLU, dropout, and GPU training; VGG demonstrated that depth helps and the uniform 3×3 recipe; ResNet introduced the residual connection that made deeper networks trainable.
