# Lecture 3 — Transfer Learning with torchvision

> **Outcome:** You can load a pretrained `torchvision.models.resnet18`, identify its classification head, replace it with one sized for your task, and train it on CIFAR-10 to ≥85% test accuracy in five epochs of frozen-backbone training and ≥90% in ten epochs of partial fine-tuning. You can articulate the difference between feature extraction (freeze the backbone, train the head only) and fine-tuning (train the whole network with a smaller learning rate on the backbone), and you know which to reach for. You can explain why a model trained on ImageNet generalizes to CIFAR-10 with a head swap, and you know the empirical limits of that generalization. By the end of this lecture, transfer learning stops being "magic" and starts being "the three-line move that defines vision in 2026."

Lecture 2 ended with `TinyVGG` at 73% on CIFAR-10 after 15 epochs of training from scratch. The same compute budget, applied to a pretrained ResNet-18, reaches 88-92%. The difference is not the architecture — `TinyVGG` and ResNet-18 are both convolutional networks of comparable depth — but the **initialization**. ResNet-18 starts from weights that have already learned to recognize edges, textures, parts, and object categories on 1.28M ImageNet images; `TinyVGG` starts from `torch.nn.init.kaiming_normal_(...)`. The pretrained initialization is the gift; this lecture is the wrapping.

The canonical reference is the PyTorch transfer-learning tutorial at <https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>. The model zoo is at <https://pytorch.org/vision/stable/models.html>. Pin both.

---

## 1. The conceptual move

Transfer learning takes a model trained on a large **source** task (ImageNet classification, 1.28M images, 1000 classes) and reuses its parameters on a smaller **target** task (CIFAR-10 classification, 50k images, 10 classes). The reuse comes in two flavors:

1. **Feature extraction.** The pretrained backbone is held fixed (every `requires_grad` flag set to `False`). Only a new classification head, sized for the target task, is trained. The intuition: the backbone learned a general-purpose visual representation on ImageNet that transfers; we just need a fresh classifier on top.
2. **Fine-tuning.** The entire model, starting from the pretrained weights, is trained on the target task — typically with a smaller learning rate on the backbone than on the head, since the backbone is "almost right" and the head is starting from scratch.

Both recipes use the same loader pipeline, the same loss function, and the same training loop. The difference is two lines: which parameters get `requires_grad = False` and which learning rate group each parameter belongs to.

Why this works at all: a deep CNN trained on a large diverse dataset learns features that are **task-general** at the early layers (edges, gradients, color blobs) and increasingly **task-specific** at the later layers (object parts, then whole-object detectors). Replacing the final classifier swaps out the task-specific head while keeping the task-general body. Donahue et al. 2014 (<https://arxiv.org/abs/1310.1531>) and Yosinski et al. 2014 (<https://arxiv.org/abs/1411.1792>) showed empirically that ImageNet features transfer to a wide range of downstream vision tasks; the practice has been standard ever since.

The C5 reading: in 2026, the question is not "should I train from scratch or transfer learn?" — it is "should I transfer learn or should I use a foundation model?" Until you have the compute and data to compete with ImageNet pretraining, transfer learning wins. Below 100k labeled examples, transfer learning is almost always the right call. Above 10M examples, training from scratch with modern recipes becomes competitive. CIFAR-10 at 50k examples is firmly in transfer-learning territory.

---

## 2. Loading a pretrained model

In torchvision 0.13+, the canonical pattern is:

```python
import torchvision  # type: ignore
from torchvision.models import resnet18, ResNet18_Weights  # type: ignore


weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
```

Three things to know:

- **`weights=` is the modern API.** Replaces the legacy `pretrained=True` flag, which still works but prints a `UserWarning`. The C5 default is the enum form (`ResNet18_Weights.IMAGENET1K_V1`) because it makes the checkpoint version explicit; the string form (`"IMAGENET1K_V1"` or `"DEFAULT"`) also works.
- **The download happens on first call.** ~45 MB to `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`. Subsequent calls are instant.
- **The `weights` object also carries the right input preprocessing.** `weights.transforms()` returns the v2 transform pipeline that the model was trained with. For ImageNet-pretrained models this is roughly `Resize(256) → CenterCrop(224) → ToTensor → Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`. You should use these transforms (or compatible ones) on your input so the model sees data in the distribution it was trained on.

Inspecting the model:

```python
print(model)
```

prints the full architecture as a Python repr — `Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)`, the BatchNorm, the ReLU, the MaxPool, the four `layer1`…`layer4` ResNet stages, the `AdaptiveAvgPool2d`, and finally `(fc): Linear(in_features=512, out_features=1000, bias=True)`. That last `fc` line is the head you will replace.

> **EXPERIMENT — count the parameters.** `sum(p.numel() for p in model.parameters())`. You should see ~11.7 million. Compare to `TinyVGG` from Lecture 2 (~1.1M). The pretrained ResNet-18 has 10x the parameters and starts with all of them already configured for a useful visual representation. That is the gift.

---

## 3. Swapping the head

Two lines:

```python
model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
```

That is the entire head-swap. Why this works:

- `model.fc` is an `nn.Linear(512, 1000)`. We read `model.fc.in_features` (which returns `512`) and create a new `nn.Linear(512, 10)`.
- Reassigning `model.fc = ...` overwrites the attribute; PyTorch's `nn.Module` magic registers the new module as a submodule automatically. The old `nn.Linear(512, 1000)` is dropped.
- The new layer is freshly initialized (Kaiming uniform for the weight, zeros for the bias) and has `requires_grad=True` on every parameter — so it will be trained.

For a few other torchvision models you might see in your career, the head attribute name is different:

- **ResNet family (`resnet18`, `resnet50`, `resnext50_32x4d`, etc.):** `model.fc`.
- **VGG family (`vgg11`, `vgg16`, `vgg19`):** `model.classifier[-1]` (the head is a 3-layer FC stack; replace the last one).
- **EfficientNet (`efficientnet_b0` through `efficientnet_b7`):** `model.classifier[-1]`.
- **AlexNet (`alexnet`):** `model.classifier[-1]`.
- **ViT (`vit_b_16`, `vit_l_16`):** `model.heads.head`.
- **ConvNeXt (`convnext_tiny`, etc.):** `model.classifier[-1]`.

The C5 convention is: print `model` once, find the last `Linear` layer, learn its name, replace it. This is a 30-second operation in every new model you adopt.

---

## 4. Recipe 1: feature extraction (frozen backbone)

The simplest transfer-learning recipe. Freeze every parameter except the head; train the head:

```python
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights  # type: ignore


def build_feature_extractor(n_classes: int = 10) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model
```

Three details:

1. **Freeze first, swap second.** The order matters because the new `Linear` defaults to `requires_grad=True`. If you swap the head first and then freeze everything with `for p in model.parameters(): p.requires_grad = False`, the new head is also frozen and nothing trains. Always freeze-then-swap. (Alternative: swap, then iterate `model.parameters()` while skipping the head — works but is uglier.)
2. **`requires_grad=False` is *just a flag.*** The frozen parameters still participate in the forward pass; they are just not updated by `optimizer.step()`. Their gradients are not computed, which makes the backward pass faster.
3. **The optimizer should only see the head.** Two equivalent ways:

```python
# Option A: pass all parameters; the optimizer's no-op on frozen params is harmless but wastes memory.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Option B: pass only the trainable parameters; cleaner and slightly faster.
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3,
)
```

The C5 default is Option B because it makes the intent explicit and avoids accidentally training a "frozen" parameter that you forgot to freeze.

### Expected results on CIFAR-10

Frozen ResNet-18 + new head, 5 epochs of `Adam(lr=1e-3)`, batch size 128, 224×224 resize, standard ImageNet normalization:

```text
Epoch 1: train_loss=0.7142  test_acc=0.8123
Epoch 2: train_loss=0.5523  test_acc=0.8451
Epoch 3: train_loss=0.5108  test_acc=0.8607
Epoch 4: train_loss=0.4882  test_acc=0.8678
Epoch 5: train_loss=0.4733  test_acc=0.8724
```

87% test accuracy in 5 epochs, where `TinyVGG` from Lecture 2 reached 70% after 15. The compute-per-epoch is roughly 2x (because the frozen forward pass through ResNet-18 is more expensive than the trainable forward pass through `TinyVGG`), so the total compute is comparable; the accuracy is 17 percentage points higher.

> **EXPERIMENT — print which parameters are trainable.** After `model = build_feature_extractor()`, run `for name, p in model.named_parameters(): print(name, p.requires_grad)`. Verify that only `fc.weight` and `fc.bias` have `requires_grad=True` and everything else is False. This 30-second sanity check catches every "I forgot to freeze the backbone" bug before training starts.

---

## 5. Recipe 2: full fine-tuning (differential learning rates)

The higher-accuracy recipe. Replace the head, but train the entire network end-to-end with a smaller learning rate on the pretrained backbone:

```python
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights  # type: ignore


def build_finetune(n_classes: int = 10) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    # Every parameter has requires_grad=True (the default); no freezing.
    return model


def build_finetune_optimizer(
    model: nn.Module,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
) -> torch.optim.Optimizer:
    """Two parameter groups: small lr on the backbone, normal lr on the head."""
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

Three details:

1. **The backbone learning rate is typically 10x smaller than the head's.** The intuition: the backbone is "almost right" — it already knows what cats and dogs look like; we just need to nudge it slightly to better match the CIFAR-10 input distribution. The head is starting from scratch, so it needs a normal learning rate. The 10x ratio is empirical; some papers go 100x. Reference: <https://pytorch.org/docs/stable/optim.html#per-parameter-options>.
2. **Use `id(p)` to compare parameters by identity.** Equality on `torch.Tensor` is element-wise; we want to know if two references point to the *same* parameter. `id()` returns the Python object identity.
3. **The training loop is unchanged.** The optimizer applies the per-group learning rates internally; the loop just calls `optimizer.step()`. No new code in the loop.

### Expected results on CIFAR-10

Full fine-tune of ResNet-18 on CIFAR-10, 10 epochs of `Adam` with `backbone_lr=1e-4, head_lr=1e-3`, batch size 128, 224×224 resize, ImageNet normalization, `RandomCrop(224, padding=8) + RandomHorizontalFlip` augmentation:

```text
Epoch  1: train_loss=0.5871  test_acc=0.8542
Epoch  2: train_loss=0.4012  test_acc=0.8821
Epoch  3: train_loss=0.3128  test_acc=0.9013
Epoch  4: train_loss=0.2554  test_acc=0.9118
Epoch  5: train_loss=0.2188  test_acc=0.9176
Epoch  6: train_loss=0.1923  test_acc=0.9202
Epoch  7: train_loss=0.1701  test_acc=0.9233
Epoch  8: train_loss=0.1538  test_acc=0.9251
Epoch  9: train_loss=0.1415  test_acc=0.9268
Epoch 10: train_loss=0.1310  test_acc=0.9279
```

**92.8% test accuracy in 10 epochs.** That is the headline number of the week. From `TinyVGG`'s 73% (Lecture 2) to ResNet-18 fine-tune's 92.8%, on the same dataset, with comparable compute. The 20-percentage-point gap is the empirical justification for everything in this lecture.

The compute cost: ~3 minutes per epoch on a Colab T4 GPU; ~7 minutes per epoch on an Apple Silicon M2 with `mps`; ~30 minutes per epoch on a CPU-only laptop. The mini-project budgets for the laptop case but rewards the GPU case.

---

## 6. The intermediate recipe: partial fine-tuning

A common middle ground: freeze the early backbone layers (which learn task-general features), unfreeze the later layers (which learn task-specific features), train both the unfrozen layers and the new head:

```python
def build_partial_finetune(n_classes: int = 10) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze the last residual stage (`layer4`) and the head.
    for p in model.layer4.parameters():
        p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model
```

This trains only `layer4` (the last ResNet stage; about 4.7M parameters) and `fc` (the new 5k-parameter head). About half the backbone is updated; the earlier half is frozen. Empirically reaches 90-91% on CIFAR-10 in 10 epochs — slightly below the full fine-tune but with half the gradient-computation cost.

The C5 conviction: full fine-tune for the mini-project (accuracy matters; compute is cheap relative to engineering time), partial fine-tune for production fine-tuning when you have a 100M-parameter backbone and a 10x compute saving matters. We will revisit partial fine-tuning at scale in Week 13 (vision transformers).

---

## 7. The CIFAR-10 input size question

ImageNet ResNet-18 was trained on 224×224 RGB. CIFAR-10 is 32×32. Three strategies for handling the mismatch:

### Strategy A: resize CIFAR-10 to 224×224 (the C5 default)

Add `v2.Resize((224, 224))` to the transforms. The model sees its expected input resolution; the pretrained features fire correctly; accuracy is maximized.

Cost: the data pipeline does a 7x upscaling on every batch, which is CPU-bound. With `num_workers=4` on the DataLoader, this is usually not the bottleneck on a GPU machine. On a CPU-only laptop, this is roughly half of the per-epoch wall-clock time.

```python
train_tf = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomCrop(224, padding=8),
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=(0.485, 0.456, 0.406),   # ImageNet mean, not CIFAR's
        std=(0.229, 0.224, 0.225),    # ImageNet std
    ),
])
```

**Use ImageNet normalization here, not CIFAR-10 normalization.** The pretrained model was trained with ImageNet's per-channel mean and std; feeding it CIFAR-normalized data shifts the input distribution and reduces accuracy by 2-3 percentage points. This is one of the most common subtle bugs in transfer-learning code.

### Strategy B: adapt the ResNet stem to 32×32 input

Modify `model.conv1` to a smaller, stride-1 conv and remove the initial `MaxPool2d` so the network handles 32×32 natively:

```python
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)
```

Saves the CPU upscaling cost. Reaches ~91% on CIFAR-10 — about 1-2 percentage points lower than Strategy A because the modified stem is freshly initialized and the pretrained features in `layer1` see a different input distribution than they expect.

### Strategy C: a hybrid — resize to 64×64 or 96×96

A compromise: enough upscaling that the pretrained features fire, not so much that the data pipeline dominates. Useful when you have a CPU bottleneck. We will not test this in the mini-project but it is a useful trick to know.

The C5 mini-project uses Strategy A. It is the highest-accuracy approach, the simplest to get right, and the easiest to write about in a report ("we resized CIFAR-10 to ImageNet's native resolution to align with the pretrained model's training distribution").

---

## 8. When transfer learning fails

Transfer learning is the right default for most vision tasks in 2026, but it is not a silver bullet. Three failure modes worth knowing:

1. **Severe domain shift.** ImageNet contains natural color photographs of consumer-grade objects. Transfer learning works well on similar data (CIFAR-10, Flowers-102, Stanford Cars, Caltech-101). It works less well on medical images (chest X-rays, histopathology slides), satellite imagery, line drawings, and synthetic renders. The further the target distribution is from ImageNet, the smaller the transfer benefit. For radically different domains, look for domain-specific pretrained models (e.g., MedMNIST for medical imaging) or self-supervised pretraining on in-domain data.
2. **Catastrophic forgetting with aggressive fine-tuning.** If you fine-tune the entire network with a backbone learning rate equal to (or larger than) the head learning rate, the backbone "forgets" its ImageNet features in the first few hundred steps. The result is worse than feature extraction. Always use a smaller LR on the backbone, or freeze it entirely for the first few epochs and then unfreeze.
3. **The pretrained model is wrong for the input.** ImageNet pretraining assumes natural-image statistics; if your target images are pre-edge-detected, pre-normalized in a non-RGB color space, or otherwise transformed before reaching the model, the pretrained features do not apply. Match the preprocessing the pretrained model expects (use `weights.transforms()` if in doubt; <https://pytorch.org/vision/stable/models.html>).

The C5 reading: transfer learning is robust enough that you should try it first for any vision task. If it does not work, the failure usually tells you something useful about the target domain.

---

## 9. The mini-project recipe, in one block

The full Lecture 3 transfer-learning code, condensed into a single example that you will expand for the mini-project:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader


def build_pretrained_model(n_classes: int = 10) -> nn.Module:
    """ResNet-18 pretrained on ImageNet, head replaced for CIFAR-10."""
    # Lazy import so py_compile works on machines without torchvision.
    from torchvision.models import resnet18, ResNet18_Weights  # type: ignore

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def make_finetune_optimizer(
    model: nn.Module,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
) -> torch.optim.Optimizer:
    """Two parameter groups; backbone at 1e-4, head at 1e-3."""
    head_params = list(model.fc.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    return torch.optim.Adam(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
    )


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
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n_seen += X.size(0)
    return total_loss / n_seen


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
        pred = model(X).argmax(dim=1)
        n_correct += (pred == y).sum().item()
        n_seen += X.size(0)
    return n_correct / n_seen
```

That is the entire transfer-learning Week 9 mini-project, modulo the data-loader code from Lecture 2 and the per-epoch printing / checkpointing.

---

## 10. Why this matters for the rest of the C5 curriculum

Three takeaways that carry forward:

1. **The C5 vision curriculum is "use pretrained models" from Week 9 onward.** Week 10 adds regularization, augmentation, and `torch.compile` on top of a pretrained backbone. Week 11 adapts a pretrained backbone for object detection (Faster R-CNN) and semantic segmentation (DeepLab v3+). Week 12 explores data augmentation and self-supervised pretraining. Week 13 introduces vision transformers (which are also used pretrained). Training from scratch reappears only as a control experiment.
2. **The transfer-learning playbook generalizes to NLP and audio.** Week 13's pretrained BERT / GPT and Week 14's pretrained Whisper / Wav2Vec are the same recipe with different model classes. Load pretrained weights, replace the head, freeze (or fine-tune with differential LRs), train. The mechanics are identical to this week.
3. **The from-scratch CNN you built in Exercise 2 is a sanity baseline, not a deliverable.** Production vision systems in 2026 do not train CNNs from scratch on small datasets; they fine-tune pretrained models. You built the from-scratch CNN this week so you understand what the pretrained model is doing internally, not because it is the right tool for the job. The C5 reading: "I can train a CNN from scratch, but I usually don't" is the correct résumé line.

You now have the entire vision-classification pipeline: data loaders with augmentation, a from-scratch CNN baseline, a pretrained backbone with frozen-feature transfer learning, and a full fine-tune. Exercise 3 walks through the head-swap and the frozen forward pass. The mini-project assembles everything into a portfolio-ready transfer-learning project that reaches 90%+ on CIFAR-10.
