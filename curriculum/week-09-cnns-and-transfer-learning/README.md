# Week 9 — Convolutional Neural Networks and Transfer Learning

> *Week 8 ended with a 2-layer MLP on FashionMNIST. The same MLP on CIFAR-10 stalls around 50%. This week we close that gap. The convolution operation, the pooling operation, the LeNet → VGG → ResNet lineage, a hand-built CNN that reaches 70% on CIFAR-10 from scratch — and then the move that changes everything about how you do vision in 2026: load a pretrained ResNet-18 from `torchvision.models`, swap the final layer, and reach 90%+ on the same dataset in fewer epochs. By Sunday you will have shipped two end-to-end PyTorch image classifiers and learned, with your own numbers on your own laptop, why nobody trains vision models from scratch anymore.*

Welcome to week nine of **C5 · Crunch AI / Data Science**. Week 8 gave you the bare PyTorch training loop. The model was a 2-layer MLP and it sufficed for FashionMNIST. Challenge 1 of Week 8 showed you the same MLP on CIFAR-10 plateaus around 50% — bad, embarrassingly so, and the gap between that 50% and what a CNN can do on the same data is the entire reason convolutional architectures replaced fully-connected ones for vision tasks in the 2010s. Week 9 is the week you measure that gap on your own machine and then close it twice: once with a hand-rolled CNN, and once with transfer learning.

Three commitments before we start:

1. **The training loop from Week 8 carries over verbatim.** Every line of the `for epoch in range(n_epochs): model.train(); for X, y in train_loader: optimizer.zero_grad(); loss = loss_fn(model(X), y); loss.backward(); optimizer.step()` pattern is unchanged. Only the model class changes. If you write the loop differently this week, you are doing extra work for no reason. The C5 conviction is that you should write the same eight lines every time, for the next thirty weeks.
2. **We build the CNN by hand before we touch a pretrained model.** `nn.Conv2d`, `nn.MaxPool2d`, `nn.AdaptiveAvgPool2d`, `nn.BatchNorm2d`. You will write a small VGG-style CNN, count its parameters, train it for fifteen epochs on CIFAR-10, and reach 70%. Only then do we load `torchvision.models.resnet18(weights="IMAGENET1K_V1")` and ride the eight-million-parameter ImageNet-pretrained backbone to 90%. The order matters: students who skip the from-scratch CNN never develop the intuition for *why* transfer learning works, and they cannot answer "what is the backbone doing?" in interview.
3. **We compare from-scratch vs. transfer learning with identical training budgets.** Same dataset (CIFAR-10), same batch size, same optimizer, same number of epochs. The from-scratch CNN reaches ~70%; the transfer-learning ResNet-18 reaches ~92%. That 22-point gap, on the same compute budget, is the headline number of the week and the experimental motivation for the rest of the C5 vision curriculum.

We target **PyTorch 2.x** (we test on 2.4 and 2.5), **torchvision 0.19+** (the CIFAR-10 loader and the pretrained `resnet18` weights), and **Python 3.11+**. The Apple Silicon `mps` backend is supported by every model this week; CUDA is what the C5 grading rig runs on. CPU-only is enough to complete the week, though the transfer-learning mini-project takes 20–40 minutes on CPU instead of 3–5 minutes on GPU.

PyTorch reference docs: <https://pytorch.org/docs/stable/index.html>. torchvision model zoo: <https://pytorch.org/vision/stable/models.html>. Pin both.

---

## Learning objectives

By the end of this week, you will be able to:

- **Explain the convolution operation.** Given an input feature map, a kernel size, a stride, a padding, and a dilation, compute the output spatial dimensions with the standard formula `out = floor((in + 2*padding - dilation*(kernel-1) - 1) / stride) + 1`. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>.
- **Compare convolution to fully-connected layers.** Three properties: weight sharing across spatial positions, locality of receptive field, translation equivariance. Argue why these properties are appropriate for images and not for, for example, tabular data.
- **Use the pooling operations.** `nn.MaxPool2d`, `nn.AvgPool2d`, `nn.AdaptiveAvgPool2d`. Know why a 2×2 max-pool with stride 2 is the canonical downsampling block and when adaptive pooling lets a CNN ingest variable-sized inputs.
- **Place the LeNet / AlexNet / VGG / ResNet lineage on a timeline.** LeNet-5 (LeCun 1998) on MNIST. AlexNet (Krizhevsky, Sutskever, Hinton 2012) on ImageNet — the result that triggered the deep-learning explosion. VGG-16 (Simonyan and Zisserman 2014) — uniform 3×3 stacks. ResNet (He, Zhang, Ren, Sun 2015) — residual connections that made networks deeper than 30 layers possible. Reference: <https://arxiv.org/abs/1512.03385>.
- **Build a small CNN end-to-end on CIFAR-10.** Two or three conv blocks with `Conv2d → BatchNorm2d → ReLU → MaxPool2d`, a `Flatten`, two fully-connected layers, and `nn.CrossEntropyLoss`. Reach **≥70% test accuracy** after 15 epochs of `optim.Adam(lr=1e-3)`.
- **Load a pretrained model from torchvision.** `torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)`. Inspect the architecture. Replace `model.fc` with a new `nn.Linear(512, 10)` for CIFAR-10. Reference: <https://pytorch.org/vision/stable/models.html>.
- **Freeze and fine-tune a backbone.** Two recipes. (a) Freeze the backbone (`for p in model.parameters(): p.requires_grad = False`), train only the new head; this is the safe default and uses the pretrained features as fixed extractors. (b) Fine-tune end-to-end with a smaller learning rate on the backbone and a normal learning rate on the head; this reaches higher accuracy but needs more care. Both recipes appear in Lecture 3.
- **Reach 90%+ test accuracy on CIFAR-10 via transfer learning.** With ResNet-18 pretrained on ImageNet, a frozen-backbone fine-tune for 5 epochs reaches ~88%; a partial fine-tune (unfreeze the last block) for 10 epochs reaches 92%. Both numbers reproduce on a free Google Colab T4.
- **Compute the receptive field of a CNN.** Given a stack of conv layers with known kernels and strides, compute the size of the input region that influences a single output pixel. Reference: <https://distill.pub/2019/computing-receptive-fields/>.
- **Use the modern torchvision transforms API.** `torchvision.transforms.v2` (introduced 2024; the v1 API is deprecated but still works). The standard CIFAR-10 pipeline: `ToImage → ToDtype(float32, scale=True) → Normalize(mean, std)`, optionally `RandomCrop(32, padding=4)` and `RandomHorizontalFlip()` for augmentation. Reference: <https://pytorch.org/vision/stable/transforms.html>.
- **Ship** a mini-project: a transfer-learning CIFAR-10 (or Oxford Flowers-102) classifier that reaches 90%+ accuracy, includes a from-scratch CNN baseline for comparison, a confusion-matrix figure, and a 600–900 word report. Pushed to your portfolio repo.
- **Pass** every `pytest` case on the Week 9 exercises.

---

## Prerequisites

- **Week 8 complete.** The bare PyTorch training loop, `nn.Module` subclassing, `DataLoader` semantics, `state_dict` save and load. If you have not pushed the Week 8 FashionMNIST classifier to your portfolio repo, go back and finish that first; Week 9's exercises assume those reflexes.
- **Python 3.11+** with PyTorch 2.x and torchvision 0.19+ installed (`pip install "torch>=2.4,<3" "torchvision>=0.19,<1"`).
- **About 200 MB of disk** for the CIFAR-10 cache (`./data/cifar-10-batches-py/`, ~170 MB) plus ~45 MB for the pretrained ResNet-18 weights cached at `~/.cache/torch/hub/checkpoints/resnet18-*.pth`. Both are downloaded on first use.
- **Optional but useful:** a CUDA GPU (any made after 2018), an Apple Silicon M1+, or a free Google Colab T4. Transfer learning on CPU is feasible but slow.
- **Familiarity with the Week 8 challenge.** Challenge 1 of Week 8 had you train an MLP on CIFAR-10 and observe the ~50% plateau. Reread your own notes from that exercise before Monday's lecture; the motivation for Week 9 is sitting in your portfolio repo.

You should already be comfortable with:

- **PyTorch tensor shapes for images.** `(N, C, H, W)` is the PyTorch convention; `N` is batch, `C` is channels, `H` is height, `W` is width. Week 8 introduced this.
- **`nn.Module` subclassing.** Week 8 Lecture 2.
- **The DataLoader pattern.** Week 8 Lecture 3.

---

## Topics covered

- **What a convolution is.** A windowed weighted sum applied at every spatial location of a feature map. Three hyperparameters: kernel size, stride, padding. A fourth one, dilation, is occasionally useful. The output spatial size formula is on every Week 9 exercise and you will memorize it. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>.
- **Why convolutions.** Weight sharing reduces parameter count by orders of magnitude versus a fully-connected layer with the same receptive field. Locality matches the inductive bias that nearby pixels are correlated. Translation equivariance means a feature detector trained at one location works at all locations. These three properties together are why CIFAR-10 gives the MLP no help and the CNN every help.
- **The receptive field.** The region of the input that any given hidden-layer activation can "see." Grows linearly with depth in a strided / pooled network. The receptive field at layer L equals the size of the input neighborhood that contributes any signal to a given activation at layer L. The intuition is "deep features see large patches of the image." Reference: <https://distill.pub/2019/computing-receptive-fields/>.
- **Pooling.** Two flavors: max pooling (the maximum activation in a window; most common) and average pooling (the mean; rare in modern nets except for the global pool at the head of a ResNet). A 2×2 max-pool with stride 2 halves the spatial dimensions and is the canonical downsampling op. `nn.AdaptiveAvgPool2d((1, 1))` is the global average pool that ResNets and EfficientNets use before the classification head; it lets the same backbone accept any input size.
- **Batch normalization in CNNs.** `nn.BatchNorm2d(num_features)` normalizes per-channel statistics across the spatial and batch dimensions. The 2015 introduction (Ioffe and Szegedy) was the unlock that made deep CNNs trainable with standard initialization. Reference: <https://arxiv.org/abs/1502.03167>. We will use it in our from-scratch CIFAR-10 CNN.
- **The architectural lineage.** LeNet-5 (1998): two convs, two pools, two fully-connected, on 32×32 grayscale digits. The architecture that defined the template. AlexNet (2012): same template at scale, plus ReLU plus dropout plus GPU training, won ImageNet by a 10-point margin and broke the field open. VGG (2014): uniform 3×3 convs stacked deep; the conceptual demonstration that depth helps more than width. GoogLeNet / Inception (2014): width-mixing modules; not on our reading list but historically important. ResNet (2015): residual connections solve the degradation problem and let you train 50- and 100-layer networks; the architecture that is *still* the default vision backbone in 2026. Reference: <https://arxiv.org/abs/1512.03385>.
- **Why ResNet still wins in 2026.** Three reasons. (a) The residual connection: `y = F(x) + x`, which makes "do nothing" the identity map and prevents the optimizer from making things worse with depth. (b) Batch normalization in every block. (c) A clean architecture that scales: ResNet-18, ResNet-50, ResNet-101, ResNet-152 are the same recipe at different depths. Vision transformers (ViT, Swin) have begun to share the throne, but for image classification under a million examples, a fine-tuned ResNet-50 is still a strong default. We use ResNet-18 because it is the smallest member of the family and trains fastest.
- **Transfer learning.** Take a model trained on a large source dataset (ImageNet-1k, 1.28M images, 1000 classes) and reuse its parameters on a smaller target dataset (CIFAR-10, 50k images, 10 classes). Two ways. (a) **Feature extraction:** freeze every parameter except the final classification layer; replace the final layer with one sized for the target task; train only the new layer. (b) **Fine-tuning:** initialize the model with pretrained weights, replace the final layer, train end-to-end with a small learning rate. Feature extraction is fast, robust, and reaches 85–90% on CIFAR-10. Fine-tuning reaches 92–95% but needs more care.
- **Replacing the final layer.** For ResNet, the head is `model.fc = nn.Linear(in_features, n_classes)`. You read `model.fc.in_features` to discover the embedding dimension (512 for ResNet-18, 2048 for ResNet-50) and create a new `nn.Linear` of the same input size and your target output size. Two lines. Reference: <https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>.
- **Freezing parameters.** `for p in model.parameters(): p.requires_grad = False`. After this, `optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)` only sees the parameters of the new head. The frozen-backbone forward pass is the same; only `loss.backward()` is cheaper because it stops at the frozen boundary.
- **Differential learning rates.** When fine-tuning, give the pretrained backbone a 10x smaller learning rate than the freshly-initialized head. `optimizer = torch.optim.Adam([{"params": model.backbone.parameters(), "lr": 1e-4}, {"params": model.head.parameters(), "lr": 1e-3}])`. The intuition: the backbone already knows what it is doing; the head is starting from scratch. PyTorch supports per-parameter-group learning rates natively.
- **The CIFAR-10 resize question.** Pretrained ImageNet models expect 224×224 input; CIFAR-10 is 32×32. Two strategies. (a) Resize CIFAR-10 to 224×224 with `transforms.v2.Resize((224, 224))` — costs CPU at data-load time, gives the pretrained backbone a familiar input scale, reaches the highest accuracy. This is the C5 default for the mini-project. (b) Adapt the ResNet stem (modify `model.conv1` to a 3×3 conv with stride 1; remove the initial maxpool) so it handles 32×32 input natively — saves data-load cost, slightly underperforms on CIFAR-10 because the pretrained features assume a larger receptive field at the input. Both are reasonable; the mini-project uses (a).
- **What we do not do.** No `torch.compile()` yet; defer to Week 10. No mixed precision; defer to Week 10. No object detection (Week 11). No segmentation (Week 11). No vision transformers; defer to Week 13. No self-supervised pretraining; that is a course in itself.

---

## Weekly schedule

Target about **38 hours**. Monday and Tuesday cover convolutions and pooling. Wednesday is the from-scratch CIFAR-10 CNN. Thursday is transfer learning. Friday–Sunday is the mini-project.

| Day       | Focus                                                       | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|-------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Convolution, kernels, stride, padding, dilation             |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    7h       |
| Tuesday   | Pooling; the LeNet / VGG / ResNet lineage; BatchNorm        |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    7h       |
| Wednesday | Build a small CNN on CIFAR-10 from scratch                  |   2h     |   2h      |     2h     |   0.5h    |   1h     |     0h       |   0h       |    7.5h     |
| Thursday  | Transfer learning: load ResNet-18, swap head, freeze/tune   |   2h     |   1h      |     0h     |   0.5h    |   1h     |     3h       |   0h       |    7.5h     |
| Friday    | Mini-project: ResNet-18 fine-tune; reach 90%+               |   0h     |   0.5h    |     0h     |   0.5h    |   1h     |     3h       |   0.5h     |    5.5h     |
| Saturday  | Mini-project: confusion matrix, write-up                    |   0h     |   0h      |     0h     |   0h      |   1h     |     2h       |   0h       |    3h       |
| Sunday    | Quiz, push to portfolio repo                                |   0h     |   0h      |     0h     |   0.5h    |   0h     |     0h       |   0h       |   0.5h      |
| **Total** |                                                             | **10h**  | **7.5h**  | **2h**     | **3h**    | **6h**   | **8h**       | **1h**     |  **38h**    |

The schedule is comfortable Monday–Tuesday because most of the material is conceptual (what is a convolution, what does it buy us) rather than algorithmic. Wednesday is when the work shows up: writing the from-scratch CNN, watching the loss curve, hitting 70% on CIFAR-10. Thursday's transfer-learning step takes 30 minutes of code and 2 hours of "wait, that worked?" reflection.

---

## How to navigate this week

| File | What is inside |
|------|----------------|
| [README.md](./README.md) | This overview |
| [resources.md](./resources.md) | torchvision model docs, ResNet / VGG / AlexNet papers, Yann LeCun's free deep-learning course, the distill.pub receptive-field essay |
| [lecture-notes/01-convolutions-and-pooling.md](./lecture-notes/01-convolutions-and-pooling.md) | The convolution operation, kernel / stride / padding / dilation, output-size formula, pooling, weight sharing, locality, translation equivariance, receptive fields |
| [lecture-notes/02-cnn-architectures-lenet-vgg-resnet.md](./lecture-notes/02-cnn-architectures-lenet-vgg-resnet.md) | The CNN architectural timeline from LeNet to ResNet; the residual connection; why ResNet is still the default vision backbone; building a small VGG-style CNN for CIFAR-10 |
| [lecture-notes/03-transfer-learning-with-torchvision.md](./lecture-notes/03-transfer-learning-with-torchvision.md) | Loading pretrained models via `torchvision.models`; replacing the final layer; freeze vs. fine-tune; differential learning rates; the CIFAR-10 90%+ recipe |
| [exercises/exercise-01-convolutions-by-hand.py](./exercises/exercise-01-convolutions-by-hand.py) | Implement 2D convolution from scratch with NumPy and verify against `torch.nn.functional.conv2d`; compute output shapes from the formula |
| [exercises/exercise-02-build-a-cnn.py](./exercises/exercise-02-build-a-cnn.py) | Build a `TinyVGG` CNN as an `nn.Module`; train on CIFAR-10 for 15 epochs; reach ≥70% test accuracy |
| [exercises/exercise-03-transfer-learning.py](./exercises/exercise-03-transfer-learning.py) | Load pretrained ResNet-18; replace the head; freeze the backbone; train for 5 epochs on CIFAR-10; reach ≥85% test accuracy |
| [exercises/SOLUTIONS.md](./exercises/SOLUTIONS.md) | Reference solutions with commentary; read only after attempting |
| [challenges/challenge-01-receptive-field.md](./challenges/challenge-01-receptive-field.md) | Compute the receptive field of a 4-layer CNN by hand; verify with the gradient-of-output-with-respect-to-input trick |
| [challenges/challenge-02-feature-extraction-vs-finetune.md](./challenges/challenge-02-feature-extraction-vs-finetune.md) | Compare frozen feature extraction against end-to-end fine-tuning on CIFAR-10; document the accuracy / compute trade-off |
| [quiz.md](./quiz.md) | 10 multiple-choice questions on convolutions, pooling, architecture history, and transfer learning |
| [homework.md](./homework.md) | Six problems for the week |
| [mini-project/README.md](./mini-project/README.md) | Build a transfer-learning image classifier that reaches 90%+ on CIFAR-10 (or Oxford Flowers-102) |
| [mini-project/starter.py](./mini-project/starter.py) | Runnable starter: data loaders with the 224×224 resize, ResNet-18 head-swap, freeze-then-finetune training loop |
| [mini-project/rubric.md](./mini-project/rubric.md) | Grading rubric for the mini-project |

---

## Stretch goals

- Read the **ResNet paper** (He, Zhang, Ren, Sun 2015) cover to cover (<https://arxiv.org/abs/1512.03385>). Twelve pages. Section 3 (residual learning) is the conceptual core; Section 4 (experiments) is the empirical proof. The deepest insight in computer vision since AlexNet.
- Read **Yann LeCun's NYU Deep Learning course Lecture 5** ("Convolutional networks") and **Lecture 6** ("CNN applications") at <https://atcold.github.io/NYU-DLSP21/>. Free, video + transcript, the most authoritative single source on CNNs available outside a textbook.
- Read the **distill.pub "A guide to receptive field arithmetic"** essay (<https://distill.pub/2019/computing-receptive-fields/>). The figures alone are worth the visit. After this you can compute the receptive field of any CNN on paper.
- Read **Sebastian Raschka — "Pre-training, Fine-Tuning, and In-Context Learning"** (<https://magazine.sebastianraschka.com/p/finetuning-large-language-models>). LLM-focused but the conceptual framework for transfer learning is the same.
- Read **the AlexNet paper** (Krizhevsky, Sutskever, Hinton 2012; <https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>). The result that started deep learning. Section 3 on the architecture is concise; Section 4 on training (two GPUs, ReLU, dropout, augmentation) is a useful checklist of what 2012 had to invent.
- Read **the VGG paper** (Simonyan and Zisserman 2014; <https://arxiv.org/abs/1409.1556>). The "stack 3×3s" insight is the simplest correct answer to "what is the right CNN depth?"
- Watch **Andrej Karpathy's CS231n lecture on convolutional networks** (the 2016 recording is still free on YouTube). Karpathy was a student when the lecture was recorded; the explanation of weight sharing at minute 15 is the cleanest version on the internet.

---

## What you will *not* do this week

You will not:

- **Train from scratch on ImageNet.** A from-scratch ResNet-50 on ImageNet-1k takes ~2 GPU-days even on an A100. That is a research-budget experiment, not a course exercise. Transfer learning is precisely the move that makes vision affordable for everyone else.
- **Implement convolution in CUDA.** `torch.nn.Conv2d` already calls `cuDNN`, which is hand-tuned by NVIDIA. Writing your own CUDA conv is a fun three-week side quest with no production payoff. We implement convolution in NumPy in Exercise 1 to prove we understand the math.
- **Use vision transformers.** Week 13. ResNet-18 is the right backbone for this week.
- **Use object-detection or segmentation models.** Faster R-CNN, YOLO, Mask R-CNN, DETR, SAM — all out of scope for Week 9. Week 11 introduces object detection.
- **Use data augmentation beyond the basic flip and crop.** AutoAugment, RandAugment, mixup, CutMix — all real and all useful, but they are a Week 12 topic. We use `RandomCrop(32, padding=4)` and `RandomHorizontalFlip()` as the standard CIFAR-10 augmentation; that is enough for the 90% mark.
- **Use Lightning, fastai, or `transformers`.** The bare loop continues. The eight lines you memorized in Week 8 are the same eight lines this week.

That list is deliberate. The point of Week 9 is to *understand the inductive biases that make CNNs work* and *to know how to ride the pretrained-model gravy train* — not to chase state-of-the-art numbers. SOTA is a moving target; the underlying recipe (pretrained backbone, replaced head, brief fine-tune) has been stable for a decade.

---

## A note on the EXPERIMENT cards

Lectures continue to use `EXPERIMENT` callouts:

> **EXPERIMENT — short title.** A one-paragraph drill you can run in a Python REPL in five to fifteen minutes. The drill makes a specific claim from the lecture concrete. Run them. Do not skip them.

The Lecture 1 experiment that builds a 3×3 edge-detection kernel by hand and convolves it over a grayscale image is the experiment that makes "convolutions detect local patterns" stop being a slogan. The Lecture 3 experiment that prints `model.fc.in_features` before and after the head swap is the experiment that makes transfer learning stop feeling like magic. Both take less than ten minutes; both are the difference between "I read about it" and "I have done it."

---

## Up next

[Week 10 — Regularization, augmentation, and the practical deep-learning toolbox](../week-10/) — once you have pushed the Week 9 transfer-learning classifier, the from-scratch CNN baseline, and the comparison report to your portfolio repo. Week 10 covers dropout, weight decay, learning-rate schedules, gradient clipping, `torch.compile()`, mixed precision (`torch.cuda.amp`), and the production tricks that turn a working PyTorch model into a robust one. The architecture stays roughly the same; what changes is the training discipline around it.
