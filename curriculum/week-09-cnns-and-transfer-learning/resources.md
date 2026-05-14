# Week 9 — Resources

Every resource on this page is **free** and **publicly accessible**. No paywalled books, no proprietary PDFs, no signup-required courses. If a link breaks, please open an issue.

## Required reading (work it into your week)

- **PyTorch official tutorial — "Transfer Learning for Computer Vision Tutorial"** (the canonical end-to-end recipe; uses ResNet-18 and the Hymenoptera ants/bees dataset; the C5 mini-project is a CIFAR-10 variant of this exact tutorial):
  <https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>
  Read **Monday evening** before Lecture 3. The tutorial is ~15 minutes; the C5 lectures go deeper on freezing semantics and differential learning rates.
- **PyTorch docs — `torchvision.models`** (the model zoo; every pretrained model in PyTorch lives here):
  <https://pytorch.org/vision/stable/models.html>
  Pin the tab. The "Classification" section lists ResNet-18 through ResNet-152, plus VGG, AlexNet, EfficientNet, ViT, ConvNeXt, and dozens more. The `weights` enums on each model (`ResNet18_Weights.IMAGENET1K_V1`, etc.) are how PyTorch 2.x advertises pretrained checkpoints — the legacy `pretrained=True` flag is deprecated.
- **Distill — "A guide to receptive field arithmetic for Convolutional Neural Networks"** (free; the cleanest visual treatment of receptive fields):
  <https://distill.pub/2019/computing-receptive-fields/>
  Read after Lecture 1. The interactive figures are the part you remember; the formulas after them are the part you use on the exercises.
- **Sebastian Raschka — "Understanding Pre-Trained Models in PyTorch"** (free Substack; the plain-English explanation of `torchvision.models`):
  <https://magazine.sebastianraschka.com/>
  ~15 minutes. Read Tuesday after Lecture 2.
- **fast.ai — *Practical Deep Learning for Coders***, **Chapter 5** ("Image Classification") and **Chapter 7** ("Training a State-of-the-Art Model"):
  <https://github.com/fastai/fastbook>
  Chapter 5 covers transfer learning with the fastai library; the C5 reading skips the library calls and absorbs the principles (one-cycle schedule, discriminative learning rates, the freeze-then-unfreeze recipe). The principles are framework-independent.

## The papers (free PDFs)

- **He, Zhang, Ren, Sun — "Deep Residual Learning for Image Recognition"** (CVPR 2016; the ResNet paper):
  <https://arxiv.org/abs/1512.03385>
  Twelve pages. The single most-cited paper in computer vision. Section 3 introduces the residual connection; Section 4 reports the ImageNet results that won 2015 ILSVRC. Read it twice. The architecture diagram on page 4 is the one you should be able to draw from memory by Sunday.
- **Krizhevsky, Sutskever, Hinton — "ImageNet Classification with Deep Convolutional Neural Networks"** (NeurIPS 2012; the AlexNet paper):
  <https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>
  Mirror PDF: <https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>
  Nine pages. The paper that started the deep-learning era. Section 3 lists the ingredients (ReLU, dropout, two-GPU training, local response normalization); Section 6 is the result that beat the second-best ILSVRC-2012 entry by 10.8 percentage points. Read it for the historical context.
- **Simonyan and Zisserman — "Very Deep Convolutional Networks for Large-Scale Image Recognition"** (ICLR 2015; the VGG paper):
  <https://arxiv.org/abs/1409.1556>
  Fourteen pages. The "stack 3×3s" demonstration. VGG-16 and VGG-19 are still useful pretrained backbones in 2026 for problems that need a simple feature extractor without skip connections.
- **LeCun, Bottou, Bengio, Haffner — "Gradient-Based Learning Applied to Document Recognition"** (Proceedings of the IEEE 1998; the LeNet-5 paper):
  <http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf>
  46 pages. You do not need to read the whole thing; sections II and III (pages 4-14) describe LeNet-5 and its training. The architecture diagram on page 8 is the ancestor of every CNN in this week's reading list.
- **Ioffe and Szegedy — "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"** (ICML 2015):
  <https://arxiv.org/abs/1502.03167>
  Eleven pages. Introduces `nn.BatchNorm2d`. The "internal covariate shift" framing has been contested by later work, but the empirical effect — networks train faster and at higher learning rates — is uncontested and the layer is in every modern CNN.
- **Szegedy et al. — "Going Deeper with Convolutions"** (CVPR 2015; the Inception / GoogLeNet paper):
  <https://arxiv.org/abs/1409.4842>
  Optional. Twelve pages. Inception modules are a useful counterpoint to VGG's "stack 3×3s" recipe. The C5 conviction is that you should know Inception exists but should reach for ResNet-18 in production.
- **Tan and Le — "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"** (ICML 2019):
  <https://arxiv.org/abs/1905.11946>
  Optional. Ten pages. The "compound scaling" idea that motivates the EfficientNet family. `torchvision.models.efficientnet_b0` through `efficientnet_b7` are all pretrained and available.

## The official PyTorch 2.x docs you will live in this week

- **`torch.nn.Conv2d`** (the workhorse 2D convolution layer):
  <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>
- **`torch.nn.functional.conv2d`** (the functional version; what `Conv2d.forward` calls under the hood):
  <https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html>
- **`torch.nn.MaxPool2d`** (max pooling; the 2×2-stride-2 downsampling block):
  <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>
- **`torch.nn.AvgPool2d`** (average pooling):
  <https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html>
- **`torch.nn.AdaptiveAvgPool2d`** (the global pool used at the head of every modern CNN):
  <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html>
- **`torch.nn.BatchNorm2d`** (per-channel batch normalization):
  <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>
- **`torch.nn.Dropout2d`** (channel-wise dropout for conv features):
  <https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html>
- **`torch.nn.Flatten`** (turns `(N, C, H, W)` into `(N, C*H*W)` for the FC head):
  <https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html>
- **`torchvision.models`** (the index of pretrained models):
  <https://pytorch.org/vision/stable/models.html>
- **`torchvision.models.resnet18`** (the smallest ResNet; our default backbone this week):
  <https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html>
- **`torchvision.models.resnet50`** (the larger ResNet; mini-project stretch goal):
  <https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html>
- **`torchvision.models.ResNet18_Weights`** (the enum that names pretrained checkpoints):
  <https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights>
- **`torchvision.transforms.v2`** (the modern transforms API; the v1 API is still importable but deprecated):
  <https://pytorch.org/vision/stable/transforms.html>
- **`torchvision.transforms.v2.Resize`** (scale to a target size; used to upsample CIFAR-10 to 224×224 for ResNet):
  <https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.Resize.html>
- **`torchvision.transforms.v2.RandomCrop`** (the canonical CIFAR-10 augmentation):
  <https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomCrop.html>
- **`torchvision.transforms.v2.RandomHorizontalFlip`**:
  <https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomHorizontalFlip.html>
- **`torchvision.datasets.CIFAR10`** (Week 9 default dataset):
  <https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html>
- **`torchvision.datasets.Flowers102`** (mini-project stretch dataset; 102-class flower classification):
  <https://pytorch.org/vision/stable/generated/torchvision.datasets.Flowers102.html>

## Yann LeCun's free deep-learning course (the gold-standard CNN course)

Yann LeCun co-invented the modern CNN. His NYU graduate course is on YouTube with full transcripts and lecture notes.

- **Course homepage** (Spring 2021 edition; PyTorch + JAX):
  <https://atcold.github.io/NYU-DLSP21/>
- **Lecture 4 — "Convolutional Neural Networks"** (the conceptual introduction):
  <https://atcold.github.io/NYU-DLSP21/en/week04/04-1/>
- **Lecture 5 — "Optimisation and Initialisation"** (the methods that make CNNs trainable):
  <https://atcold.github.io/NYU-DLSP21/en/week05/05-1/>
- **Lecture 6 — "Applications of Convolutional Network"** (computer vision case studies):
  <https://atcold.github.io/NYU-DLSP21/en/week06/06-1/>
- **YouTube playlist**:
  <https://www.youtube.com/playlist?list=PLLHTzKZzVU9e6xUfG10TkTWApKSZCzuBI>
- **LeCun's older 2020 edition** (still available; some students prefer the longer lectures):
  <https://cds.nyu.edu/deep-learning/>

## Stanford CS231n (the classical CNN course)

CS231n is the source of half the educational vocabulary in modern computer vision. The 2017 lectures are on YouTube; the 2022 course notes are on the course site.

- **Course site** (free notes; updated 2022 edition):
  <https://cs231n.github.io/>
- **Convolutional networks chapter** (the canonical written introduction):
  <https://cs231n.github.io/convolutional-networks/>
- **Transfer learning notes**:
  <https://cs231n.github.io/transfer-learning/>
- **YouTube playlist** (2017 Stanford recording; Karpathy and Johnson lecturing):
  <https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv>

## Free textbooks

- **Goodfellow, Bengio, Courville — *Deep Learning*** (free online; MIT Press 2016):
  <https://www.deeplearningbook.org/>
  **Chapter 9** ("Convolutional Networks") is the canonical textbook treatment. Sections 9.1–9.3 cover the convolution operation; 9.4–9.5 cover pooling; 9.6 covers practical considerations. Read sections 9.1–9.6 this week.
- **Bishop, Bishop — *Deep Learning: Foundations and Concepts*** (free online; 2024):
  <https://www.bishopbook.com/>
  **Chapter 10** ("Convolutional Networks"). More current than Goodfellow's treatment; includes vision transformers as a counterpoint. Read sections 10.1–10.3.
- **Raschka, Liu, Mirjalili — *Machine Learning with PyTorch and Scikit-Learn*** (chapter source free on GitHub):
  <https://github.com/rasbt/machine-learning-book>
  **Chapter 14** ("Classifying Images with Deep Convolutional Neural Networks"). The bare-PyTorch CIFAR-10 walkthrough. The code matches the C5 style.
- **Lippe — *UvA Deep Learning Tutorials*** (free; University of Amsterdam):
  <https://uvadlc-notebooks.readthedocs.io/>
  **Tutorial 5 ("Inception, ResNet and DenseNet")** and **Tutorial 9 ("Convolutional Neural Networks")** are the most thorough PyTorch tutorials on CNN architecture available online.
- **Dive into Deep Learning** (free online; Aston Zhang, Zachary Lipton, Mu Li, Alex Smola; 2024 edition; PyTorch + JAX + TensorFlow side by side):
  <https://d2l.ai/>
  **Chapter 7** ("Convolutional Neural Networks") and **Chapter 8** ("Modern Convolutional Neural Networks"). Chapter 8 walks through implementations of LeNet, AlexNet, VGG, NiN, GoogLeNet, ResNet, and DenseNet in PyTorch — the historical lineage in code.

## Open-source projects to read (in this order)

You learn more about a framework from one hour of reading idiomatic source than from three hours of tutorials.

1. **`torchvision.models.resnet`** — the canonical PyTorch ResNet implementation. ~400 lines. Read the `BasicBlock` class (Section 3.3 of the paper, in code) and the `ResNet` class's `__init__` and `forward`:
   <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>
2. **`torchvision.models.vgg`** — the VGG implementation; even simpler than ResNet. ~150 lines. The `make_layers` function shows what "stack 3×3s" means in code:
   <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>
3. **`torchvision.transforms.v2`** — the new transforms API. Skim `_functional.py` and `_transform.py` once. You will see how PyTorch handles transform composition and tensor / PIL interop:
   <https://github.com/pytorch/vision/tree/main/torchvision/transforms/v2>
4. **PyTorch examples — `imagenet/main.py`** — the official training script for ImageNet classification with ResNet. ~400 lines. The C5 mini-project's `train.py` is a smaller variant of this file:
   <https://github.com/pytorch/examples/blob/main/imagenet/main.py>
5. **timm — `timm.models.resnet`** (Ross Wightman's PyTorch Image Models library; the unofficial production-grade model zoo):
   <https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py>
   We will not use `timm` this week, but knowing it exists is useful — it has hundreds of variants of every architecture on this page, all PyTorch-native and all pretrained.

## Videos (free, no signup)

- **Yann LeCun NYU Deep Learning 2021** (the lectures listed above):
  <https://www.youtube.com/playlist?list=PLLHTzKZzVU9e6xUfG10TkTWApKSZCzuBI>
- **Stanford CS231n 2017** (Karpathy and Johnson; the convolutional networks lecture is at <https://www.youtube.com/watch?v=bNb2fEVKeEo>):
  <https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv>
- **MIT 6.S191 (Introduction to Deep Learning)** (annual; free; the CNN lecture is the most up-to-date in this list):
  <http://introtodeeplearning.com/>
- **fast.ai 2024 course** (the lectures that pair with the fast.ai book; the CNN content is in lectures 5 and 6):
  <https://course.fast.ai/>
- **Andrej Karpathy — "Stanford CS231n Winter 2016 Lecture 7" on convolutional networks** (the lecture students rewatch):
  <https://www.youtube.com/watch?v=LxfUGhug-iQ>
- **3Blue1Brown — "But what is a convolution?"** (the math, with the rotating-image animation that explains every convolution diagram you have ever seen):
  <https://www.youtube.com/watch?v=KuXjwB4LzSA>

## The original dataset sources

- **CIFAR-10 / CIFAR-100** (Krizhevsky 2009; the canonical 32×32 color image benchmarks):
  <https://www.cs.toronto.edu/~kriz/cifar.html>
- **Krizhevsky's technical report** ("Learning Multiple Layers of Features from Tiny Images"; the paper that introduced CIFAR-10 and CIFAR-100):
  <https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf>
- **ImageNet** (Deng et al. 2009; the dataset that the pretrained ResNet weights were trained on):
  <https://www.image-net.org/>
- **Oxford Flowers-102** (Nilsback and Zisserman 2008; the mini-project stretch dataset):
  <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>
- **MS COCO** (the next-step object-detection dataset; not used this week):
  <https://cocodataset.org/>

## Tools you will use this week

- **`torch`** ≥ 2.4: `pip install "torch>=2.4,<3"`. Same install as Week 8.
- **`torchvision`** ≥ 0.19: `pip install "torchvision>=0.19,<1"`. Provides `models.resnet18`, the CIFAR-10 dataset, and the transforms.v2 API.
- **`numpy`** ≥ 2.0 (from Week 1): used in Exercise 1 (convolution by hand) and for one-off array ops in the mini-project.
- **`matplotlib`** ≥ 3.8 (from Week 3): for training-curve plots and confusion-matrix heatmaps.
- **`scikit-learn`** ≥ 1.5 (from Week 4): for `confusion_matrix` in the mini-project evaluation script.
- **`pytest`** ≥ 8: for the exercise tests.

A `requirements.txt` snippet for the week:

```text
torch>=2.4,<3
torchvision>=0.19,<1
numpy>=2,<3
scikit-learn>=1.5,<2
matplotlib>=3.8,<4
pytest>=8
```

### Installation gotchas

- **The pretrained ResNet-18 weights are downloaded on first use.** `torchvision.models.resnet18(weights="IMAGENET1K_V1")` triggers a ~45 MB download to `~/.cache/torch/hub/checkpoints/`. First call takes ~30 seconds on a fast connection; subsequent calls are instant.
- **The `weights=` argument vs. the legacy `pretrained=True`.** `pretrained=True` is deprecated in torchvision 0.13+; it still works but prints a DeprecationWarning. The C5 convention is to use the `weights=` enum form (`weights=ResNet18_Weights.IMAGENET1K_V1` or the string `"IMAGENET1K_V1"` or the convenience `weights="DEFAULT"`). The enum form is the only one we use in C5 lectures.
- **CIFAR-10 download.** `torchvision.datasets.CIFAR10(root="./data", download=True)` downloads ~170 MB and caches it in `./data/cifar-10-batches-py/`. First call takes 60-120 seconds; subsequent calls are instant.
- **The 224×224 resize is the most expensive step in the data pipeline.** Without GPU augmentation pipelines (which we cover in Week 10), CPU-side `transforms.v2.Resize((224, 224))` on every batch dominates a transfer-learning run's wall-clock time. On a CPU-only laptop, transfer-learning on CIFAR-10 at 224×224 can take 30+ minutes per epoch. The mitigation: use `num_workers=4` on the DataLoader; if still too slow, fall back to 32×32 native input and a modified ResNet stem.
- **`torch.backends.mps`** on Apple Silicon supports every layer we use this week; the runtime is roughly 5x faster than CPU for transfer learning at 224×224.
- **`torch.compile()`** can speed up transfer learning by 20-40% on GPU but adds ~30 seconds to the first batch and obscures error tracebacks. We do not use it this week; revisit in Week 10.

## Cheat sheets (one-page references)

- **PyTorch official cheat sheet** (one-pager; current with PyTorch 2.x):
  <https://pytorch.org/tutorials/beginner/ptcheat.html>
- **CS231n cheat sheet on ConvNets**:
  <https://cs231n.github.io/convolutional-networks/>
- **The "Conv2d output shape" formula** (the formula you will run a hundred times this week):
  ```text
  out_size = floor((in_size + 2*padding - dilation*(kernel_size - 1) - 1) / stride) + 1
  ```
  From the `nn.Conv2d` docs: <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>.

## Glossary cheat sheet

Keep this open in a tab while you work.

| Term | Plain English |
|------|---------------|
| **Convolution** | A windowed weighted sum applied at every spatial location of a feature map. A `Conv2d(in_c, out_c, k, stride, padding)` layer slides an `out_c × in_c × k × k` weight tensor over the input. |
| **Kernel** | The weight tensor of a convolution; equivalently, the "filter" that gets applied. A 3×3 kernel has 9 weights per `(in_channel, out_channel)` pair. |
| **Stride** | How far the kernel moves between applications. Stride 1 = every pixel, stride 2 = every other pixel (also halves the output size). |
| **Padding** | Extra zeros around the input edges so the kernel can apply at boundaries. `padding=1` with `kernel=3` keeps the spatial size constant at stride 1. |
| **Dilation** | The spacing between kernel elements. `dilation=2` with `kernel=3` covers a 5×5 receptive area with 9 weights (every other pixel). Used in dense prediction / segmentation. |
| **Feature map** | The output of a convolutional layer: an `(N, C_out, H_out, W_out)` tensor. |
| **Channel** | One slice of a feature map along the channel dimension. Conceptually, one "feature detector" output for each spatial location. |
| **Receptive field** | The size of the input region that influences one output activation. Grows with depth, kernel size, stride, and dilation. |
| **Pooling** | A non-parametric downsampling op. `MaxPool2d(2)` halves H and W by taking the max within each 2×2 window. |
| **`nn.Conv2d`** | The PyTorch 2D convolution layer. Most-used signature: `Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)`. |
| **`nn.MaxPool2d`** | The PyTorch max-pool layer. Most-used signature: `MaxPool2d(kernel_size=2, stride=2)`. |
| **`nn.AdaptiveAvgPool2d`** | Global pooling: averages each channel down to a target spatial size, typically `(1, 1)`. The head of every ResNet uses this. |
| **`nn.BatchNorm2d`** | Per-channel normalization across the batch + spatial dimensions. Lets you train deeper CNNs at higher learning rates. |
| **LeNet** | LeCun's 1998 CNN. Two convs, two pools, two FCs. The archetype. |
| **AlexNet** | Krizhevsky 2012. Five convs, three FCs, ReLU, dropout, two-GPU training. The result that started deep learning. |
| **VGG** | Simonyan and Zisserman 2014. Stacks of 3×3 convs. VGG-16 has 138M parameters; useful as a teaching architecture. |
| **ResNet** | He et al. 2015. Residual connections (`y = F(x) + x`) make 50+-layer networks trainable. Still the 2026 vision backbone. |
| **Residual block** | The `y = F(x) + x` motif. `F` is typically two or three convs with BN and ReLU; the skip connection adds the input back at the end. |
| **`torchvision.models.resnet18`** | The smallest ResNet, 11.7M parameters, designed for ImageNet-1k. The Week 9 default backbone. |
| **`weights=ResNet18_Weights.IMAGENET1K_V1`** | The pretrained-on-ImageNet-1k checkpoint shipped with torchvision. The default in 2026. |
| **Transfer learning** | Reusing parameters trained on one (large) dataset for a different (smaller) task. |
| **Feature extraction** | The transfer-learning recipe where the backbone is frozen and only a new head is trained. |
| **Fine-tuning** | The transfer-learning recipe where the entire network (or part of it) is trained end-to-end starting from pretrained weights. |
| **Backbone** | The body of a CNN that produces feature embeddings; everything before the classification head. |
| **Head** | The final layer(s) that map embeddings to class scores. For ResNet, `model.fc`. |
| **Freezing** | Setting `requires_grad=False` on a parameter so it is not updated during training. |
| **Differential learning rates** | Using different `lr` values for different parameter groups, typically a smaller `lr` for the pretrained backbone and a normal `lr` for the new head. |
| **`weights="IMAGENET1K_V1"`** | The string form of the pretrained-weights selector; equivalent to the enum form and shorter to type. |
| **Translation equivariance** | The property that shifting the input shifts the output the same way. True for convolution; not true for fully-connected layers. |
| **Weight sharing** | The same kernel weights are applied at every spatial position, so a 3×3 conv layer with `out_channels=64` on a 32×32 input has `9 * in_c * 64` weights regardless of the input resolution. |
| **CIFAR-10** | Ten classes of 32×32 RGB images, 50k train + 10k test. The Week 9 default dataset. |
| **CIFAR-10 normalization** | `mean=(0.4914, 0.4822, 0.4465)`, `std=(0.2470, 0.2435, 0.2616)`. Standard channel statistics. |
| **ImageNet** | 1.28M training images, 1000 classes. The pretraining dataset for every `torchvision.models` weight set. |
| **ImageNet normalization** | `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`. Use these when feeding a pretrained ResNet, not CIFAR-10's stats. |

---

*If a link 404s, please open an issue so we can replace it.*
