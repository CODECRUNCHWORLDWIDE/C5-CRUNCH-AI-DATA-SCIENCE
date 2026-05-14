# Week 8 — Resources

Every resource on this page is **free** and **publicly accessible**. No paywalled books, no proprietary PDFs, no signup-required courses. If a link breaks, please open an issue.

## Required reading (work it into your week)

- **PyTorch official tutorial — "Learn the Basics"** (the canonical eight-chapter introduction; uses FashionMNIST as the running example, identical to our Week 8):
  <https://pytorch.org/tutorials/beginner/basics/intro.html>
  Read **chapters 0 through 7** before Wednesday. The chapters are short (10-15 minutes each); the content overlaps directly with the C5 Week 8 lectures and uses the same FashionMNIST dataset. Treat this tutorial as the second pass.
- **PyTorch official tutorial — "A 60 Minute Blitz"** (Quick start; condensed version of the above):
  <https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html>
  Read Monday morning before Lecture 1. It is one hour; it covers tensors, autograd, `nn.Module`, and a training loop end to end. Useful as orientation; the C5 lectures go deeper on each topic.
- **PyTorch docs — Autograd Mechanics** (the definitive reference for how the autograd engine works under the hood; eight pages):
  <https://pytorch.org/docs/stable/notes/autograd.html>
  Read on Monday after Lecture 1. The sections on "How autograd encodes the history" and "Locally disabling gradient computation" are the ones you will refer back to repeatedly.
- **Sebastian Raschka — "PyTorch Deep Learning Fundamentals" essays** (free Substack series; the most readable plain-English explanations of the PyTorch API in 2024-2026):
  <https://magazine.sebastianraschka.com/>
  Start with "Understanding PyTorch nn.Linear" and "Understanding PyTorch DataLoader". Each is a 15-minute read.
- **fast.ai — *Practical Deep Learning for Coders* (book)** (free online; we are using bare PyTorch this week, not the fastai library, but the book's Chapter 4 walkthrough of "what's inside the training loop" is the cleanest narrative explanation in any text):
  <https://github.com/fastai/fastbook>
  Read **Chapter 4** ("Under the Hood: Training a Digit Classifier"). The fastai *library* is hidden until Chapter 5; Chapter 4 is bare PyTorch.
- **Andrej Karpathy — "Neural Networks: Zero to Hero" Episode 3 onwards** (the episodes that switch from `micrograd` to PyTorch; free on YouTube):
  <https://karpathy.ai/zero-to-hero.html>
  Episodes 3 and 4 ("Building makemore") build a character-level MLP in PyTorch with the same bare-loop discipline we use in C5. Two hours each; recommended over the weekend.

## The official PyTorch 2.x docs you will live in this week

- **PyTorch 2.x main reference**:
  <https://pytorch.org/docs/stable/index.html>
- **`torch.Tensor`** (every attribute and method):
  <https://pytorch.org/docs/stable/tensors.html>
- **Tensor creation ops** (`zeros`, `ones`, `arange`, `linspace`, `from_numpy`, `as_tensor`, `tensor`):
  <https://pytorch.org/docs/stable/torch.html#creation-ops>
- **`torch.autograd`** (the autograd engine):
  <https://pytorch.org/docs/stable/autograd.html>
- **Autograd mechanics note** (the under-the-hood explanation):
  <https://pytorch.org/docs/stable/notes/autograd.html>
- **`torch.nn`** (every layer type):
  <https://pytorch.org/docs/stable/nn.html>
- **`torch.nn.Module`** (the base class):
  <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>
- **`torch.nn.Linear`** (the fully-connected layer):
  <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>
- **`torch.nn.ReLU`** (the activation):
  <https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html>
- **`torch.nn.CrossEntropyLoss`** (combined logsoftmax + NLL; what you use for classification):
  <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>
- **`torch.nn.Sequential`** (chain layers without defining a Module):
  <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>
- **`torch.optim`** (every optimizer):
  <https://pytorch.org/docs/stable/optim.html>
- **`torch.optim.SGD`**:
  <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>
- **`torch.optim.Adam`**:
  <https://pytorch.org/docs/stable/generated/torch.optim.Adam.html>
- **`torch.utils.data`** (`Dataset`, `DataLoader`, `random_split`):
  <https://pytorch.org/docs/stable/data.html>
- **`torch.utils.data.Dataset`** (the abstract class):
  <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>
- **`torch.utils.data.DataLoader`** (batching, shuffling, multi-worker):
  <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>
- **`torch.device`** (device objects; `"cuda"`, `"mps"`, `"cpu"`):
  <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>
- **`torch.save` and `torch.load`** (serialization; the `weights_only=True` flag matters):
  <https://pytorch.org/docs/stable/generated/torch.save.html>
  <https://pytorch.org/docs/stable/generated/torch.load.html>
- **Saving and loading models tutorial** (the canonical `state_dict` pattern):
  <https://pytorch.org/tutorials/beginner/saving_loading_models.html>
- **PyTorch installation page** (the OS-specific `pip install torch` command):
  <https://pytorch.org/get-started/locally/>

## torchvision (the data sources for this week)

- **`torchvision.datasets`** (the index of ready-made image datasets):
  <https://pytorch.org/vision/stable/datasets.html>
- **`torchvision.datasets.FashionMNIST`** (the default for Week 8):
  <https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html>
- **`torchvision.datasets.CIFAR10`** (the stretch goal):
  <https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html>
- **`torchvision.transforms.v2`** (the modern transforms API; replaced `torchvision.transforms` in 2024):
  <https://pytorch.org/vision/stable/transforms.html>
- **`torchvision.transforms.v2.ToTensor`** and **`Normalize`** (the two transforms you will use this week):
  <https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToTensor.html>
  <https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.Normalize.html>

## The original dataset sources

- **FashionMNIST** (Zalando Research; GitHub repo with the raw IDX files and the paper):
  <https://github.com/zalandoresearch/fashion-mnist>
- **FashionMNIST paper** (Xiao, Rasul, Vollgraf 2017; six pages; the explicit "MNIST is too easy" argument):
  <https://arxiv.org/abs/1708.07747>
- **CIFAR-10** (Krizhevsky 2009; the canonical 32×32 color benchmark):
  <https://www.cs.toronto.edu/~kriz/cifar.html>
- **Krizhevsky's technical report** ("Learning Multiple Layers of Features from Tiny Images"; the paper that introduced CIFAR-10 and CIFAR-100):
  <https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf>

## Free textbooks (the canon, continued from Week 7)

- **Ian Goodfellow, Yoshua Bengio, Aaron Courville — *Deep Learning*** (free online; MIT Press, 2016):
  <https://www.deeplearningbook.org/>
  Re-skim **chapter 6** (deep feedforward networks; the framework-free version of what we are doing now) and **chapter 8** (optimization for training deep models). The Adam derivation in Section 8.5 is the canonical reference.
- **Sebastian Raschka, Yuxi (Hayden) Liu, Vahid Mirjalili — *Machine Learning with PyTorch and Scikit-Learn*** (the chapter source code is free and CC-BY on GitHub; book PDF is free for some chapters at the author's site):
  <https://github.com/rasbt/machine-learning-book>
  **Chapter 12** ("Parallelizing Neural Network Training with PyTorch") is the bare-PyTorch counterpart to Week 8. The code is well-commented and uses the modern API.
- **Aurélien Géron — *Hands-On Machine Learning, 3rd ed.*** (early chapters free; the author's notebooks are CC-BY on GitHub):
  <https://github.com/ageron/handson-ml3>
  The chapter-12 notebook on "Custom Models and Training with TensorFlow" has a parallel PyTorch version maintained by the community; useful for cross-framework comparison.
- **Lippe — *UvA Deep Learning Tutorials*** (free; University of Amsterdam; the tutorials are written in PyTorch and JAX side by side):
  <https://uvadlc-notebooks.readthedocs.io/>
  **Tutorial 2 ("Introduction to PyTorch")** and **Tutorial 3 ("Activation Functions")** cover the Week 8 material with extra emphasis on numerical-stability and initialization.

## The papers (free PDFs)

- **Adam Paszke et al. — "PyTorch: An Imperative Style, High-Performance Deep Learning Library"** (NeurIPS 2019; the official PyTorch paper):
  <https://arxiv.org/abs/1912.01703>
  Twelve pages. Read it for the design rationale, particularly Section 3 on the autograd engine and Section 4 on the C++ backend. The paper is the reason "define-by-run" replaced "define-and-run" as the field's default.
- **Diederik Kingma, Jimmy Ba — "Adam: A Method for Stochastic Optimization"** (ICLR 2015):
  <https://arxiv.org/abs/1412.6980>
  Fifteen pages. The Adam optimizer is the default in `torch.optim`; this paper is its definition. Algorithm 1 on page 2 is the update rule you will use this week.
- **Han Xiao, Kashif Rasul, Roland Vollgraf — "Fashion-MNIST: a Novel Image Dataset for Benchmarking ML Algorithms"** (2017):
  <https://arxiv.org/abs/1708.07747>
  Six pages. The FashionMNIST paper. The explicit motivation is that MNIST is too easy and the baseline floor is too high.
- **Léon Bottou — "Stochastic Gradient Descent Tricks"** (2012; from Week 7):
  <https://leon.bottou.org/publications/pdf/tricks-2012.pdf>
  Still relevant; the practical advice on learning rates and batch sizes applies verbatim in PyTorch.
- **Loshchilov and Hutter — "Decoupled Weight Decay Regularization"** (ICLR 2019; the AdamW paper):
  <https://arxiv.org/abs/1711.05101>
  Eleven pages. Introduces `torch.optim.AdamW`, the variant of Adam that we will reach for in Week 11 and later. Read it once for the conceptual difference between L2 regularization in the loss and decoupled weight decay in the optimizer.

## Tools you will use this week

- **`torch`** ≥ 2.4: `pip install "torch>=2.4,<3"`. The CPU build works on every OS. For CUDA, follow <https://pytorch.org/get-started/locally/> to get the right wheel. For Apple Silicon, no extra step (the `mps` backend ships in the standard wheel).
- **`torchvision`** ≥ 0.19: `pip install "torchvision>=0.19,<1"`. Provides FashionMNIST, CIFAR-10, the transforms, and a handful of pre-trained models we will not touch this week.
- **`numpy`** ≥ 2.0 (from Week 1): still used for one-off array ops, for `torch.from_numpy`, and for the gradient-check experiments in Lecture 1.
- **`matplotlib`** ≥ 3.8 (from Week 3): for training-curve plots, confusion-matrix heatmaps, and the sample-image grid.
- **`scikit-learn`** ≥ 1.5 (from Week 4): for `StandardScaler` in one homework problem and for `confusion_matrix` in the mini-project.
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

- **`pip install torch` is large.** The CPU wheel is ~200 MB on Linux/Mac and ~2 GB with CUDA. Be patient on a slow connection.
- **CUDA version mismatches.** If `torch.cuda.is_available()` returns `False` on a machine with a known-working NVIDIA GPU, the wheel you installed was compiled for the wrong CUDA version. The fix is at <https://pytorch.org/get-started/locally/>; pick the OS, package manager, and CUDA version explicitly.
- **`mps` on Apple Silicon.** Works out of the box on PyTorch 2.x. `torch.backends.mps.is_available()` returns `True` on M1 or later. Some operations are still slower on `mps` than on `cpu` (e.g., some pooling ops as of PyTorch 2.4); the C5 default is to detect and prefer `mps` for this week's models but to fall back to `cpu` if you see a warning about an unsupported op.
- **FashionMNIST download.** `torchvision.datasets.FashionMNIST(root="./data", download=True)` downloads ~30 MB and caches it in `./data/FashionMNIST/raw/`. First call takes 30-60 seconds; subsequent calls are instant. If the Zalando mirror is unreachable, the `urls` attribute on the dataset class lists alternates.
- **Multi-worker DataLoader on macOS.** `DataLoader(num_workers=4)` on macOS works but the first batch can take 10-15 seconds to spin up the subprocesses (the `fork` start method on macOS prints a warning; PyTorch handles it). For Week 8 the default `num_workers=0` is fine.
- **`weights_only=True` in `torch.load`.** Required as of PyTorch 2.4 to silence a UserWarning about untrusted pickle deserialization. Always pass it when loading checkpoints you saved with `state_dict`. We will see the deprecation play out across PyTorch 2.5 and 2.6.

## Videos (free, no signup)

- **PyTorch official YouTube channel** (free; the "PyTorch Tutorials" playlist is the video version of the written tutorials):
  <https://www.youtube.com/@PyTorch>
- **Andrej Karpathy — "Neural Networks: Zero to Hero" episodes 3-5** (the PyTorch episodes; ~6 hours total; the cleanest end-to-end walkthrough of writing PyTorch by hand):
  <https://karpathy.ai/zero-to-hero.html>
- **fast.ai "Practical Deep Learning for Coders" lectures 1-2** (free; the introduction is fastai-flavored but lecture 2 dives into the bare-PyTorch foundations):
  <https://course.fast.ai/>
- **Sebastian Raschka — Statistics & ML lecture series** (free; recorded UW-Madison lectures; the PyTorch series covers the same material as the C5 Week 8 lectures with a different presenter):
  <https://www.youtube.com/@SebastianRaschka>
- **Yann LeCun — NYU Deep Learning** (free; the lecture series from NYU; lectures 5 and 6 cover MLPs and CNNs in PyTorch):
  <https://atcold.github.io/NYU-DLSP21/>

## Open-source projects to read (in this order)

You learn more about a framework from one hour of reading idiomatic source than from three hours of tutorials.

1. **Andrej Karpathy's `nanoGPT`** — a 600-line PyTorch GPT-2 reimplementation. Read the `train.py` file once. The training loop is identical to the eight-line C5 pattern (with checkpointing and gradient accumulation added):
   <https://github.com/karpathy/nanoGPT>
2. **`torchvision.models.resnet`** — the canonical PyTorch ResNet implementation. Read it after Week 9 introduces convolutions, but skim the `__init__` and `forward` methods now to see what idiomatic `nn.Module` code looks like:
   <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>
3. **`torch.optim.Adam`** — the source of the Adam optimizer in PyTorch. ~150 lines. Reading it reveals what `param.data` and `param.grad` look like in practice:
   <https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py>
4. **PyTorch's `examples/mnist/main.py`** — the official PyTorch MNIST example, maintained by the PyTorch team. This file is the gold standard for what a tiny PyTorch training script should look like:
   <https://github.com/pytorch/examples/blob/main/mnist/main.py>
5. **Sebastian Raschka's `LLMs-from-scratch` chapter 4** — the chapter where Raschka builds a small GPT in bare PyTorch with a hand-written training loop, exactly the style we teach:
   <https://github.com/rasbt/LLMs-from-scratch>

## Cheat sheets (one-page references)

- **PyTorch official cheat sheet** (the PyTorch team's own one-pager; the most up-to-date single-page reference):
  <https://pytorch.org/tutorials/beginner/ptcheat.html>
- **Stanford CS231n cheat sheet on backprop** (still useful; PyTorch's autograd is exactly what this page describes):
  <https://cs231n.github.io/optimization-2/>
- **NumPy → PyTorch translation table** (Sebastian Raschka):
  <https://magazine.sebastianraschka.com/p/numpy-to-pytorch-translation>

## Glossary cheat sheet

Keep this open in a tab while you work.

| Term | Plain English |
|------|---------------|
| **`torch.Tensor`** | The fundamental data structure in PyTorch. A multi-dimensional array with dtype, device, and optional gradient tracking. Roughly a NumPy array with two extra fields. |
| **`requires_grad`** | A boolean on a `torch.Tensor`. When `True`, every op on the tensor is recorded into the autograd graph and contributes to the backward pass. Leaf tensors with `requires_grad=True` are the parameters of your model. |
| **Leaf tensor** | A tensor that was created directly (not as the output of another op). Model parameters are leaf tensors. The `.grad` attribute is only populated on leaf tensors. |
| **Autograd graph** | A directed acyclic graph built on the fly during the forward pass. Each node is an operation; each edge is a tensor flowing into or out of an op. Calling `.backward()` walks the graph in reverse. |
| **`.backward()`** | The method on a scalar tensor that triggers the backward pass. Accumulates gradients into the `.grad` attribute of every leaf tensor in the graph. |
| **`.grad`** | The accumulated gradient on a leaf tensor. Reset to zero between training steps with `optimizer.zero_grad()` (or `param.grad.zero_()` if you are managing manually). |
| **`nn.Module`** | The base class for every neural-network component. Subclass it to define a model. Override `__init__` to register submodules; override `forward(x)` to define the forward pass. |
| **`nn.Linear`** | A fully-connected layer. `nn.Linear(in_features, out_features)` holds a `(out_features, in_features)` weight matrix and an `(out_features,)` bias. The matrix-vector product is `x @ W.T + b`. |
| **`nn.ReLU`** | An activation module that applies `max(0, x)` elementwise. Stateless; no parameters. |
| **`nn.Sequential`** | A container that chains modules in order. `nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))` is a complete two-layer MLP in one line. |
| **`nn.CrossEntropyLoss`** | The standard loss for multi-class classification. Combines `log_softmax` and negative-log-likelihood into one numerically stable op. Takes raw logits (not probabilities) as input. |
| **`torch.optim.SGD`** | Stochastic gradient descent. `SGD(model.parameters(), lr=0.01)` constructs the optimizer; `.zero_grad()` clears gradients, `.step()` applies the update. |
| **`torch.optim.Adam`** | Adam optimizer (Kingma and Ba 2015). Adaptive per-parameter learning rates plus momentum. The 2026 default for new projects. |
| **`Dataset`** | The PyTorch abstract class with `__len__` and `__getitem__`. Subclass it to wrap your own data; PyTorch's `DataLoader` consumes it. |
| **`DataLoader`** | The class that wraps a `Dataset` and produces shuffled, batched, multi-worker iterators. Most-used signature: `DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)`. |
| **`torchvision.datasets.FashionMNIST`** | The drop-in replacement for MNIST: ten classes of 28×28 grayscale clothing images, 60k train + 10k test. The Week 8 default. |
| **`torchvision.datasets.CIFAR10`** | Ten classes of 32×32 color images, 50k train + 10k test. The Week 8 stretch goal and the Week 9 default. |
| **`transforms.v2.ToTensor`** | Converts a PIL Image or NumPy array to a `torch.Tensor` in `[0, 1]` with the channel dimension first (CHW). |
| **`transforms.v2.Normalize`** | Standardizes a tensor by subtracting a per-channel mean and dividing by a per-channel std. For FashionMNIST: `mean=(0.2860,)`, `std=(0.3530,)`. |
| **Device** | A `torch.device` object: `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`. The location where the tensor's memory lives. Operations between tensors require them on the same device. |
| **`.to(device)`** | The method that copies a tensor (or moves a model's parameters) to a device. No-op when source and destination are the same device. |
| **`state_dict`** | A Python dict mapping parameter and buffer names to tensors. The portable serialization format for PyTorch models. |
| **`weights_only=True`** | A flag on `torch.load` (PyTorch 2.4+) that disables the pickle deserializer for the safe loading path. Should be passed whenever loading a state_dict from disk. |
| **`model.train()` / `model.eval()`** | Flags that toggle layers with different train/eval behavior (Dropout, BatchNorm). No effect on a vanilla MLP, but always call them anyway; it is the convention. |
| **`with torch.no_grad():`** | A context manager that disables autograd tracking inside the block. Used for evaluation and inference; halves memory usage and roughly doubles speed for forward-only passes. |
| **`optimizer.zero_grad()`** | Resets `.grad` on every parameter the optimizer manages. The most-forgotten line in a training loop; if you forget it, gradients accumulate across batches and the loss explodes. |
| **`optimizer.step()`** | Applies the update rule (SGD, Adam, etc.) using the current `.grad` values on the managed parameters. |
| **Pin memory** | A `DataLoader` option (`pin_memory=True`) that places batch tensors in page-locked CPU memory, speeding up the transfer to GPU. Useful when training on GPU; ignored on CPU-only setups. |
| **`num_workers`** | A `DataLoader` option that spawns subprocess workers to prefetch batches. `num_workers=4` is a sensible default on a machine with 4+ cores; `num_workers=0` keeps everything in the main process (default; safe; slower). |
| **`drop_last`** | A `DataLoader` option that discards the final partial batch. Useful when batch-norm or other layers misbehave on tiny batches; safe to leave `False` for an MLP. |
| **Eager mode** | PyTorch's default execution model: every line of Python runs as written; the autograd graph is built on the fly. Contrast with TensorFlow 1.x's static graph and JAX's `jit`-compiled functions. |
| **`torch.compile()`** | A PyTorch 2.x feature that JIT-compiles a model to a faster backend (Triton, by default). Speeds up training 20-50% on supported architectures. Not used this week; covered in Week 10. |

---

*If a link 404s, please open an issue so we can replace it.*
