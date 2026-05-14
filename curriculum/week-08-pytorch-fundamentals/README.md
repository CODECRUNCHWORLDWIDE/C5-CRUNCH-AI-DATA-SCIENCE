# Week 8 — PyTorch Fundamentals

> *Last week you wrote backpropagation by hand. This week the framework writes it for you. We move every line of the Week 7 MLP from NumPy to PyTorch 2.x, then keep going: the autograd engine, `nn.Module`, `DataLoader`, `optim.SGD` and `optim.Adam`, GPU device placement, `state_dict` serialization, and the bare training loop without Lightning or fastai. By Sunday you will have an image classifier on FashionMNIST (or CIFAR-10 for the ambitious) trained on a real DataLoader pipeline, saved to disk, and reloaded for evaluation. The autograd graph is no longer a mystery — you wrote one in NumPy last week.*

Welcome to week eight of **C5 · Crunch AI / Data Science**. Week 7 ended with a NumPy MLP that reached 95%+ on MNIST. That model was honest: every line was yours, every gradient was derived from the chain rule, and `loss.backward()` did not exist anywhere in the file. Week 8 keeps the honesty but trades the manual derivation for a framework that handles it correctly, on GPU, with one line.

The trade is real. You give up: (1) the satisfaction of writing `dW1 = X.T @ dZ1` and (2) the precision of knowing exactly what is happening inside `backward()`. You gain: (1) thirty-times-faster iteration on architecture changes, (2) GPU acceleration without writing CUDA, (3) automatic mixed precision, (4) a model-serialization format the rest of the field actually uses, and (5) the option to ship something larger than a two-layer MLP without re-deriving Jacobians for every architecture change. By the end of Week 8 you should be able to argue for both sides of that trade — and you should reach for the framework, because that is what the field does.

Three commitments before we start:

1. **No `pytorch-lightning`, no `fastai`, no `accelerate`, no `transformers`.** Those abstractions exist because the bare PyTorch training loop is repetitive; you should still write the repetitive version first. The C5 conviction is that a student who has written `for batch in dataloader: optimizer.zero_grad(); loss.backward(); optimizer.step()` five times can read Lightning code in an hour and understand every callback. A student who started with Lightning cannot debug a stalled training loop, because the loop has been hidden from them. Bare PyTorch first. Lightning in Week 12 or later.
2. **You will write the training loop by hand, every line.** `model.train()` and `model.eval()` flags. `optimizer.zero_grad()`. `loss.backward()`. `optimizer.step()`. Gradient accumulation if relevant. Per-epoch validation. Per-epoch checkpointing. This is the loop that ships every PyTorch model in production at every company. You will memorize it this week.
3. **You will run on CPU first, then move to GPU only after the CPU version works.** GPU debugging is harder than CPU debugging because errors are asynchronous and the stack traces are in C++. The C5 rule is: get the loop right on a tiny batch on CPU, verify the loss goes down, *then* move to GPU. If you do not have a GPU, the week is still completable — FashionMNIST on a 2-layer MLP trains in three minutes on a laptop CPU. Section 9 of Lecture 3 covers GPU when you have one.

We target **PyTorch 2.x** (we test on 2.4 and 2.5; the API is stable across the 2.x series), **torchvision 0.19+** (the FashionMNIST and CIFAR-10 loaders), and **Python 3.11+**. The Apple Silicon `mps` backend is supported by PyTorch 2.x and works for this week's models; the `cuda` backend is what we test on Linux.

PyTorch 2.x reference documentation lives at <https://pytorch.org/docs/stable/index.html>. Pin a tab; you will refer to it every twenty minutes this week.

---

## Learning objectives

By the end of this week, you will be able to:

- **Construct and manipulate `torch.Tensor`** objects. Know the difference between `torch.tensor(data)` (copies), `torch.as_tensor(data)` (shares memory when possible), `torch.from_numpy(arr)` (zero-copy view of a NumPy array), and `torch.zeros / ones / arange / linspace`. Reference: <https://pytorch.org/docs/stable/tensors.html>.
- **Use `autograd`** to compute gradients of scalar-valued functions. Set `requires_grad=True` on a leaf tensor; build a computation; call `.backward()`; read `.grad`. Understand that the graph is built on the fly (eager mode, not symbolic). Reference: <https://pytorch.org/docs/stable/autograd.html>.
- **Subclass `nn.Module`** to define a model. Override `__init__` to register submodules and parameters; override `forward(x)` to define the forward pass. Use `parameters()` and `state_dict()` to enumerate trainable tensors. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>.
- **Pick an optimizer.** `torch.optim.SGD` for the baseline; `torch.optim.Adam` for the default in 2026; know the three-line `optimizer.zero_grad(); loss.backward(); optimizer.step()` pattern in your sleep. Reference: <https://pytorch.org/docs/stable/optim.html>.
- **Build a `Dataset` and a `DataLoader`.** Subclass `torch.utils.data.Dataset` for a custom data source; use `torch.utils.data.DataLoader` for batching, shuffling, and multi-worker prefetch. Use `torchvision.datasets.FashionMNIST` as the canonical example. References: <https://pytorch.org/docs/stable/data.html>, <https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html>.
- **Write the bare training loop.** Outer loop over epochs; inner loop over batches; `model.train()` at start of each epoch; `optimizer.zero_grad()`; forward; loss; `loss.backward()`; `optimizer.step()`; per-epoch `model.eval()` validation pass inside `with torch.no_grad():`. The eight lines that ship every PyTorch model.
- **Place tensors and models on a device.** Detect the available device (`cuda`, `mps`, `cpu`); call `.to(device)` on the model once and on every batch in the loop; know why CPU-GPU transfers are the slowest part of a naive training loop. Reference: <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>.
- **Save and load model state.** Use `torch.save(model.state_dict(), path)` and `model.load_state_dict(torch.load(path, weights_only=True))`. Avoid `torch.save(model, path)` (the pickle approach) for portability and security reasons. Reference: <https://pytorch.org/tutorials/beginner/saving_loading_models.html>.
- **Diagnose three common training failures.** Loss is NaN (usually a learning-rate or numerical-stability issue), loss does not decrease (usually a forgotten `optimizer.zero_grad()` or an incorrect device placement), validation accuracy decreases while training accuracy rises (overfitting). Each has a specific debugging recipe in Lecture 3.
- **Ship** a mini-project: an end-to-end image classifier on FashionMNIST or CIFAR-10 with a saved `state_dict`, a training-curve plot, a reload-and-evaluate script, and a 600-word report explaining the architecture choice and the trade-offs. Pushed to your portfolio repo.
- **Pass** every `pytest` case on the Week 8 exercises.

---

## Prerequisites

- **Week 7 complete.** In particular, you have the NumPy MLP-on-MNIST notebook checked into your `crunch-ai-portfolio-<yourhandle>` repo with a verified gradient check and ≥95% test accuracy. You know what `dW1 = X.T @ dZ1` means and why the chain rule produces it. Week 8 builds on that intuition; you will see the framework recover those same gradients, automatically, in seconds.
- **Python 3.11+** working in your virtualenv. CI runs on 3.12.
- **PyTorch 2.x** installed: `pip install "torch>=2.4,<3" torchvision`. The default wheel installs the CPU build; for CUDA, see <https://pytorch.org/get-started/locally/> for the OS-specific command. The `mps` backend on Apple Silicon needs no extra install.
- **About 300 MB of disk** for the FashionMNIST and CIFAR-10 caches (`~/data/FashionMNIST/raw/` and `~/data/cifar-10-batches-py/`). `torchvision.datasets.FashionMNIST(root, download=True)` handles the first call; subsequent calls are instant.
- **Optional but useful:** a CUDA GPU (any GPU made after 2018), an Apple Silicon Mac (M1 or later), or a free Google Colab T4 session. None of these are required to pass the week, but the optional Lecture 3 GPU experiments need one of them.
- **scikit-learn** 1.5+ (from Week 4) and **matplotlib** 3.8+ (from Week 3) are still used; we no longer need `numpy` to be the implementation language but we still use it for one-off array ops and for the StandardScaler.

You should be comfortable with:

- **NumPy broadcasting.** PyTorch broadcasts the same way. `tensor.shape`, `tensor.unsqueeze(dim)`, `tensor.squeeze(dim)`, and `tensor.view(shape)` are the moves you will make most often. Week 1.
- **Object-oriented Python.** `class Foo(Bar):`, `super().__init__()`, `@property`. `nn.Module` is the most-subclassed class in deep learning. Week 0 / Week 6 if you need a refresher.
- **The Week 7 MLP forward pass.** This week's `nn.Module` subclass is a one-to-one translation of last week's `forward(X, W1, b1, W2, b2)` function — Linear, ReLU, Linear — but written with `nn.Linear` and `nn.ReLU` modules. You should be able to read the PyTorch code and point to the matching line of Week 7 NumPy.

---

## Topics covered

- **What PyTorch is.** A Python library for tensor computation with GPU acceleration and reverse-mode autodiff. Created at FAIR in 2016 as a Python rewrite of Torch (Lua). The 1.0 release was 2018; the 2.0 release (with `torch.compile`) was 2023. We target 2.x for the stable API and the Apple Silicon support. Reference: <https://pytorch.org/>.
- **`torch.Tensor` vs. `numpy.ndarray`.** Same memory layout (with `torch.from_numpy`), same broadcasting rules, same dtype conventions. The differences: PyTorch tensors can live on a GPU, can track gradients, and can be operated on by `nn.Module` subclasses. The `torch.Tensor` API is roughly 95% a superset of NumPy's; the 5% difference is intentional (PyTorch has explicit dtype tracking; NumPy has more loose promotion rules).
- **The autograd graph.** Every operation on a `requires_grad=True` tensor records a node in a directed acyclic graph. Calling `.backward()` on a scalar walks the graph backwards, accumulating gradients into `.grad` attributes of leaf tensors. The graph is rebuilt on every forward pass (this is "define-by-run" or "eager mode"); contrast with TensorFlow 1.x's static graph or JAX's `jit`. Reference: <https://pytorch.org/docs/stable/notes/autograd.html>.
- **`nn.Module` and friends.** The base class for every neural-network layer in PyTorch. The pattern is: define `__init__` to register submodules; define `forward(x)` to use them. `nn.Linear(in_features, out_features)` is the equivalent of your Week 7 `(W, b)` pair. `nn.ReLU()`, `nn.Sigmoid()`, `nn.Tanh()` are stateless activation modules. `nn.Sequential(...)` chains modules. Reference: <https://pytorch.org/docs/stable/nn.html>.
- **Optimizers.** `torch.optim.SGD(params, lr=0.01, momentum=0.0)` is the workhorse; `torch.optim.Adam(params, lr=1e-3)` is the modern default. Both take a list of parameters (typically `model.parameters()`) and a learning rate. The three-method API is `zero_grad()`, `step()`, and `state_dict()`. Reference: <https://pytorch.org/docs/stable/optim.html>.
- **`DataLoader` and `Dataset`.** `torch.utils.data.Dataset` is an abstract class with `__len__` and `__getitem__`. `torch.utils.data.DataLoader` wraps a `Dataset` and produces batches with shuffling, multiprocessing, and pin-memory transfers. `torchvision.datasets.FashionMNIST` is a ready-made `Dataset`. References: <https://pytorch.org/docs/stable/data.html>, <https://pytorch.org/vision/stable/datasets.html>.
- **FashionMNIST.** The "drop-in replacement for MNIST" introduced by Zalando in 2017: 60,000 training and 10,000 test 28×28 grayscale images across ten clothing categories. Harder than MNIST (a two-layer MLP reaches ~88% rather than 98%) and a more honest benchmark. Reference: <https://github.com/zalandoresearch/fashion-mnist>.
- **CIFAR-10.** The standard 32×32 color-image benchmark: 50,000 training and 10,000 test images across ten categories (airplane, car, bird, cat, etc.). Introduced by Krizhevsky 2009. A two-layer MLP reaches ~50% (poor); a small CNN reaches ~70% (Week 9). For Week 8, FashionMNIST is the default; CIFAR-10 is the stretch goal that motivates Week 9's CNNs.
- **The bare training loop.** Eight lines, memorize them:

```python
for epoch in range(n_epochs):
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
```

- **Device placement.** `device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")`. Call `model.to(device)` once; call `X.to(device)` on every batch. The `.to(device)` call is a copy on a different-device tensor and a no-op on a same-device tensor. The most common GPU bug is forgetting to move the batch; the error message is "Expected all tensors to be on the same device" and it is unambiguous.
- **`state_dict`.** A Python dict mapping parameter and buffer names to tensors. `torch.save(model.state_dict(), "model.pt")` writes the dict to disk; `model.load_state_dict(torch.load("model.pt", weights_only=True))` restores it. The `weights_only=True` flag (PyTorch 2.4+) disables the pickle deserializer for the safe loading path and is now the recommended default. Reference: <https://pytorch.org/tutorials/beginner/saving_loading_models.html>.
- **What we do not do.** No `torch.compile()` (a 2.x feature; speeds things up but adds complexity; defer to Week 10). No mixed-precision (`torch.cuda.amp`; defer to Week 9). No `DistributedDataParallel` (multi-GPU training; defer to Week 14). No `torch.fx` (graph manipulation; specialist). No Lightning. No fastai. No pre-trained models — those return in Week 11 (transfer learning).

---

## Weekly schedule

Target about **38 hours**. Monday and Tuesday rebuild Week 7's MLP in PyTorch (the "wait, that's it?" moment). Wednesday introduces DataLoader and FashionMNIST. Thursday is the full training loop on FashionMNIST. Friday-Sunday is the mini-project.

| Day       | Focus                                                       | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|-------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Tensors; autograd; the chain rule, framework-style          |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    7h       |
| Tuesday   | `nn.Module`; optimizers; rebuild the Week 7 MLP             |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    7h       |
| Wednesday | DataLoader; FashionMNIST; the bare training loop            |   2.5h   |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0h       |    6h       |
| Thursday  | GPU / mps device placement; save/load `state_dict`          |   1.5h   |   1h      |     2h     |   0.5h    |   1h     |     2h       |   0h       |    8h       |
| Friday    | Mini-project: train FashionMNIST end-to-end                 |   0h     |   0.5h    |     0h     |   0.5h    |   1h     |     3h       |   0.5h     |   5.5h      |
| Saturday  | Mini-project: confusion matrix, write-up                    |   0h     |   0h      |     0h     |   0h      |   1h     |     3h       |   0h       |    4h       |
| Sunday    | Quiz, push to portfolio repo                                |   0h     |   0h      |     0h     |   0.5h    |   0h     |     0h       |   0h       |   0.5h      |
| **Total** |                                                             | **10h**  | **7.5h**  | **2h**     | **3h**    | **6h**   | **8h**       | **1h**     |  **37.5h**  |

The schedule is comfortable on Monday and Tuesday because PyTorch hides the algebra that occupied last week. The schedule is tight on Wednesday because `DataLoader` semantics (shuffle, num_workers, pin_memory, drop_last) have a surprising amount of detail and the FashionMNIST download takes a few minutes the first time. Thursday is the first GPU-or-mps run; expect a 30-minute detour if your environment is not already configured.

---

## How to navigate this week

| File | What is inside |
|------|----------------|
| [README.md](./README.md) | This overview |
| [resources.md](./resources.md) | PyTorch 2.x docs by URL, the official tutorials, Sebastian Raschka's free essays, the fast.ai free book, Karpathy's "Zero to Hero" episodes that use PyTorch |
| [lecture-notes/01-tensors-and-autograd.md](./lecture-notes/01-tensors-and-autograd.md) | `torch.Tensor`; the autograd engine; `requires_grad`; the computation graph; manual gradient comparison against Week 7's NumPy MLP |
| [lecture-notes/02-modules-optimizers-and-the-training-loop.md](./lecture-notes/02-modules-optimizers-and-the-training-loop.md) | `nn.Module`; `nn.Linear`, `nn.ReLU`, `nn.Sequential`; optimizers (SGD, Adam); the eight-line training loop |
| [lecture-notes/03-datasets-dataloaders-devices-and-checkpoints.md](./lecture-notes/03-datasets-dataloaders-devices-and-checkpoints.md) | `Dataset` and `DataLoader`; FashionMNIST via torchvision; device placement; `state_dict` save / load; the three common training failures and their fixes |
| [exercises/exercise-01-tensors-and-autograd.py](./exercises/exercise-01-tensors-and-autograd.py) | Tensor creation, broadcasting, autograd on a scalar function, manual `.backward()` vs. Week 7's NumPy gradient |
| [exercises/exercise-02-nn-module-and-training-loop.py](./exercises/exercise-02-nn-module-and-training-loop.py) | Build a 2-layer MLP as an `nn.Module`; the optimizer / loss / step pattern; train on synthetic data; verify the loss decreases |
| [exercises/exercise-03-fashionmnist-end-to-end.py](./exercises/exercise-03-fashionmnist-end-to-end.py) | The full training loop on FashionMNIST via `torchvision.datasets`; reach ≥85% test accuracy; save and reload the state dict |
| [exercises/SOLUTIONS.md](./exercises/SOLUTIONS.md) | Reference solutions with commentary; read only after attempting |
| [challenges/challenge-01-cifar10-mlp-vs-cnn-baseline.md](./challenges/challenge-01-cifar10-mlp-vs-cnn-baseline.md) | Train an MLP and a tiny CNN on CIFAR-10; document the gap; preview Week 9 |
| [challenges/challenge-02-write-your-own-dataset.md](./challenges/challenge-02-write-your-own-dataset.md) | Implement a `Dataset` subclass for an image folder on disk with on-the-fly transforms; integrate into the training loop |
| [quiz.md](./quiz.md) | 10 multiple-choice questions on tensors, autograd, `nn.Module`, `DataLoader`, and devices |
| [homework.md](./homework.md) | Six problems for the week |
| [mini-project/README.md](./mini-project/README.md) | Build an image classifier on FashionMNIST (or CIFAR-10) end to end with checkpointing and a reload-and-evaluate script |
| [mini-project/starter.py](./mini-project/starter.py) | Runnable starter: data loaders, model class, training-loop scaffold, save / load helpers |
| [mini-project/rubric.md](./mini-project/rubric.md) | Grading rubric for the mini-project |

---

## Stretch goals

- Read **the official PyTorch "Learn the Basics" tutorial** end to end (eight chapters, about ninety minutes; <https://pytorch.org/tutorials/beginner/basics/intro.html>). The C5 lectures are a parallel pass; the official tutorial uses the same FashionMNIST example and is the canonical reference when in doubt.
- Read **Sebastian Raschka — "Understanding Mixed Precision Training"** (free; <https://magazine.sebastianraschka.com/p/understanding-mixed-precision-training>). We do not use mixed precision this week, but the essay is the cleanest free explanation of why 2026 production training is mostly bf16 and not fp32. File it for Week 9.
- Watch **Karpathy — "Let's build GPT: from scratch, in code, spelled out"** (free; YouTube). The video is famous for its transformer focus but the first hour is the most patient introduction to writing PyTorch by hand in any format. Watch the first hour this week; save the rest for Week 13 (transformers).
- Read **the PyTorch autograd notes** (<https://pytorch.org/docs/stable/notes/autograd.html>). The "Gradients for non-differentiable functions" section explains why ReLU's gradient at zero is defined as zero by convention, which is exactly the question Week 7 left dangling.
- Read the **FashionMNIST paper** (Xiao, Rasul, Vollgraf 2017; <https://arxiv.org/abs/1708.07747>). Six pages; the explicit motivation is "MNIST is too easy; researchers report 99.7% and call the problem solved." FashionMNIST raised the baseline floor; your MLP at 88% this week is honest.

---

## What you will *not* do this week

You will not:

- **Use Lightning, fastai, or any other training-loop wrapper.** The eight-line loop is the one to memorize. Wrappers come back in Week 12.
- **Train a transformer or use a pre-trained model.** Transformers are Weeks 13–14. Transfer learning is Week 11. The model this week is an MLP (or a tiny CNN at the stretch).
- **Use `torch.compile()`.** It speeds up training; it also adds 30 seconds to the first batch and obscures error messages during debugging. Add it later, once the model works.
- **Use mixed precision (`torch.cuda.amp`).** It is the production default in 2026, but the bare loop in fp32 is what you learn first. Week 9 introduces fp16 / bf16.
- **Use `DistributedDataParallel` or multi-GPU.** Week 14.
- **Skip the CPU-first rule.** A NaN loss on GPU is a much harder bug than a NaN loss on CPU. Get the loop right on CPU on a small batch; *then* move to GPU.

That list is deliberate. The point of Week 8 is to *write the bare loop without help*. The list of things Lightning hides is the list of things you are learning to do by hand. When you reach for Lightning in Week 12, you will know what every callback is replacing.

---

## A note on the EXPERIMENT cards

Lectures continue to use `EXPERIMENT` callouts:

> **EXPERIMENT — short title.** A one-paragraph drill you can run in a Python REPL in five to fifteen minutes. The drill makes a specific claim from the lecture concrete. Run them. Do not skip them.

The Lecture 1 experiment that builds the autograd graph by hand and prints `.grad` on every leaf tensor is the experiment that makes "the chain rule, mechanized" stop being a slogan and start being something you have watched a Python interpreter do. The Lecture 3 experiment that times CPU vs. GPU on the same batch is the experiment that calibrates your intuition for when GPU is worth the transfer cost (answer: not for a 2-layer MLP on FashionMNIST; yes for a CNN on CIFAR-10 in Week 9).

---

## Up next

[Week 9 — Convolutional Neural Networks](../week-09/) — once you have pushed the Week 8 FashionMNIST classifier, the reload script, and the report to your portfolio repo. Week 9 introduces convolutional layers (`nn.Conv2d`), pooling, and the LeNet / VGG / ResNet architectural progression on CIFAR-10. The bare training loop from Week 8 carries over verbatim; only the model class changes. The discipline you built this week — train on CPU first, write the eight-line loop by hand, save the state dict, plot the curves — is the routine for the next five weeks.
