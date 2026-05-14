# Week 7 — Resources

Every resource on this page is **free** and **publicly accessible**. No paywalled books, no proprietary PDFs, no signup-required courses. If a link breaks, please open an issue.

## Required reading (work it into your week)

- **Michael Nielsen — *Neural Networks and Deep Learning*** (free online; the single best introduction to MLPs and backprop in any format):
  <http://neuralnetworksanddeeplearning.com/>
  Read **chapters 1 and 2** before Tuesday. Chapter 1 builds the perceptron and the MLP, the cross-entropy loss, and the sigmoid neuron. Chapter 2 derives backprop in matrix form with the same notation we use this week. The MNIST examples in Chapter 1 are the canonical reference; your code this week is more or less an updated version of Nielsen's Chapter 1 code with He initialization and ReLU.
- **Stanford CS231n notes — "Backpropagation, Intuitions"** (Karpathy and Li; the cleanest derivation of backprop on a computational graph):
  <https://cs231n.github.io/optimization-2/>
- **Stanford CS231n notes — "Neural Networks Part 1: Setting up the Architecture"** (the activation-function survey and the initialization discussion):
  <https://cs231n.github.io/neural-networks-1/>
- **Stanford CS231n notes — "Neural Networks Part 2: Setting up the Data and the Loss"** (cross-entropy, softmax, regularization):
  <https://cs231n.github.io/neural-networks-2/>
- **3Blue1Brown — "Neural Networks" video series** (Grant Sanderson; the cleanest visual explanation of MLPs and backprop in print or video; four episodes, ~15 minutes each):
  <https://www.3blue1brown.com/topics/neural-networks>
- **Andrej Karpathy — "Neural Networks: Zero to Hero" episodes 1 and 2** (free on YouTube; episode 1 builds an autograd library, episode 2 builds a character-level MLP; together they cover everything in this week's curriculum):
  <https://karpathy.ai/zero-to-hero.html>

## The official docs you will bounce between all week

- **numpy reference manual**:
  <https://numpy.org/doc/stable/reference/index.html>
- **numpy `ndarray`**:
  <https://numpy.org/doc/stable/reference/arrays.ndarray.html>
- **numpy linear algebra (`np.linalg`)**:
  <https://numpy.org/doc/stable/reference/routines.linalg.html>
- **numpy random number generation (`np.random.default_rng`)**:
  <https://numpy.org/doc/stable/reference/random/generator.html>
- **`numpy.typing.NDArray`** (the type alias we use on every function signature):
  <https://numpy.org/doc/stable/reference/typing.html>
- **Python docs — `typing` module**:
  <https://docs.python.org/3/library/typing.html>
- **Python docs — `math` module** (for `math.exp`, `math.log` where used scalar-wise):
  <https://docs.python.org/3/library/math.html>
- **scikit-learn `fetch_openml`** (the MNIST download):
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html>
- **matplotlib `pyplot`** (the figure API we use for training curves):
  <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html>

## Free textbooks (the canon)

- **Ian Goodfellow, Yoshua Bengio, Aaron Courville — *Deep Learning*** (free online; MIT Press, 2016):
  <https://www.deeplearningbook.org/>
  Read **chapter 6** (deep feedforward networks) before Wednesday — it covers the MLP, activations, output units, and the universal approximation theorem in forty pages. Skim **chapter 8** (optimization for training deep models) for the sections on mini-batch SGD and learning rates. The math is consistent with the notation we use this week.
- **Michael Nielsen — *Neural Networks and Deep Learning*** (free online; rough sketches and exercises along the way):
  <http://neuralnetworksanddeeplearning.com/>
  The full book is six chapters, ~150 pages of reading. Chapters 1 and 2 are required this week; chapters 3 (improvements to learning), 5 (vanishing gradients), and 6 (convolutions) are for Weeks 8 and 9.
- **Kevin Murphy — *Probabilistic Machine Learning: An Introduction*** (free PDF; MIT Press, 2022):
  <https://probml.github.io/pml-book/book1.html>
  Chapter 13 (neural networks for unstructured data) is the Bayesian-flavored MLP treatment. Useful if you want a cleaner derivation of cross-entropy as the negative-log-likelihood of a categorical distribution.
- **Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*** (free PDF from Stanford):
  <https://hastie.su.domains/ElemStatLearn/>
  **Chapter 11** is the classic statistical-learning treatment of MLPs. The notation is older (this is the textbook where "neural network" still means "a sigmoid MLP fit by quasi-Newton") but the math is correct and the warnings about over-fitting are sober and worth re-reading.
- **Aurélien Géron — *Hands-On Machine Learning, 3rd edition*** (early chapters free as preview from O'Reilly; the [author's notebooks are public and CC-BY on GitHub](https://github.com/ageron/handson-ml3)):
  The chapter-10 notebook on "Introduction to Artificial Neural Networks with Keras" is the practical reference; we are doing the same exercise in NumPy this week, framework-free.

## The papers (free PDFs, the originals)

The neural-network field has a long history; the founding papers are still readable.

- **Frank Rosenblatt — "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"** (Psychological Review, 1958). The original perceptron paper. Twenty pages.
  <https://psycnet.apa.org/doi/10.1037/h0042519> (abstract; full PDF in many archives)
- **David Rumelhart, Geoffrey Hinton, Ronald Williams — "Learning representations by back-propagating errors"** (Nature, 1986). The paper that introduced backpropagation to the modern community. Three pages; foundational.
  <https://www.nature.com/articles/323533a0>
- **Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner — "Gradient-based learning applied to document recognition"** (Proceedings of the IEEE, 1998). The paper that defined the MNIST benchmark and introduced LeNet. Forty pages; the MLP baseline results are in Table II.
  <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>
- **Xavier Glorot, Yoshua Bengio — "Understanding the difficulty of training deep feedforward neural networks"** (AISTATS, 2010). The paper that introduced Xavier (a.k.a. Glorot) initialization. Eight pages; the figure that compares sigmoid and tanh saturation is the one to print and tape to your monitor.
  <https://proceedings.mlr.press/v9/glorot10a.html>
- **Xavier Glorot, Antoine Bordes, Yoshua Bengio — "Deep Sparse Rectifier Neural Networks"** (AISTATS, 2011). The paper that made ReLU the default. Nine pages.
  <https://proceedings.mlr.press/v15/glorot11a.html>
- **Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun — "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"** (ICCV, 2015). The paper that introduced He initialization (`std = sqrt(2 / n_in)` for ReLU networks). Eleven pages.
  <https://arxiv.org/abs/1502.01852>
- **Sergey Ioffe, Christian Szegedy — "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"** (ICML, 2015). The paper that introduced batch normalization. Nine pages.
  <https://arxiv.org/abs/1502.03167>
- **Nitish Srivastava et al. — "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"** (JMLR, 2014). The dropout paper. Thirty pages; the figures on the MNIST test-error reduction are the headline.
  <https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf>
- **Diederik Kingma, Jimmy Ba — "Adam: A Method for Stochastic Optimization"** (ICLR, 2015). The Adam paper. Fifteen pages. We do not use Adam this week, but it is the optimizer you will reach for in Week 8.
  <https://arxiv.org/abs/1412.6980>
- **Léon Bottou — "Stochastic Gradient Descent Tricks"** (in *Neural Networks: Tricks of the Trade*, Springer 2012). The single best practical reference on SGD; covers learning rates, batch sizes, and convergence diagnostics. Twenty pages.
  <https://leon.bottou.org/publications/pdf/tricks-2012.pdf>

## The math you need this week

A bit more than last week, but still bounded:

- **Matrix multiplication** in `R^p`. `(A @ B)[i, j] = sum_k A[i, k] · B[k, j]`. Shapes: `(m, k) @ (k, n) = (m, n)`. Week 1.
- **The chain rule.** For composed functions `h(x) = f(g(x))`, `h'(x) = f'(g(x)) · g'(x)`. In multiple dimensions with `f: R^n → R^m` and `g: R^p → R^n`, the Jacobian of the composition is the matrix product of the Jacobians: `J_h = J_f · J_g`. We re-derive this in Lecture 2.
- **The gradient** of a scalar-valued function `f: R^p → R`. `∇f(x) ∈ R^p` is the vector of partial derivatives. The gradient *is* the direction of steepest ascent at `x`; gradient descent moves in the opposite direction.
- **The Jacobian** of a vector-valued function `f: R^n → R^m`. `J_f(x) ∈ R^{m × n}` is the matrix whose `(i, j)` entry is `∂f_i / ∂x_j`. The backward pass is a sequence of Jacobian-times-vector products.
- **Softmax derivative.** Given `p = softmax(z)`, the Jacobian `∂p_i / ∂z_j = p_i (δ_{ij} - p_j)`. We almost never use this in raw form; we use the combined softmax + cross-entropy gradient `∂L/∂z = p - y`, which is one line.
- **Cross-entropy.** `H(y, p) = -sum_i y_i log p_i`. For one-hot `y`, this reduces to `-log p_{true class}`, the negative log-likelihood of the correct class. The cleanest loss for classification; the partner of softmax.
- **The Hadamard (element-wise) product** `A ⊙ B`. In NumPy this is just `A * B`. The backward pass through a ReLU uses the Hadamard product: `dZ1 = dA1 ⊙ 1[Z1 > 0]`.
- **No measure theory, no convex optimization, no functional analysis.** All the math you need fits on two pages.

The "real" math is in Goodfellow chapters 2-5 (linear algebra, probability, numerical computation, machine-learning basics). Read them if you want a foundation; skip them if you already have it.

## Tools you will use this week

- **`numpy`** ≥ 2.0: `pip install "numpy>=2,<3"`. Every line of the MLP is NumPy.
- **`scikit-learn`** ≥ 1.5: `pip install "scikit-learn>=1.5,<2"`. Only for `fetch_openml` to download MNIST and `train_test_split`.
- **`matplotlib`** ≥ 3.8: `pip install "matplotlib>=3.8,<4"`. For training curves, weight histograms, and the sample-prediction grid.
- **`pytest`** ≥ 8: `pip install "pytest>=8"`. For the exercise smoke tests.

A `requirements.txt` snippet for the week:

```text
numpy>=2,<3
scipy>=1.11,<2
scikit-learn>=1.5,<2
matplotlib>=3.8,<4
pytest>=8
```

(We pin `scipy` because `scikit-learn` requires it; we do not import scipy directly except in one optional code path.)

### Installation gotchas

- **MNIST download.** `fetch_openml("mnist_784", version=1, as_frame=False)` downloads ~17 MB (compressed) and caches it under `~/scikit_learn_data/openml/`. The first call takes 30-60 seconds on a typical home connection; subsequent calls are instant. If `openml.org` is unreachable, the alternative is to download from `yann.lecun.com/exdb/mnist` directly and parse the IDX files (Nielsen's Chapter 1 code includes an `mnist_loader.py` that does this); we provide a fallback in Exercise 3's helpers.
- **Numerical-overflow warnings.** A naive `softmax` will overflow on logits with magnitude > 700 (the limit of `float64`). The fix is the "subtract the max" trick (`z - z.max(axis=1, keepdims=True)`), covered in Lecture 1 Section 6. If you see `RuntimeWarning: overflow in exp`, this is the bug.
- **`np.random.default_rng` vs `np.random.seed`.** Use the new `Generator` API. `np.random.seed` is global state; `default_rng(seed)` returns a local generator. The exercises and lectures use the latter consistently.

## Videos (free, no signup)

- **3Blue1Brown's "Neural Networks" series** (free on YouTube; four episodes; ~60 minutes total). The single best visual introduction. Watch all four before Tuesday.
- **Andrej Karpathy's "Neural Networks: Zero to Hero" series** (free on YouTube; eight episodes). Episodes 1 and 2 are this week's material. Episode 1 builds `micrograd`; Episode 2 builds a character-level MLP. Each episode is ~2 hours; the slow pace is deliberate.
- **CS229 Stanford Lectures (Andrew Ng)** (free on YouTube; the 2018 series is the most recent freely available). Lectures 11 and 12 cover MLPs and backprop. Two hours total.
- **Yann LeCun — "Deep Learning Lectures" (NYU; free on YouTube)**. LeCun's own course. The 2020 series is the definitive academic introduction; the first three lectures cover the MLP and SGD.
- **fast.ai "Practical Deep Learning for Coders"** (free; the first two lectures introduce PyTorch and MLPs at a higher level than this week's curriculum). Save for Week 8.

## Open-source projects to read (in this order)

You can learn more from one hour reading the source of `micrograd` than from three hours of tutorials.

1. **Andrej Karpathy's `micrograd`** — a scalar-valued reverse-mode autograd engine in 100 lines. The single cleanest implementation of backprop in any language. Read it twice:
   <https://github.com/karpathy/micrograd>
2. **Nielsen's `network.py` from chapter 1** — the original from-scratch MLP, in 74 lines of Python 2 (the version on GitHub is updated for Python 3). Read it after Exercise 1:
   <https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py>
3. **Nielsen's `network2.py` from chapters 3 and 4** — adds cross-entropy loss, L2 regularization, and better initialization. The differences between `network.py` and `network2.py` are the C5 Week 7 reading list as code:
   <https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py>
4. **Karpathy's `makemore`** — a character-level MLP built in PyTorch but with the gradient flow done by hand for educational reasons:
   <https://github.com/karpathy/makemore>
5. **scikit-learn `MLPClassifier`** — the sklearn MLP. Useful as a baseline; the actual training code is in `_multilayer_perceptron.py`:
   <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neural_network/_multilayer_perceptron.py>

## Cheat sheets (one-page references)

- **CS231n cheat sheet on backprop** (Karpathy's hand-drawn whiteboard summary):
  <https://cs231n.github.io/optimization-2/#patterns>
- **Stanford CS229 supervised-learning cheat sheet** (Afshine and Shervine Amidi):
  <https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning>
- **The "neural network zoo"** (Fjodor van Veen; a visual map of every architecture; cite it; do not memorize it):
  <https://www.asimovinstitute.org/neural-network-zoo/>

## Glossary cheat sheet

Keep this open in a tab while you work.

| Term | Plain English |
|------|---------------|
| **Neuron / unit** | A single linear function `z = w · x + b` followed by an activation `a = σ(z)`. The atomic building block of a neural network. |
| **Layer** | A vector of neurons computed in parallel from the same input. A "fully connected" or "dense" layer has weights connecting every input to every output. |
| **MLP (multi-layer perceptron)** | A feed-forward network with one or more hidden layers between input and output. The "perceptron" in the name is historical; modern MLPs use sigmoid or ReLU activations, not the step function of Rosenblatt 1958. |
| **Activation function** | A nonlinear function applied element-wise to a layer's pre-activation: sigmoid, tanh, ReLU, Leaky ReLU, ELU, GELU. Without nonlinearity, an MLP is a single linear function regardless of depth. |
| **ReLU** | Rectified Linear Unit: `f(z) = max(0, z)`. Derivative is `1` for `z > 0` and `0` for `z < 0`. The 2010 default. |
| **Softmax** | A function that maps a vector of logits `z ∈ R^k` to a probability distribution `p ∈ Δ^{k-1}`: `p_i = exp(z_i) / sum_j exp(z_j)`. The output layer for multi-class classification. |
| **Cross-entropy loss** | `L(y, p) = -sum_i y_i log p_i` for one-hot `y`. The standard loss for classification. Equivalent to negative log-likelihood. |
| **Forward pass** | Compute the output of the network given the input. For a 2-layer MLP: `Z1 = X W1 + b1; A1 = ReLU(Z1); Z2 = A1 W2 + b2; A2 = softmax(Z2)`. |
| **Backward pass / backprop** | Compute the gradient of the loss with respect to every parameter, by applying the chain rule layer-by-layer from output to input. |
| **Chain rule** | The rule that says `(f ∘ g)'(x) = f'(g(x)) · g'(x)`. The mathematical engine of backprop. |
| **Gradient descent** | The iterative algorithm `θ ← θ - η · ∇L(θ)` that minimizes a loss by moving in the direction of steepest descent. |
| **SGD (stochastic GD)** | Gradient descent where the gradient is estimated on a single example or a mini-batch instead of the full training set. Noisier but cheaper per step; modern default. |
| **Mini-batch** | A small random subset of the training data (typically 32, 64, 128, or 256 examples) used to estimate the gradient at each SGD step. |
| **Epoch** | One full pass through the training data. With `n = 60_000` MNIST examples and `batch_size = 128`, one epoch is `60_000 / 128 ≈ 469` mini-batch steps. |
| **Learning rate** | The scalar `η` in `θ ← θ - η · g`. Too high: loss diverges. Too low: loss decreases too slowly. For a 2-layer MNIST MLP, `η ≈ 0.1` works. |
| **Initialization** | The choice of initial values for `W1, W2, b1, b2`. Zero initialization fails (symmetry breaking); random `N(0, 0.01)` works for shallow nets; He initialization (`std = sqrt(2 / n_in)`) is the modern default for ReLU networks. |
| **Vanishing gradients** | The problem that arises when activation derivatives are < 1 and gradients become a product of many such terms, shrinking to zero in deep networks. The reason sigmoid lost to ReLU. |
| **Exploding gradients** | The opposite problem: gradients grow without bound. Common in deep networks without careful initialization or in recurrent networks; fixed by gradient clipping. |
| **Dead ReLU** | A ReLU unit whose pre-activation is always negative. Its output is always zero and its gradient is always zero; the unit never learns. Cause: bad initialization or too-large a learning rate. |
| **Dropout** | A regularization technique that randomly zeroes a fraction of activations during training. Reduces overfitting; equivalent in expectation to training an ensemble of subnetworks. Introduced 2014. |
| **Batch normalization** | A layer that normalizes each feature's mean and variance over the mini-batch, then re-scales by learned parameters. Stabilizes training; speeds up convergence; introduced 2015. |
| **One-hot encoding** | A categorical target `y ∈ {0, ..., k-1}` represented as a vector `Y ∈ R^k` with `Y[i] = 1` if `y = i` and `0` otherwise. The natural input to the cross-entropy loss. |
| **Logits** | The pre-activation outputs of the final layer, before the softmax. Often the convenient quantity to compute gradients with respect to because of the softmax + CE simplification. |
| **Gradient check** | Verify the analytical gradient against a numerical (finite-difference) gradient. The single most important debugging tool for from-scratch backprop. If your gradient check fails, your derivation is wrong; do not train until it passes. |
| **He initialization** | `W ~ N(0, sqrt(2 / n_in))`. The variance scaling that preserves activations in ReLU networks. He et al. 2015. The default for this week. |
| **Xavier initialization** | `W ~ N(0, sqrt(1 / n_in))` or `~ U(-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out)))`. The variance scaling for sigmoid/tanh networks. Glorot and Bengio 2010. |
| **MNIST** | A dataset of 70,000 28×28 grayscale images of handwritten digits, ten classes, 60k train / 10k test. The hello-world of supervised image classification; introduced by LeCun et al. 1998. |
| **Universal approximation theorem** | Any continuous function on a compact set can be approximated to arbitrary precision by an MLP with one hidden layer and sufficient width. Cybenko 1989, Hornik 1991. The theorem says deep nets *can* fit; it does not say they will, or that they will generalize. |

---

*If a link 404s, please open an issue so we can replace it.*
