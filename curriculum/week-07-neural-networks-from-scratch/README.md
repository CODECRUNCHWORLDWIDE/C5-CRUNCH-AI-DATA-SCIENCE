# Week 7 — Neural Networks from Scratch

> *Six weeks of supervised and unsupervised learning have taught you to call `model.fit(X, y)` on a library that has already worked out the math. This week we open the box. You will derive backpropagation from the chain rule, implement the forward pass and the backward pass in pure NumPy in under 200 lines, train a two-layer multi-layer perceptron on MNIST, and reach test accuracy in the mid-90s. No PyTorch. No TensorFlow. No JAX. By Sunday you will know, line by line, what every deep-learning framework is doing on your behalf — and that knowledge does not expire.*

Welcome to week seven of **C5 · Crunch AI / Data Science**. Weeks 1 through 6 stayed inside the world of tabular data, libraries, and human-interpretable models. The Week 4 Ridge regressor was a closed-form solve; the Week 5 boosted-trees model was a thousand decision stumps in a `for` loop; the Week 6 k-means was forty lines of NumPy. Every model so far has been a stack of operations you can defend on a whiteboard in five minutes.

This week the models are still defensible on a whiteboard, but the whiteboard fills up. A two-layer MLP for MNIST has roughly 100,000 parameters and two non-linearities; the loss surface is non-convex; the optimizer is stochastic gradient descent; the gradients flow backwards through the network by the chain rule. None of those pieces are new mathematics — backpropagation is a hundred lines of multivariable calculus — but the *combination* is the moment supervised learning changes shape. From this week forward, "fitting a model" means "running a gradient-descent loop that updates a parameter vector by the gradient of a loss function."

Three commitments before we start, because Week 7 is also where students reach reflexively for PyTorch and miss the point of the exercise:

1. **You will write the backward pass with your hands.** Not a `loss.backward()` call. Not an `autograd` wrapper. You will derive `∂L/∂W` for a two-layer ReLU MLP from the chain rule, compare it to a finite-difference numerical gradient, and verify they agree to six decimal places. The point is not that you will keep doing this in production — you will not — the point is that when a future PyTorch model fails to converge, you will have a mental model of what is happening inside `backward()` and where it can go wrong (vanishing gradients, ReLU dead neurons, dtype mismatches, shape bugs). That mental model is the asset.
2. **You will train on MNIST, not on synthetic data.** MNIST is the hello-world of deep learning for a reason: 60,000 training images of handwritten digits, 10,000 test images, 28×28 grayscale, ten classes. It is small enough to fit in memory, easy enough that a two-layer MLP gets ≥95% test accuracy without exotic tricks, and famous enough that you can compare your number against the LeCun 1998 baseline (LeNet-1: 95.0% on the 28×28 MLP variant; modern MLPs reach 98% with care). When the literature says "MNIST is solved," the literature is partly right (the *task* is solved) and partly wrong (the *exercise* of solving it from scratch teaches more than any tutorial).
3. **No PyTorch this week. No TensorFlow. No JAX. No `keras`. No pre-trained anything.** You are allowed `numpy`, `scipy` (for `scipy.special.expit` if you want a numerically stable sigmoid, though you will write your own), `matplotlib` for plots, and `pytest` for the exercises. We import `sklearn.datasets.fetch_openml` once to download the MNIST images, but every line of the model is yours. Deep-learning frameworks return in Week 8.

We target **numpy 2.x**, **scikit-learn 1.5+** (for the MNIST download via `fetch_openml`), and **matplotlib 3.8+**. Python 3.11 or higher.

---

## Learning objectives

By the end of this week, you will be able to:

- **State** the forward pass for a two-layer MLP with ReLU activations and a softmax output: `Z1 = X W1 + b1`, `A1 = ReLU(Z1)`, `Z2 = A1 W2 + b2`, `A2 = softmax(Z2)`. Know which matrices have which shapes for an input of `(batch_size, 784)` going to ten classes.
- **Derive** the gradient of the cross-entropy loss with respect to every parameter (`W1`, `b1`, `W2`, `b2`) from the chain rule. State the famous "softmax + cross-entropy" simplification: `∂L/∂Z2 = A2 - Y` (the predicted probabilities minus the one-hot targets). Use it to skip a layer of algebra.
- **Implement** the forward pass and the backward pass in pure NumPy, in under 200 lines of code. Verify the analytical gradient against a finite-difference numerical gradient to within `1e-6`.
- **Recognize** the four implementation bugs that account for >90% of failed MLPs from scratch: shape mismatches between `W` and `X`, the off-by-one in the batch dimension during averaging, the wrong sign on the gradient update, and the un-stable softmax (which overflows without the `max` subtraction trick).
- **Train** the MLP on MNIST with mini-batch stochastic gradient descent. Reach **≥95% test accuracy** in 10 to 20 epochs at a learning rate of `0.1` with batch size `128`. Plot the training-loss and test-accuracy curves and read them honestly: a smooth descending loss and a plateauing accuracy are both signals; a thrashing loss is a learning-rate bug.
- **Explain** the vanishing-gradient problem in one paragraph: as a sigmoid activation saturates, its derivative `σ(z)(1−σ(z))` approaches zero, and the gradient `∂L/∂W1` becomes vanishingly small for deep networks. Know why ReLU fixed this in 2010 (the derivative is `1` on the positive side and `0` on the negative — no saturation in the forward direction). Know the cost: dead ReLU neurons, which is why Leaky ReLU and ELU exist.
- **Distinguish** between batch gradient descent (the full training set per step; slow, smooth), mini-batch SGD (a random subset per step; fast, noisy, modern default), and pure SGD (one example per step; very fast, very noisy, the historical default).
- **Identify** when a from-scratch implementation is the right move (educational, prototyping a novel layer, deeply-embedded systems) and when a framework is the right move (production training, anything beyond two layers, anything that needs GPU acceleration).
- **Ship** a mini-project: build a two-layer NumPy MLP that classifies a non-MNIST tabular dataset (the Wisconsin Breast Cancer Dataset, the Adult Census income dataset, or the Iris dataset for the gentle path), with the same forward/backward/SGD machinery you wrote on MNIST. Push the trained model and the training-curve plot to your portfolio repo.
- **Pass** every `pytest` case on the Week 7 exercises.

---

## Prerequisites

- **Weeks 1, 2, 3, 4, 5, and 6 complete.** In particular, you have the Week 6 segmentation notebook checked into your `crunch-ai-portfolio-<yourhandle>` repo with a defended `k`, a three-panel 2D embedding, and an honest paragraph. That is the signal you can ship an unsupervised analysis end-to-end. Week 7 is the first deep-learning artifact.
- A working **Python 3.11+** install (we use 3.12 in CI; 3.11 works identically).
- **numpy** 2.x (from Week 1). `pip install "numpy>=2,<3"`. Every line of the MLP runs on NumPy alone.
- **scikit-learn** 1.5+ (from Week 4) for `sklearn.datasets.fetch_openml("mnist_784")` and `sklearn.model_selection.train_test_split`. We do not use any sklearn model this week, only its data loader.
- **matplotlib** 3.8+ (from Week 3) for the loss / accuracy curves, the weight histograms, and the sample-prediction grid.
- **pytest** 8+ for the exercise tests.
- About **300 MB of disk space** to cache the MNIST dataset (`~/scikit_learn_data/openml/`). `fetch_openml` downloads it once and caches it; subsequent runs are instant.

No GPU required. A two-layer MLP on MNIST trains in 2-4 minutes per epoch on a laptop CPU. Twenty epochs is well under an hour. We are deliberately at the upper edge of "CPU-trainable" this week; Weeks 8 and 9 will introduce JAX and PyTorch, at which point GPU becomes relevant.

You should be comfortable with:

- **Matrix multiplication.** `A @ B` in NumPy. Shapes: `(m, k) @ (k, n) = (m, n)`. If you have not seen `@` since Week 1, re-read Week 1 Lecture 2.
- **Broadcasting.** Adding a `(n,)` bias to a `(batch, n)` activation works without an explicit `np.tile`. Week 1 Lecture 1 if you need a refresher.
- **The chain rule** from single-variable calculus: `(f ∘ g)'(x) = f'(g(x)) · g'(x)`. The multivariable version is the same, with Jacobians replacing derivatives. We re-derive it in Lecture 2.
- **Mini-batch sampling.** `np.random.default_rng(seed).choice(n, size=batch_size, replace=False)`. The `replace=False` matters; sampling with replacement gives noisier gradients.

---

## Topics covered

- **The single neuron.** The perceptron from Rosenblatt 1958: `y = sign(w · x + b)`. The linear regression neuron: `y = w · x + b` (no activation). The logistic neuron: `y = σ(w · x + b)`. Three neurons; three lessons. The point is that "a neural network" is a composition of these simple units arranged in layers — there is no new building block, only a new pattern of composition.
- **The two-layer MLP.** Input layer `(batch, 784)`, hidden layer `(batch, h)` with ReLU activation, output layer `(batch, 10)` with softmax. Two weight matrices (`W1` shape `(784, h)` and `W2` shape `(h, 10)`), two bias vectors. The "two" in "two-layer" is the convention of counting weight matrices, not counting input + output.
- **Activation functions.** Sigmoid (`σ(z) = 1 / (1 + e^{-z})`), the historical default; tanh (`tanh(z)`), the 1990s upgrade; ReLU (`max(0, z)`), the 2010 revolution; Leaky ReLU and ELU, the modern refinements. Each has a derivative that matters; the derivative is the part you write into the backward pass.
- **Softmax and cross-entropy.** Softmax maps logits in `R^k` to a probability distribution: `p_i = exp(z_i) / sum_j exp(z_j)`. Cross-entropy measures the disagreement between two distributions: `L = -sum_i y_i log(p_i)`. The composition `softmax + cross-entropy` has a uniquely clean derivative: `∂L/∂z = p - y`. Memorize this; it is the most-used identity in deep learning.
- **The forward pass.** `Z1 = X W1 + b1; A1 = ReLU(Z1); Z2 = A1 W2 + b2; A2 = softmax(Z2); L = cross_entropy(A2, Y)`. Five lines of NumPy. Each line is one matrix operation. The cache of intermediate values (`Z1, A1, Z2, A2`) is what the backward pass will reuse.
- **The chain rule.** `dL/dW1 = (dL/dZ2) · (dZ2/dA1) · (dA1/dZ1) · (dZ1/dW1)`. Each term is a Jacobian; the product is the gradient. In matrix form, every term becomes a matrix multiplication or an element-wise mask. The whole derivation is one page of algebra.
- **Backpropagation.** Apply the chain rule recursively, layer by layer, from the loss back to the parameters. The "back" in backprop is the order of computation: you compute the gradient at the output, then propagate it backward to the hidden layer, then to the input weights. The total work is roughly equal to the forward pass — backprop is "cheap" relative to numerical differentiation, which would cost `O(p)` forward passes.
- **Gradient checking.** The single most-useful debugging technique for from-scratch backprop. Pick a parameter `θ_i`; compute the analytical gradient `g_i = ∂L/∂θ_i`; compute the numerical gradient `g̃_i = (L(θ + ε·e_i) - L(θ - ε·e_i)) / (2ε)`; verify `|g_i - g̃_i| / max(|g_i|, |g̃_i|) < 1e-6`. If this check fails, your analytical derivation has a bug; do not start training until it passes.
- **Stochastic gradient descent (SGD).** Update rule: `θ ← θ - η · g` where `η` is the learning rate and `g` is the gradient estimated on a mini-batch. Mini-batch SGD is the workhorse: a batch of 128 examples gives a reasonable gradient estimate at a fraction of the cost of full-batch GD, and the noise in the gradient helps escape shallow local minima.
- **Learning-rate selection.** Too high: the loss thrashes. Too low: the loss decreases but never converges. The right rate for a two-layer MNIST MLP is in `[0.01, 0.5]`; `0.1` is the C5 default. We do not use Adam this week — Adam belongs in Week 8 — but we acknowledge that Adam exists and that it does most of what learning-rate scheduling does, automatically.
- **Initialization.** Zero initialization breaks the network (every neuron computes the same gradient, the network never learns). Random initialization from `N(0, 0.01)` works for shallow networks. Xavier initialization (variance scaled to `1/n_in`) and He initialization (variance scaled to `2/n_in`, for ReLU networks) are the modern defaults. We use He initialization for the MLP this week.
- **Vanishing and exploding gradients.** Sigmoid networks suffer from vanishing gradients past two or three layers because `σ'(z) ≤ 0.25` and the gradient is a product of these terms. ReLU networks suffer instead from "dead neurons" (a ReLU unit whose pre-activation is always negative outputs zero forever) and exploding gradients without careful initialization. Both are why frameworks default to ReLU + He initialization.
- **When to stop using NumPy.** Anything beyond two or three layers; anything that benefits from autodiff (you would re-derive the backward pass for every architecture change); anything that needs GPU; anything in production. Week 8 introduces PyTorch and JAX. Week 7 is the "I understand what they are doing" week, not the "I will rewrite PyTorch" week.

---

## Weekly schedule

Target about **38 hours**. The schedule is tight; the second half of the week is one long training run with twenty distinct debugging moments.

| Day       | Focus                                                       | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|-------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Forward pass; activation functions; softmax + CE            |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    7h       |
| Tuesday   | Backprop from the chain rule; gradient check               |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    7h       |
| Wednesday | SGD; learning rate; initialization; full MLP                |   2.5h   |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0h       |    6h       |
| Thursday  | MNIST training; debug the curve; reach 95%                  |   1.5h   |   1h      |     2h     |   0.5h    |   1h     |     2h       |   0h       |    8h       |
| Friday    | Mini-project: NumPy MLP on a non-MNIST dataset              |   0h     |   0.5h    |     0h     |   0.5h    |   1h     |     3h       |   0.5h     |   5.5h      |
| Saturday  | Mini-project: training curves + write-up                    |   0h     |   0h      |     0h     |   0h      |   1h     |     3h       |   0h       |    4h       |
| Sunday    | Quiz, review, push to portfolio repo                        |   0h     |   0h      |     0h     |   0.5h    |   0h     |     0h       |   0h       |   0.5h      |
| **Total** |                                                             | **10h**  | **7.5h**  | **2h**     | **3h**    | **6h**   | **8h**       | **1h**     |  **37.5h**  |

The schedule is tight on Thursday. If your MNIST run is not converging by 8 PM on Wednesday, do not push through — re-read Lecture 2 and re-run the gradient check. A failed gradient check on Tuesday is recoverable; a failed gradient check on Thursday is two days of training a network with the wrong gradient.

---

## How to navigate this week

| File | What is inside |
|------|----------------|
| [README.md](./README.md) | This overview |
| [resources.md](./resources.md) | Free books (Nielsen, Goodfellow et al.), the original papers (Rumelhart 1986, LeCun 1998, He 2015), the courses (Ng CS229, Karpathy nn-zero-to-hero, 3Blue1Brown) |
| [lecture-notes/01-forward-pass-and-loss.md](./lecture-notes/01-forward-pass-and-loss.md) | The single neuron; the MLP forward pass; activations; softmax + cross-entropy; the numerical-stability trick |
| [lecture-notes/02-backprop-from-chain-rule.md](./lecture-notes/02-backprop-from-chain-rule.md) | The chain rule, written out for a two-layer MLP; the analytical gradient for every parameter; gradient checking; the four canonical bugs |
| [lecture-notes/03-training-loop-and-mnist.md](./lecture-notes/03-training-loop-and-mnist.md) | Mini-batch SGD; learning rate; initialization; the full MNIST training loop; reading the training curves |
| [exercises/exercise-01-numpy-mlp-skeleton.py](./exercises/exercise-01-numpy-mlp-skeleton.py) | A runnable scaffold: parameter init, forward pass, prediction. No backward pass yet. |
| [exercises/exercise-02-implement-relu-softmax-ce.py](./exercises/exercise-02-implement-relu-softmax-ce.py) | Fill in `relu`, `relu_grad`, `softmax`, `cross_entropy_loss`, with numerical-stability tests |
| [exercises/exercise-03-train-on-mnist-subset.py](./exercises/exercise-03-train-on-mnist-subset.py) | Full training loop on a 10,000-row MNIST subset; reach ≥90% test accuracy in 10 epochs |
| [exercises/SOLUTIONS.md](./exercises/SOLUTIONS.md) | Reference solutions with commentary; read only after attempting |
| [challenges/challenge-01-implement-dropout-and-batch-norm.md](./challenges/challenge-01-implement-dropout-and-batch-norm.md) | Add dropout and batch normalization to the two-layer MLP, in NumPy, and measure the effect on test accuracy |
| [challenges/challenge-02-write-your-own-autograd.md](./challenges/challenge-02-write-your-own-autograd.md) | Implement a tiny reverse-mode automatic-differentiation engine over scalar values; reproduce Karpathy's `micrograd` |
| [quiz.md](./quiz.md) | 10 multiple-choice questions on the chain rule, vanishing gradients, vectorization, and the softmax + CE identity |
| [homework.md](./homework.md) | Six practice problems for the week |
| [mini-project/README.md](./mini-project/README.md) | Build a NumPy MLP classifier on a non-MNIST tabular dataset; full training-curve report |
| [mini-project/starter.py](./mini-project/starter.py) | Runnable starter for the mini-project: data loader, model skeleton, training-loop scaffold |

---

## Stretch goals

- Read **Rumelhart, Hinton, Williams — "Learning representations by back-propagating errors"** (Nature, 1986). The paper that introduced backprop to the modern era. Three pages. Read it for the historical context, not the implementation; the modern derivation is cleaner.
- Read **LeCun, Bottou, Bengio, Haffner — "Gradient-based learning applied to document recognition"** (Proc. IEEE, 1998). The paper that introduced LeNet on MNIST. The MLP results are in Table II; you are aiming for the line in the table that says "MLP, 300 hidden, 10 output: 4.7% error" — that is your target.
- Watch **Andrej Karpathy's "Neural Networks: Zero to Hero" series**, episodes 1 and 2. Episode 1 builds `micrograd` (a scalar-valued autograd library); Episode 2 builds a character-level MLP from scratch. Each episode is two hours and unusually patient with the math. Strongly recommended in addition to the lectures.
- Read **3Blue1Brown's "Neural Networks" series** (four videos, ~15 minutes each). The first video introduces the MLP visually; the second and third derive backprop with the cleanest animations in print or video; the fourth visualizes gradient descent. Watch all four before Tuesday.
- Read **Goodfellow, Bengio, Courville — *Deep Learning*** (free online), chapters 6 (deep feedforward networks) and 8 (optimization for training). The MLP and gradient-descent treatment is the densest correct introduction in any textbook; the chapter on optimization is your reference for the next two weeks.

---

## What you will *not* do this week

You will not:

- **Use PyTorch, TensorFlow, JAX, Keras, or any deep-learning framework.** They return in Week 8. The point of Week 7 is to know what `loss.backward()` is doing under the hood; the point of Week 8 is to use the framework without having to remember.
- **Train a convolutional network.** CNNs are Week 9. A vanilla MLP on the flattened 784-pixel MNIST images can reach 98% test accuracy with care; that is good enough for Week 7.
- **Use Adam, RMSProp, or learning-rate schedulers.** Plain mini-batch SGD with a fixed learning rate is the algorithm this week. The fancier optimizers are Week 8.
- **Run on GPU.** A laptop CPU is enough. If you have a GPU and want to use it, save it for Weeks 8 and 9.
- **Use a pre-trained model, transfer learning, or any data beyond MNIST.** The mini-project uses a small tabular dataset; the exercises use MNIST. That is all.
- **Skip the gradient check.** This is the one rule that is non-negotiable. Every from-scratch backprop in the history of deep learning has had a bug at some point; the gradient check is the only systematic way to find it. If your gradient check fails, do not train; debug the derivation.

That is deliberate. The point of Week 7 is not to ship a competitive MNIST classifier — the state-of-the-art on MNIST is 99.8% and the gap between your 96% and SOTA's 99.8% is irrelevant to your career. The point is to *know what the framework will do for you* when you reach for PyTorch on Monday of Week 8.

---

## A note on the EXPERIMENT cards

Lectures continue to use `EXPERIMENT` callouts:

> **EXPERIMENT — short title.** A one-paragraph drill you can run in a Python REPL in five to fifteen minutes. The drill makes a specific claim from the lecture concrete. Run them. Do not skip them.

The cards are not graded and they are not optional. The Lecture 2 experiment on gradient checking is the experiment that will save you on Thursday — the one analytical gradient that disagrees with the finite-difference check by a factor of 10 is the bug that would otherwise cost you a day of training a network that cannot learn. The `EXPERIMENT` cards exist for exactly this kind of moment.

---

## Up next

[Week 8 — Deep Learning with PyTorch and JAX](../week-08/) — once you have pushed your Week 7 MLP-on-MNIST notebook and the mini-project to your portfolio repo with ≥95% test accuracy and an honest training-curve plot. Week 8 will retrain the same architecture in PyTorch in fifteen lines (the framework hides the backward pass you wrote by hand this week) and introduce convolutional layers, the Adam optimizer, and learning-rate scheduling. The discipline you built this week — verify the gradient, train on a baseline, plot the loss, write the honest paragraph — carries over verbatim.
