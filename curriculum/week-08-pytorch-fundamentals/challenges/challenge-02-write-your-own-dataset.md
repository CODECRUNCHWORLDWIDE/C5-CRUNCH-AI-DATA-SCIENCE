# Challenge 2 — Write Your Own `Dataset`

**Time estimate:** 2 hours.

## Problem statement

Implement a `Dataset` subclass that reads images from a folder on disk, applies on-the-fly transforms, and integrates cleanly with `DataLoader`. The point is to internalize the `Dataset` API by writing one — the same API you would use for any custom data source (text, audio, time series, multimodal). After this challenge, you will never again be surprised by "how do I get my data into PyTorch."

The deliverable is a small image classifier on the **Hymenoptera dataset** (a 240-image subset of ants vs. bees, used in the official PyTorch transfer-learning tutorial). The dataset is small enough to download in seconds, big enough to be non-trivial, and laid out in the canonical "class-named subfolder per class" structure that you will see in every real-world image dataset.

PyTorch references for this challenge:

- `torch.utils.data.Dataset`: <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>
- `torchvision.transforms.v2`: <https://pytorch.org/vision/stable/transforms.html>
- `torchvision.datasets.ImageFolder` (the canonical reference implementation; your job is to *not* use it and write your own):
  <https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html>
- Hymenoptera dataset (download URL in the PyTorch transfer-learning tutorial; ~45 MB):
  <https://download.pytorch.org/tutorial/hymenoptera_data.zip>

## What you will produce

A single script `image_folder_dataset.py` and a one-page markdown writeup `notes.md` that:

1. Downloads and unzips the Hymenoptera dataset to `./data/hymenoptera_data/` (or assumes the dataset is already present; either is fine). The dataset layout is:

   ```text
   data/hymenoptera_data/
   ├── train/
   │   ├── ants/
   │   │   ├── 0013035.jpg
   │   │   ├── ...
   │   └── bees/
   │       ├── 16838648_415acd9e3f.jpg
   │       ├── ...
   └── val/
       ├── ants/
       └── bees/
   ```

2. Implements `class ImageFolderDataset(Dataset)` with:
   - `__init__(self, root, transform=None)` that walks the directory tree and records `(image_path, class_index)` tuples.
   - `__len__` returning the number of images.
   - `__getitem__(idx)` loading the image (use `PIL.Image.open(path).convert("RGB")`), applying `self.transform` if provided, and returning `(image_tensor, class_index_as_int)`.
   - A `classes` attribute: a list of class names in alphabetical order (`["ants", "bees"]`).
   - A `class_to_idx` attribute: a dict mapping class name to integer index.

3. Wraps the dataset in a `DataLoader` with `batch_size=16`, `shuffle=True` for train, `shuffle=False` for val.

4. Defines a tiny 3-layer CNN classifier (224x224 RGB → ants/bees binary). The architecture is not the point; any sensible classifier head works. The C5 reference is:

   ```text
   Conv2d(3, 16, 3, padding=1) -> ReLU -> MaxPool(2)   # 224 -> 112
   Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool(2)  # 112 -> 56
   Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool(2)  # 56 -> 28
   Flatten -> Linear(64*28*28, 64) -> ReLU -> Linear(64, 2)
   ```

5. Trains with the eight-line loop from Lecture 2 for `n_epochs=10`, prints per-epoch train loss and val accuracy. The val accuracy should reach **≥ 75%** with this small dataset and small architecture. (Anything substantially higher requires pretrained features, which is Week 11.)

## Acceptance criteria

- [ ] `image_folder_dataset.py` runs end-to-end with `python image_folder_dataset.py`.
- [ ] `python -m py_compile image_folder_dataset.py` succeeds.
- [ ] `ImageFolderDataset` does *not* use `torchvision.datasets.ImageFolder` or any other ready-made image-folder utility. The whole point of the challenge is to write the class.
- [ ] `ImageFolderDataset.__init__` discovers the classes alphabetically and assigns indices in alphabetical order. `classes == ["ants", "bees"]`; `class_to_idx == {"ants": 0, "bees": 1}`.
- [ ] `__len__` returns the total image count (train: 244, val: 153 for the standard hymenoptera split).
- [ ] `__getitem__` returns `(tensor, int)` where the tensor has the shape produced by your transform (after `Resize(224)` and `ToTensor`, this is `(3, 224, 224)`).
- [ ] The transform pipeline uses `torchvision.transforms.v2`, not the deprecated `torchvision.transforms`.
- [ ] The final val accuracy after 10 epochs is at least **0.75**.
- [ ] `notes.md` (~200 words) answers: what was the hardest part of the `Dataset` implementation (likely the directory walk or the PIL conversion); what does `torchvision.datasets.ImageFolder` do that yours does not; whether you would use yours or torchvision's in a production project.

## Hints

<details>
<summary>The directory walk (the part that trips everyone)</summary>

```python
import os
from typing import List, Tuple

def _discover_files(root: str) -> Tuple[List[str], List[str], List[Tuple[str, int]]]:
    """Walk root/<class>/<image> and return classes, class_to_idx, samples."""
    classes = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples: List[Tuple[str, int]] = []
    for cls in classes:
        cls_dir = os.path.join(root, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(cls_dir, fname), class_to_idx[cls]))
    return classes, class_to_idx, samples
```

The `sorted` is what makes the class indexing deterministic. The lowercase extension check is a convention that matches `ImageFolder`. Hidden files (`.DS_Store` on macOS) get filtered automatically because the extension check rejects them.

</details>

<details>
<summary>The transform pipeline</summary>

```python
from torchvision.transforms import v2

train_transform = v2.Compose([
    v2.Resize(256),
    v2.RandomCrop(224),
    v2.RandomHorizontalFlip(),
    v2.ToTensor(),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet stats
])

val_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToTensor(),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
```

The ImageNet channel stats are conventional for natural images even when you are not using a pretrained model — they happen to be close to the empirical statistics of most photographic images.

</details>

<details>
<summary>The minimal Dataset class</summary>

```python
from typing import Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageFolderDataset(Dataset):
    def __init__(self, root: str, transform=None) -> None:
        self.root = root
        self.transform = transform
        self.classes, self.class_to_idx, self.samples = _discover_files(root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
```

That is the entire class. About fifteen lines. `torchvision.datasets.ImageFolder` adds a `loader` callable, a `target_transform` argument, and some validation on `is_valid_file` — useful in production, omitted here.

</details>

## Submission checklist

- [ ] `image_folder_dataset.py` (the dataset class, the model, the training loop)
- [ ] `notes.md` (the reflection)
- [ ] `python -m py_compile image_folder_dataset.py` exits cleanly
- [ ] The script downloads (or assumes downloaded) the hymenoptera dataset and trains to ≥75% val accuracy

When you can read `torchvision.datasets.ImageFolder` source and recognize every line of it, you have completed this challenge.
