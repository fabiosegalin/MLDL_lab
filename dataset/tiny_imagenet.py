from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATASET_DIRNAME = "tiny-imagenet-200"
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def download_tiny_imagenet(data_root: str = "data") -> Path:
    data_root_path = Path(data_root)
    dataset_root = data_root_path / DATASET_DIRNAME

    if dataset_root.exists():
        return dataset_root

    data_root_path.mkdir(parents=True, exist_ok=True)
    response = requests.get(TINY_IMAGENET_URL, timeout=120)
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall(data_root_path)

    return dataset_root


def get_default_transform(image_size: int = 64) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )


def get_tiny_imagenet_datasets(
    data_root: str = "data",
    transform: transforms.Compose | None = None,
    download: bool = True,
):
    if transform is None:
        transform = get_default_transform()

    dataset_root = download_tiny_imagenet(data_root) if download else Path(data_root) / DATASET_DIRNAME
    train_dataset = ImageFolder(root=dataset_root / "train", transform=transform)
    val_dataset = TinyImageNetValDataset(
        val_root=dataset_root / "val",
        class_to_idx=train_dataset.class_to_idx,
        transform=transform,
    )

    return train_dataset, val_dataset


def get_tiny_imagenet_dataloaders(
    data_root: str = "data",
    batch_size: int = 64,
    num_workers: int = 0,
    transform: transforms.Compose | None = None,
    download: bool = True,
):
    train_dataset, val_dataset = get_tiny_imagenet_datasets(
        data_root=data_root,
        transform=transform,
        download=download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose((1, 2, 0))
    mean = np.array(DEFAULT_MEAN)
    std = np.array(DEFAULT_STD)
    image = image * std + mean
    return np.clip(image, 0, 1)


class TinyImageNetValDataset(Dataset):
    def __init__(self, val_root: str | Path, class_to_idx: dict[str, int], transform=None):
        self.val_root = Path(val_root)
        self.images_root = self.val_root / "images"
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = self._load_samples()

    def _load_samples(self):
        annotations_path = self.val_root / "val_annotations.txt"
        samples = []

        with annotations_path.open("r", encoding="utf-8") as file:
            for line in file:
                image_name, class_name, *_ = line.strip().split("\t")
                image_path = self.images_root / image_name
                target = self.class_to_idx[class_name]
                samples.append((image_path, target))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def plot_class_examples(dataloader: DataLoader, class_names: list[str] | None = None, max_classes: int = 10) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    sampled_classes = set()

    for inputs, targets in dataloader:
        for image, target in zip(inputs, targets):
            class_idx = int(target.item())
            if class_idx in sampled_classes:
                continue

            axis = axes[len(sampled_classes)]
            axis.imshow(denormalize(image))
            title = class_names[class_idx] if class_names is not None else f"Class {class_idx}"
            axis.set_title(title)
            axis.axis("off")
            sampled_classes.add(class_idx)

            if len(sampled_classes) == max_classes:
                plt.tight_layout()
                plt.show()
                return

    for axis in axes[len(sampled_classes):]:
        axis.axis("off")

    plt.tight_layout()
    plt.show()
