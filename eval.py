from pathlib import Path

import torch
from torch import nn

from dataset.tiny_imagenet import get_default_transform, get_tiny_imagenet_dataloaders
from models.custom_net import CustomNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_default_transform(image_size=224)
    train_dataset, val_dataset, train_loader, val_loader = get_tiny_imagenet_dataloaders(
        data_root="data",
        batch_size=32,
        num_workers=0,
        transform=transform,
        download=False,
    )
    _ = train_dataset
    _ = train_loader

    model = CustomNet(num_classes=200).to(device)
    checkpoint_path = Path("checkpoints") / "best_custom_net.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = 100.0 * correct / total
    print(f"Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%")
    print(f"Validation samples: {len(val_dataset)}")


if __name__ == "__main__":
    main()
