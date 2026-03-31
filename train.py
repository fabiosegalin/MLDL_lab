from pathlib import Path

import torch
import wandb
from torch import nn

from dataset.tiny_imagenet import get_default_transform, get_tiny_imagenet_dataloaders, plot_class_examples
from models.custom_net import CustomNet


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100.0 * correct / total
    print(f"Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%")
    return train_loss, train_accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
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
    return val_loss, val_accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    momentum = 0.9
    image_size = 224

    wandb.init(
        project="mldl-lab3",
        config={
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "image_size": image_size,
            "optimizer": "SGD",
            "model": "CustomNet",
        },
    )

    transform = get_default_transform(image_size=224)
    train_dataset, val_dataset, train_loader, val_loader = get_tiny_imagenet_dataloaders(
        data_root="data",
        batch_size=batch_size,
        num_workers=0,
        transform=transform,
        download=True,
    )

    print(f"Using device: {device}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    plot_class_examples(train_loader, class_names=train_dataset.classes)

    model = CustomNet(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    best_acc = 0.0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_custom_net.pt"

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")
            wandb.log({"best_val_accuracy": best_acc})

    print(f"Best validation accuracy: {best_acc:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    main()
