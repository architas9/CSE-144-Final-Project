import random
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models


SEED = 42
NUM_CLASSES = 100
BATCH_SIZE = 16
EPOCHS_HEAD = 5
EPOCHS_FINE = 30
IMG_SIZE = 260

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FoodTrainDataset(torch.utils.data.Dataset):

    def __init__(self, train_dir, transform):
        self.samples = []
        self.transform = transform

        train_dir = Path(train_dir)

        for label in range(NUM_CLASSES):
            class_dir = train_dir / str(label)

            if not class_dir.exists():
                raise FileNotFoundError(f"missing class directory: {class_dir}")

            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, label))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, test_dir, transform):
        self.test_dir = Path(test_dir)
        self.transform = transform

        self.files = sorted(
            self.test_dir.glob("*.jpg"),
            key=lambda x: int(x.stem),
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), path.name


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def predict_test(model, test_loader):
    model.eval()

    predictions = {}

    for x, ids in test_loader:
        x = x.to(DEVICE)

        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()

        for img_id, pred in zip(ids, preds):
            predictions[img_id] = int(pred)

    return predictions


def main():
    set_seed(SEED)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    full_dataset = FoodTrainDataset("train", train_transform)

    val_size = 200
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    val_dataset.dataset.transform = val_test_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    model = models.efficientnet_b2(
        weights=models.EfficientNet_B2_Weights.DEFAULT
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES,
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # train head
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    train_acc_history = []
    val_acc_history = []
    epochs_history = []

    for epoch in range(EPOCHS_HEAD):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
        )

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        epochs_history.append(epoch + 1)

        print(
            f"Train Head Epoch {epoch + 1}/{EPOCHS_HEAD}: "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # fine-tune all
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS_FINE,
    )

    best_val_acc = 0.0

    for epoch in range(EPOCHS_FINE):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
        )

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        epochs_history.append(EPOCHS_HEAD + epoch + 1)

        scheduler.step()

        print(
            f"Fine-tuning Epoch {epoch + 1}/{EPOCHS_FINE}: "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs_history,
        train_acc_history,
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        epochs_history,
        val_acc_history,
        marker="s",
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        "training_validation_accuracy.png",
        bbox_inches="tight"
    )

    plt.show()

    # predict test set
    model.load_state_dict(
        torch.load("best_model.pt", map_location=DEVICE)
    )

    test_dataset = TestDataset("test", val_test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    predictions = predict_test(model, test_loader)

    ids = sorted(predictions.keys(), key=lambda x: int(x.replace(".jpg", "")))

    submission = pd.DataFrame({
        "ID": ids,
        "Label": [predictions[i] for i in ids],
    })

    assert len(submission) == 1036
    assert submission["ID"].iloc[0] == "0.jpg"
    assert submission["ID"].iloc[-1] == "1035.jpg"
    assert submission["Label"].isnull().sum() == 0

    submission.to_csv("submission.csv", index=False)

    print(submission.head())
    print(submission.tail())
    print(submission.shape)


if __name__ == "__main__":
    main()
