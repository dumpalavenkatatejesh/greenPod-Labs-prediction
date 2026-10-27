import argparse
import json
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES = ["blackspot", "canker", "fresh", "grenning"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed=42):
    """Set random seeds to make the run more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_paths(root, split):
    """Return (image_path, class_label) pairs for one dataset split."""
    rows = []
    for class_name in CLASSES:
        folder = Path(root) / split / class_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing class folder: {folder}")

        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append((str(path), class_name))

    if not rows:
        raise ValueError(
            f"No supported images found in {Path(root) / split}. "
            f"Expected class folders: {', '.join(CLASSES)}"
        )
    return rows


def inspect_dataset(root, output_dir):
    """Create a dataset inventory, simple quality flags, class counts, and plot."""
    rows = []
    corrupt = []

    for split in ("train", "test"):
        for path, label in image_paths(root, split):
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    image.verify()

                rows.append(
                    {
                        "split": split,
                        "label": label,
                        "path": path,
                        "width": width,
                        "height": height,
                        "aspect_ratio": round(width / height, 4),
                        "bytes": os.path.getsize(path),
                    }
                )
            except Exception as exc:
                corrupt.append({"path": path, "error": str(exc)})

    inventory = pd.DataFrame(rows)
    inventory.to_csv(output_dir / "dataset_inventory.csv", index=False)

    flagged = []
    if not inventory.empty:
        flagged = inventory[
            (inventory["width"] < 64)
            | (inventory["height"] < 64)
            | (inventory["aspect_ratio"] < 0.5)
            | (inventory["aspect_ratio"] > 2.0)
        ]["path"].tolist()

    with open(output_dir / "data_quality_notes.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "valid_images": len(inventory),
                "unreadable_images": corrupt,
                "dimension_outlier_rule": (
                    "Flag images whose width or height is below 64 pixels, "
                    "or whose aspect ratio is outside [0.5, 2.0]. "
                    "This is a review flag, not an automatic deletion rule."
                ),
                "flagged_images": flagged,
            },
            file,
            indent=2,
        )

    if not inventory.empty:
        counts = inventory.groupby(["split", "label"]).size().unstack(fill_value=0)
        counts.to_csv(output_dir / "class_counts.csv")

        ax = counts.T.plot(kind="bar", figsize=(9, 5))
        ax.set_title("Image count by class and split")
        ax.set_ylabel("Images")
        ax.set_xlabel("Class")
        plt.tight_layout()
        plt.savefig(output_dir / "class_distribution.png", dpi=160)
        plt.close()

    return inventory


def classical_features(image_path, size=(128, 128)):
    """Extract a colour histogram and HOG features for the SVM."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    histogram = cv2.calcHist(
        [hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]
    ).flatten()
    histogram = histogram / (histogram.sum() + 1e-8)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
    hog_features = hog.compute(gray).flatten()

    return np.concatenate([histogram, hog_features]).astype(np.float32)


def calculate_scores(y_true, y_pred, labels):
    """Calculate accuracy, macro precision/recall/F1, and weighted F1."""
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )[2]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def save_evaluation(name, y_true, y_pred, labels, output_dir, target_names=None):
    """Save per-class report and confusion-matrix plot."""
    if target_names is None:
        target_names = [str(label) for label in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    with open(output_dir / f"{name}_classification_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(matrix, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, xticks_rotation=35, colorbar=False)
    ax.set_title(f"{name} — test confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    return calculate_scores(y_true, y_pred, labels)


class OrangeDataset(Dataset):
    """PyTorch dataset returning transformed image tensors and numeric labels."""

    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform
        self.class_to_index = {class_name: i for i, class_name in enumerate(CLASSES)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, class_name = self.rows[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), self.class_to_index[class_name]


class SmallCNN(nn.Module):
    """A compact convolutional neural network for four classes."""

    def __init__(self, number_of_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(160, number_of_classes),
        )

    def forward(self, inputs):
        return self.head(self.features(inputs))


def predict_cnn_image(model, image_path, transform, device):
    """Return CNN class probabilities for one image."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    predicted_index = int(np.argmax(probabilities))
    return CLASSES[predicted_index], float(probabilities[predicted_index]), probabilities


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate SVM and CNN orange-image classifiers."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Dataset root containing train/ and test/ class folders.",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    train_rows = image_paths(args.data_dir, "train")
    test_rows = image_paths(args.data_dir, "test")

    print(
        "Train class counts:",
        pd.Series([label for _, label in train_rows])
        .value_counts()
        .reindex(CLASSES, fill_value=0)
        .to_dict(),
    )
    print(
        "Test class counts:",
        pd.Series([label for _, label in test_rows])
        .value_counts()
        .reindex(CLASSES, fill_value=0)
        .to_dict(),
    )

    inventory = inspect_dataset(args.data_dir, output_dir)
    print(
        f"Data audit: {len(inventory)} readable images; "
        f"inventory and flagged cases saved to {output_dir}"
    )

    # Approach 1: HOG + HSV + SVM. Use a validation subset for model selection.
    train_svm_rows, val_svm_rows = train_test_split(
        train_rows,
        test_size=0.2,
        random_state=args.seed,
        stratify=[label for _, label in train_rows],
    )

    print("Extracting HOG + HSV features...")
    x_train_svm = np.stack([classical_features(path) for path, _ in train_svm_rows])
    y_train_svm = np.array([label for _, label in train_svm_rows])
    x_val_svm = np.stack([classical_features(path) for path, _ in val_svm_rows])
    y_val_svm = np.array([label for _, label in val_svm_rows])

    svm = make_pipeline(
        StandardScaler(),
        SVC(C=3.0, kernel="rbf", class_weight="balanced", probability=True),
    )
    svm.fit(x_train_svm, y_train_svm)
    val_svm_pred = svm.predict(x_val_svm)
    print(
        "SVM validation:",
        json.dumps(calculate_scores(y_val_svm, val_svm_pred, CLASSES), indent=2),
    )

    # Refit SVM using all training images and evaluate on the held-out test set.
    x_all_svm = np.stack([classical_features(path) for path, _ in train_rows])
    y_all_svm = np.array([label for _, label in train_rows])
    x_test_svm = np.stack([classical_features(path) for path, _ in test_rows])
    y_test_labels = np.array([label for _, label in test_rows])

    svm.fit(x_all_svm, y_all_svm)
    svm_test_pred = svm.predict(x_test_svm)
    svm_scores = save_evaluation(
        "svm_hog_hsv",
        y_test_labels,
        svm_test_pred,
        CLASSES,
        output_dir,
        target_names=CLASSES,
    )

    import joblib
    joblib.dump(svm, output_dir / "svm_hog_hsv.joblib")

    # Approach 2: CNN with augmentation and validation-based checkpoint selection.
    train_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_cnn_rows, val_cnn_rows = train_test_split(
        train_rows,
        test_size=0.2,
        random_state=args.seed,
        stratify=[label for _, label in train_rows],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training CNN on: {device}")

    model = SmallCNN(len(CLASSES)).to(device)
    train_loader = DataLoader(
        OrangeDataset(train_cnn_rows, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        OrangeDataset(val_cnn_rows, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        OrangeDataset(test_rows, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    counts = pd.Series([label for _, label in train_cnn_rows]).value_counts()
    class_weights = torch.tensor(
        [
            len(train_cnn_rows) / (len(CLASSES) * counts[class_name])
            for class_name in CLASSES
        ],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = []
    best_macro_f1 = -1.0
    best_model_path = output_dir / "small_cnn_best.pt"

    for epoch in range(args.epochs):
        model.train()
        total_seen = 0
        loss_sum = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_seen += len(labels)
            loss_sum += loss.item() * len(labels)

        train_loss = loss_sum / max(total_seen, 1)

        model.eval()
        val_true = []
        val_pred = []
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images.to(device))
                predictions = logits.argmax(dim=1).cpu().numpy()
                val_pred.extend(predictions)
                val_true.extend(labels.numpy())

        val_metrics = calculate_scores(
            np.array(val_true),
            np.array(val_pred),
            list(range(len(CLASSES))),
        )
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, **val_metrics}
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"loss={train_loss:.4f}, "
            f"val_acc={val_metrics['accuracy']:.3f}, "
            f"val_macro_f1={val_metrics['macro_f1']:.3f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_model_path)

    pd.DataFrame(history).to_csv(
        output_dir / "cnn_training_history.csv", index=False
    )

    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )
    model.eval()

    cnn_true = []
    cnn_pred = []
    cnn_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            cnn_true.extend(labels.numpy())
            cnn_pred.extend(probabilities.argmax(axis=1))
            cnn_probabilities.extend(probabilities)

    cnn_true = np.array(cnn_true)
    cnn_pred = np.array(cnn_pred)
    cnn_probabilities = np.array(cnn_probabilities)

    cnn_scores = save_evaluation(
        "small_cnn",
        cnn_true,
        cnn_pred,
        list(range(len(CLASSES))),
        output_dir,
        target_names=CLASSES,
    )

    comparison = pd.DataFrame(
        [
            {"model": "SVM (HSV histogram + HOG)", **svm_scores},
            {"model": "Small CNN", **cnn_scores},
        ]
    )
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    with open(output_dir / "model_comparison.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "class_order": CLASSES,
                "test_metrics": {
                    "SVM_HOG_HSV": svm_scores,
                    "Small_CNN": cnn_scores,
                },
                "note": (
                    "Metrics are from the supplied test split; "
                    "not evidence of production generalization."
                ),
            },
            file,
            indent=2,
        )

    # Save predictions for EVERY test image from both models.
    all_predictions = []
    for index, (image_path, actual_class) in enumerate(test_rows):
        cnn_prob = cnn_probabilities[index]
        cnn_index = int(np.argmax(cnn_prob))
        svm_prediction = str(svm_test_pred[index])

        row = {
            "image": image_path,
            "actual_class": actual_class,
            "cnn_predicted_class": CLASSES[cnn_index],
            "cnn_confidence": float(cnn_prob[cnn_index]),
            "svm_predicted_class": svm_prediction,
            "cnn_correct": CLASSES[cnn_index] == actual_class,
            "svm_correct": svm_prediction == actual_class,
        }

        for class_index, class_name in enumerate(CLASSES):
            row[f"prob_{class_name}"] = float(cnn_prob[class_index])

        all_predictions.append(row)

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(output_dir / "all_test_predictions.csv", index=False)

    with open(output_dir / "all_test_predictions.json", "w", encoding="utf-8") as file:
        json.dump(all_predictions, file, indent=2)

    # Keep one sample prediction for use in the report.
    if all_predictions:
        with open(output_dir / "example_prediction.json", "w", encoding="utf-8") as file:
            json.dump(all_predictions[0], file, indent=2)

    print("\nTEST METRICS\n", comparison.to_string(index=False))
    print(f"\nPredictions generated for {len(all_predictions)} test images.")
    print(f"All predictions CSV: {(output_dir / 'all_test_predictions.csv').resolve()}")
    print(f"All predictions JSON: {(output_dir / 'all_test_predictions.json').resolve()}")

    print("\nFIRST 5 PREDICTIONS")
    if not predictions_df.empty:
        print(predictions_df.head().to_string(index=False))

    print(f"\nAll outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
