"""
Plant disease classifier for synthetic→real transfer validation.

Train on synthetic data (rendered from Gaussian splats), test on real images.

Usage:
    # Train binary classifier (healthy vs diseased)
    uv run src/classify.py train --task binary --epochs 30

    # Train 5-way disease classifier
    uv run src/classify.py train --task multiclass --epochs 30

    # Evaluate on real data
    uv run src/classify.py eval --checkpoint results/best.pt --data data/plantsegv2

    # Run full experiment suite
    uv run src/classify.py experiment --output results/
"""

import argparse
import json
import random
from pathlib import Path

import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# Disease type mapping
DISEASE_CLASSES = ["powdery_mildew", "rust", "leaf_spot", "blight", "chlorosis"]
DISEASE_TO_IDX = {d: i for i, d in enumerate(DISEASE_CLASSES)}

# PlantSegV2 category → our 5 disease types
PLANTSEG_MAPPING = {
    "rust": ["rust"],
    "blight": ["blight", "anthracnose"],
    "powdery_mildew": ["mildew", "powdery"],
    "chlorosis": ["chlorosis", "yellow", "mosaic"],
    "leaf_spot": ["spot", "scab"],
}


def map_plantseg_category(category_name: str) -> str | None:
    """Map PlantSegV2 category to our 5 disease types."""
    name_lower = category_name.lower()
    for our_disease, keywords in PLANTSEG_MAPPING.items():
        if any(kw in name_lower for kw in keywords):
            return our_disease
    return None


class SyntheticDataset(Dataset):
    """Load synthetic plant images (healthy or diseased)."""

    def __init__(
        self,
        root: Path,
        split: str = "train",
        task: str = "binary",
        transform=None,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            root: Path to data/synthetic or data/synthetic_diseased
            split: "train" or "val"
            task: "binary" (healthy=0, diseased=1) or "multiclass" (5 diseases)
            transform: torchvision transforms
            val_ratio: Fraction for validation
            seed: Random seed for split
        """
        self.root = Path(root)
        self.task = task
        self.transform = transform

        # Load annotations
        ann_path = self.root / "annotations.json"
        with open(ann_path) as f:
            data = json.load(f)

        # Determine if this is healthy or diseased dataset
        self.is_diseased = "diseases" in data

        # Build samples list
        all_samples = []
        for img_info in data["images"]:
            img_path = self.root / img_info["image"]
            if self.is_diseased:
                disease = img_info["disease"]["disease_type"]
                label = DISEASE_TO_IDX[disease] if task == "multiclass" else 1
            else:
                label = 0  # healthy

            all_samples.append((img_path, label))

        # Split train/val deterministically
        random.seed(seed)
        indices = list(range(len(all_samples)))
        random.shuffle(indices)
        n_val = int(len(indices) * val_ratio)

        if split == "val":
            indices = indices[:n_val]
        else:
            indices = indices[n_val:]

        self.samples = [all_samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class PlantSegDataset(Dataset):
    """Load PlantSegV2 real images for evaluation."""

    def __init__(
        self,
        root: Path,
        task: str = "binary",
        transform=None,
        max_samples: int | None = None,
    ):
        """
        Args:
            root: Path to data/plantsegv2
            task: "binary" (all diseased=1) or "multiclass" (mapped to our 5)
            transform: torchvision transforms
            max_samples: Limit samples (for quick testing)
        """
        self.root = Path(root)
        self.task = task
        self.transform = transform

        # Load Metadatav2.csv (has correct filenames, unlike coco_annotations.json)
        import csv
        self.samples = []
        with open(self.root / "Metadatav2.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["Name"]
                disease = row["Disease"]

                if task == "binary":
                    label = 1  # all PlantSegV2 images are diseased
                else:
                    # Map to our 5 diseases
                    mapped = map_plantseg_category(disease)
                    if mapped is None:
                        continue  # skip unmappable
                    label = DISEASE_TO_IDX[mapped]

                img_path = self.root / "images" / filename
                if img_path.exists():
                    self.samples.append((img_path, label))

        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class CombinedDataset(Dataset):
    """Combine healthy + diseased synthetic datasets for binary classification."""

    def __init__(
        self,
        healthy_root: Path,
        diseased_root: Path,
        split: str = "train",
        transform=None,
    ):
        self.healthy = SyntheticDataset(
            healthy_root, split=split, task="binary", transform=transform
        )
        self.diseased = SyntheticDataset(
            diseased_root, split=split, task="binary", transform=transform
        )

    def __len__(self):
        return len(self.healthy) + len(self.diseased)

    def __getitem__(self, idx):
        if idx < len(self.healthy):
            return self.healthy[idx]
        return self.diseased[idx - len(self.healthy)]


def get_transforms(train: bool = True, img_size: int = 224):
    """Get image transforms."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def create_model(num_classes: int, backbone: str = "resnet50") -> nn.Module:
    """Create classifier with timm backbone."""
    model = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
    return model


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-4,
    output_dir: Path = Path("results"),
    device: str = "cuda",
):
    """Train model with validation and checkpointing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(output_dir / "logs")
    best_val_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Log to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, output_dir / "best.pt")
            print(f"  → Saved best model (acc={val_acc:.4f})")

    writer.close()
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    return model


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    device: str = "cuda",
) -> dict:
    """Evaluate model and return metrics."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    _, acc, preds, labels = validate(model, test_loader, criterion, device)

    # Compute metrics
    f1_macro = float(f1_score(labels, preds, average="macro"))
    f1_per_class_arr = f1_score(labels, preds, average=None, labels=list(range(len(class_names))))
    f1_per_class = list(f1_per_class_arr) if hasattr(f1_per_class_arr, "__iter__") else [f1_per_class_arr]
    cm = confusion_matrix(labels, preds)

    print(f"\n{'='*50}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"\nPer-class F1:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {f1_per_class[i]:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_per_class": {name: float(f1_per_class[i]) for i, name in enumerate(class_names)},
        "confusion_matrix": cm.tolist(),
    }


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cmd_train(args):
    """Train classifier."""
    device = get_device()
    print(f"Using device: {device}")

    # Setup data
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    if args.task == "binary":
        # Combine healthy + diseased
        train_dataset = CombinedDataset(
            Path("data/synthetic"),
            Path("data/synthetic_diseased"),
            split="train",
            transform=train_transform,
        )
        val_dataset = CombinedDataset(
            Path("data/synthetic"),
            Path("data/synthetic_diseased"),
            split="val",
            transform=val_transform,
        )
        num_classes = 2
        class_names = ["healthy", "diseased"]
    else:
        # 5-way disease classification
        train_dataset = SyntheticDataset(
            Path("data/synthetic_diseased"),
            split="train",
            task="multiclass",
            transform=train_transform,
        )
        val_dataset = SyntheticDataset(
            Path("data/synthetic_diseased"),
            split="val",
            task="multiclass",
            transform=val_transform,
        )
        num_classes = 5
        class_names = DISEASE_CLASSES

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Create model
    model = create_model(num_classes, backbone=args.backbone)
    print(f"Model: {args.backbone} with {num_classes} classes")

    # Train
    output_dir = Path(args.output) / args.task
    train(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, output_dir=output_dir, device=device)

    # Final evaluation on val set
    checkpoint = torch.load(output_dir / "best.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluate(model, val_loader, class_names, device=device)


def cmd_eval(args):
    """Evaluate trained model on test data."""
    device = get_device()
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, weights_only=False)

    # Infer task from checkpoint path
    task = "multiclass" if "multiclass" in str(args.checkpoint) else "binary"

    if task == "binary":
        num_classes = 2
        class_names = ["healthy", "diseased"]
    else:
        num_classes = 5
        class_names = DISEASE_CLASSES

    # Create model and load weights
    model = create_model(num_classes, backbone=args.backbone)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Setup test data
    transform = get_transforms(train=False)
    data_path = Path(args.data)

    if "plantseg" in str(data_path):
        test_dataset = PlantSegDataset(data_path, task=task, transform=transform)
        print(f"Evaluating on PlantSegV2 ({len(test_dataset)} samples)")
    else:
        test_dataset = SyntheticDataset(data_path, split="val", task=task, transform=transform)
        print(f"Evaluating on synthetic ({len(test_dataset)} samples)")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Evaluate
    metrics = evaluate(model, test_loader, class_names, device=device)

    # Save metrics
    output_path = Path(args.checkpoint).parent / f"eval_{data_path.name}.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


def cmd_experiment(args):
    """Run full experiment suite."""
    device = get_device()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Experiment 1: Binary classification
    print("\n" + "="*60)
    print("EXPERIMENT 1: Binary (Healthy vs Diseased)")
    print("="*60)

    # Train
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    train_dataset = CombinedDataset(
        Path("data/synthetic"), Path("data/synthetic_diseased"),
        split="train", transform=train_transform
    )
    val_dataset = CombinedDataset(
        Path("data/synthetic"), Path("data/synthetic_diseased"),
        split="val", transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    binary_ckpt_path = output_dir / "binary" / "best.pt"
    if binary_ckpt_path.exists():
        print(f"Checkpoint exists, skipping training: {binary_ckpt_path}")
    else:
        model = create_model(2, backbone=args.backbone)
        train(model, train_loader, val_loader, epochs=args.epochs, output_dir=output_dir / "binary", device=device)

    # Evaluate on synthetic
    model = create_model(2, backbone=args.backbone)
    checkpoint = torch.load(binary_ckpt_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("\n[Binary] Synthetic test:")
    results["binary_synthetic"] = evaluate(model, val_loader, ["healthy", "diseased"], device=device)

    # Evaluate on real (PlantSegV2)
    real_dataset = PlantSegDataset(Path("data/plantsegv2"), task="binary", transform=val_transform)
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("\n[Binary] PlantSegV2 test:")
    results["binary_real"] = evaluate(model, real_loader, ["healthy", "diseased"], device=device)

    # Experiment 2: 5-way classification
    print("\n" + "="*60)
    print("EXPERIMENT 2: 5-Way Disease Classification")
    print("="*60)

    train_dataset = SyntheticDataset(
        Path("data/synthetic_diseased"), split="train", task="multiclass", transform=train_transform
    )
    val_dataset = SyntheticDataset(
        Path("data/synthetic_diseased"), split="val", task="multiclass", transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    multiclass_ckpt_path = output_dir / "multiclass" / "best.pt"
    if multiclass_ckpt_path.exists():
        print(f"Checkpoint exists, skipping training: {multiclass_ckpt_path}")
    else:
        model = create_model(5, backbone=args.backbone)
        train(model, train_loader, val_loader, epochs=args.epochs, output_dir=output_dir / "multiclass", device=device)

    model = create_model(5, backbone=args.backbone)
    checkpoint = torch.load(multiclass_ckpt_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("\n[Multiclass] Synthetic test:")
    results["multiclass_synthetic"] = evaluate(model, val_loader, DISEASE_CLASSES, device=device)

    # Evaluate on real (mapped PlantSegV2)
    real_dataset = PlantSegDataset(Path("data/plantsegv2"), task="multiclass", transform=val_transform)
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"\n[Multiclass] PlantSegV2 test ({len(real_dataset)} mapped samples):")
    results["multiclass_real"] = evaluate(model, real_loader, DISEASE_CLASSES, device=device)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Binary - Synthetic: {results['binary_synthetic']['accuracy']:.4f}")
    print(f"Binary - Real:      {results['binary_real']['accuracy']:.4f}")
    print(f"5-Way  - Synthetic: {results['multiclass_synthetic']['accuracy']:.4f}")
    print(f"5-Way  - Real:      {results['multiclass_real']['accuracy']:.4f}")

    # Save all results
    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'experiment_results.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Plant disease classifier for synthetic→real validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train classifier")
    train_parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    train_parser.add_argument("--backbone", default="resnet50")
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--output", default="results")
    train_parser.set_defaults(func=cmd_train)

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--data", required=True)
    eval_parser.add_argument("--backbone", default="resnet50")
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.set_defaults(func=cmd_eval)

    # Experiment command
    exp_parser = subparsers.add_parser("experiment", help="Run full experiment suite")
    exp_parser.add_argument("--backbone", default="resnet50")
    exp_parser.add_argument("--epochs", type=int, default=30)
    exp_parser.add_argument("--batch-size", type=int, default=32)
    exp_parser.add_argument("--output", default="results")
    exp_parser.set_defaults(func=cmd_experiment)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
