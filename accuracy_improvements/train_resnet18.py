"""Train an isolated, competition-safe ResNet-18 accuracy candidate.

The legacy ``train.py``/``predict.py`` path is deliberately untouched.  This
script uses the same permitted architecture as ``final_model/best_model.pth``:
``torchvision.models.resnet18(weights=None)`` with a six-class linear head.
By default it reads only the six seed-label folders under ``data/train`` and
the frozen ``data/val`` split.  An explicitly supplied CSV can additionally
include *human-verified* 3LC labels from ``data/train/undefined``; this is
validated strictly and is opt-in, so the existing seed-only behavior is
unchanged.

Every run must use a new output directory.  The structured checkpoint written
by this script is compatible with ``final_inference.py --checkpoint ...``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)


@dataclass
class Evaluation:
    loss: float
    accuracy: float
    confusion: list[list[int]]
    errors: list[dict[str, object]]


@dataclass
class EpochRecord:
    epoch: int
    learning_rate: float
    train_loss: float
    validation_accuracy: float
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, required=True, help="New, empty directory for this candidate.")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.2, help="MixUp Beta distribution alpha; 0 disables it.")
    parser.add_argument("--mixup-probability", type=float, default=0.5, help="Per-batch MixUp probability.")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=22)
    parser.add_argument("--workers", type=int, default=0, help="Keep 0 on Windows for stable PIL loading.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verified-manifest",
        type=Path,
        default=None,
        help=(
            "Optional CSV of reviewed 3LC pool labels. It must contain filename and label "
            "columns; paths are resolved only inside data/train/undefined."
        ),
    )
    parser.add_argument(
        "--max-active-training-examples",
        type=int,
        default=3000,
        help="Safety cap for seed images plus reviewed pool images (competition limit: 3000).",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def image_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def parse_class_label(value: str, *, row_number: int) -> int:
    """Convert a 3LC manifest label (class name or 0--5) to its canonical id."""
    normalized = value.strip().lower()
    if normalized in CLASS_NAMES:
        return CLASS_NAMES.index(normalized)
    try:
        label = int(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Manifest row {row_number}: label must be one of {CLASS_NAMES} or 0--5, got {value!r}."
        ) from exc
    if not 0 <= label < len(CLASS_NAMES):
        raise ValueError(f"Manifest row {row_number}: label {label} is outside 0--{len(CLASS_NAMES) - 1}.")
    return label


def load_verified_samples(manifest_path: Path, data_root: Path) -> list[tuple[Path, int]]:
    """Load only explicitly reviewed pool files, rejecting unsafe or ambiguous CSV rows.

    The CSV is intentionally small and auditable.  It needs ``filename`` and
    ``label`` columns.  ``weight`` is optional, but if present every row must
    have active weight 1; this prevents accidentally training on a stale or
    disabled 3LC row.  Optional ``table_id`` values must be unique.
    """
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Reviewed-label manifest does not exist: {manifest_path}")

    pool_root = (data_root / "train" / "undefined").resolve()
    if not pool_root.is_dir():
        raise FileNotFoundError(f"Undefined-pool directory does not exist: {pool_root}")

    reviewed: list[tuple[Path, int]] = []
    filenames: set[str] = set()
    table_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {"filename", "label"}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"Reviewed-label manifest is missing required column(s): {', '.join(sorted(missing))}."
            )
        for row_number, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            if not filename:
                raise ValueError(f"Manifest row {row_number}: filename is empty.")
            relative_name = Path(filename)
            if relative_name.name != filename or relative_name.parent != Path("."):
                raise ValueError(
                    f"Manifest row {row_number}: filename must be a bare pool filename, got {filename!r}."
                )
            if relative_name.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Manifest row {row_number}: unsupported image extension in {filename!r}.")
            if filename in filenames:
                raise ValueError(f"Manifest row {row_number}: duplicate filename {filename!r}.")

            weight_value = (row.get("weight") or "").strip()
            if weight_value and weight_value not in {"1", "1.0", "1.00"}:
                raise ValueError(
                    f"Manifest row {row_number}: expected reviewed 3LC weight 1, got {weight_value!r}."
                )
            table_id = (row.get("table_id") or "").strip()
            if table_id:
                if table_id in table_ids:
                    raise ValueError(f"Manifest row {row_number}: duplicate 3LC table_id {table_id!r}.")
                table_ids.add(table_id)

            path = (pool_root / relative_name).resolve()
            if path.parent != pool_root or not path.is_file():
                raise FileNotFoundError(
                    f"Manifest row {row_number}: pool file is missing or escapes undefined/: {filename!r}."
                )
            label_value = row.get("label") or ""
            reviewed.append((path, parse_class_label(label_value, row_number=row_number)))
            filenames.add(filename)

    if not reviewed:
        raise ValueError("Reviewed-label manifest contains no usable rows.")
    return reviewed


class SceneDataset(Dataset):
    """Explicit class-order dataset with optional reviewed pool examples."""

    def __init__(
        self,
        split_root: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        extra_samples: list[tuple[Path, int]] | None = None,
    ) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for label, class_name in enumerate(CLASS_NAMES):
            folder = split_root / class_name
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing class directory: {folder}")
            self.samples.extend((path, label) for path in image_files(folder))
        if extra_samples:
            self.samples.extend(extra_samples)
        if not self.samples:
            raise RuntimeError(f"No labelled images found in {split_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label, str(path)


def make_train_transform(image_size: int) -> transforms.Compose:
    """The retained, scene-safe augmentation policy from the verified baseline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.90, 1.10), shear=4),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.03)
        ], p=0.50),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])


def make_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])


def make_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def maybe_mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp only when explicitly enabled for an isolated experiment."""
    if alpha <= 0.0 or torch.rand((), device=images.device).item() >= probability:
        return images, labels, labels, 1.0
    weight = torch.distributions.Beta(alpha, alpha).sample().item()
    weight = max(weight, 1.0 - weight)
    permutation = torch.randperm(images.size(0), device=images.device)
    return weight * images + (1.0 - weight) * images[permutation], labels, labels[permutation], weight


def make_scheduler(optimizer: Optimizer, epochs: int, warmup_epochs: int) -> LambdaLR:
    """Five-epoch linear warm-up followed by cosine decay, stepped per epoch."""

    def multiplier(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=multiplier)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Evaluation:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    errors: list[dict[str, object]] = []
    for images, labels, paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        for path, true_label, predicted_label, score in zip(
            paths, labels.cpu().tolist(), predictions.cpu().tolist(), confidence.cpu().tolist()
        ):
            confusion[true_label][predicted_label] += 1
            if true_label != predicted_label:
                errors.append({
                    "path": path,
                    "true_label": true_label,
                    "true_class": CLASS_NAMES[true_label],
                    "predicted_label": predicted_label,
                    "predicted_class": CLASS_NAMES[predicted_label],
                    "confidence": round(float(score), 6),
                })
    return Evaluation(
        loss=loss_sum / max(total, 1),
        accuracy=100.0 * correct / max(total, 1),
        confusion=confusion,
        errors=errors,
    )


def clone_state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_active_training_examples < 1:
        raise ValueError("--epochs, --batch-size, and --max-active-training-examples must be positive")
    if args.mixup_alpha < 0.0 or not 0.0 <= args.mixup_probability <= 1.0:
        raise ValueError("--mixup-alpha must be non-negative and --mixup-probability must be in [0, 1]")
    if args.output_dir.exists():
        if any(args.output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite a non-empty output directory: {args.output_dir}")
    else:
        args.output_dir.mkdir(parents=True)
    if not (args.data_root / "train").is_dir() or not (args.data_root / "val").is_dir():
        raise FileNotFoundError("--data-root must contain train/ and val/")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}; AMP: {use_amp}")
    print("Model: ResNet-18 from random initialization, direct six-class linear head.")
    reviewed_samples = (
        load_verified_samples(args.verified_manifest, args.data_root)
        if args.verified_manifest is not None
        else []
    )
    train_dataset = SceneDataset(
        args.data_root / "train", make_train_transform(args.image_size), extra_samples=reviewed_samples
    )
    if len(train_dataset) > args.max_active_training_examples:
        raise ValueError(
            f"Refusing to train on {len(train_dataset)} active examples; cap is "
            f"{args.max_active_training_examples}."
        )
    if reviewed_samples:
        print(
            f"Training data: 600 seed images + {len(reviewed_samples)} explicitly reviewed 3LC pool images. "
            "Test data are excluded."
        )
    else:
        print("Training data: six seed class folders only; undefined and test data are excluded.")

    validation_dataset = SceneDataset(args.data_root / "val", make_eval_transform(args.image_size))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=use_amp, persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=use_amp, persistent_workers=args.workers > 0,
    )

    model = make_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9,
        weight_decay=args.weight_decay, nesterov=True,
    )
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = GradScaler(device.type, enabled=use_amp)

    best_accuracy = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_evaluation: Evaluation | None = None
    stale_epochs = 0
    history: list[EpochRecord] = []

    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        loss_sum = 0.0
        examples = 0
        for images, labels, _paths in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images, first_labels, second_labels, mixup_weight = maybe_mixup(
                images, labels, args.mixup_alpha, args.mixup_probability
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = (
                    mixup_weight * criterion(logits, first_labels)
                    + (1.0 - mixup_weight) * criterion(logits, second_labels)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach().item() * labels.size(0)
            examples += labels.size(0)

        model_evaluation = evaluate(model, validation_loader, criterion, device)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        elapsed = time.monotonic() - started
        history.append(EpochRecord(
            epoch=epoch,
            learning_rate=learning_rate,
            train_loss=loss_sum / max(examples, 1),
            validation_accuracy=model_evaluation.accuracy,
            seconds=elapsed,
        ))
        print(
            f"Epoch {epoch:03d}/{args.epochs} | lr={learning_rate:.2e} | "
            f"train_loss={loss_sum / max(examples, 1):.4f} | val={model_evaluation.accuracy:.2f}% | {elapsed:.1f}s"
        )

        if model_evaluation.accuracy > best_accuracy:
            best_accuracy = model_evaluation.accuracy
            best_epoch = epoch
            best_state = clone_state_to_cpu(model)
            best_evaluation = model_evaluation
            stale_epochs = 0
            print("  New best checkpoint copied to CPU memory.")
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stopping_patience:
                print(f"Early stopping after {stale_epochs} epochs without validation improvement.")
                break

    if best_state is None or best_evaluation is None:
        raise RuntimeError("Training finished without a checkpoint")

    training_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    checkpoint = {
        "format_version": 2,
        "model_kind": "resnet18_linear_head",
        "model_state_dict": best_state,
        "class_names": list(CLASS_NAMES),
        "image_size": args.image_size,
        "normalization_mean": list(NORMALIZATION_MEAN),
        "normalization_std": list(NORMALIZATION_STD),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "validation_accuracy": best_accuracy,
        "training_args": training_args,
        "undefined_examples_used_for_training": len(reviewed_samples),
        "test_examples_used_for_training": 0,
    }
    checkpoint_path = args.output_dir / "best_resnet18.pth"
    torch.save(checkpoint, checkpoint_path)

    write_csv(
        args.output_dir / "training_history.csv",
        list(EpochRecord.__dataclass_fields__),
        [asdict(record) for record in history],
    )
    write_csv(
        args.output_dir / "validation_confusion_matrix.csv",
        ["true_class", *CLASS_NAMES],
        [{"true_class": CLASS_NAMES[index], **dict(zip(CLASS_NAMES, row))} for index, row in enumerate(best_evaluation.confusion)],
    )
    write_csv(
        args.output_dir / "validation_errors.csv",
        ["path", "true_label", "true_class", "predicted_label", "predicted_class", "confidence"],
        best_evaluation.errors,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_validation_accuracy_percent": best_accuracy,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "undefined_examples_used_for_training": len(reviewed_samples),
        "test_examples_used_for_training": 0,
        "train_class_counts": {
            CLASS_NAMES[label]: count
            for label, count in sorted(Counter(label for _path, label in train_dataset.samples).items())
        },
        "reviewed_manifest": str(args.verified_manifest.resolve()) if args.verified_manifest else None,
        "device": str(device),
        "training_config": training_args,
    }
    with (args.output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"\nBest validation accuracy: {best_accuracy:.2f}% at epoch {best_epoch}")
    print(f"Checkpoint: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
