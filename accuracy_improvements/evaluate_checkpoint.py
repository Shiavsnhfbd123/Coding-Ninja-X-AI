"""Independently evaluate a structured ResNet-18 checkpoint on frozen validation data.

This tool is intentionally validation-only.  It never opens ``data/test`` and
creates a report, confusion matrix, and error list in a caller-supplied output
directory.  Use the same optional horizontal-flip logit blend as
``final_inference.py`` before producing a submission.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_NAMES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ValidationDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for label, name in enumerate(CLASS_NAMES):
            folder = root / name
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing validation class folder: {folder}")
            self.samples.extend(
                (path, label) for path in sorted(folder.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        if not self.samples:
            raise RuntimeError(f"No validation images found in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        from PIL import Image

        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label, str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hflip-logit-weight", type=float, default=0.0)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {path}")
    required = {"model_state_dict", "class_names", "image_size", "normalization_mean", "normalization_std"}
    if not required - set(checkpoint):
        if tuple(checkpoint["class_names"]) != CLASS_NAMES:
            raise ValueError(f"Unexpected class mapping: {checkpoint['class_names']}")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        checkpoint.setdefault("model_kind", "resnet18_linear_head")
        return model.to(device).eval(), checkpoint

    if any(key.startswith("classifier.") for key in checkpoint):
        class LegacyStarterResNet18(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.resnet = models.resnet18(weights=None)
                features = self.resnet.fc.in_features
                self.resnet.fc = nn.Identity()
                self.classifier = nn.Sequential(
                    nn.Linear(features, 256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, len(CLASS_NAMES)),
                )

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return self.classifier(self.resnet(images))

        model = LegacyStarterResNet18()
        model.load_state_dict(checkpoint, strict=True)
        return model.to(device).eval(), {
            "model_kind": "legacy_resnet18_mlp_head",
            "image_size": 150,
            "class_names": list(CLASS_NAMES),
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "validation_accuracy": None,
            "best_epoch": None,
        }

    raise ValueError("Expected a structured direct-head checkpoint or the legacy ResNet-18 MLP checkpoint")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not 0.0 <= args.hflip_logit_weight <= 1.0:
        raise ValueError("--hflip-logit-weight must be in [0, 1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    transform = transforms.Compose([
        transforms.Resize((int(checkpoint["image_size"]), int(checkpoint["image_size"])), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["normalization_mean"], checkpoint["normalization_std"]),
    ])
    dataset = ValidationDataset(args.data_root / "val", transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    errors: list[dict[str, object]] = []
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels, paths in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        if args.hflip_logit_weight:
            flipped_logits = model(torch.flip(images, dims=(3,)))
            logits = (1.0 - args.hflip_logit_weight) * logits + args.hflip_logit_weight * flipped_logits
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)
        correct += (predictions == labels).sum().item()
        for path, true_label, prediction, score in zip(
            paths, labels.cpu().tolist(), predictions.cpu().tolist(), confidence.cpu().tolist()
        ):
            confusion[true_label][prediction] += 1
            if true_label != prediction:
                errors.append({
                    "path": path,
                    "true_label": true_label,
                    "true_class": CLASS_NAMES[true_label],
                    "predicted_label": prediction,
                    "predicted_class": CLASS_NAMES[prediction],
                    "confidence": round(float(score), 6),
                })

    total = len(dataset)
    accuracy = 100.0 * correct / total
    per_class = {
        name: 100.0 * confusion[index][index] / max(1, sum(confusion[index]))
        for index, name in enumerate(CLASS_NAMES)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "confusion_matrix.csv",
        ["true_class", *CLASS_NAMES],
        [{"true_class": CLASS_NAMES[index], **dict(zip(CLASS_NAMES, row))} for index, row in enumerate(confusion)],
    )
    write_csv(
        args.output_dir / "validation_errors.csv",
        ["path", "true_label", "true_class", "predicted_label", "predicted_class", "confidence"],
        errors,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_model_kind": checkpoint.get("model_kind"),
        "checkpoint_metadata_accuracy_percent": checkpoint.get("validation_accuracy"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "hflip_logit_weight": args.hflip_logit_weight,
        "validation_examples": total,
        "validation_accuracy_percent": accuracy,
        "validation_loss": loss_sum / total,
        "per_class_accuracy_percent": per_class,
        "error_count": len(errors),
        "uses_undefined_data": False,
        "uses_test_data": False,
    }
    with (args.output_dir / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"Validation accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Report: {(args.output_dir / 'validation_report.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
