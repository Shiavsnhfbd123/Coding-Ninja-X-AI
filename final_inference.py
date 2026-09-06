"""Reproduce the final 3LC HackBlox Scene Classification submission.

This script is self-contained for structured ResNet-18 checkpoints such as
final_model/best_model.pth. It uses test images only for inference and
validates the exact sample-submission image-ID order before writing a CSV.
The optional horizontal-flip logit blend is disabled by default, preserving the
original inference behavior.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class TestImageDataset(Dataset):
    def __init__(self, folder: Path, transform: transforms.Compose) -> None:
        self.transform = transform
        self.paths = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No test images found in {folder}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), path.stem


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Support older PyTorch releases.
        checkpoint = torch.load(path, map_location=device)
    required = {"model_state_dict", "class_names", "image_size", "normalization_mean", "normalization_std"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Final checkpoint is missing keys: {sorted(missing)}")
    if tuple(checkpoint["class_names"]) != CLASS_NAMES:
        raise ValueError(f"Unexpected class mapping: {checkpoint['class_names']}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def sample_ids(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "image_id" not in (reader.fieldnames or []):
            raise ValueError(f"sample submission has no image_id column: {path}")
        image_ids = [row["image_id"] for row in reader]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("sample submission contains duplicate image IDs")
    return image_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "final_model" / "best_model.pth")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "data" / "test")
    parser.add_argument("--sample-submission", type=Path, default=PROJECT_ROOT / "sample_submission.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "final_submission" / "submission.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hflip-logit-weight",
        type=float,
        default=0.0,
        help="Blend this fraction of horizontal-flip logits with original logits (default: 0.0).",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if not 0.0 <= args.hflip_logit_weight <= 1.0:
        raise ValueError("--hflip-logit-weight must be in [0, 1]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    transform = transforms.Compose([
        transforms.Resize((int(checkpoint["image_size"]), int(checkpoint["image_size"])), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["normalization_mean"], checkpoint["normalization_std"]),
    ])
    loader = DataLoader(TestImageDataset(args.test_dir, transform), batch_size=args.batch_size, shuffle=False, num_workers=0)
    predicted: dict[str, tuple[int, float]] = {}
    for images, image_ids in loader:
        images = images.to(device)
        logits = model(images)
        if args.hflip_logit_weight:
            flipped_logits = model(torch.flip(images, dims=(3,)))
            logits = (1.0 - args.hflip_logit_weight) * logits + args.hflip_logit_weight * flipped_logits
        probabilities = torch.softmax(logits, dim=1)
        confidences, labels = probabilities.max(dim=1)
        for image_id, label, confidence in zip(image_ids, labels.cpu().tolist(), confidences.cpu().tolist()):
            predicted[image_id] = (int(label), float(confidence))
    required_ids = sample_ids(args.sample_submission)
    missing = [image_id for image_id in required_ids if image_id not in predicted]
    extras = set(predicted) - set(required_ids)
    if missing or extras:
        raise RuntimeError(f"Test/sample ID mismatch: missing={len(missing)}, extras={len(extras)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_id", "prediction", "confidence"))
        writer.writeheader()
        for image_id in required_ids:
            label, confidence = predicted[image_id]
            writer.writerow({"image_id": image_id, "prediction": label, "confidence": confidence})
    print(
        f"Wrote {len(required_ids)} validated predictions to {args.output.resolve()} "
        f"(hflip logit weight: {args.hflip_logit_weight:.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
