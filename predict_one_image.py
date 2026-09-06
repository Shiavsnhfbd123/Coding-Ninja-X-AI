"""Predict one image with the saved final scene-classification model.

Example (PowerShell):
    python predict_one_image.py "C:\\Users\\shiva\\Pictures\\my_scene.jpg"

This script is read-only: it never changes the model, dataset, or submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_CLASSES = ("buildings", "forest", "glacier", "mountain", "sea", "street")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a .jpg, .jpeg, or .png image")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "final_model" / "best_model.pth",
        help="Saved structured checkpoint to use (default: final_model/best_model.pth)",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # Supports older PyTorch versions.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    required = {
        "model_state_dict",
        "class_names",
        "image_size",
        "normalization_mean",
        "normalization_std",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            "This script requires the structured final checkpoint. "
            f"Missing keys: {sorted(missing)}"
        )
    if tuple(checkpoint["class_names"]) != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected class mapping: {checkpoint['class_names']}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(EXPECTED_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    transform = transforms.Compose(
        [
            transforms.Resize(
                (int(checkpoint["image_size"]), int(checkpoint["image_size"])),
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                checkpoint["normalization_mean"], checkpoint["normalization_std"]
            ),
        ]
    )

    with Image.open(args.image) as image:
        image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    probabilities = torch.softmax(model(image_tensor), dim=1)[0]
    confidence, class_index = probabilities.max(dim=0)
    label_number = int(class_index.item())
    class_name = checkpoint["class_names"][label_number]

    print(f"Image:       {args.image.resolve()}")
    print(f"Prediction:  {class_name}")
    print(f"Class code:  {label_number}")
    print(f"Confidence:  {float(confidence.item()):.2%}")
    print(f"Device used: {device.type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
