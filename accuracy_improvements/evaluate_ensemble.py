"""Independently evaluate an equal-probability checkpoint ensemble on validation.

No training-pool or test images are opened.  Every member can use its own
checkpoint-defined image size and normalization, while the final probabilities
are averaged only after each model has completed its own preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_checkpoint import ValidationDataset, load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--hflip-logit-weight", type=float, action="append", required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


@torch.inference_mode()
def member_probabilities(
    checkpoint_path: Path,
    data_root: Path,
    device: torch.device,
    batch_size: int,
    hflip_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    transform = transforms.Compose([
        transforms.Resize((int(checkpoint["image_size"]), int(checkpoint["image_size"])), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["normalization_mean"], checkpoint["normalization_std"]),
    ])
    dataset = ValidationDataset(data_root / "val", transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities: list[torch.Tensor] = []
    labels_out: list[torch.Tensor] = []
    paths_out: list[str] = []
    for images, labels, paths in loader:
        images = images.to(device)
        logits = model(images)
        if hflip_weight:
            logits = (1.0 - hflip_weight) * logits + hflip_weight * model(torch.flip(images, dims=(3,)))
        probabilities.append(torch.softmax(logits, dim=1).cpu())
        labels_out.append(labels.cpu())
        paths_out.extend(paths)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(probabilities), torch.cat(labels_out), paths_out


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if len(args.checkpoint) < 2:
        raise ValueError("An ensemble requires at least two --checkpoint values.")
    if len(args.checkpoint) != len(args.hflip_logit_weight):
        raise ValueError("Supply exactly one --hflip-logit-weight per --checkpoint, in the same order.")
    if any(not 0.0 <= weight <= 1.0 for weight in args.hflip_logit_weight):
        raise ValueError("Each --hflip-logit-weight must be in [0, 1].")
    if any(not checkpoint.is_file() for checkpoint in args.checkpoint):
        missing = [str(checkpoint) for checkpoint in args.checkpoint if not checkpoint.is_file()]
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members: list[torch.Tensor] = []
    labels: torch.Tensor | None = None
    expected_paths: list[str] | None = None
    for checkpoint, hflip_weight in zip(args.checkpoint, args.hflip_logit_weight):
        probabilities, candidate_labels, paths = member_probabilities(
            checkpoint, args.data_root, device, args.batch_size, hflip_weight
        )
        if labels is None:
            labels, expected_paths = candidate_labels, paths
        elif not torch.equal(labels, candidate_labels) or paths != expected_paths:
            raise RuntimeError("Validation ordering mismatch between ensemble members.")
        members.append(probabilities)

    assert labels is not None and expected_paths is not None
    ensemble = torch.stack(members).mean(dim=0)
    confidence, prediction = ensemble.max(dim=1)
    correct = int((prediction == labels).sum().item())
    confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    errors: list[dict[str, object]] = []
    for path, true_label, predicted_label, score in zip(
        expected_paths, labels.tolist(), prediction.tolist(), confidence.tolist()
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
    total = len(labels)
    report = {
        "checkpoints": [str(path) for path in args.checkpoint],
        "hflip_logit_weights": args.hflip_logit_weight,
        "ensemble_method": "equal_probability_average",
        "validation_examples": total,
        "validation_accuracy_percent": 100.0 * correct / total,
        "per_class_accuracy_percent": {
            name: 100.0 * confusion[index][index] / max(1, sum(confusion[index]))
            for index, name in enumerate(CLASS_NAMES)
        },
        "error_count": len(errors),
        "uses_undefined_data": False,
        "uses_test_data": False,
    }
    with (args.output_dir / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"Validation accuracy: {report['validation_accuracy_percent']:.2f}% ({correct}/{total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
