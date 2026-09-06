"""Create a Kaggle CSV from an equal-probability ensemble of saved checkpoints.

This is deliberately separate from ``final_inference.py``: the existing
single-checkpoint command and its defaults remain unchanged.  Each checkpoint
retains its own stored input size and normalization settings, so compatible
150px and 224px ResNet-18 checkpoints can be combined safely.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from final_inference import TestImageDataset, load_checkpoint, sample_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Checkpoint to include. Repeat once for each ensemble member.",
    )
    parser.add_argument(
        "--hflip-logit-weight",
        type=float,
        action="append",
        required=True,
        help="Flip-logit blend for the corresponding checkpoint. Repeat in checkpoint order.",
    )
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "data" / "test")
    parser.add_argument("--sample-submission", type=Path, default=PROJECT_ROOT / "sample_submission.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


@torch.inference_mode()
def probabilities_for_checkpoint(
    checkpoint_path: Path,
    test_dir: Path,
    device: torch.device,
    batch_size: int,
    hflip_weight: float,
) -> tuple[list[str], torch.Tensor]:
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    transform = transforms.Compose([
        transforms.Resize((int(checkpoint["image_size"]), int(checkpoint["image_size"])), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["normalization_mean"], checkpoint["normalization_std"]),
    ])
    dataset = TestImageDataset(test_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    image_ids: list[str] = []
    outputs: list[torch.Tensor] = []
    for images, ids in loader:
        images = images.to(device)
        logits = model(images)
        if hflip_weight:
            logits = (1.0 - hflip_weight) * logits + hflip_weight * model(torch.flip(images, dims=(3,)))
        outputs.append(torch.softmax(logits, dim=1).cpu())
        image_ids.extend(ids)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return image_ids, torch.cat(outputs)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if len(args.checkpoint) < 2:
        raise ValueError("An ensemble requires at least two --checkpoint values.")
    if len(args.checkpoint) != len(args.hflip_logit_weight):
        raise ValueError("Supply exactly one --hflip-logit-weight for each --checkpoint, in the same order.")
    if any(not 0.0 <= weight <= 1.0 for weight in args.hflip_logit_weight):
        raise ValueError("Each --hflip-logit-weight must be in [0, 1].")
    if any(not checkpoint.is_file() for checkpoint in args.checkpoint):
        missing = [str(checkpoint) for checkpoint in args.checkpoint if not checkpoint.is_file()]
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expected_ids: list[str] | None = None
    probabilities: list[torch.Tensor] = []
    for checkpoint, hflip_weight in zip(args.checkpoint, args.hflip_logit_weight):
        image_ids, member_probabilities = probabilities_for_checkpoint(
            checkpoint, args.test_dir, device, args.batch_size, hflip_weight
        )
        if expected_ids is None:
            expected_ids = image_ids
        elif image_ids != expected_ids:
            raise RuntimeError("Checkpoint test-image ordering mismatch.")
        probabilities.append(member_probabilities)

    assert expected_ids is not None
    ensemble = torch.stack(probabilities).mean(dim=0)
    confidence, prediction = ensemble.max(dim=1)
    predicted = {
        image_id: (int(label), float(score))
        for image_id, label, score in zip(expected_ids, prediction.tolist(), confidence.tolist())
    }
    ordered_ids = sample_ids(args.sample_submission)
    missing = [image_id for image_id in ordered_ids if image_id not in predicted]
    extras = set(predicted) - set(ordered_ids)
    if missing or extras:
        raise RuntimeError(f"Test/sample ID mismatch: missing={len(missing)}, extras={len(extras)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_id", "prediction", "confidence"))
        writer.writeheader()
        for image_id in ordered_ids:
            label, score = predicted[image_id]
            writer.writerow({"image_id": image_id, "prediction": label, "confidence": score})
    print(f"Wrote {len(ordered_ids)} validated ensemble predictions to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
