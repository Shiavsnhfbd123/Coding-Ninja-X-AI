# Accuracy Improvement Record

This file contains only measured accuracy-improvement work. It does not
replace, modify, or duplicate the project README. Every validation result below
uses the supplied frozen `data/val` split (1,200 images, 200 per class). No
trial trained on `data/test`, and no trial used an `undefined` image for
training. All models are `torchvision.models.resnet18(weights=None)` with a
six-class head, so no pretrained weights or external image data were used.

## Validation protocol and promotion rule

- Training data: 600 balanced seed images (100 per class).
- Validation data: 1,200 balanced images, used only for evaluation and model
  selection.
- Evaluation: reload the saved checkpoint with
  `accuracy_improvements/evaluate_checkpoint.py`; do not rely only on the
  in-training metric.
- A candidate is selected only when its independently reloaded validation score
  exceeds the previous selected score without changing the data split or using
  test data.
- Existing checkpoints and all earlier experiment folders are preserved. New
  checkpoints are written only to new `accuracy_improvements/artifacts/...`
  folders.

## Accuracy journey

| Stage | Technique / configuration | Validation accuracy | Correct / 1,200 | Change from previous selected stage | Status |
|---|---|---:|---:|---:|---|
| Original structured final model | Existing 150x150 direct-head ResNet-18 | 77.83% | 934 | — | Previous baseline preserved |
| Previous selected model | 150x150 MixUp + 20% flip-logit blend | 78.33% | 940 | +0.50 pp | Previous selected model preserved |
| Re-evaluated previous EMA artifact | 150x150 EMA artifact + 50% flip-logit blend | 79.50% | 954 | +1.17 pp | Better historical inference candidate, later superseded |
| New retained model | **224x224 input + retained MixUp + 20% flip-logit blend** | **79.75%** | **957** | **+1.42 pp** | **Selected** |
| New trial | 224x224 + EMA 0.995 | 78.92% | 947 | -0.83 pp | Rejected |
| New trial | 224x224 + label smoothing 0.05 | 79.50% | 954 | -0.25 pp | Rejected |
| Evaluation-only trial | Multi-view affine/flip TTA | 80.00% full split | 960 | Not confirmed on held-out half | Rejected |
| New retained submission candidate | Equal-probability ensemble of four preserved checkpoints | **80.75%** | **969** | **+1.00 pp** | **Selected for submission** |
| New retained submission candidate | Seed-diverse three-checkpoint probability ensemble | **81.00%** | **972** | **+0.25 pp** | **Selected for submission** |

The selected model is a measured **+1.92 percentage-point** improvement over
the preserved 77.83% structured-final baseline (23 more correct validation
images), and a **+1.42 percentage-point** improvement over the prior 78.33%
selected path.

## Previous accuracy-improvement steps

These were already present before this continuation and were preserved.

### P1. Direct linear ResNet-18 final checkpoint

**Why it was used.** The legacy root checkpoint uses a larger MLP classifier
and independently scored 67.17%. The preserved final checkpoint uses the
competition-permitted direct six-class ResNet-18 head and a longer regularized
schedule.

**Files retained.**

- `final_model/best_model.pth`
- `final_model/model_info.txt`
- `final_inference.py`

**Architecture actually used.**

```python
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 6)
```

**Recorded training configuration.** 70 epochs, batch size 32, seed 42,
SGD/Nesterov (momentum 0.9), learning rate 0.015, weight decay 0.0005,
five-epoch warm-up followed by cosine decay, label smoothing 0.05, horizontal
flip, mild affine augmentation, and mild color jitter.

**Measured before / after.** Legacy root checkpoint: 67.17% (806/1,200).
Preserved direct-head checkpoint: 77.83% (934/1,200), a +10.66 pp historical
improvement.

**Status.** Kept as the stable, untouched fallback.

### P2. Reproducible evaluator and 150x150 MixUp model

**Why it was used.** A fixed evaluator was needed before promoting a candidate;
MixUp was tested to regularize training from only 600 seed labels.

**Files added previously.**

- `accuracy_improvements/train_resnet18.py`
- `accuracy_improvements/evaluate_checkpoint.py`

**Kept training change.**

```python
weight = torch.distributions.Beta(alpha, alpha).sample().item()
weight = max(weight, 1.0 - weight)
images = weight * images + (1.0 - weight) * images[permutation]
loss = weight * criterion(logits, first_labels) + (1.0 - weight) * criterion(logits, second_labels)
```

**Actual command.**

```powershell
$py = 'D:\Codex\2026-09-05\https-docs-3lc-ai-3lc-latest-3\work\accuracy-env\Scripts\python.exe'
& $py .\accuracy_improvements\train_resnet18.py `
  --data-root .\data `
  --output-dir .\accuracy_improvements\artifacts\mixup_a02_p05_seed42 `
  --epochs 70 --batch-size 32 --seed 42 `
  --label-smoothing 0.0 --mixup-alpha 0.2 --mixup-probability 0.5
```

**Measured before / after.** The saved 150x150 MixUp checkpoint scored 78.00%
with identity inference. A 20% flipped-logit contribution increased it to
78.33% (940/1,200), +0.50 pp over the original structured final checkpoint.

**Status.** Kept and preserved; later superseded by the 224x224 model below.

### P3. Optional horizontal-flip logit blend

**Why it was used.** A conservative blend of logits from the original and
horizontally flipped image improved the 150x150 MixUp candidate.

**File modified previously.** `final_inference.py`.

**Kept code.** Its default is `0.0`, so prior inference behavior is unchanged.

```python
logits = model(images)
if args.hflip_logit_weight:
    flipped_logits = model(torch.flip(images, dims=(3,)))
    logits = (1.0 - args.hflip_logit_weight) * logits + args.hflip_logit_weight * flipped_logits
```

**Status.** Kept. The selected model below uses the same explicitly supplied
`--hflip-logit-weight 0.20` setting.

## New accuracy-improvement steps in this continuation

### N1. Re-evaluate the preserved EMA artifact with a symmetric flip blend

**Why it was selected.** The preserved `ema_0995_seed42` checkpoint was
originally rejected by its identity score. The allowed flip-logit weight was
rechecked on the frozen validation set before beginning new training.

**Files modified.** None. Existing artifact and evaluator only.

**Actual evaluation command.**

```powershell
& $py .\accuracy_improvements\evaluate_checkpoint.py `
  --checkpoint .\accuracy_improvements\artifacts\ema_0995_seed42\best_resnet18.pth `
  --data-root .\data `
  --output-dir .\accuracy_improvements\reports\ema_0995_hflip_050 `
  --hflip-logit-weight 0.50
```

**Configuration.** Existing 150x150 EMA checkpoint; original-logit weight
0.50, horizontally flipped-logit weight 0.50; batch size 64.

**Measured before / after.** Its previous selected-path score was 78.33%.
The re-evaluated artifact scored 79.50% (954/1,200), +1.17 pp.

**Status.** Retained as an important historical experiment, but superseded by
N2. No source or checkpoint was overwritten.

### N2. Increase the ResNet-18 input size from 150x150 to 224x224

**Why it was selected.** The supplied images are 150 pixels wide/high, but the
standard ResNet-18 spatial pipeline retains a more useful feature-map layout
at 224x224. Upsampling was tested as one isolated preprocessing change while
keeping the model, seed, optimizer, scheduler, loss, augmentation, MixUp,
batch size, and training data unchanged.

**Files modified.** No source file. The image size is an existing trainer
argument and is stored in the new checkpoint metadata. New artifacts only:

- `accuracy_improvements/artifacts/image224_mixup_seed42/`
- `accuracy_improvements/reports/image224_mixup_hflip_020/`

**Previous configuration.** `--image-size 150`.

**Updated configuration.** `--image-size 224`.

**Actual training command.**

```powershell
& $py .\accuracy_improvements\train_resnet18.py `
  --data-root .\data `
  --output-dir .\accuracy_improvements\artifacts\image224_mixup_seed42 `
  --epochs 70 --batch-size 32 --image-size 224 --seed 42 `
  --label-smoothing 0.0 --mixup-alpha 0.2 --mixup-probability 0.5 `
  --learning-rate 0.015 --weight-decay 0.0005 `
  --warmup-epochs 5 --early-stopping-patience 22
```

**Training result.** Best in-training validation accuracy: 79.58% at epoch
62. Final epoch training loss: 0.2631. The saved checkpoint was then reloaded
independently.

**Actual validation command.**

```powershell
& $py .\accuracy_improvements\evaluate_checkpoint.py `
  --checkpoint .\accuracy_improvements\artifacts\image224_mixup_seed42\best_resnet18.pth `
  --data-root .\data `
  --output-dir .\accuracy_improvements\reports\image224_mixup_hflip_020 `
  --hflip-logit-weight 0.20
```

**Measured before / after.** Prior selected path: 78.33% (940/1,200). New
224x224 path: 79.75% (957/1,200), **+1.42 pp**.

**Status.** Kept and selected.

**Observation.** Per-class accuracy for the selected configuration is buildings
77.0%, forest 92.5%, glacier 71.5%, mountain 73.5%, sea 74.5%, and street
89.5%. The remaining error concentration is glacier/mountain/sea, indicating a
label/data limitation rather than a class-count imbalance.

### N3. Exponential moving average (EMA) on the 224x224 MixUp model

**Why it was selected.** The historical 150x150 EMA artifact had benefited
from symmetric flip TTA, so EMA was tested as the sole added training change to
the retained 224x224 configuration.

**File temporarily modified and then reverted.**
`accuracy_improvements/train_resnet18.py` received an optional
`--ema-decay 0.995` update rule and EMA-state validation. It was removed after
the experiment because it did not improve the selected model. The normal
trainer interface and default behavior are therefore unchanged.

**Trial code actually used.**

```python
ema_parameter.mul_(0.995).add_(parameter.detach(), alpha=0.005)
```

**Actual training command used during the trial.**

```powershell
& $py .\accuracy_improvements\train_resnet18.py `
  --data-root .\data `
  --output-dir .\accuracy_improvements\artifacts\image224_mixup_ema0995_seed42 `
  --epochs 70 --batch-size 32 --image-size 224 --seed 42 `
  --label-smoothing 0.0 --mixup-alpha 0.2 --mixup-probability 0.5 `
  --learning-rate 0.015 --weight-decay 0.0005 `
  --warmup-epochs 5 --early-stopping-patience 22 --ema-decay 0.995
```

**Measured before / after.** Non-EMA selected model: 79.75%. EMA checkpoint:
78.92% (947/1,200) at identity, 20% flip blend, and 50% flip blend; **-0.83
pp**.

**Status.** Rejected. The new checkpoint and reports remain as experiment
evidence; EMA source changes were reverted and are not part of the final
training path.

### N4. Multi-view affine / flip test-time augmentation

**Why it was selected.** The selected model was trained with mild affine
augmentation. A read-only validation sweep tested whether small rotations and
translations improved its predictions without retraining.

**Files modified.** Temporary, non-project analysis script only:
`D:\Codex\2026-09-05\https-docs-3lc-ai-3lc-latest-3\work\tta_224_sweep.py`.
No project inference code was changed.

**Views actually evaluated.** Original, horizontal flip, rotations of -4 and
+4 degrees, and translations of 5 pixels left/right/up/down. The strongest
full-split blend used original 85%, horizontal flip 5%, and each of four affine
views 2.5%.

**Measured before / after.** The full split reached 80.00% (960/1,200), but an
even-index validation tuning half selected the same 80.50% score on the
untouched odd-index half as the simpler retained 20% flip blend. It did not
show a held-out gain.

**Status.** Rejected. It would add more inference complexity for no confirmed
generalization benefit.

### N5. Label smoothing at 0.05 with the 224x224 MixUp model

**Why it was selected.** The historical direct-head final model used label
smoothing 0.05. It was tested as one loss-function change while holding the
224x224 configuration constant.

**Files modified.** No source file; new artifact and reports only:

- `accuracy_improvements/artifacts/image224_mixup_ls005_seed42/`
- `accuracy_improvements/reports/image224_mixup_ls005_hflip_020/`

**Previous configuration.** `CrossEntropyLoss(label_smoothing=0.0)`.

**Updated configuration.** `CrossEntropyLoss(label_smoothing=0.05)`.

**Actual training command.**

```powershell
& $py .\accuracy_improvements\train_resnet18.py `
  --data-root .\data `
  --output-dir .\accuracy_improvements\artifacts\image224_mixup_ls005_seed42 `
  --epochs 70 --batch-size 32 --image-size 224 --seed 42 `
  --label-smoothing 0.05 --mixup-alpha 0.2 --mixup-probability 0.5 `
  --learning-rate 0.015 --weight-decay 0.0005 `
  --warmup-epochs 5 --early-stopping-patience 22
```

**Training result.** Best in-training validation accuracy: 79.33% at epoch
62. Final epoch training loss: 0.4467.

**Measured before / after.** Selected non-smoothed model: 79.75%. The saved
label-smoothed checkpoint reached 79.50% (954/1,200) with both 20% and 50%
flip blends; **-0.25 pp**.

**Status.** Rejected. It is not included in the final training configuration.

### N6. Equal-probability ensemble of complementary retained checkpoints

**Why it was selected.** The existing checkpoints make different validation
errors despite using the same permitted random-init ResNet-18 architecture.
An equal average of their post-softmax probabilities was tested as a separate
inference-only change. This keeps every original checkpoint and the normal
single-checkpoint inference command unchanged.

**Files added.**

- `accuracy_improvements/evaluate_ensemble.py`
- `accuracy_improvements/ensemble_inference.py`

**Actual implementation.** Each member first applies its own checkpoint
metadata (`image_size`, normalization) and its fixed flip-logit blend. The
six-class probability vectors are then averaged with equal weight:

```python
ensemble = torch.stack(members).mean(dim=0)
confidence, prediction = ensemble.max(dim=1)
```

**Members actually used.**

1. `image224_mixup_seed42/best_resnet18.pth`, flip-logit weight `0.20`
2. `image224_mixup_ls005_seed42/best_resnet18.pth`, flip-logit weight `0.20`
3. `mixup_a02_p05_seed42/best_resnet18.pth`, flip-logit weight `0.20`
4. `ema_0995_seed42/best_resnet18.pth`, flip-logit weight `0.50`

**Actual validation command.**

```powershell
& $py .\accuracy_improvements\evaluate_ensemble.py `
  --data-root .\data `
  --output-dir .\accuracy_improvements\reports\ensemble_equal_4_hflip `
  --batch-size 64 `
  --checkpoint .\accuracy_improvements\artifacts\image224_mixup_seed42\best_resnet18.pth --hflip-logit-weight 0.20 `
  --checkpoint .\accuracy_improvements\artifacts\image224_mixup_ls005_seed42\best_resnet18.pth --hflip-logit-weight 0.20 `
  --checkpoint .\accuracy_improvements\artifacts\mixup_a02_p05_seed42\best_resnet18.pth --hflip-logit-weight 0.20 `
  --checkpoint .\accuracy_improvements\artifacts\ema_0995_seed42\best_resnet18.pth --hflip-logit-weight 0.50
```

**Measured before / after.** Previous selected single-checkpoint result:
79.75% (957/1,200). The ensemble result is **80.75% (969/1,200)**,
**+1.00 pp** (12 additional correct validation images).

**Validation robustness observation.** The four-member configuration scored
80.00% on the even-index half versus 79.00% for the prior single model, and
81.50% on the odd-index half versus 80.50% for the prior single model. This
does not turn the validation set into a second training set; it is reported to
show that the measured gain is present in both halves rather than only one.

**Actual prediction command.**

```powershell
& $py .\accuracy_improvements\ensemble_inference.py `
  --test-dir .\data\test `
  --sample-submission .\sample_submission.csv `
  --output .\accuracy_improvements\submissions\ensemble_equal_4_hflip.csv `
  --batch-size 64 `
  --checkpoint .\accuracy_improvements\artifacts\image224_mixup_seed42\best_resnet18.pth --hflip-logit-weight 0.20 `
  --checkpoint .\accuracy_improvements\artifacts\image224_mixup_ls005_seed42\best_resnet18.pth --hflip-logit-weight 0.20 `
  --checkpoint .\accuracy_improvements\artifacts\mixup_a02_p05_seed42\best_resnet18.pth --hflip-logit-weight 0.20 `
  --checkpoint .\accuracy_improvements\artifacts\ema_0995_seed42\best_resnet18.pth --hflip-logit-weight 0.50
```

**Output validation.** The resulting `ensemble_equal_4_hflip.csv` has 1,800
rows, the exact `sample_submission.csv` ID order, labels only in 0--5, and
confidence values in [0, 1].

**Status.** Kept as a validated interim submission path, then superseded by
N7's better 81.00% candidate. The individual 79.75% single-checkpoint path
remains available and unmodified.

### N7. Second random seed and seed-diverse three-member ensemble

**Why it was selected.** N6 showed that independently trained retained
checkpoints make complementary errors. One additional run changed only the
random seed from N2 (`42` to `17`), then its independently reloaded checkpoint
was tested as a concise ensemble member. The architecture, data, optimizer,
schedule, augmentation, MixUp configuration, and inference interface were not
replaced.

**Files changed.** No existing source file was changed for the seed run. New
artifact and reports only:

- `accuracy_improvements/artifacts/image224_mixup_seed17/`
- `accuracy_improvements/reports/image224_mixup_seed17_hflip020/`
- `accuracy_improvements/reports/ensemble_equal_3_seed42_seed17_ema_hflip/`
- `accuracy_improvements/submissions/ensemble_equal_3_seed42_seed17_ema_hflip.csv`

**Only training configuration change.**

```powershell
--seed 17   # N2 used --seed 42; all other N2 settings were retained
```

**Actual seed-17 training command.**

```powershell
& $py .\accuracy_improvements\train_resnet18.py `
  --data-root .\data `
  --output-dir .\accuracy_improvements\artifacts\image224_mixup_seed17 `
  --epochs 70 --batch-size 32 --image-size 224 --seed 17 `
  --label-smoothing 0.0 --mixup-alpha 0.2 --mixup-probability 0.5 `
  --learning-rate 0.015 --weight-decay 0.0005 `
  --warmup-epochs 5 --early-stopping-patience 22
```

**Training result.** Early stopping occurred after epoch 68. The best
in-training checkpoint was epoch 46 at 79.50%; the trainer's cloned CPU state
selection prevents later epochs from mutating that saved best state.

**Independent single-checkpoint evaluation.** With the fixed 20% flip-logit
blend, the reloaded seed-17 checkpoint scored **79.83% (958/1,200)**. That is
only +0.08 pp above N2 alone, so it was not promoted as a replacement for the
single-model path.

**Final three ensemble members actually used.**

1. `image224_mixup_seed42/best_resnet18.pth`, flip-logit weight `0.20`
2. `image224_mixup_seed17/best_resnet18.pth`, flip-logit weight `0.20`
3. `ema_0995_seed42/best_resnet18.pth`, flip-logit weight `0.50`

**Actual validation command.**

```powershell
& $py .\accuracy_improvements\evaluate_ensemble.py `
  --data-root .\data `
  --output-dir .\accuracy_improvements\reports\ensemble_equal_3_seed42_seed17_ema_hflip `
  --batch-size 64 `
  --checkpoint .\accuracy_improvements\artifacts\image224_mixup_seed42\best_resnet18.pth --hflip-logit-weight 0.20 `
  --checkpoint .\accuracy_improvements\artifacts\image224_mixup_seed17\best_resnet18.pth --hflip-logit-weight 0.20 `
  --checkpoint .\accuracy_improvements\artifacts\ema_0995_seed42\best_resnet18.pth --hflip-logit-weight 0.50
```

**Measured before / after.** N6 interim ensemble: 80.75% (969/1,200). This
three-member ensemble: **81.00% (972/1,200)**, **+0.25 pp** (three additional
correct validation images). It is +1.25 pp above the prior 79.75% single-model
selected path.

**Validation robustness observation.** The final three-member ensemble scored
80.17% on even-indexed validation images versus 79.00% for the old primary
single model, and 81.83% on odd-indexed images versus 80.50% for that model.
The difference appears on both halves, but the frozen full validation result
remains the formal comparison metric.

**Actual prediction command.**

```powershell
& $py .\accuracy_improvements\ensemble_inference.py `
  --test-dir .\data\test `
  --sample-submission .\sample_submission.csv `
  --output .\accuracy_improvements\submissions\ensemble_equal_3_seed42_seed17_ema_hflip.csv `
  --batch-size 64 `
  --checkpoint .\accuracy_improvements\artifacts\image224_mixup_seed42\best_resnet18.pth --hflip-logit-weight 0.20 `
  --checkpoint .\accuracy_improvements\artifacts\image224_mixup_seed17\best_resnet18.pth --hflip-logit-weight 0.20 `
  --checkpoint .\accuracy_improvements\artifacts\ema_0995_seed42\best_resnet18.pth --hflip-logit-weight 0.50
```

**Status.** Kept and selected for the next Kaggle upload. No old model,
checkpoint, README, or single-checkpoint prediction path was removed or
overwritten.

## Final selected model and exact commands

### Final configuration

| Item | Selected value |
|---|---|
| Validation accuracy | **81.00% (972/1,200)** |
| Architecture | Equal-probability ensemble of three random-init ResNet-18 direct `Linear(512, 6)` checkpoints; no architecture replacement or pretrained weights |
| Training images | Each member used the same 600 balanced seed images only; no undefined or test images |
| Input size / preprocessing | Each checkpoint uses its stored resize and ImageNet normalization: two 224x224 members and one 150x150 member |
| Training augmentation | Member-specific retained settings; the leading member uses horizontal flip p=0.5, mild affine/color augmentation, and MixUp alpha=0.2 at p=0.5 |
| Optimizer | Existing member optimizers retained; no new training was introduced by the ensemble |
| Final inference | Equal average of post-softmax probabilities after each member's fixed flip-logit blend |
| Batch size / workers | 64 / 0 for ensemble evaluation and prediction |
| Final model files | The three members listed in N7; all original files are preserved |
| Final prediction file | `accuracy_improvements/submissions/ensemble_equal_3_seed42_seed17_ema_hflip.csv` |

### Train

```powershell
$py = 'D:\Codex\2026-09-05\https-docs-3lc-ai-3lc-latest-3\work\accuracy-env\Scripts\python.exe'
& $py .\accuracy_improvements\train_resnet18.py `
  --data-root .\data `
  --output-dir .\accuracy_improvements\artifacts\image224_mixup_seed42 `
  --epochs 70 --batch-size 32 --image-size 224 --seed 42 `
  --label-smoothing 0.0 --mixup-alpha 0.2 --mixup-probability 0.5 `
  --learning-rate 0.015 --weight-decay 0.0005 `
  --warmup-epochs 5 --early-stopping-patience 22
```

### Evaluate

```powershell
& $py .\accuracy_improvements\evaluate_checkpoint.py `
  --checkpoint .\accuracy_improvements\artifacts\image224_mixup_seed42\best_resnet18.pth `
  --data-root .\data `
  --output-dir .\accuracy_improvements\reports\image224_mixup_hflip_020 `
  --hflip-logit-weight 0.20
```

### Generate the selected Kaggle prediction CSV

```powershell
& $py .\accuracy_improvements\ensemble_inference.py `
  --test-dir .\data\test `
  --sample-submission .\sample_submission.csv `
  --output .\accuracy_improvements\submissions\ensemble_equal_3_seed42_seed17_ema_hflip.csv `
  --batch-size 64 `
  --checkpoint .\accuracy_improvements\artifacts\image224_mixup_seed42\best_resnet18.pth --hflip-logit-weight 0.20 `
  --checkpoint .\accuracy_improvements\artifacts\image224_mixup_seed17\best_resnet18.pth --hflip-logit-weight 0.20 `
  --checkpoint .\accuracy_improvements\artifacts\ema_0995_seed42\best_resnet18.pth --hflip-logit-weight 0.50
```

The generated CSV has 1,800 rows, the exact `sample_submission.csv` ID order,
valid labels 0–5, and confidence values in [0, 1]. The preserved single-model
file remains available if Kaggle requires a single-checkpoint fallback.
