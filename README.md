# STIRDiff

Code supporting the associated STIRDiff manuscript.

## Installation

Requires Python 3.10 or later and PyTorch 2.0 or later.

```bash
pip install .
```

## Import

```python
from stirdiff import (
    AxialDiffusionImputer,
    ImputerConfig,
    PredictorConfig,
    RiskPredictor,
)

imputer = AxialDiffusionImputer(ImputerConfig())
predictor = RiskPredictor(PredictorConfig())
```

## Data loading

`stirdiff/data.py` defines the CSV fields, per-admission sequence windows, and
padded batches.

| Columns | Format |
| --- | --- |
| `subject_id`, `hadm_id` | Patient and admission identifiers; read as strings |
| `hospital_observationoffset`, `scene_observationoffset`, `vital_observationoffset` | Time offsets in minutes |
| `scene`, `age`, `gender` | Scene code 1–5, age, gender code 0/1 (2 = unknown) |
| `heartrate`, `sbp`, `dbp`, `resprate`, `temperature`, `spo2` | Numeric vital signs; empty cells indicate missing values |
| `death` | Binary outcome, required for training and testing |
| `__row_id` | Optional unique nonnegative integer row identifier |

Optional `<vital>_observed` columns must be supplied for all six vital signs
together (1 = measured, 0 = missing). Otherwise, masks are derived from finite
values. Records are ordered within each admission and split into consecutive
windows; batches are padded with a separate validity mask.

```python
from torch.utils.data import DataLoader
from stirdiff import read_table, WindowDataset, collate_windows

frame = read_table("your_table.csv", require_labels=False)
dataset = WindowDataset(frame, max_length=512)
loader = DataLoader(dataset, batch_size=8, collate_fn=collate_windows)
batch = next(iter(loader))
```

## Training

Provide separate training and development CSVs with disjoint patient IDs.

```bash
python -m stirdiff.train --train train.csv --dev dev.csv --output runs/stirdiff --device cuda
```

The command runs imputer warmup, predictor training with the imputer frozen,
and joint fine-tuning. Input statistics are fitted on training data; development
data select checkpoints and the classification cutoff. The cutoff maximizes
Youden's J (sensitivity + specificity − 1) on development predictions from the
selected checkpoint. A prediction is positive when its probability is at least
this cutoff. Stage limits and other
settings are configurable with `--help` and `TrainingConfig` in `stirdiff/train.py`.
Checkpoint selection uses mean development AUROC across scenes 1, 2, 3 and 5
by default; use `--selection-scenes` to specify a different set. Each selected
scene must contain both outcomes. Use a new output directory for each run.

### Five-fold cross-validation

```bash
python -m stirdiff.cross_validation --data development.csv --output runs/cv --device cuda
```

This creates five patient-disjoint folds from the supplied development data,
with approximate outcome stratification at the admission level. All records
and admissions belonging to one patient stay in the same fold. In each run,
four folds provide training data and the remaining fold is the development
split for checkpoint selection and its Youden cutoff. Test data are not used.

Run only selected folds (numbered 1–5) by adding `--folds`:

```bash
python -m stirdiff.cross_validation --data development.csv --output runs/cv --folds 1 3 --device cuda
```

Omit `--folds` to run all five. Use the same data, split seed and training
settings when adding other folds to the same output directory sequentially. Verified
completed folds are skipped; incomplete folds are not overwritten. Each fold
saves its own model and development cutoff under `fold_1` through `fold_5`.
For example, test fold 1 with `--checkpoint runs/cv/fold_1/model.pt`.

## Testing and prediction

```bash
python -m stirdiff.test --checkpoint runs/stirdiff/model.pt --input test.csv --output results/test --device cuda
python -m stirdiff.predict --checkpoint runs/stirdiff/model.pt --input observations.csv --output predictions.csv --device cuda
```

Testing saves row-level predictions and basic metrics: AUROC, average precision,
Brier score, accuracy, sensitivity, specificity, PPV and NPV. Classification
metrics use the saved development-set Youden cutoff without re-estimating it
on test data. Prediction also accepts unlabelled
tables. Use `--device cpu` to run on a CPU.
