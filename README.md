# STIRDiff

Model architecture code supporting the associated manuscript.

This repository contains the imputer and risk predictor modules only. Data,
preprocessing, training and evaluation workflows, comparison models, and trained
parameters are not included.

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

Both modules are randomly initialized and accept prepared tensors. Input
preparation and the application pipeline are outside the scope of this repository.
