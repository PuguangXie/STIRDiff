"""Input statistics fitted on training data and stored with the model."""
import numpy as np
import pandas as pd

from .data import VITALS

TIMES = ("vital_observationoffset", "scene_observationoffset")


def fit_statistics(frame):
    """Fit means/scales on the supplied training table; time uses log1p minutes.

    The imputer uses observed training values. The predictor uses the completed
    training table after imputer warmup. Never fit these statistics on dev/test.
    """
    means, scales, medians, support = {}, {}, {}, {}
    for column in [*VITALS, "age", *TIMES]:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        support[column] = int(np.isfinite(values).sum())
        if column in TIMES:
            values = np.log1p(np.maximum(values, 0))
        values = values[np.isfinite(values)]
        if not len(values):
            if column in VITALS or column == "age":
                raise ValueError(f"No finite training observations: {column}")
            values = np.array([0.])
        means[column] = float(values.mean())
        medians[column] = float(np.median(values))
        sd = float(values.std())
        scales[column] = sd if sd > 1e-6 else 1.0
    return dict(means=means, scales=scales, medians=medians, support=support)
