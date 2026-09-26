"""MPDD-2026 development metrics on the raw 0-27 PHQ-9 scale."""
from __future__ import annotations

import numpy as np


METRIC_CONTRACT = 'mpdd2026_raw_phq_track'


def raw_phq_metrics(y_true, y_pred):
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if (truth.ndim != 1 or prediction.ndim != 1 or not truth.size
            or truth.shape != prediction.shape):
        raise ValueError('raw PHQ targets and predictions must be equal nonempty 1D arrays')
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('raw PHQ targets and predictions must be finite')
    if np.any((truth < 0.0) | (truth > 27.0)) or np.any(
            (prediction < 0.0) | (prediction > 27.0)):
        raise ValueError('raw PHQ targets and predictions must be within 0..27')
    truth_mean, prediction_mean = float(truth.mean()), float(prediction.mean())
    covariance = float(np.mean((truth - truth_mean) * (prediction - prediction_mean)))
    denominator = float(truth.var() + prediction.var()
                        + (truth_mean - prediction_mean) ** 2)
    ccc = float(2.0 * covariance / denominator) if denominator > 1e-10 else 0.0
    error = truth - prediction
    total = float(np.sum((truth - truth_mean) ** 2))
    return {
        'ccc': ccc,
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'mae': float(np.mean(np.abs(error))),
        'r2': float(1.0 - np.sum(error ** 2) / total) if total > 1e-12 else 0.0,
    }
