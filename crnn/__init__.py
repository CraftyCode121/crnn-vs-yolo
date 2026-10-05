"""
CRNN OCR package for synthetic document OCR line recognition.
"""

from .dataset import CharsetVocab, CRNNDatasetBuilder, preprocess_image
from .model import CRNNModel, build_crnn_model
from .metrics import (
    compute_cer,
    compute_wer,
    compute_sequence_accuracy,
    compute_epoch_metrics,
    normalize_lookalikes
)
from .callbacks import EpochAccuracyReportCallback, build_callbacks

__all__ = [
    "CharsetVocab",
    "CRNNDatasetBuilder",
    "preprocess_image",
    "CRNNModel",
    "build_crnn_model",
    "compute_cer",
    "compute_wer",
    "compute_sequence_accuracy",
    "compute_epoch_metrics",
    "normalize_lookalikes",
    "EpochAccuracyReportCallback",
    "build_callbacks"
]
