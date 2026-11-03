"""
Utility functions for saving experiment outputs.

Provides helper functions for consistently writing research metrics,
text reports, and other experiment artifacts to disk.
"""

import json
from pathlib import Path

import numpy as np
import torch


def ensure_parent_directory(path):
    """
    Create the parent directory for a file path if it does not already exist.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def _make_json_serializable(value):
    """
    Convert common PyTorch and NumPy values into standard Python types
    that can be written using json.dump().
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()

        return value.detach().cpu().tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            key: _make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _make_json_serializable(item)
            for item in value
        ]

    return value


def save_metrics(metrics, output_path):
    """
    Save a dictionary of experiment metrics as formatted JSON.
    """
    output_path = ensure_parent_directory(output_path)

    serializable_metrics = _make_json_serializable(
        metrics
    )

    with open(output_path, "w") as file:
        json.dump(
            serializable_metrics,
            file,
            indent=4,
        )

    print(f"Saved metrics to {output_path}")


def save_text_report(report, output_path):
    """
    Save a plain-text experiment report.
    """
    output_path = ensure_parent_directory(output_path)

    with open(output_path, "w") as file:
        file.write(report)

    print(f"Saved report to {output_path}")