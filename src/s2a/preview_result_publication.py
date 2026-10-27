"""Pre-serialize NumPy-valued preview receipts before exclusive publication."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import reference


def result_json(value):
    """Keep strict JSON numbers; never stringify unsupported measurement types."""

    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported preview result type: {type(item).__name__}")

    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=convert) + "\n"


def publish_result(path, value):
    """Reject serialization errors before creating a file; preserve existing evidence.

    This prevents the observed ndarray exception from leaving a partial receipt.
    It does not claim atomicity for filesystem failures during the subsequent write.
    """
    path = Path(path).resolve()
    if not path.is_relative_to(REPO / "outputs"):
        raise ValueError("output must stay inside S2A outputs")
    payload = result_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(payload)
    return reference(path)
