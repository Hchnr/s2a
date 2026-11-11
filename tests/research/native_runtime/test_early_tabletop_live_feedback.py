"""Independent replay cannot silently switch a branch's live WAM reference."""

import copy

import numpy as np
import pytest

from scripts.simulation.runtime import run_early_tabletop_live_feedback as live


def test_checker_rejects_a_different_model_reference(monkeypatch):
    monkeypatch.setattr(live.previous, "check_pair", lambda *_args: None)
    continuation = {
        "model_image_ref": {"path": "image", "sha256": "a"},
        "model_state_sha256": "b",
        "model_response_ref": {"path": "response", "sha256": "c"},
        "original_targets": np.repeat(np.eye(4)[None], 8, axis=0),
    }
    producer = {"runtime": {"continuation": continuation}}
    checker = copy.deepcopy(producer)
    live.check_pair(producer, checker)
    checker["runtime"]["continuation"]["model_response_ref"]["sha256"] = "other"
    with pytest.raises(ValueError):
        live.check_pair(producer, checker)
