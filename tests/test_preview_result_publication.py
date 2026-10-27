"""Measured-anchor serialization and preservation of failed-run evidence."""

import json

import numpy as np
import pytest

from s2a import preview_result_publication as publication


def test_measured_anchor_roundtrip():
    pose = np.eye(4)
    pose[:3, 3] = [0.12345678901234567, -0.25, 0.7]
    value = {"images": [{"current_source_pose": pose, "current_opening": np.float64(0.2)}]}
    with pytest.raises(TypeError, match="ndarray"):
        json.dumps(value, allow_nan=False)
    decoded = json.loads(publication.result_json(value))
    np.testing.assert_array_equal(decoded["images"][0]["current_source_pose"], pose)
    assert decoded["images"][0]["current_opening"] == value["images"][0]["current_opening"]


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.float64(np.inf), object()])
def test_bad_measurements_leave_no_partial_file(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(publication, "REPO", tmp_path)
    destination = tmp_path / "outputs" / "worker.json"
    with pytest.raises((TypeError, ValueError)):
        publication.publish_result(destination, {"measurement": bad})
    assert not destination.exists()


def test_existing_failure_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(publication, "REPO", tmp_path)
    destination = tmp_path / "outputs" / "worker.json"
    destination.parent.mkdir()
    destination.write_text('{"images": [')
    with pytest.raises(FileExistsError):
        publication.publish_result(destination, {"pose": np.eye(4)})
    assert destination.read_text() == '{"images": ['


def test_outside_outputs_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(publication, "REPO", tmp_path)
    with pytest.raises(ValueError, match="outputs"):
        publication.publish_result(tmp_path / "result.json", {})
