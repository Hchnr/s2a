"""Reproduce the stopped preview run's serialization error without new inference."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import reference
from s2a.preview_result_publication import publish_result, result_json


def main():
    root = REPO / "outputs/current_coupled_wam_development_20261026"
    case = root / "development/cases/task0_init0_s42_raw"
    failed = reference(case / "worker_result.json")
    inputs = reference(case / "actual_trajectory.npz")
    with np.load(inputs["path"], allow_pickle=False) as arrays:
        images = [
            dict(
                frame_index=i,
                current_source_pose=arrays["source_pose"][i].copy(),
                current_opening=float(arrays["opening"][i]),
            )
            for i in (0, 16, 56)
        ]
    payload = dict(images=images)
    try:
        json.dumps(payload, allow_nan=False)
    except TypeError as error:
        if "ndarray" not in str(error):
            raise
        original_error = str(error)
    else:
        raise AssertionError("Original ndarray failure was not reproduced")
    decoded = json.loads(result_json(payload))
    for actual, expected in zip(decoded["images"], images, strict=True):
        np.testing.assert_array_equal(actual["current_source_pose"], expected["current_source_pose"])
        assert actual["current_opening"] == expected["current_opening"]
    assert reference(Path(failed["path"])) == failed
    result = dict(
        complete=True,
        passed=True,
        scope="serialization of saved measured anchor values only; not worker recovery or task acceptance",
        original_failure_reproduced=original_error,
        failed_worker_ref=failed,
        arrays_ref=inputs,
        anchors=[0, 16, 56],
        pose_values_equal=True,
        original_failure_unchanged=True,
        source_refs=[reference(Path(__file__)), reference(REPO / "src/s2a/preview_result_publication.py")],
        model_requests=0,
        physics_steps=0,
        development_qualified=False,
        fresh_authorized=False,
    )
    print(json.dumps(publish_result(REPO / "outputs/preview_result_publication_20261026/verified.json", result)))


if __name__ == "__main__":
    main()
