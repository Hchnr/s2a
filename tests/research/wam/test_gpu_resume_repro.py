import copy
import json

import numpy as np
import pytest

from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro


def fixed_inputs():
    config = json.loads(repro.CONFIG.read_text())
    repro.validate_config(config)
    _prepared, _bridge, completed, request, response = repro.inputs(config)
    return config, completed, request, response


def test_registered_historical_request_and_response():
    _config, _completed, _request, response = fixed_inputs()
    compared = repro.compare_responses(response, response)
    assert all(row["elementwise_equal"] and row["max_absolute_difference"] == 0 for row in compared.values())


@pytest.mark.parametrize("change", ["future", "mask", "seed", "image", "video"])
def test_rejects_changed_or_privileged_request(change):
    _config, completed, request, _response = fixed_inputs()
    request = copy.deepcopy(request)
    if change == "future":
        request["future_state"] = [1]
    elif change == "mask":
        request["extra"]["action128"]["action_dim_mask"][10] = True
    elif change == "seed":
        request["seed"] = 43
    elif change == "image":
        request["vision_path"] += ".other"
    else:
        request["extra"]["action128"]["return_video"] = True
    with pytest.raises(ValueError):
        repro.validate_request(request, completed["image_ref"])


def test_output_validation_and_actual_difference():
    _config, _completed, _request, expected = fixed_inputs()
    actual = copy.deepcopy(expected)
    actual["outputs"][0]["content"]["action"][0][0] += 0.125
    compared = repro.compare_responses(actual, expected)
    assert compared["action"]["different_elements"] == 1
    assert compared["action"]["max_absolute_difference"] == pytest.approx(0.125)
    assert compared["normalized_action"]["elementwise_equal"]
    actual["outputs"][0]["content"]["action"][0][10] = 1
    with pytest.raises(ValueError):
        repro.response_arrays(actual)
    actual["outputs"][0]["content"]["action"][0][10] = 0
    actual["outputs"][0]["content"]["action"][0][0] = np.nan
    with pytest.raises(ValueError):
        repro.response_arrays(actual)


def test_manifest_detects_changed_bytes(tmp_path):
    source = tmp_path / "model"
    source.write_bytes(b"original")
    manifest = {str(source): repro.sha256(source)}
    repro.verify_manifest(manifest, workers=1)
    source.write_bytes(b"modified")
    with pytest.raises(ValueError, match="Manifest SHA256 mismatch"):
        repro.verify_manifest(manifest, workers=1)
