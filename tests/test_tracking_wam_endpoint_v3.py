"""Reject endpoint drift at the physical entry before any model or simulator call."""

import pytest

from s2a.clean_training_runtime import GPU
from s2a.tracking_preview_wam_control_v3 import PORT, validate_checkpoint_endpoint
from s2a.tracking_preview_wam_simulation_v3 import run_tracking_wam_case


def config():
    return dict(checkpoints=[dict(name="8400_masked", gpu_uuid=GPU, port=PORT)])


def test_registered_endpoint_matches_shared_profile_contract():
    value = config()
    assert validate_checkpoint_endpoint(value, "8400_masked") is value["checkpoints"][0]


@pytest.mark.parametrize("fault", ["old_port", "other_gpu", "duplicate", "missing"])
def test_actual_physical_entry_rejects_endpoint_before_prerequisite_or_io(fault):
    value = config()
    if fault == "old_port":
        value["checkpoints"][0]["port"] = 31365
    elif fault == "other_gpu":
        value["checkpoints"][0]["gpu_uuid"] = "forbidden"
    elif fault == "duplicate":
        value["checkpoints"] *= 2
    else:
        value["checkpoints"] = []
    # No tracking_admission, output directory, parameters, HTTP server or env
    # is supplied: the real entry must reject the endpoint before reading any.
    with pytest.raises(ValueError, match="registered tracking|Registered tracking"):
        run_tracking_wam_case(
            value,
            None,
            {},
            source="wam",
            variant_name="position_hold",
            sampling_seed=42,
            checkpoint="8400_masked",
            config_path=None,
            oracle_reference=None,
            endpoint_observer=lambda *args: None,
            response_parameters={},
        )
