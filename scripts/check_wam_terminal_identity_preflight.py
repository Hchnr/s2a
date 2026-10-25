"""Exercise actual060 sampler MRO identity in the model interpreter, no model."""

import os
from pathlib import Path

from sample_wam_terminal_identity_state import ConversionWitness

from s2a.clean_training_runtime import publish, read, reference, require
from s2a.terminal_candidate import candidate_head
from s2a.wam_clean_capture import CleanCaptureBridge, json_sha


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    root = Path("outputs/terminal_identity_state_validation_20261025")
    prepared = read(reference(root / "prepared.json"))
    for path in (
        "scripts/sample_wam_terminal_identity_state.py",
        "src/s2a/terminal_capture_identity.py",
        "src/s2a/terminal_capture.py",
        "src/s2a/terminal_readout_policy.py",
    ):
        ref = reference(Path(path))
        require(prepared["source_sha256"].get(ref["path"]) == ref["sha256"], "Frozen actual identity source differs")
    policy, params_ref = candidate_head(prepared["candidate_proof_ref"])
    bridge = object.__new__(ConversionWitness)
    bridge.readout = policy
    fingerprint = bridge._readout_sha()
    require(
        isinstance(fingerprint, str) and len(fingerprint) == 64 and fingerprint == bridge._readout_sha(),
        "Actual sampler MRO policy fingerprint failed",
    )
    bridge.readout = policy.baseline
    require(
        bridge._readout_sha() == CleanCaptureBridge._readout_sha(bridge) == json_sha(policy.baseline.to_parameters()),
        "Original single head identity differs",
    )
    bridge.readout = None
    require(bridge._readout_sha() is None, "Original raw identity differs")
    import sys

    publish(
        root / "actual_identity_preflight.json",
        dict(
            passed=True,
            complete=True,
            prepared_ref=reference(root / "prepared.json"),
            params_ref=params_ref,
            interpreter=sys.executable,
            source_ref=reference(Path(__file__)),
            policy_fingerprint=fingerprint,
            mro=[c.__module__ + "." + c.__name__ for c in ConversionWitness.__mro__],
            new_model_initializations=0,
            new_model_calls=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
