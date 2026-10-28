"""Own exact-prefix failure capture with the original native CPU environment."""

import json
from pathlib import Path

from run_tracking_preview_wam_development_v3 import cpu_environment
from run_wam_readout import owned_command

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require


def main():
    root = REPO / "outputs/tracking_wam_qp_capture_execution_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    prior_ref = reference(REPO / "outputs/tracking_preview_wam_development_v3_20261027/registration.json")
    prior = read(prior_ref)
    prepared = read(prior["prepared_ref"])
    sources = dict(prior["source_sha256"])
    for path in [Path(__file__), REPO / "scripts/capture_tracking_wam_qp_failure.py"]:
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    reg = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            prior_registration_ref=prior_ref,
            physics_budget=210,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, registration_ref=reg)
    try:
        command = owned_command(
            [str(REPO / "outputs/libero_env311/bin/python"), "scripts/capture_tracking_wam_qp_failure.py"],
            cpu_environment(prepared),
            root / "capture",
            timeout=900,
        )
        ref = reference(REPO / "outputs/tracking_wam_qp_failure_capture_20261027/result.json")
        require(read(ref)["complete"] is read(ref)["passed"] is True, "Exact failure capture")
        for path, digest in sources.items():
            checked(dict(path=path, sha256=digest))
        result.update(complete=True, passed=True, owned_command=command, result_ref=ref)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
