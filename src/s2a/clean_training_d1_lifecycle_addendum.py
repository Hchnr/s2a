"""050 read-only qualification of complete D1 despite the frozen046 field bug."""

from pathlib import Path

from s2a.clean_training_d1_admission import ROOT as CONTROL_ROOT
from s2a.clean_training_d1_admission import admit as admit046
from s2a.clean_training_w1_handoff import PREPARED_ROOT
from s2a.durable_stage_supervisor import REPO, check_sources, checked, read, reference, require
from s2a.owned_receipt_contract import d1_postcheck_failure, normal_receipts

ROOT = REPO / "outputs/clean_training_d1_lifecycle_addendum_20261024"
D1_REF = dict(
    path=str(CONTROL_ROOT / "execution/admitted.json"),
    sha256="76d8751d0213e585b9660b704556dff4c4ae7849aa80c7c067dbdcb694749d49",
)
SCHEMA = "s2a.clean-training-d1-lifecycle-addendum.v1"
DATA_PATHS = dict(
    d1_registration_ref=CONTROL_ROOT / "execution/admitted.json",
    d1_handoff_owner_ref=CONTROL_ROOT / "supervisor/result.json",
    d1_wrapper_owner_ref=CONTROL_ROOT / "supervisor/D1/result.json",
    d1_worker_ref=CONTROL_ROOT / "worker/result.json",
    d1_owner_ref=PREPARED_ROOT / "orchestration/D1/result.json",
    d1_collection_ref=PREPARED_ROOT / "capture/D1/result.json",
    d1_capture_audit_ref=PREPARED_ROOT / "review/D1_verified.json",
)


class Evidence:
    def __init__(self):
        self.consumed = {}

    def file(self, ref):
        path = Path(ref["path"])
        old = Path("/share/project/eai_pwm/home/hcr/repos/s2a")
        require(
            path.is_absolute() and not path.is_relative_to(old) and not path.resolve().is_relative_to(old),
            "forbidden old workspace is not an input",
        )
        actual = checked(ref)
        require(actual.resolve() == actual, "canonical evidence path required")
        prior = self.consumed.setdefault(str(actual), ref["sha256"])
        require(prior == ref["sha256"], "conflicting evidence SHA")
        return actual

    def json(self, ref):
        require(self.file(ref).suffix == ".json", "metadata reader refuses array decode")
        return read(ref)

    def mapping(self, values):
        for path, digest in values.items():
            self.file(dict(path=path, sha256=digest))

    def finish(self):
        check_sources(self.consumed)


def complete(value, schema):
    require(
        value.get("schema_version") == schema and value.get("complete") is value.get("passed") is True,
        "050 required original result incomplete",
    )


def core(spec, e, *, hash_inputs):
    require(spec["d1_registration_ref"] == D1_REF, "050 exact046 registration required")
    registered = admit046(D1_REF)
    for key, path in DATA_PATHS.items():
        require(spec[key] == reference(path), "050 fixed original artifact differs: " + key)
    outer = e.json(spec["d1_handoff_owner_ref"])
    require(
        outer["complete"] is outer["passed"] is False
        and outer["registration_ref"] == D1_REF
        and outer["error"]["type"] == "ValueError"
        and str(CONTROL_ROOT / "supervisor/D1/result.json") in outer["error"]["message"],
        "050 original supervisor failure was changed",
    )
    require(outer["failed_owned_wrapper_ref"] == spec["d1_wrapper_owner_ref"], "050 failed-wrapper binding differs")
    owned = e.json(spec["d1_wrapper_owner_ref"])
    require(owned["schema_version"] == "s2a.durable-stage-owner.v1", "050 durable wrapper schema differs")
    wrapper_launch = d1_postcheck_failure(e, owned)
    require(
        wrapper_launch["command"]
        == [
            str(REPO / ".venv/bin/python"),
            "scripts/execute_wam_clean_training_d1_handoff.py",
            "--registration",
            D1_REF["path"],
            "--registration-sha256",
            D1_REF["sha256"],
            "--seed",
            "42",
        ],
        "050 wrapper command differs",
    )
    cleanup = e.json(owned["descendant_cleanup_ref"])
    require(
        cleanup["all_recorded_descendants_reaped"] is True and cleanup["observer_signals"] == 0,
        "050 recorded descendants not released",
    )
    worker = e.json(spec["d1_worker_ref"])
    complete(worker, "s2a.clean-training-d1-handoff-worker.v1")
    require(
        worker["registration_ref"] == D1_REF
        and worker["delegate_calls_attempted"] == 1
        and worker["delegate_result_ref"] == spec["d1_owner_ref"]
        and worker["collection_ref"] == spec["d1_collection_ref"]
        and worker["verification_ref"] == spec["d1_capture_audit_ref"]
        and worker["source_before_after_exact"] is True
        and worker["handoff"]["w1_lookups"] == 1
        and worker["handoff"]["restored"] is worker["handoff"]["source_guard_completed"] is True,
        "050 original wrapper provenance differs",
    )
    original = e.json(spec["d1_owner_ref"])
    complete(original, "s2a.clean-training-owner.v1")
    require(
        original["stage"] == "D1"
        and original["prepared_ref"] == registered["prepared_ref"]
        and original["execution_registration_ref"] == registered["execution_registration_ref"]
        and original["producer_ref"] == spec["d1_collection_ref"]
        and original["verification_ref"] == spec["d1_capture_audit_ref"]
        and len(original["owned_commands"]) == 2,
        "050 original D1 owner differs",
    )
    prepared = e.json(registered["prepared_ref"])
    for index, command in enumerate(original["owned_commands"]):
        launch = normal_receipts(e, command["launch"], command["exit"], command["release"], format="legacy")
        expected = registered["collector_ref"] if index == 0 else registered["capture_verifier_ref"]
        python = prepared["model_config"]["model_python"] if index == 0 else str(REPO / ".venv/bin/python")
        require(
            launch["command"][0] == python and Path(launch["command"][1]).resolve() == Path(expected["path"]),
            "050 original model/audit command differs",
        )
    require(
        original["owned_commands"][0]["launch"] != original["owned_commands"][1]["launch"],
        "050 model/audit reused a launch",
    )
    gpu = e.json(original["gpu_after_ref"])
    require(
        gpu["released"] is True
        and gpu["uuid"] == registered["gpu_uuid"]
        and gpu["memory_mib"] == 0
        and gpu["compute_apps"] == []
        and gpu["device_minor"] == 6
        and gpu["visible_gpu"] == 2,
        "050 GPU normal release missing",
    )
    capture = e.json(spec["d1_collection_ref"])
    audit = e.json(spec["d1_capture_audit_ref"])
    complete(capture, "s2a.clean-training-capture.v1")
    complete(audit, "s2a.clean-training-capture-audit.v1")
    for value in (capture, audit):
        require(
            value["round_name"] == "D1"
            and value["prepared_ref"] == registered["prepared_ref"]
            and value["execution_registration_ref"] == registered["execution_registration_ref"]
            and value["d0_audit_ref"] == registered["d0_capture_audit_ref"]
            and value["actual_requests"] == 300
            and value["actual_denoise_calls"] == 10500
            and all(value[k] == 0 for k in ("future_label_reads", "producer_fits", "physics_steps", "fresh_cases")),
            "050 full D1 numerical/causal boundary differs",
        )
    require(
        capture["actual_head_calls"] == 10500
        and capture["model_initialization_attempts"] == capture["model_initializations"] == 1
        and capture["w1_fit_ref"] == registered["w1_fit_ref"]
        and capture["w1_verification_ref"] == registered["w1_verification_ref"]
        and capture["readout_ref"] == registered["w1_params_ref"]
        and audit["actual_online_points_verified"] == 10500
        and audit["collection_ref"] == spec["d1_collection_ref"],
        "050 D1 W1/independent point qualification differs",
    )
    require(
        all(
            audit[key] == original["owned_commands"][0][entry]
            for key, entry in (
                ("worker_launch_ref", "launch"),
                ("worker_exit_ref", "exit"),
                ("worker_release_ref", "release"),
            )
        )
        and audit["gpu_after_ref"] == original["gpu_after_ref"],
        "050 audit lifecycle refs differ",
    )
    fault = REPO / "scripts/run_wam_clean_training_d1_handoff.py"
    require(
        reference(fault)["sha256"] == registered["source_sha256"][str(fault)]
        and 'released["actual_wait"]' in fault.read_text(),
        "050 exact frozen post-check bug source differs",
    )
    e.mapping(registered["source_sha256"])
    e.mapping(registered["snapshot_sha256"])
    if hash_inputs:
        e.mapping(audit["input_sha256"])
    e.finish()
    return registered


def verify(manifest_ref):
    e = Evidence()
    m = e.json(manifest_ref)
    require(
        m["schema_version"] == "s2a.d1-lifecycle-addendum-manifest.v1" and m["read_only"] is True,
        "050 fixed read-only manifest required",
    )
    e.mapping(m["source_sha256"])
    e.mapping(m["snapshot_sha256"])
    require(
        m["source_sha256"].get(str(REPO / "src/s2a/clean_training_d1_lifecycle_addendum.py"))
        == reference(REPO / "src/s2a/clean_training_d1_lifecycle_addendum.py")["sha256"],
        "050 actual verifier source is not frozen",
    )
    registered = core(m["data_refs"], e, hash_inputs=True)
    e.finish()
    return dict(
        schema_version=SCHEMA,
        complete=True,
        passed=True,
        eligible_for_w2=True,
        manifest_ref=manifest_ref,
        verifier_ref=reference(REPO / "src/s2a/clean_training_d1_lifecycle_addendum.py"),
        data_refs=m["data_refs"],
        source_sha256=m["source_sha256"],
        snapshot_sha256=m["snapshot_sha256"],
        original_supervisor_false_preserved=True,
        postcheck_error_only="KeyError('actual_wait')",
        original_numerical_qualification_passed=True,
        full_capture_input_fingerprints_current=True,
        source_before_after_exact=True,
        d1_requests=300,
        d1_nodes=10500,
        d1_head_calls=10500,
        source_base_count=len(registered["source_sha256"]),
        legacy_wait_from_exit=True,
        legacy_reap_from_release=True,
        new_model_calls=0,
        new_denoise_calls=0,
        new_fit_calls=0,
        new_qr_calls=0,
        new_physics_steps=0,
        new_fresh_cases=0,
        array_decodes=0,
        consumed_sha256=e.consumed,
    )


def read_qualified(spec, e=None):
    e = Evidence() if e is None else e
    require("d1_lifecycle_addendum_ref" in spec, "050 D1 lifecycle addendum is required")
    ref = spec["d1_lifecycle_addendum_ref"]
    require(ref == reference(ROOT / "result.json"), "050 exact addendum result required")
    result = e.json(ref)
    complete(result, SCHEMA)
    require(
        result["eligible_for_w2"]
        is result["original_supervisor_false_preserved"]
        is result["original_numerical_qualification_passed"]
        is result["full_capture_input_fingerprints_current"]
        is result["source_before_after_exact"]
        is True
        and result["postcheck_error_only"] == "KeyError('actual_wait')"
        and result["d1_requests"] == 300
        and result["d1_nodes"] == 10500
        and result["array_decodes"] == 0,
        "050 result does not qualify the exact original data",
    )
    m = e.json(result["manifest_ref"])
    require(
        m["data_refs"] == result["data_refs"]
        and m["source_sha256"] == result["source_sha256"]
        and m["snapshot_sha256"] == result["snapshot_sha256"],
        "050 manifest/result envelope differs",
    )
    e.mapping(m["source_sha256"])
    e.mapping(m["snapshot_sha256"])
    require(
        result["verifier_ref"] == reference(REPO / "src/s2a/clean_training_d1_lifecycle_addendum.py"),
        "050 actual source differs",
    )
    require(all(spec[key] == ref for key, ref in result["data_refs"].items()), "050 consumed D1 references differ")
    registered = core(spec, e, hash_inputs=False)
    for key in (
        "prepared_ref",
        "execution_registration_ref",
        "w1_owner_ref",
        "w1_params_ref",
        "d0_owner_ref",
        "d0_collection_ref",
        "d0_capture_audit_ref",
    ):
        require(spec[key] == registered[key], "050 original D0/W1 handoff differs")
    return registered
