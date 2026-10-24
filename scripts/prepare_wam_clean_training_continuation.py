"""041 metadata-only derivation, immutable proposal and independent admission.

The interrupted attempt stays at its original paths. This entrypoint cannot
launch models, decode future labels, fit parameters or initialize a simulator.
"""

import argparse
import copy
import json
from pathlib import Path

from prepare_wam_clean_training_execution import (
    DISK_REQUIRED,
    RAM_REQUIRED,
    Evidence,
    cpu_environment,
    normalization,
    observe_resources,
)

from s2a.artifacts import REPO
from s2a.clean_execution_proof import check_execution_proof, validate_envelope
from s2a.clean_training_continuation_bundle import BUNDLE_REL, NEW_ROOT_REL, publish_bundle
from s2a.clean_training_runtime import GPU, checked, publish, read, reference, require, sha
from s2a.clean_training_seed_plan import build_seed_plan

CONFIG = REPO / "configs/wam_clean_training_continuation_20261023.json"
PLAN = "plans/v0.0.1/041_clean_training_continuation_20261023.md"
STAGES = ("D0", "W1", "D1", "W2", "STATE", "DEVELOPMENT")
TIMEOUTS = dict(D0=88400, W1=15000, D1=88400, W2=15000, STATE=45500, DEVELOPMENT=175000)
AGGREGATE = dict(
    original_complete_requests=20,
    original_partial_requests=1,
    original_saved_denoise=725,
    original_denoise_lower_bound=725,
    original_denoise_upper_bound=735,
    original_model_initializations=1,
    new_request_attempts=1599,
    new_denoise=55965,
    new_model_initializations=4,
    total_request_attempts_upper_bound=1620,
    total_denoise_upper_bound=56700,
    total_model_initializations_upper_bound=5,
    producer_fits=2,
    independent_qr_rebuilds=2,
    physics_steps=7200,
)
EXTRA_SOURCES = (
    "configs/wam_clean_training_20261023.json",
    "plans/v0.0.1/038_clean_residual_training_20261023.md",
    "configs/wam_clean_training_execution_20261023.json",
    "plans/v0.0.1/040_clean_training_execution_20261023.md",
    "configs/wam_clean_training_continuation_20261023.json",
    PLAN,
    "pyproject.toml",
)


def config_contract(config):
    require(
        config["schema_version"] == "s2a.clean-training-continuation-config.v1"
        and config["seed"] == 42
        and config["output_dir"] == NEW_ROOT_REL
        and config["plan_path"] == PLAN,
        "041 config, seed, isolated root or plan differs",
    )
    require(
        config["aggregate_budget"] == AGGREGATE
        and config["stage_timeouts_s"] == TIMEOUTS
        and config["old_data_used_for_fit"]
        is config["automatic_retry"]
        is config["fresh_performance_authorized"]
        is False,
        "single new attempt or explicitly cumulative budget differs",
    )
    require(
        config["gpu_uuid"] == GPU
        and config["disk_required_bytes"] == DISK_REQUIRED
        and config["ram_required_bytes"] == RAM_REQUIRED
        and config["master_port"] == 31338
        and config["development_port"] == 31339,
        "resource, UUID or port contract differs",
    )
    require(
        len(config["required_test_paths"]) == len(set(config["required_test_paths"])) == 23
        and all((REPO / path).is_file() for path in config["required_test_paths"]),
        "complete related23 test modules required",
    )


def continuation_contract(config):
    return dict(
        protocol="041",
        original_execution_ref=config["original_execution_ref"],
        incident_ref=config["incident_ref"],
        old_data_used_for_fit=False,
        one_new_complete_attempt=True,
        automatic_retry=False,
    )


def derive_prepared(original):
    value = copy.deepcopy(original)
    require(
        original["root"] == str(REPO / "outputs/wam_stability_20261021/clean_residual_training_20261023")
        and original["execution_authorized"] is False
        and original["readiness"]["ready_for_execution"] is False,
        "root-only derivation requires the unchanged original metadata proposal",
    )
    value["root"] = str(REPO / NEW_ROOT_REL)
    return value


def original_evidence(config, evidence):
    original = evidence.json(config["original_prepared_ref"])
    registration = evidence.json(config["original_execution_ref"])
    check_execution_proof(
        registration,
        prepared_ref=config["original_prepared_ref"],
        read_json=evidence.json,
        check_file=evidence.check,
    )
    require(
        registration["prepared_ref"] == config["original_prepared_ref"]
        and registration["metadata_verification_ref"] == config["original_metadata_verification_ref"]
        and registration["execution_verification_ref"] == config["original_execution_verification_ref"]
        and registration["development_control_ref"] == config["development_control_ref"],
        "original metadata, execution or control chain differs",
    )
    metadata = evidence.json(config["original_metadata_verification_ref"])
    require(
        metadata["complete"] is metadata["passed"] is metadata["metadata_qualified"] is True
        and metadata["execution_ready"] is metadata["execution_authorized"] is False,
        "original metadata qualification missing",
    )
    for record in (original, registration):
        for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
            evidence.mapping(record[key])
    evidence.mapping(metadata["consumed_sha256"])
    inventory = evidence.json(config["interrupted_inventory_ref"])
    require(
        inventory["file_count"] == len(inventory["files"]) == 1661
        and inventory["old_registration_ref"] == config["original_execution_ref"]
        and inventory["source_changes"] == inventory["snapshot_changes"] == [],
        "immutable interrupted-file inventory differs",
    )
    for item in inventory["files"]:
        path = evidence.check(dict(path=item["path"], sha256=item["sha256"]))
        require(path.stat().st_size == item["bytes"], "interrupted artifact size changed")
    incident = evidence.json(config["incident_ref"])
    require(
        incident["audit_complete"] is incident["audit_consistency_passed"] is True
        and incident["original_stage_complete"] is incident["original_stage_passed"] is False
        and incident["capture_summary"]["complete_ledgers"] == 20,
        "original incomplete attempt is not independently preserved",
    )
    design = evidence.json(config["recovery_design_ref"])
    require(
        design["complete"] is design["passed"] is True
        and design["execution_authorized"] is False
        and design["new_protocol_required"] is True,
        "new bounded protocol/persistence review missing",
    )
    for path, expected in original["weight_files"].items():
        stat = Path(path).stat()
        require(stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"], "weight stat drift")
    return original, registration


def metadata(config_ref):
    cpu_environment()
    evidence = Evidence()
    config = evidence.json(config_ref)
    config_contract(config)
    original, registration = original_evidence(config, evidence)
    root = REPO / NEW_ROOT_REL
    require(not root.exists(), "refuses existing041 attempt, overwritten history or retry")
    derived = derive_prepared(original)
    evidence.finish()
    root.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema_version="s2a.clean-training-continuation-metadata-preparation.v1",
        complete=False,
        passed=False,
        config_ref=config_ref,
        original_prepared_ref=config["original_prepared_ref"],
        new_model_calls=0,
        new_fits=0,
        physics_steps=0,
        future_archive_decodes=0,
        execution_authorized=False,
    )
    try:
        prepared_ref = publish(root / "preparation/prepared.json", derived)
        manifest = publish_bundle(registration, output_root=root, old_admitted_ref=config["original_execution_ref"])
        bundle_ref = publish(root / "source/bundle_manifest.json", manifest)
        evidence.finish()
        result.update(complete=True, passed=True, prepared_ref=prepared_ref, bundle_manifest_ref=bundle_ref)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        publish(root / "metadata_preparation_result.json", result)
    return result


def source_paths(bundle):
    paths = {path.resolve() for name in ("src", "scripts", "tests") for path in (REPO / name).rglob("*.py")}
    paths.update(REPO / name for name in EXTRA_SOURCES)
    paths.update(Path(path) for path in bundle["source_sha256"])
    require(
        len(bundle["source_sha256"]) == 11 and all(path.is_file() and path.is_relative_to(REPO) for path in paths),
        "complete main/qualified bundle source closure missing or escaped",
    )
    return sorted(paths)


def freeze_sources(execution, values):
    snapshots, refs = {}, {}
    bundle_root = REPO / BUNDLE_REL
    for filename, expected in values.items():
        source = Path(filename)
        require(source.is_relative_to(REPO) and sha(source) == expected, "source changed before snapshot")
        if source.is_relative_to(bundle_root):
            target = execution / "source/bundle" / source.relative_to(bundle_root)
        else:
            require(not source.is_relative_to(REPO / "outputs"), "only qualified bundle code may come from outputs")
            target = execution / "source/repo" / source.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        require(sha(target) == sha(source) == expected, "source changed during snapshot")
        snapshots[str(target)] = expected
        refs[filename] = reference(target)
    return snapshots, refs


def supervisor_contract():
    commands = {}
    for stage in STAGES:
        basename = (
            "run_wam_clean_training.py"
            if stage in ("D0", "W1", "D1", "W2")
            else "run_wam_clean_training_state.py"
            if stage == "STATE"
            else "run_wam_clean_training_development.py"
        )
        commands[stage] = dict(
            entrypoint=f"{BUNDLE_REL}/{basename}",
            arguments=["--stage", stage] if stage in ("D0", "W1", "D1", "W2") else [],
        )
    return dict(
        python=str(REPO / ".venv/bin/python"),
        stage_order=list(STAGES),
        stage_commands=commands,
        stage_timeouts_s=TIMEOUTS.copy(),
        result_paths={stage: f"orchestration/{stage}/result.json" for stage in STAGES},
    )


def stage_entrypoints():
    return dict(
        D0=f"{BUNDLE_REL}/collect_wam_clean_training.py",
        W1="scripts/fit_wam_clean_training.py",
        D1=f"{BUNDLE_REL}/collect_wam_clean_training.py",
        W2="scripts/fit_wam_clean_training.py",
        STATE=f"{BUNDLE_REL}/sample_wam_clean_training_state.py",
        DEVELOPMENT=f"{BUNDLE_REL}/run_wam_clean_training_development.py",
    )


def propose(config_ref, qa_ref):
    cpu_environment()
    evidence = Evidence()
    config = evidence.json(config_ref)
    config_contract(config)
    original, old_registration = original_evidence(config, evidence)
    root = REPO / NEW_ROOT_REL
    prepared_ref = reference(root / "preparation/prepared.json")
    prepared = evidence.json(prepared_ref)
    require(prepared == derive_prepared(original), "prepared is not the exact root-only derivation")
    metadata_ref = reference(root / "metadata_verified.json")
    verified = evidence.json(metadata_ref)
    require(
        verified["schema_version"] == "s2a.clean-training-preparation-independent.v1"
        and verified["prepared_ref"] == prepared_ref
        and verified["original_prepared_ref"] == config["original_prepared_ref"]
        and verified["original_metadata_verification_ref"] == config["original_metadata_verification_ref"]
        and verified["complete"] is verified["passed"] is verified["metadata_qualified"] is True
        and verified["execution_ready"] is verified["execution_authorized"] is False,
        "independent041 metadata derivation did not pass",
    )
    evidence.mapping(verified["consumed_sha256"])
    evidence.check(verified["verifier_ref"])
    bundle_ref = verified["bundle_manifest_ref"]
    bundle = evidence.json(bundle_ref)
    evidence.mapping(bundle["source_sha256"])
    paths = source_paths(bundle)
    sources = {str(path): sha(path) for path in paths}
    require(
        sources.get(verified["verifier_ref"]["path"]) == verified["verifier_ref"]["sha256"],
        "metadata verifier changed after its independent result",
    )
    qa = evidence.json(qa_ref)
    require(
        qa["complete"] is qa["passed"] is True
        and qa["exit_code"] == 0
        and qa["source_sha256"] == sources
        and qa["test_paths"] == config["required_test_paths"]
        and qa["tests_passed"] > 0,
        "final-source related23 QA is stale, incomplete or failed",
    )
    for command in qa["commands"]:
        require(command["exit_code"] == 0, "final-source QA command failed")
        evidence.check(command["log_ref"])
    bridge = evidence.json(original["prerequisite_refs"]["bridge_prepared"])
    config_yaml = next(
        dict(path=path, sha256=digest)
        for path, digest in bridge["checkpoint_small_metadata_sha256"].items()
        if Path(path).name == "config.yaml"
    )
    normal = normalization(config_yaml, evidence)
    plan = build_seed_plan(prepared_ref, prepared)
    old_seed_plan = evidence.json(old_registration["training_seed_plan_ref"])
    require(
        dict(plan, prepared_ref=old_seed_plan["prepared_ref"]) == old_seed_plan,
        "actual training seeds changed instead of only their prepared reference",
    )
    execution = root / "execution"
    require(not execution.exists(), "refuses existing041 proposal/attempt or retry")
    evidence.finish()
    execution.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema_version="s2a.clean-training-continuation-proposal-attempt.v1",
        complete=False,
        passed=False,
        config_ref=config_ref,
        qa_ref=qa_ref,
        execution_authorized=False,
        new_model_calls=0,
        new_fits=0,
        physics_steps=0,
    )
    try:
        resource_ref = publish(execution / "resources.json", observe_resources(root))
        normal_ref = publish(execution / "normalization.json", normal)
        seed_ref = publish(execution / "seed_plan.json", plan)
        for ref in (resource_ref, normal_ref, seed_ref):
            evidence.check(ref)
        snapshots, snapshot_refs = freeze_sources(execution, sources)
        stages = stage_entrypoints()
        proposal = dict(
            schema_version="s2a.clean-training-execution-proposal.v1",
            complete=True,
            passed=True,
            readiness=dict(ready_for_execution=False, reason="independent041 execution verification required"),
            execution_authorized=False,
            automatic_retry=False,
            fresh_performance_authorized=False,
            root=str(root),
            config_ref=config_ref,
            plan_ref=reference(REPO / PLAN),
            prepared_ref=prepared_ref,
            metadata_verification_ref=metadata_ref,
            development_control_ref=config["development_control_ref"],
            original_execution_ref=config["original_execution_ref"],
            incident_ref=config["incident_ref"],
            continuation_contract=continuation_contract(config),
            aggregate_budget=config["aggregate_budget"],
            bundle_manifest_ref=bundle_ref,
            supervisor=supervisor_contract(),
            qa_ref=qa_ref,
            source_sha256=sources,
            snapshot_sha256=snapshots,
            snapshot_refs=snapshot_refs,
            input_sha256=evidence.inputs,
            weight_files=prepared["weight_files"],
            resource_observation_ref=resource_ref,
            normalization_ref=normal_ref,
            training_seed_plan_ref=seed_ref,
            sampler_trace_ref=old_registration["sampler_trace_ref"],
            resources=prepared["resources"],
            budget=prepared["budget"],
            master_port=31338,
            stage_entrypoints=stages,
            entrypoint_refs={stage: reference(REPO / path) for stage, path in stages.items()},
            development=old_registration["development"],
            new_model_calls=0,
            new_fits=0,
            physics_steps=0,
            fresh_cases=0,
            future_archive_decodes=0,
        )
        evidence.finish()
        require(
            sources == {str(path): sha(path) for path in source_paths(bundle)},
            "source inventory or bytes changed during proposal preparation",
        )
        proposal_ref = publish(execution / "proposal.json", proposal)
        result.update(complete=True, passed=True, proposal_ref=proposal_ref)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        publish(execution / "preparation_result.json", result)
    return result


def admit(proposal_ref, verification_ref):
    cpu_environment()
    proposal, verified = read(proposal_ref), read(verification_ref)
    execution = REPO / NEW_ROOT_REL / "execution"
    require(
        Path(proposal_ref["path"]) == execution / "proposal.json"
        and Path(verification_ref["path"]) == execution / "verified.json",
        "fixed041 proposal and independent verification paths required",
    )
    value = copy.deepcopy(proposal)
    value.update(
        schema_version="s2a.clean-training-execution-registration.v1",
        execution_authorized=True,
        proposal_ref=proposal_ref,
        execution_verification_ref=verification_ref,
    )
    value["readiness"]["ready_for_execution"] = True
    checked(validate_envelope(value, proposal, verified, prepared_ref=proposal["prepared_ref"]))
    for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
        for path, digest in proposal[key].items():
            checked(dict(path=path, sha256=digest))
    return publish(execution / "admitted.json", value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("metadata", "propose", "admit"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    for name in ("qa", "proposal", "verification"):
        parser.add_argument(f"--{name}", type=Path)
        parser.add_argument(f"--{name}-sha256")
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    args = parser.parse_args()

    def ref(name):
        path, digest = getattr(args, name), getattr(args, name + "_sha256")
        require(path is not None and digest is not None, "stage requires exact path/SHA arguments")
        return dict(path=str(path.resolve()), sha256=digest)

    result = (
        metadata(reference(args.config))
        if args.stage == "metadata"
        else propose(reference(args.config), ref("qa"))
        if args.stage == "propose"
        else admit(ref("proposal"), ref("verification"))
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
