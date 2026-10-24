"""Adversarial040 admission metadata tests; no model or archive decoding."""

import ast
import copy
import importlib.util
import json
import os
import subprocess
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_wam_clean_training_execution.py"
SPEC = importlib.util.spec_from_file_location("independent_execution", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=audit.sha(path))


@pytest.fixture
def base():
    prepared = dict(
        schema_version="s2a.clean-training-preparation.v1",
        budget=copy.deepcopy(audit.BUDGET),
        resources=copy.deepcopy(audit.RESOURCES),
        execution_authorized=False,
        readiness=dict(ready_for_execution=False),
        config=dict(gpu_uuid=audit.GPU),
    )
    proposal = dict(
        schema_version="s2a.clean-training-execution-proposal.v1",
        complete=True,
        passed=True,
        readiness=dict(ready_for_execution=False),
        execution_authorized=False,
        automatic_retry=False,
        fresh_performance_authorized=False,
        budget=copy.deepcopy(audit.BUDGET),
        resources=copy.deepcopy(audit.RESOURCES),
        master_port=31338,
        development=dict(port=31339, case_order=audit.case_order()),
        stage_entrypoints=audit.STAGES.copy(),
    )
    return proposal, prepared


def test_fixed_full_budget_and_nonadmitted_proposal(base):
    audit.proposal_contract(*base)
    assert len(base[0]["development"]["case_order"]) == 12
    assert [case["variant"] for case in audit.case_order()] == ["raw", "w2"] * 6


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(schema_version="s2a.clean-training-execution-registration.v1"),
        lambda p: p["readiness"].update(ready_for_execution=True),
        lambda p: p.update(execution_authorized=True),
        lambda p: p.update(automatic_retry=True),
        lambda p: p.update(fresh_performance_authorized=True),
        lambda p: p["budget"].update(total_denoise=55964),
        lambda p: p["resources"].update(d0_timeout_s=7200),
        lambda p: p["stage_entrypoints"].update(STATE="scripts/run_wam_clean_training_state.py"),
        lambda p: p["development"]["case_order"].reverse(),
        lambda p: p["development"]["case_order"][0].update(expected_requests=74),
        lambda p: p.update(master_port=31335),
    ],
)
def test_proposal_refuses_budget_stage_order_or_permission_change(base, mutation):
    mutation(base[0])
    with pytest.raises(ValueError):
        audit.proposal_contract(*base)


def test_preparation_cannot_promote_itself(base):
    base[1]["readiness"]["ready_for_execution"] = True
    with pytest.raises(ValueError, match="metadata preparation"):
        audit.proposal_contract(*base)


def test_evidence_hashes_opaque_future_without_parsing_and_detects_change(tmp_path):
    path = tmp_path / "opaque.npz"
    path.write_bytes(b"not-a-valid-archive")
    ref = dict(path=str(path), sha256=audit.sha(path))
    evidence = audit.Evidence()
    assert evidence.check(ref) == path
    with pytest.raises(ValueError, match="opaque"):
        evidence.json(ref)
    path.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="changed"):
        evidence.unchanged()


def test_evidence_rejects_invalid_sha_and_json_nonfinite(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text('{"unsafe":NaN}')
    evidence = audit.Evidence()
    with pytest.raises(ValueError, match="SHA256"):
        evidence.check(dict(path=str(path), sha256="1"))
    with pytest.raises(ValueError, match="NaN"):
        evidence.json(dict(path=str(path), sha256=audit.sha(path)))


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REPO", tmp_path)
    monkeypatch.setattr(audit, "REQUIRED_SOURCES", ("scripts/check.py", "tests/test_check.py"))
    monkeypatch.setattr(audit, "EXTRA_SOURCES", ())
    monkeypatch.setattr(audit, "STAGES", dict(D0="scripts/check.py"))
    monkeypatch.setattr(audit, "VERIFIER", tmp_path / "scripts/check.py")
    mapping, snapshots, snapshot_refs = {}, {}, {}
    for name in audit.REQUIRED_SOURCES:
        source, target = tmp_path / name, tmp_path / "outputs/execution/source" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# source " + name)
        target.write_bytes(source.read_bytes())
        digest = audit.sha(source)
        mapping[str(source)] = digest
        snapshots[str(target)] = digest
        snapshot_refs[str(source)] = dict(path=str(target), sha256=digest)
    input_ref = write(tmp_path / "input.json", dict(passed=True))
    prepared = dict(
        root=str(tmp_path / "outputs"),
        source_sha256=mapping.copy(),
        snapshot_sha256=snapshots.copy(),
        input_sha256={input_ref["path"]: input_ref["sha256"]},
        qualified_model_source_sha256=mapping.copy(),
    )
    proposal = dict(
        source_sha256=mapping.copy(),
        snapshot_sha256=snapshots.copy(),
        input_sha256=prepared["input_sha256"].copy(),
        snapshot_refs=snapshot_refs,
        entrypoint_refs=dict(D0=dict(path=str(audit.VERIFIER), sha256=audit.sha(audit.VERIFIER))),
    )
    return proposal, prepared


def test_source_and_snapshot_bind_each_actual_source(sources):
    counts = audit.sources(*sources, audit.Evidence())
    assert counts["source_snapshots"] == counts["final_sources"] == 2


def test_unfrozen_new_python_file_cannot_be_omitted_after_qa(sources):
    (audit.REPO / "scripts/added_after_qa.py").write_text("# executable added after source inventory")
    with pytest.raises(ValueError, match="complete src/scripts/tests"):
        audit.sources(*sources, audit.Evidence())


def test_execution_config_is_constant_metadata_and_binds_exact_refs(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REPO", tmp_path)
    config = dict(
        schema_version="s2a.clean-training-execution-config.v1",
        seed=42,
        output_dir="outputs/wam_stability_20261021/clean_residual_training_20261023/execution",
        plan_path="plans/v0.0.1/040_clean_training_execution_20261023.md",
        prepared_ref=dict(path="/prepared.json", sha256="a" * 64),
        metadata_verification_ref=dict(path="/metadata.json", sha256="b" * 64),
        development_control_ref=dict(path="/control.json", sha256="c" * 64),
        disk_required_bytes=audit.DISK_REQUIRED,
        ram_required_bytes=audit.RAM_REQUIRED,
        gpu_uuid=audit.GPU,
        master_port=31338,
        development_port=31339,
        automatic_retry=False,
        fresh_performance_authorized=False,
        required_test_paths=list(audit.REQUIRED_TESTS),
    )
    config_ref = write(tmp_path / "configs/wam_clean_training_execution_20261023.json", config)
    proposal = {key: config[key] for key in ("prepared_ref", "metadata_verification_ref", "development_control_ref")}
    proposal.update(config_ref=config_ref, source_sha256={config_ref["path"]: config_ref["sha256"]})
    prepared = dict(root=str(tmp_path / "outputs/wam_stability_20261021/clean_residual_training_20261023"))
    assert audit.execution_config(proposal, prepared, audit.Evidence())["execution_config_exact"] is True
    proposal["prepared_ref"] = dict(path="/replacement.json", sha256="d" * 64)
    with pytest.raises(ValueError, match="constant execution config"):
        audit.execution_config(proposal, prepared, audit.Evidence())


@pytest.mark.parametrize("problem", ["missing", "swapped", "unmapped", "changed"])
def test_source_closure_refuses_missing_swapped_unmapped_or_modified_source(sources, problem):
    proposal, prepared = sources
    paths = list(proposal["source_sha256"])
    if problem == "missing":
        proposal["source_sha256"].pop(paths[1])
    elif problem == "swapped":
        proposal["snapshot_refs"][paths[0]] = proposal["snapshot_refs"][paths[1]]
    elif problem == "unmapped":
        proposal["snapshot_sha256"] = {key: value for key, value in list(proposal["snapshot_sha256"].items())[:1]}
    else:
        Path(paths[0]).write_text("changed executable after QA")
    with pytest.raises(ValueError):
        audit.sources(proposal, prepared, audit.Evidence())


@pytest.fixture
def normalize(tmp_path):
    transform = dict(
        normalization_mask=[True] * 3 + [False] * 125, scale=[1.25] * 128, bias=[0.1] * 128, clip_min=-1.0, clip_max=1.0
    )
    spec = dict(
        registry=dict(domains=[audit.DOMAIN] * 67, domain_projection_ids=[0] * 67),
        transforms=[copy.deepcopy(transform) for _ in range(67)],
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(dict(model=dict(config=dict(action128_spec=spec)))))
    config_ref = dict(path=str(path), sha256=audit.sha(path))
    receipt = dict(
        schema_version="s2a.clean-training-normalization.v1",
        config_ref=config_ref,
        domain=audit.DOMAIN,
        normalization_id=31,
        projection_id=0,
        transform=transform,
    )
    proposal = dict(normalization_ref=write(tmp_path / "normalization.json", receipt))
    bridge = dict(checkpoint_small_metadata_sha256={str(path): config_ref["sha256"]})
    return proposal, bridge, spec, receipt


def test_real_safe_yaml_list31_routing_and_complete_transform(normalize):
    assert audit.normalization(*normalize[:2], audit.Evidence())["transforms"] == 67


@pytest.mark.parametrize(
    "problem", ["projection", "domain", "list", "interpolation", "mask", "clip", "source", "receipt"]
)
def test_wrong_normalization_source_or_routing_is_refused(tmp_path, normalize, problem):
    proposal, bridge, spec, receipt = normalize
    if problem == "projection":
        spec["registry"]["domain_projection_ids"][31] = 1
    elif problem == "domain":
        spec["registry"]["domains"][31] = "wrong source"
    elif problem == "list":
        spec["transforms"] = {audit.DOMAIN: spec["transforms"][31]}
    elif problem == "interpolation":
        spec["registry"]["domains"][0] = "${model.config.domain}"
    elif problem == "mask":
        spec["transforms"][31]["normalization_mask"][9] = True
    elif problem == "clip":
        spec["transforms"][31]["clip_max"] = 5
    elif problem == "source":
        bridge["checkpoint_small_metadata_sha256"] = {"/wrong/config.yaml": "a" * 64}
    else:
        receipt["transform"]["scale"][0] = 7
    if problem not in ("source", "receipt"):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(dict(model=dict(config=dict(action128_spec=spec)))))
        receipt["config_ref"]["sha256"] = audit.sha(path)
        bridge["checkpoint_small_metadata_sha256"][str(path)] = audit.sha(path)
    proposal["normalization_ref"] = write(tmp_path / "replacement.json", receipt)
    with pytest.raises(ValueError):
        audit.normalization(proposal, bridge, audit.Evidence())


@pytest.fixture
def trace(tmp_path):
    source = {"/model.py": "a" * 64}
    sigmas = [float(35 - i) / 36 for i in range(35)] + [0.0]
    timesteps = list(range(999, 964, -1))
    steps = [
        dict(
            step_index=i,
            solver_sigma=sigmas[i],
            solver_timestep=timesteps[i],
            head_calls=1,
            denoise_calls=1,
            conversion_calls=1,
            velocity_delegations=1,
        )
        for i in range(35)
    ]
    value = dict(
        complete=True,
        purpose="evaluation",
        seed=42,
        sampler_calls=1,
        parameters_before={"sha": "a"},
        parameters_after={"sha": "a"},
        source_sha256=source,
        steps=steps,
        sigma_values=sigmas,
        scheduler_timesteps=timesteps,
        terminal_sigma=0.0,
        terminal_neural_calls=0,
        solver_sigma_dtype="torch.float32",
        actual_sampler_arguments=dict(num_steps=35, shift=5.0, seed=[42]),
    )
    return value, dict(qualified_model_source_sha256=source)


def test_sigma_is_observed_not_derived_from_network_t(tmp_path, trace):
    value, prepared = trace
    proposal = dict(sampler_trace_ref=write(tmp_path / "trace.json", value))
    assert audit.sampler(proposal, prepared, audit.Evidence())["actual_positive_sigma_calls"] == 35


@pytest.mark.parametrize("problem", ["terminal", "call", "count", "source", "sigma", "seed"])
def test_incomplete_grid_or_multiple_delegation_is_refused(tmp_path, trace, problem):
    value, prepared = trace
    if problem == "terminal":
        value["terminal_neural_calls"] = 1
    elif problem == "call":
        value["steps"][17]["head_calls"] = 2
    elif problem == "count":
        value["steps"].pop()
    elif problem == "source":
        value["source_sha256"] = {"/foreign.py": "b" * 64}
    elif problem == "sigma":
        value["steps"][0]["solver_sigma"] = 0.999
    else:
        value["actual_sampler_arguments"]["seed"] = [43]
    with pytest.raises(ValueError):
        audit.sampler(dict(sampler_trace_ref=write(tmp_path / "trace.json", value)), prepared, audit.Evidence())


@pytest.fixture
def machine():
    return dict(
        disk_free=300 * 1024**3,
        host_MemAvailable=300 * 1024**3,
        cgroup_memory_max=256 * 1024**3,
        cgroup_memory_current=10 * 1024**3,
        MemAvailable=246 * 1024**3,
        CPU=16,
        GPU=dict(
            uuid=audit.GPU,
            pci_bus_id=audit.GPU_PCI,
            device_minor=audit.GPU_MINOR,
            memory_mib=0,
            compute_apps=[],
            utilization_gpu=0,
        ),
        ports={"31338": True, "31339": True},
    )


def resource_receipt(machine):
    return dict(
        machine,
        complete=True,
        passed=True,
        required_disk_bytes=audit.DISK_REQUIRED,
        required_ram_bytes=audit.RAM_REQUIRED,
        observed_utc=datetime.now(timezone.utc).isoformat(),
        threads=audit.THREADS.copy(),
    )


def test_resources_reobserve_effective_cgroup_memory_and_actual_ports(tmp_path, monkeypatch, machine):
    receipt = resource_receipt(machine)
    monkeypatch.setattr(audit, "machine_observation", lambda root: copy.deepcopy(machine))
    result = audit.resources(
        dict(resource_observation_ref=write(tmp_path / "resources.json", receipt)),
        dict(root=str(tmp_path)),
        audit.Evidence(),
    )
    assert result["resources_reobserved"] is True


@pytest.mark.parametrize(
    "problem", ["unmeasured", "lower_threshold", "host_only", "ram", "disk", "cpu", "gpu", "port", "threads"]
)
def test_unmeasured_or_insufficient_recorded_resources_are_refused(tmp_path, monkeypatch, machine, problem):
    receipt = resource_receipt(machine)
    if problem == "unmeasured":
        receipt["passed"] = False
    elif problem == "lower_threshold":
        receipt["required_disk_bytes"] = 1
    elif problem == "host_only":
        receipt["MemAvailable"] = receipt["host_MemAvailable"]
    elif problem == "ram":
        receipt["MemAvailable"] = 1
    elif problem == "disk":
        receipt["disk_free"] = 1
    elif problem == "cpu":
        receipt["CPU"] = 7
    elif problem == "gpu":
        receipt["GPU"]["memory_mib"] = 100
    elif problem == "port":
        receipt["ports"]["31339"] = False
    else:
        receipt["threads"]["OPENBLAS_NUM_THREADS"] = "64"
    monkeypatch.setattr(audit, "machine_observation", lambda root: machine)
    with pytest.raises(ValueError):
        audit.resources(
            dict(resource_observation_ref=write(tmp_path / "resources.json", receipt)),
            dict(root=str(tmp_path)),
            audit.Evidence(),
        )


def test_recorded_headroom_cannot_hide_current_occupied_gpu(tmp_path, monkeypatch, machine):
    receipt = resource_receipt(machine)
    current = copy.deepcopy(machine)
    current["GPU"]["compute_apps"] = ["foreign pid"]
    monkeypatch.setattr(audit, "machine_observation", lambda root: current)
    with pytest.raises(ValueError, match="GPU"):
        audit.resources(
            dict(resource_observation_ref=write(tmp_path / "resources.json", receipt)),
            dict(root=str(tmp_path)),
            audit.Evidence(),
        )


@pytest.fixture
def final_qa(tmp_path):
    paths = list(audit.REQUIRED_TESTS)
    sources = {"/source.py": "a" * 64}
    commands = []
    for name, argv, text in (
        ("pytest", ["python", "-m", "pytest", *paths], "================ 123 passed in 4.0s ================\n"),
        ("ruff", ["python", "-m", "ruff", "check", "src", "tests", "scripts"], "All checks passed!\n"),
        (
            "format",
            ["python", "-m", "ruff", "format", "--check", "src", "tests", "scripts"],
            "49 files already formatted\n",
        ),
    ):
        path = tmp_path / (name + ".log")
        path.write_text(text)
        commands.append(dict(argv=argv, exit_code=0, log_ref=dict(path=str(path), sha256=audit.sha(path))))
    return dict(
        complete=True,
        passed=True,
        exit_code=0,
        tests_passed=123,
        test_paths=paths,
        source_sha256=sources,
        commands=commands,
    ), dict(source_sha256=sources.copy())


def test_qa_requires_final_source_sha_actual_counts_and_three_commands(tmp_path, final_qa):
    receipt, proposal = final_qa
    proposal["qa_ref"] = write(tmp_path / "qa.json", receipt)
    assert audit.qa(proposal, audit.Evidence())["actual_cpu_tests_passed"] == 123


@pytest.mark.parametrize("problem", ["stale", "count", "missing_test", "command", "selective", "failed", "log_tamper"])
def test_qa_cannot_relabel_stale_incomplete_or_failed_commands(tmp_path, final_qa, problem):
    receipt, proposal = final_qa
    if problem == "stale":
        receipt["source_sha256"] = {"/source.py": "b" * 64}
    elif problem == "count":
        receipt["tests_passed"] = 999
    elif problem == "missing_test":
        receipt["test_paths"].pop()
    elif problem == "command":
        receipt["commands"].pop()
    elif problem == "selective":
        receipt["commands"][0]["argv"] += ["-k", "passing_only"]
    elif problem == "failed":
        receipt["commands"][0]["exit_code"] = 1
    else:
        Path(receipt["commands"][0]["log_ref"]["path"]).write_text("999 passed")
    proposal["qa_ref"] = write(tmp_path / "qa.json", receipt)
    with pytest.raises(ValueError):
        audit.qa(proposal, audit.Evidence())


def test_normal_owned_lifecycle_needs_pid_birth_and_no_signal(tmp_path):
    command = dict(
        launch=write(tmp_path / "launch.json", dict(child=dict(pid=17, start_ticks=90))),
        exit=write(tmp_path / "exit.json", dict(actual_wait=True, returncode=0)),
        release=write(
            tmp_path / "release.json", dict(pid=17, start_ticks=90, reaped=True, returncode=0, signal_sent=False)
        ),
    )
    audit.normal_lifecycle(command, audit.Evidence())
    command["release"] = write(
        tmp_path / "wrong_release.json", dict(pid=17, start_ticks=91, reaped=True, returncode=0, signal_sent=False)
    )
    with pytest.raises(ValueError, match="PID/birth"):
        audit.normal_lifecycle(command, audit.Evidence())


@pytest.fixture
def qualified_metadata(tmp_path):
    path = tmp_path / "verifier.py"
    path.write_text("# independent metadata verifier")
    verifier = dict(path=str(path), sha256=audit.sha(path))
    opaque = tmp_path / "future.npz"
    opaque.write_bytes(b"opaque future bytes; not a valid array archive")
    value = dict(
        schema_version="s2a.clean-training-preparation-independent.v1",
        prepared_ref=dict(path=str(tmp_path / "prepared.json"), sha256="b" * 64),
        complete=True,
        passed=True,
        metadata_qualified=True,
        consumed_before_after_exact=True,
        execution_ready=False,
        execution_authorized=False,
        verifier_ref=verifier,
        new_model_calls=0,
        new_fit_calls=0,
        new_physics_steps=0,
        new_fresh_cases=0,
        future_label_archives_decoded=0,
        weight_stat_files=163,
        full_weight_content_rehash=False,
        consumed_sha256={str(opaque): audit.sha(opaque)},
    )
    proposal = dict(prepared_ref=value["prepared_ref"].copy(), source_sha256={str(path): verifier["sha256"]})
    return value, proposal


def test_independent_metadata_receipt_is_bound_to_producer_and_actual_verifier(tmp_path, qualified_metadata):
    value, proposal = qualified_metadata
    proposal["metadata_verification_ref"] = write(tmp_path / "independent.json", value)
    evidence = audit.Evidence()
    audit.independent_metadata(proposal, evidence)
    assert any(path.endswith("future.npz") for path in evidence.consumed)


@pytest.mark.parametrize(
    "problem", ["schema", "source", "prepared", "promoted", "count", "rewrite_boundary", "decoded"]
)
def test_generic_true_cannot_replace_independent_metadata(tmp_path, qualified_metadata, problem):
    value, proposal = qualified_metadata
    if problem == "schema":
        value["schema_version"] = "generic.verdict.v1"
    elif problem == "source":
        proposal["source_sha256"] = {}
    elif problem == "prepared":
        value["prepared_ref"]["sha256"] = "c" * 64
    elif problem == "promoted":
        value["execution_authorized"] = True
    elif problem == "count":
        value["weight_stat_files"] = 162
    elif problem == "rewrite_boundary":
        value["full_weight_content_rehash"] = True
    else:
        value["future_label_archives_decoded"] = 1
    proposal["metadata_verification_ref"] = write(tmp_path / "independent.json", value)
    with pytest.raises(ValueError):
        audit.independent_metadata(proposal, audit.Evidence())


def test_development_guard_is_before_actual_heavy_import_and_strict_after():
    source = """
def execute():
    strict_native_environment()
    readiness = restore_cv2_environment(probe, runtime)
    from s2a.stability_position_simulation import run_position_case
    strict_native_environment()
    run_position_case()
"""
    audit.guarded_worker(source)
    with pytest.raises(ValueError, match="heavy imports"):
        audit.guarded_worker(
            source.replace("    readiness = restore_cv2_environment(probe, runtime)\n", "").replace(
                "    run_position_case()", "    restore_cv2_environment(probe, runtime)\n    run_position_case()"
            )
        )
    with pytest.raises(ValueError, match="top-level"):
        audit.guarded_worker("import cv2\n" + source)
    with pytest.raises(ValueError, match="heavy imports"):
        audit.guarded_worker(
            source.replace("    strict_native_environment()\n    run_position_case()", "    run_position_case()")
        )


@pytest.fixture
def seed_revision(tmp_path):
    anchors, requests, entries = [], [], []
    for task in range(3):
        for demo in range(8, 18):
            identity = dict(task_index=task, demo=f"demo_{demo}", initial_state_sha256=f"{1 + task * 10 + demo:064x}")
            for anchor_index, frame in enumerate((0, 15, 30, 45, 123)):
                anchor_name = f"task{task}_demo_{demo}_a{anchor_index}"
                anchors.append(dict(name=anchor_name, source_identity=identity.copy(), anchor_frame=frame))
                for root in (101, 102):
                    canonical = json.dumps(
                        dict(task=task, demo=f"demo_{demo}", anchor_frame=frame, root_seed=root),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    )
                    original_digest = audit.hashlib.sha256(canonical.encode()).digest()
                    original = int.from_bytes(original_digest[:8], "big") & ((1 << 63) - 1)
                    index = len(requests)
                    name = f"train_{anchor_name}_r{root}"
                    requests.append(
                        dict(
                            index=index,
                            name=name,
                            anchor_name=anchor_name,
                            **identity,
                            anchor_frame=frame,
                            root_seed=root,
                            sampling_seed=original,
                            canonical_json=canonical,
                            identity_sha256=original_digest.hex(),
                            request=dict(seed=original),
                        )
                    )
                    payload = json.dumps(
                        [audit.SEED_DOMAIN, anchor_name, root, original], ensure_ascii=True, separators=(",", ":")
                    ).encode()
                    digest = audit.hashlib.sha256(payload).digest()
                    actual = int.from_bytes(digest[:4], "big") & ((1 << 31) - 1)
                    entries.append(
                        dict(
                            index=index,
                            name=name,
                            anchor_name=anchor_name,
                            root_seed=root,
                            original_sampling_seed=original,
                            sampling_seed=actual,
                            payload_sha256=digest.hex(),
                        )
                    )
    prepared = dict(
        training_anchors=anchors,
        training_requests=requests,
        noise_roster=dict(training_seeds=[r["sampling_seed"] for r in requests], evaluation_seeds=[42, 43]),
    )
    prepared_ref = dict(path=str(tmp_path / "original.json"), sha256="a" * 64)
    plan = dict(
        schema_version="s2a.clean-training-seed-plan.v1",
        complete=True,
        passed=True,
        prepared_ref=prepared_ref,
        algorithm=audit.SEED_ALGORITHM,
        entries=entries,
        evaluation_seeds=[42, 43],
        actual_seed_bits=31,
        original_seed_bits=63,
        model_calls=0,
        fits=0,
        physics_steps=0,
    )
    proposal = dict(prepared_ref=prepared_ref)
    return prepared, plan, proposal


def test_full_seed_revision_binds_original63_identity_without_mutation(tmp_path, seed_revision):
    prepared, plan, proposal = seed_revision
    before = copy.deepcopy(prepared)
    proposal["training_seed_plan_ref"] = write(tmp_path / "seed_plan.json", plan)
    result = audit.training_seed_plan(proposal, prepared, audit.Evidence())
    assert result["training_requests_with_revised_seeds"] == 300
    assert result["actual_training_seed_bits"] == 31
    assert prepared == before
    assert all(0 <= row["sampling_seed"] <= 2**31 - 1 for row in plan["entries"])
    assert 0 < result["training_captions_without_task_plan"] < 300


@pytest.mark.parametrize(
    "problem",
    [
        "prepared",
        "domain",
        "actual",
        "original",
        "root",
        "source",
        "canonical",
        "order",
        "row_type",
        "bits_type",
        "evaluation_type",
        "extra",
        "counter",
        "roster",
        "overlay_twice",
    ],
)
def test_seed_revision_refuses_wrong_domain_identity_mapping_or_type(tmp_path, seed_revision, problem):
    prepared, plan, proposal = seed_revision
    if problem == "prepared":
        plan["prepared_ref"] = dict(path="/foreign.json", sha256="b" * 64)
    elif problem == "domain":
        plan["algorithm"] += " silently truncate63"
    elif problem == "actual":
        plan["entries"][0]["sampling_seed"] = 42
    elif problem == "original":
        plan["entries"][0]["original_sampling_seed"] += 1
    elif problem == "root":
        prepared["training_requests"][0]["root_seed"] = 102
    elif problem == "source":
        prepared["training_requests"][0]["initial_state_sha256"] = "c" * 64
    elif problem == "canonical":
        prepared["training_requests"][0]["canonical_json"] += " "
    elif problem == "order":
        plan["entries"].reverse()
    elif problem == "row_type":
        plan["entries"][1]["index"] = True
    elif problem == "bits_type":
        plan["actual_seed_bits"] = 31.0
    elif problem == "evaluation_type":
        plan["evaluation_seeds"] = [42.0, 43]
    elif problem == "extra":
        plan["retry"] = 1
    elif problem == "counter":
        plan["model_calls"] = False
    elif problem == "roster":
        prepared["noise_roster"]["training_seeds"].reverse()
    else:
        prepared["training_seed_plan_ref"] = dict(path="/already_applied.json", sha256="d" * 64)
    proposal["training_seed_plan_ref"] = write(tmp_path / "seed_plan.json", plan)
    with pytest.raises(ValueError):
        audit.training_seed_plan(proposal, prepared, audit.Evidence())


def test_seed_revision_collision_or_evaluation_overlap_never_redraws(tmp_path, monkeypatch, seed_revision):
    prepared, plan, proposal = seed_revision
    original_hash = audit.hashlib.sha256
    calls = []
    digest = (42).to_bytes(4, "big") + bytes(28)

    class ForcedCollision:
        def digest(self):
            return digest

    def collide(payload=b""):
        if payload.startswith(b'["s2a.clean-training-request-seed31.v1",'):
            calls.append(payload)
            return ForcedCollision()
        return original_hash(payload)

    for entry in plan["entries"]:
        entry.update(sampling_seed=42, payload_sha256=digest.hex())
    proposal["training_seed_plan_ref"] = write(tmp_path / "seed_plan.json", plan)
    monkeypatch.setattr(audit.hashlib, "sha256", collide)
    with pytest.raises(ValueError, match="without retry"):
        audit.training_seed_plan(proposal, prepared, audit.Evidence())
    assert len(calls) == 300


def test_failed_verification_publishes_false_without_promoting_or_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REPO", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    ref = write(tmp_path / "bad.json", dict(schema_version="old", prepared_ref=write(tmp_path / "prepared.json", {})))
    args = Namespace(
        proposal=Path(ref["path"]), proposal_sha256=ref["sha256"], output=tmp_path / "outputs/failed.json", seed=42
    )
    with pytest.raises(ValueError, match="schema"):
        audit.verify(args)
    receipt = json.loads(args.output.read_text())
    assert receipt["eligible_for_execution"] is receipt["complete"] is receipt["passed"] is False
    assert receipt["new_model_calls"] == receipt["new_fits"] == receipt["future_label_archives_decoded"] == 0
    with pytest.raises(ValueError, match="overwrite"):
        audit.verify(args)


def test_import_and_real_cli_help_are_metadata_only():
    tree = ast.parse(SCRIPT.read_text())
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(
        name and (name.startswith(("s2a", "numpy", "torch", "cosmos")) or "prepare_wam" in name) for name in imported
    )
    result = subprocess.run(
        [os.sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        timeout=15,
    )
    assert result.returncode == 0 and "--proposal-sha256" in result.stdout and "--seed" in result.stdout
