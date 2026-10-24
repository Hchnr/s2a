"""Adversarial041 admission contracts; no array decoding or live model work."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_wam_clean_training_continuation.py"
SPEC = importlib.util.spec_from_file_location("independent041", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return dict(path=str(path), sha256=audit.sha(path))


def test_verifier_uses_no_producer_or_array_imports():
    tree = ast.parse(SCRIPT.read_text())
    imports = [
        name
        for node in ast.walk(tree)
        for name in (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [node.module]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    ]
    assert not any(name.split(".")[0] in {"s2a", "numpy", "torch", "h5py", "mujoco", "yaml"} for name in imports)
    loader = ast.get_source_segment(
        SCRIPT.read_text(),
        next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "legacy_independent"),
    )
    assert '"scripts/verify_wam_clean_training_execution.py"' in loader
    assert "evidence.check(ref)" in loader
    assert "prepare_wam" not in loader and "continuation_bundle" not in loader


def test_root_only_derivation_preserves_nested_objects_and_metadata_status():
    original = dict(
        root=str(audit.REPO / audit.ORIGINAL_ROOT_REL),
        schema_version="s2a.clean-training-preparation.v1",
        execution_authorized=False,
        readiness=dict(ready_for_execution=False),
        training_requests=[dict(seed=123)],
        mask=[True, False],
    )
    before = copy.deepcopy(original)
    derived = copy.deepcopy(original)
    derived["root"] = str(audit.REPO / audit.NEW_ROOT_REL)
    assert audit.root_only_preparation(original, derived)["root_only_derivation_exact"]
    assert original == before
    derived["mask"][0] = 1
    with pytest.raises(ValueError, match="only root"):
        audit.root_only_preparation(original, derived)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(extra=True),
        lambda d: d.update(execution_authorized=True),
        lambda d: d["readiness"].update(ready_for_execution=True),
        lambda d: d.update(root=str(audit.REPO / audit.ORIGINAL_ROOT_REL)),
    ],
)
def test_metadata_cannot_change_any_field_beyond_root(change):
    original = dict(
        root=str(audit.REPO / audit.ORIGINAL_ROOT_REL),
        schema_version="s2a.clean-training-preparation.v1",
        execution_authorized=False,
        readiness=dict(ready_for_execution=False),
    )
    derived = copy.deepcopy(original)
    derived["root"] = str(audit.REPO / audit.NEW_ROOT_REL)
    change(derived)
    with pytest.raises(ValueError):
        audit.root_only_preparation(original, derived)


def test_evidence_hashes_opaque_future_bytes_without_parsing(tmp_path):
    file = tmp_path / "future.npz"
    file.write_bytes(b"not an archive")
    ref = dict(path=str(file), sha256=audit.sha(file))
    ev = audit.Evidence()
    assert ev.check(ref) == file
    with pytest.raises(ValueError, match="opaque"):
        ev.json(ref)
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        ev.unchanged()


def test_evidence_rejects_nonfinite_json_and_conflicting_sha(tmp_path):
    file = tmp_path / "source.json"
    file.write_text('{"number":NaN}')
    ev = audit.Evidence()
    with pytest.raises(ValueError, match="NaN"):
        ev.json(dict(path=str(file), sha256=audit.sha(file)))
    with pytest.raises(ValueError, match="SHA256"):
        ev.check(dict(path=str(file), sha256="bad"))


@pytest.fixture
def seeds():
    requests, entries = [], []
    for index in range(300):
        request = dict(
            name=f"train{index}",
            anchor_name=f"anchor{index // 2}",
            root_seed=(101, 102)[index % 2],
            sampling_seed=100000000000 + index,
        )
        payload = json.dumps(
            [audit.SEED_DOMAIN, request["anchor_name"], request["root_seed"], request["sampling_seed"]],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(payload).digest()
        entries.append(
            dict(
                index=index,
                name=request["name"],
                anchor_name=request["anchor_name"],
                root_seed=request["root_seed"],
                original_sampling_seed=request["sampling_seed"],
                sampling_seed=int.from_bytes(digest[:4], "big") & ((1 << 31) - 1),
                payload_sha256=digest.hex(),
            )
        )
        requests.append(request)
    old_ref, new_ref = dict(path="/original.json", sha256="a" * 64), dict(path="/new.json", sha256="b" * 64)
    old = dict(
        schema_version="s2a.clean-training-seed-plan.v1",
        prepared_ref=old_ref,
        entries=entries,
        evaluation_seeds=[42, 43],
        actual_seed_bits=31,
        original_seed_bits=63,
        model_calls=0,
        fits=0,
        physics_steps=0,
        complete=True,
        passed=True,
    )
    new = copy.deepcopy(old)
    new["prepared_ref"] = copy.deepcopy(new_ref)
    return new, old, dict(training_requests=requests), new_ref, old_ref


def test_seed_plan_preserves_original300_values_and_recomputes_hash(seeds):
    assert audit.seed_plan(*seeds)["seed31_independently_rebuilt"]


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["entries"][0].update(sampling_seed=42),
        lambda p: p["entries"].reverse(),
        lambda p: p["entries"][12].update(payload_sha256="a" * 64),
        lambda p: p.update(fits=1),
        lambda p: p.update(extra=True),
        lambda p: p["prepared_ref"].update(sha256="c" * 64),
    ],
)
def test_recovery_cannot_choose_new_seed_reorder_or_change_manifest(seeds, change):
    change(seeds[0])
    with pytest.raises(ValueError, match="only rebind"):
        audit.seed_plan(*seeds)


def test_matching_forged_old_and_new_seed_value_still_fails_recomputation(seeds):
    seeds[0]["entries"][17]["sampling_seed"] += 1
    seeds[1]["entries"][17]["sampling_seed"] += 1
    with pytest.raises(ValueError, match="independent derivation"):
        audit.seed_plan(*seeds)


def test_finite_source_patch_changes_only_exact_tokens():
    source = (
        "from pathlib import Path\nfrom s2a.clean_training_runtime import GPU\nROOT = '"
        + audit.ORIGINAL_ROOT_REL
        + "'\nENTRY = \"scripts/collect_wam_clean_training.py\"\nTEXT = f'unchanged {ROOT}'\n# scripts/collect_wam_clean_training.py must stay in comment\ndef calculate(x):\n    return x * 0.1\n"
    )
    patched, edits = audit.finite_bundle_source(source, "scripts/collect_wam_clean_training.py")
    assert "from continuation_runtime import GPU" in patched
    assert "ROOT = '" + audit.NEW_ROOT_REL + "'" in patched
    assert "# scripts/collect_wam_clean_training.py must stay in comment" in patched
    assert "return x * 0.1" in patched and "TEXT = f'unchanged {ROOT}'" in patched
    assert len(edits) == 3 and all(e["count"] == 1 for e in edits)


def test_only_two_registered_local_repo_guards_can_change():
    source = "from pathlib import Path\nREPO = Path(__file__).resolve().parents[1]\n"
    patched, edits = audit.finite_bundle_source(source, "scripts/verify_wam_clean_training_capture.py")
    assert "REPO = Path(" + json.dumps(str(audit.REPO)) + ")" in patched
    assert [e["kind"] for e in edits] == ["repository_guard"]
    with pytest.raises(ValueError, match="unregistered"):
        audit.finite_bundle_source(source, "scripts/fit_wam_clean_training.py")


@pytest.fixture
def bundled(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REPO", tmp_path)
    original, actual, files = {}, {}, []
    for relative in ["scripts/" + name for name in audit.SCRIPT_NAMES] + [audit.RUNTIME_REL]:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('ENTRY = "scripts/collect_wam_clean_training.py"\ndef calculate(x):\n    return x * 0.1\n')
        name = "continuation_runtime.py" if relative == audit.RUNTIME_REL else source.name
        target = tmp_path / audit.BUNDLE_REL / name
        target.parent.mkdir(parents=True, exist_ok=True)
        text, edits = audit.finite_bundle_source(source.read_text(), relative)
        target.write_text(text)
        old_ref, new_ref = (
            dict(path=str(source), sha256=audit.sha(source)),
            dict(path=str(target), sha256=audit.sha(target)),
        )
        original[str(source)] = old_ref["sha256"]
        actual[str(target)] = new_ref["sha256"]
        files.append(dict(original_ref=old_ref, actual_ref=new_ref, edits=edits))
    old_ref = dict(path=str(tmp_path / "old_admitted.json"), sha256="a" * 64)
    manifest = dict(
        schema_version="s2a.clean-training-continuation-bundle.v1",
        complete=True,
        passed=True,
        old_admitted_ref=old_ref,
        original_root=str(tmp_path / audit.ORIGINAL_ROOT_REL),
        output_root=str(tmp_path / audit.NEW_ROOT_REL),
        bundle_root=str(tmp_path / audit.BUNDLE_REL),
        files=files,
        source_sha256=actual,
        original_sources_before_after_exact=True,
        ast_and_source_compiled=True,
        entrypoints_executed=0,
        future_archive_decodes=0,
    )
    path = tmp_path / audit.NEW_ROOT_REL / "source/bundle_manifest.json"
    return path, manifest, dict(source_sha256=original), dict(original_execution_ref=old_ref)


def test_bundle_full_eleven_files_pass_independent_byte_and_edit_check(bundled):
    path, value, original, config = bundled
    ref = write(path, value)
    assert len(audit.bundle(ref, original, config, audit.Evidence())["files"]) == 11


def test_bundle_rejects_math_change_even_with_updated_self_hash(bundled):
    path, value, original, config = bundled
    entry = value["files"][0]
    file = Path(entry["actual_ref"]["path"])
    file.write_text(file.read_text().replace("x * 0.1", "x * 0.2"))
    entry["actual_ref"]["sha256"] = audit.sha(file)
    value["source_sha256"][str(file)] = audit.sha(file)
    with pytest.raises(ValueError, match="unregistered byte"):
        audit.bundle(write(path, value), original, config, audit.Evidence())


def test_bundle_rejects_unreported_allowed_patch_and_wrong_original_identity(bundled):
    path, value, original, config = bundled
    value["files"][0]["edits"][0]["count"] = 2
    with pytest.raises(ValueError, match="edit witness"):
        audit.bundle(write(path, value), original, config, audit.Evidence())
    value["files"][0]["original_ref"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="040-frozen"):
        audit.bundle(write(path, value), original, config, audit.Evidence())


@pytest.fixture
def source_closure(bundled, monkeypatch):
    _, manifest, original, _ = bundled
    root = audit.REPO
    verifier = root / "scripts/verify_wam_clean_training_continuation.py"
    monkeypatch.setattr(audit, "VERIFIER", verifier)
    for relative in (
        "src/s2a/clean_training_continuation_bundle.py",
        "src/s2a/durable_stage_supervisor.py",
        "scripts/run_wam_clean_training_continuation.py",
        "scripts/prepare_wam_clean_training_continuation.py",
        "scripts/verify_wam_clean_training_continuation.py",
        *audit.NEW_TESTS,
        *audit.EXTRA_SOURCES,
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# independent source closure fixture\n")
    main = {p for folder in ("src", "scripts", "tests") for p in (root / folder).rglob("*.py")}
    main.update(root / relative for relative in audit.EXTRA_SOURCES)
    sources = {str(p): audit.sha(p) for p in main}
    sources.update(manifest["source_sha256"])
    refs, snapshots = {}, {}
    for path, digest in sources.items():
        source = Path(path)
        relative = (
            Path("bundle") / source.name
            if path in manifest["source_sha256"]
            else Path("repo") / source.relative_to(root)
        )
        target = root / audit.NEW_ROOT_REL / "execution/source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        refs[path] = dict(path=str(target), sha256=digest)
        snapshots[str(target)] = digest
    return dict(source_sha256=sources, snapshot_refs=refs, snapshot_sha256=snapshots), manifest, original


def test_complete_main_and_bundle_source_snapshots_are_distinct_and_exact(source_closure):
    proposal, manifest, original = source_closure
    result = audit.execution_sources(proposal, manifest, original, audit.Evidence())
    assert result["final_sources"] == result["source_snapshots"]
    assert result["isolated_bundle_sources"] == 11


def test_execution_source_closure_cannot_omit_current_python_file(source_closure):
    proposal, manifest, original = source_closure
    omitted = next(path for path in proposal["source_sha256"] if path.endswith("durable_stage_supervisor.py"))
    proposal["source_sha256"].pop(omitted)
    with pytest.raises(ValueError, match="complete main"):
        audit.execution_sources(proposal, manifest, original, audit.Evidence())


def test_execution_source_closure_rejects_old_source_change_even_if_new_sha_updated(source_closure):
    proposal, manifest, original = source_closure
    path = Path(next(iter(original["source_sha256"])))
    path.write_text(path.read_text() + "# changed immutable original\n")
    proposal["source_sha256"][str(path)] = audit.sha(path)
    with pytest.raises(ValueError, match="old462"):
        audit.execution_sources(proposal, manifest, original, audit.Evidence())


def test_execution_snapshots_cannot_reuse_old_or_wrong_root(source_closure):
    proposal, manifest, original = source_closure
    path = next(iter(proposal["snapshot_refs"]))
    proposal["snapshot_refs"][path] = dict(path=path, sha256=proposal["source_sha256"][path])
    with pytest.raises(ValueError, match="snapshot path"):
        audit.execution_sources(proposal, manifest, original, audit.Evidence())


@pytest.fixture
def contract():
    config = json.loads((audit.REPO / audit.CONFIG_REL).read_text())
    legacy_spec = importlib.util.spec_from_file_location(
        "original041_test", audit.REPO / "scripts/verify_wam_clean_training_execution.py"
    )
    legacy = importlib.util.module_from_spec(legacy_spec)
    legacy_spec.loader.exec_module(legacy)
    prepared = dict(
        root=str(audit.REPO / audit.NEW_ROOT_REL),
        budget=copy.deepcopy(legacy.BUDGET),
        resources=copy.deepcopy(legacy.RESOURCES),
    )
    ref, manifest_ref = dict(path="/new_prepared.json", sha256="b" * 64), dict(path="/manifest.json", sha256="c" * 64)
    stage_entrypoints = {
        stage: audit.BUNDLE_REL + "/collect_wam_clean_training.py"
        if stage in ("D0", "D1")
        else "scripts/fit_wam_clean_training.py"
        if stage in ("W1", "W2")
        else audit.BUNDLE_REL + "/sample_wam_clean_training_state.py"
        if stage == "STATE"
        else audit.BUNDLE_REL + "/run_wam_clean_training_development.py"
        for stage in audit.STAGES
    }
    source = {str(audit.REPO / p): "d" * 64 for p in stage_entrypoints.values()}
    sampler_ref = dict(path="/actual_grid.json", sha256="e" * 64)
    proposal = dict(
        schema_version="s2a.clean-training-execution-proposal.v1",
        complete=True,
        passed=True,
        execution_authorized=False,
        readiness=dict(ready_for_execution=False),
        automatic_retry=False,
        fresh_performance_authorized=False,
        root=prepared["root"],
        prepared_ref=ref,
        original_execution_ref=config["original_execution_ref"],
        incident_ref=config["incident_ref"],
        bundle_manifest_ref=manifest_ref,
        development_control_ref=config["development_control_ref"],
        budget=copy.deepcopy(prepared["budget"]),
        resources=copy.deepcopy(prepared["resources"]),
        aggregate_budget=copy.deepcopy(audit.AGGREGATE),
        continuation_contract=audit.continuation_contract(config),
        supervisor=audit.supervisor_contract(),
        stage_entrypoints=stage_entrypoints,
        source_sha256=source,
        entrypoint_refs={
            stage: dict(path=str(audit.REPO / p), sha256="d" * 64) for stage, p in stage_entrypoints.items()
        },
        master_port=31338,
        development=dict(port=31339, case_order=legacy.case_order()),
        sampler_trace_ref=sampler_ref,
        new_model_calls=0,
        new_fits=0,
        physics_steps=0,
        fresh_cases=0,
        future_archive_decodes=0,
    )
    return proposal, config, prepared, ref, manifest_ref, dict(sampler_trace_ref=sampler_ref), legacy


def test_new_complete_budget_and_durable_six_stage_specs(contract):
    audit.execution_contract(*contract)
    supervisor = contract[0]["supervisor"]
    assert supervisor["stage_order"] == list(audit.STAGES)
    assert all("sha256" not in arg for c in supervisor["stage_commands"].values() for arg in c["arguments"])


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["aggregate_budget"].update(original_saved_denoise=700),
        lambda p: p["aggregate_budget"].update(total_denoise_upper_bound=55965),
        lambda p: p["budget"].update(raw_training_requests=280),
        lambda p: p["continuation_contract"].update(old_data_used_for_fit=True),
        lambda p: p["supervisor"]["stage_order"].reverse(),
        lambda p: p["supervisor"]["stage_commands"]["D0"].update(entrypoint="scripts/run_wam_clean_training.py"),
        lambda p: p["supervisor"]["stage_timeouts_s"].update(D0=999999),
        lambda p: p.update(automatic_retry=True),
        lambda p: p.update(fresh_performance_authorized=True),
        lambda p: p["readiness"].update(ready_for_execution=True),
    ],
)
def test_recovery_refuses_reusing_partial_dropping_budget_or_changing_runtime(contract, change):
    change(contract[0])
    with pytest.raises(ValueError):
        audit.execution_contract(*contract)


def test_configuration_checks_real_frozen_aggregate_and_reference_identity():
    config = json.loads((audit.REPO / audit.CONFIG_REL).read_text())
    audit.configuration(config)
    config["aggregate_budget"]["original_model_initializations"] = False
    with pytest.raises(ValueError, match="cumulative"):
        audit.configuration(config)


def test_actual_supervisor_declares_two_modes_and_one_ordered_owner_loop():
    source = (audit.REPO / "scripts/run_wam_clean_training_continuation.py").read_text()
    assert audit.supervisor_source_contract(source)["supervisor_launch_run_interface_checked"]


@pytest.mark.parametrize("old,new", [('"--run"', '"--resume"'), ("for stage in STAGES:", 'for stage in ["D0"]:')])
def test_supervisor_rejects_resume_mode_or_partial_stage_loop(old, new):
    source = (audit.REPO / "scripts/run_wam_clean_training_continuation.py").read_text()
    assert old in source
    with pytest.raises(ValueError):
        audit.supervisor_source_contract(source.replace(old, new))


def test_cli_help_does_not_initialize_any_phase():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0
    assert "--stage {metadata,execution}" in completed.stdout
    assert "--proposal-sha256" in completed.stdout


def test_failed_metadata_keeps_false_receipt_and_rejects_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REPO", tmp_path)
    monkeypatch.chdir(tmp_path)
    for name, value in dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", **audit.THREADS).items():
        monkeypatch.setenv(name, value)
    output = tmp_path / audit.NEW_ROOT_REL / "metadata_verified.json"
    args = Namespace(
        seed=42,
        stage="metadata",
        output=output,
        config=tmp_path / audit.CONFIG_REL,
        config_sha256="a" * 64,
        prepared=tmp_path / "prepared.json",
        prepared_sha256="b" * 64,
        proposal=None,
        proposal_sha256=None,
    )
    monkeypatch.setattr(audit, "metadata_audit", lambda *_: (_ for _ in ()).throw(ValueError("actual rejection")))
    with pytest.raises(ValueError, match="actual rejection"):
        audit.verify(args)
    raw = output.read_bytes()
    receipt = json.loads(raw)
    assert receipt["complete"] is receipt["passed"] is receipt["metadata_qualified"] is False
    assert receipt["execution_authorized"] is False and receipt["new_model_calls"] == receipt["new_fit_calls"] == 0
    with pytest.raises(ValueError, match="overwrite"):
        audit.verify(args)
    assert output.read_bytes() == raw


def test_explicit_cpu_thread_environment_is_required(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REPO", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with pytest.raises(ValueError, match="thread values8"):
        audit.verify(SimpleNamespace(seed=42))
