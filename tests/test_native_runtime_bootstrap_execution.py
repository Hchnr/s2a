"""Bootstrap CLI admission and publication tests with synthetic files only."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_native_runtime_bootstrap.py"


@pytest.fixture
def execution(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("_synthetic_native_bootstrap_execution", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(cli, "REPO", repo)
    monkeypatch.setattr(cli, "CONFIG", repo / "configs/native_runtime_bootstrap_20261029.json")
    monkeypatch.setattr(cli, "PLAN", repo / "plans/v0.0.1/native_runtime_bootstrap_20261029.md")
    monkeypatch.setattr(cli, "RESULT", repo / "outputs/native_runtime_bootstrap_20261029/result.json")
    monkeypatch.setattr(cli, "NATIVE_PYTHON", Path(sys.executable))
    monkeypatch.chdir(repo)
    for key, value in cli.CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)

    def write(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return cli.reference(path)

    for path in cli.SOURCE_PATHS:
        write(repo / path, f"synthetic source: {path}".encode())
    qa = dict(
        complete=True,
        passed=True,
        actual_wait=True,
        reaped=True,
        source_unchanged=True,
        driver_unchanged=True,
        pytest_disable_plugin_autoload="1",
        source_count=17,
        test_module_count=5,
        returncode=0,
        signal_sent=False,
        automatic_retry=False,
        environment=dict(cli.CPU_ENVIRONMENT),
        python_version="3.11.15",
        python_executable=str(cli.NATIVE_PYTHON),
        command=[str(cli.NATIVE_PYTHON), "-m", "pytest", "-q", *cli.QA_TESTS, f"--basetemp={repo}/outputs/qa_tmp"],
        source_sha256={path: cli.reference(repo / path)["sha256"] for path in cli.SOURCE_PATHS},
        log_ref=write(repo / "outputs/qa/pytest.log", b"synthetic successful scoped tests"),
        driver_ref=write(repo / "outputs/qa/driver.py", b"synthetic QA driver"),
    )
    return SimpleNamespace(cli=cli, repo=repo, write=write, qa=qa, qa_ref=write(repo / "outputs/qa/result.json", qa))


def test_exact_small_qa_covers_bootstrap_and_frozen_dependencies(execution):
    e = execution
    assert len(e.cli.SOURCE_PATHS) == len(set(e.cli.SOURCE_PATHS)) == 17
    assert len(e.cli.QA_TESTS) == 5
    assert e.cli.verify_native_qa(e.qa_ref) == e.qa
    for path in e.cli.SOURCE_PATHS:
        changed = copy.deepcopy(e.qa)
        del changed["source_sha256"][path]
        with pytest.raises(ValueError, match="Exact bootstrap"):
            e.cli.verify_native_qa(e.write(e.repo / "outputs/qa/changed.json", changed))
    for module in e.cli.QA_TESTS:
        changed = copy.deepcopy(e.qa)
        changed["command"].remove(module)
        with pytest.raises(ValueError, match="test module"):
            e.cli.verify_native_qa(e.write(e.repo / "outputs/qa/changed.json", changed))


@pytest.mark.parametrize(
    "field,value", [("actual_wait", False), ("returncode", 1), ("signal_sent", True), ("automatic_retry", True)]
)
def test_qa_must_be_a_successfully_closed_native_process(execution, field, value):
    e = execution
    e.qa[field] = value
    with pytest.raises(ValueError, match="actual successful wait"):
        e.cli.verify_native_qa(e.write(e.repo / "outputs/qa/changed.json", e.qa))


def test_changed_source_or_qa_log_is_rejected(execution):
    e = execution
    path = e.repo / e.cli.SOURCE_PATHS[0]
    original = path.read_bytes()
    path.write_bytes(b"changed after QA")
    with pytest.raises(ValueError, match="SHA mismatch"):
        e.cli.verify_native_qa(e.qa_ref)
    path.write_bytes(original)
    Path(e.qa["log_ref"]["path"]).write_bytes(b"changed log")
    with pytest.raises(ValueError, match="SHA mismatch"):
        e.cli.verify_native_qa(e.qa_ref)


def test_sources_cannot_be_rebound_or_output_escape(execution, tmp_path):
    e = execution
    refs = {}
    ref = e.write(e.repo / "outputs/input.json", {"source": 1})
    e.cli.add_ref(ref, refs)
    changed = e.write(ref["path"], {"source": 2})
    with pytest.raises(ValueError, match="conflicting digests"):
        e.cli.add_ref(changed, refs)
    with pytest.raises(ValueError, match="repository outputs"):
        e.cli.repository_output(tmp_path / "external/result.json")
    (e.repo / "outputs/link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        e.cli.repository_output(e.repo / "outputs/link/result.json")


@pytest.fixture
def admitted(execution, monkeypatch):
    e = execution
    import prepare_native_runtime_path_snapshot as preparation

    import s2a.native_runtime_bootstrap as core

    artifact = e.repo / "outputs/inputs"
    original = dict(record=dict(initial_state=[0.0] * 79, model_xml="<mujoco/>"), runtime_config={})
    original_ref = e.write(artifact / "original.json", original)
    roots = [
        e.repo / "outputs/snapshot/LIBERO/libero/libero/assets",
        e.repo / "outputs/libero_env311/lib/python3.11/site-packages/robosuite/models/assets",
    ]
    for root in roots:
        root.mkdir(parents=True)
    runtime = dict(task_suite="libero_goal", task_id=8, seed=42, libero_root=str(e.repo / "outputs/snapshot/LIBERO"))
    manifest = dict(record=copy.deepcopy(original["record"]), runtime_config=runtime)
    record_ref = e.write(artifact / "runtime.json", manifest)
    asset_ref = e.write(roots[0] / "asset.obj", b"immutable mesh")
    snapshot = dict(
        file_snapshot_verified=True,
        runtime_loaded=False,
        bootstrap_verified=False,
        control_qualified=False,
        runtime_manifest_ref=record_ref,
        original_manifest_ref=original_ref,
        file_inventory=[dict(source_ref=asset_ref, snapshot_ref=asset_ref)],
    )
    for key in (
        "runtime_config_ref",
        "original_manifest_snapshot_ref",
        "original_config_snapshot_ref",
        "source_inventory_ref",
    ):
        snapshot[key] = e.write(artifact / f"{key}.json", {"metadata": key})
    snapshot_ref = e.write(artifact / "snapshot.json", snapshot)
    driver_ref = e.write(artifact / "crosscheck_driver.py", b"synthetic closed independent checker")
    cross = dict(
        complete=True,
        passed=True,
        refs_rechecked_unchanged=True,
        input_manifest_ref=snapshot_ref,
        historical_libero_sources=45,
        historical_robosuite_sources=120,
        outer_path_changes=3,
        inner_path_changes=5,
        inverse_config_object_equal=True,
        inverse_manifest_object_equal=True,
        original_manifest_and_config_byte_equal=True,
        driver_ref=driver_ref,
        preparation_receipt_ref=e.write(artifact / "preparation.json", {"passed": True}),
        registered_source_refs_unchanged=[asset_ref],
    )
    cross_ref = e.write(artifact / "crosscheck.json", cross)
    prior_execution = dict(
        passed=True,
        actual_wait=True,
        reaped=True,
        driver_unchanged=True,
        returncode=0,
        signal_sent=False,
        automatic_retry=False,
        result_ref=cross_ref,
        driver_ref=driver_ref,
        command=[str(e.cli.NATIVE_PYTHON), driver_ref["path"]],
    )
    for key in ("launch_ref", "repair_ref", "stdout_ref", "stderr_ref"):
        prior_execution[key] = e.write(artifact / f"{key}.json", {"preserved": key})
    prior_execution_ref = e.write(artifact / "crosscheck_execution.json", prior_execution)
    inputs = dict(
        original_manifest_ref=original_ref,
        current_only_refs=[dict(role="test_dependency", identity="current_only", ref=asset_ref)],
    )
    input_ref = e.write(artifact / "snapshot_inputs.json", inputs)
    controller = dict(robot_model="MountedPanda", control_freq=20)
    config = dict(
        schema="s2a.native-runtime-bootstrap-inputs.v1",
        seed=42,
        trial=0,
        timeout_seconds=600,
        render=dict(camera_name="agentview", depth=False, height=256, width=256),
        automatic_retry=False,
        control_qualified=False,
        historical_runtime_equivalence_claimed=False,
        actual_consumed_reference_count=0,
        fresh_cases=0,
        s2a_commands=0,
        world_model_calls=0,
        limits={key: core.LIMITS[value] for key, value in e.cli.LIMIT_NAMES.items()},
        output_dir=str(e.cli.RESULT.parent),
        snapshot_ref=snapshot_ref,
        snapshot_crosscheck_ref=cross_ref,
        snapshot_crosscheck_execution_ref=prior_execution_ref,
        snapshot_inputs_ref=input_ref,
        record_ref=record_ref,
        original_manifest_ref=original_ref,
        controller_ref=e.write(artifact / "controller.json", dict(contract=controller)),
        allowed_asset_roots=list(map(str, roots)),
        cv2_guard=dict(
            helper_ref=e.cli.reference(e.repo / "src/s2a/native_environment.py"),
            exact_probe_ref=e.write(artifact / "cv2.json", {"synthetic": True}),
            runtime_source_refs=[asset_ref],
        ),
    )
    e.write(e.cli.CONFIG, config)
    e.qa["source_sha256"]["configs/native_runtime_bootstrap_20261029.json"] = e.cli.reference(e.cli.CONFIG)["sha256"]
    e.qa_ref = e.write(e.qa_ref["path"], e.qa)
    monkeypatch.setattr(e.cli, "verify_runtime_snapshot", lambda ref: copy.deepcopy(snapshot))
    monkeypatch.setattr(preparation, "inspect_inputs", lambda source: ({}, [asset_ref]))
    counters = {name: dict(attempted=0, completed=0) for name in core.LIMITS}
    for name, count in dict(
        configure=1, model_construct=3, context=3, forward=16, step1=2, reset=1, render=1, read_pixels=1
    ).items():
        counters[name] = dict(attempted=count, completed=count)
    receipt = dict(
        complete=True,
        passed=True,
        runtime_loaded=True,
        bootstrap_verified=True,
        cleanup_complete=True,
        loaded_sources_unchanged=True,
        control_qualified=False,
        limits=dict(core.LIMITS),
        counters=counters,
        new_s2a_actions=0,
        new_physics_steps=0,
        new_world_model_calls=0,
        new_fresh_episodes=0,
        automatic_retries=0,
        controller_contract=controller,
        initial_state_check=dict(exact=True),
        renderer=dict(backend="osmesa"),
        frame=dict(pixel_min=0, pixel_max=255),
        source_refs=[dict(asset_ref, size_bytes=14)],
    )
    calls = []

    def restore(*args, **kwargs):
        calls.append("cv2")
        return dict(complete=True, passed=True)

    def bootstrap(runtime_config, record, output, **kwargs):
        calls.append("bootstrap")
        assert runtime_config == runtime and record == original["record"]
        assert output == e.cli.RESULT.parent / "work"
        assert kwargs == dict(allowed_asset_roots=list(map(str, roots)))
        return copy.deepcopy(receipt)

    monkeypatch.setattr(e.cli, "restore_cv2_environment", restore)
    monkeypatch.setattr(core, "run_native_bootstrap", bootstrap)
    return SimpleNamespace(
        **vars(e),
        core=core,
        config=config,
        snapshot=snapshot,
        manifest=manifest,
        controller=controller,
        cross=cross,
        prior_execution=prior_execution,
        asset_ref=asset_ref,
        calls=calls,
        receipt=receipt,
    )


def test_dry_verifies_files_but_never_imports_or_constructs_runtime(admitted):
    a = admitted
    result = a.cli.main(a.qa_ref, dry=True)
    assert result["passed"] and result["environment_initializations"] == 0
    assert a.calls == [] and not a.cli.RESULT.parent.exists()


def test_success_binds_all_sources_preserves_core_receipt_and_never_grants_control(admitted):
    a = admitted
    result = a.cli.main(a.qa_ref)
    assert a.calls == ["cv2", "bootstrap"]
    assert result["complete"] is result["passed"] is result["source_unchanged"] is True
    assert result["control_qualified"] is result["historical_runtime_equivalence_claimed"] is False
    assert result["bootstrap"] == a.receipt
    assert all(set(ref) == {"path", "sha256"} for ref in result["source_refs_unchanged"])
    assert a.asset_ref in result["source_refs_unchanged"]
    assert json.loads(a.cli.RESULT.read_text()) == result
    with pytest.raises(ValueError, match="exclusive"):
        a.cli.main(a.qa_ref)
    assert a.calls == ["cv2", "bootstrap"]


@pytest.mark.parametrize(
    "field,value",
    [("actual_wait", False), ("reaped", False), ("returncode", 1), ("signal_sent", True), ("automatic_retry", True)],
)
def test_crosscheck_requires_closed_success_not_partial_metadata(admitted, field, value):
    a = admitted
    a.prior_execution[field] = value
    a.config["snapshot_crosscheck_execution_ref"] = a.write(
        a.config["snapshot_crosscheck_execution_ref"]["path"], a.prior_execution
    )
    with pytest.raises(ValueError, match="actual wait/reap"):
        a.cli.inspect_inputs(a.config)
    assert a.calls == []


def test_crosscheck_result_and_snapshot_manifest_cannot_be_rebound(admitted):
    a = admitted
    a.prior_execution["result_ref"] = a.config["controller_ref"]
    a.config["snapshot_crosscheck_execution_ref"] = a.write(
        a.config["snapshot_crosscheck_execution_ref"]["path"], a.prior_execution
    )
    with pytest.raises(ValueError, match="exact result binding"):
        a.cli.inspect_inputs(a.config)
    a.prior_execution["result_ref"] = a.config["snapshot_crosscheck_ref"]
    a.config["snapshot_crosscheck_execution_ref"] = a.write(
        a.config["snapshot_crosscheck_execution_ref"]["path"], a.prior_execution
    )
    a.snapshot["runtime_manifest_ref"] = a.config["controller_ref"]
    with pytest.raises(ValueError, match="migrated manifests"):
        a.cli.inspect_inputs(a.config)


def test_changed_current_dependency_is_rejected_before_runtime(admitted):
    a = admitted
    Path(a.asset_ref["path"]).write_bytes(b"mutated dependency")
    with pytest.raises(ValueError, match="SHA mismatch"):
        a.cli.inspect_inputs(a.config)
    assert a.calls == []


def test_bootstrap_failure_keeps_actual_partial_counts_and_original_error(admitted, monkeypatch):
    a = admitted
    partial = copy.deepcopy(a.receipt)
    partial.update(
        complete=False, passed=False, bootstrap_verified=False, failure_stage="hard_reset", cleanup_complete=True
    )
    partial["counters"]["model_construct"] = dict(attempted=2, completed=1)

    def fail(*args, **kwargs):
        a.calls.append("bootstrap")
        raise a.core.BootstrapFailure("original reset failure", partial)

    monkeypatch.setattr(a.core, "run_native_bootstrap", fail)
    with pytest.raises(a.core.BootstrapFailure, match="original reset failure"):
        a.cli.main(a.qa_ref)
    result = json.loads(a.cli.RESULT.read_text())
    assert result["bootstrap"] == partial and result["failure_stage"] == "hard_reset"
    assert result["passed"] is result["complete"] is result["bootstrap_verified"] is False
    assert result["runtime_loaded"] is True and result["source_unchanged"] is True
    assert a.calls == ["cv2", "bootstrap"]


def test_cv2_failure_never_dispatches_bootstrap(admitted, monkeypatch):
    a = admitted
    partial = dict(complete=False, passed=False, after_import_cv2=dict(LD_LIBRARY_PATH="unexpected"))

    def fail(*args, **kwargs):
        a.calls.append("cv2")
        raise a.cli.EnvironmentAdmissionError("unknown import mutation", partial)

    monkeypatch.setattr(a.cli, "restore_cv2_environment", fail)
    with pytest.raises(a.cli.EnvironmentAdmissionError):
        a.cli.main(a.qa_ref)
    result = json.loads(a.cli.RESULT.read_text())
    assert result["cv2_admission"] == partial and result["runtime_loaded"] is False
    assert "bootstrap" not in result and a.calls == ["cv2"]


def test_source_mutation_after_bootstrap_fails_without_overwriting_old_digest(admitted, monkeypatch):
    a = admitted

    def changed(*args, **kwargs):
        a.calls.append("bootstrap")
        Path(a.asset_ref["path"]).write_bytes(b"mutation after dispatch")
        return copy.deepcopy(a.receipt)

    monkeypatch.setattr(a.core, "run_native_bootstrap", changed)
    with pytest.raises(ValueError, match="SHA mismatch"):
        a.cli.main(a.qa_ref)
    result = json.loads(a.cli.RESULT.read_text())
    assert result["passed"] is result["source_unchanged"] is False
    assert result["source_refs_unchanged"] == []
    assert a.asset_ref in result["source_refs_before"]


@pytest.mark.parametrize(
    "mutation",
    [
        "controller",
        "forbidden_step",
        "over_budget",
        "incomplete_model",
        "false_initial",
        "wrong_renderer",
        "changed_loaded_sources",
        "constant_frame",
    ],
)
def test_misleading_success_receipt_is_rejected(admitted, mutation):
    a = admitted
    receipt = copy.deepcopy(a.receipt)
    if mutation == "controller":
        receipt["controller_contract"]["robot_model"] = "Panda"
    elif mutation == "forbidden_step":
        receipt["counters"]["mj_step"] = dict(attempted=1, completed=0)
    elif mutation == "over_budget":
        receipt["counters"]["forward"] = dict(attempted=33, completed=33)
    elif mutation == "incomplete_model":
        receipt["counters"]["model_construct"] = dict(attempted=3, completed=2)
    elif mutation == "false_initial":
        receipt["initial_state_check"]["exact"] = False
    elif mutation == "constant_frame":
        receipt["frame"]["pixel_max"] = receipt["frame"]["pixel_min"]
    elif mutation == "changed_loaded_sources":
        receipt["loaded_sources_unchanged"] = False
    else:
        receipt["renderer"]["backend"] = "egl"
    with pytest.raises(ValueError):
        a.cli.validate_bootstrap(receipt, a.controller)


@pytest.mark.parametrize("bootstrap_failed,caller_handling_error", [(True, False), (False, False), (False, True)])
def test_publication_failure_preserves_original_exception_or_propagates_itself(
    admitted, monkeypatch, bootstrap_failed, caller_handling_error
):
    a = admitted
    primary = a.core.BootstrapFailure(
        "original bootstrap failure", dict(runtime_loaded=True, failure_stage="hard_reset")
    )
    if bootstrap_failed:

        def fail(*args, **kwargs):
            a.calls.append("bootstrap")
            raise primary

        monkeypatch.setattr(a.core, "run_native_bootstrap", fail)
    original_open = Path.open
    publication_error = OSError("synthetic result write failure")

    def fail_result_open(path, *args, **kwargs):
        if path == a.cli.RESULT and args == ("x",):
            raise publication_error
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_result_open)

    def invoke():
        with pytest.raises(type(primary) if bootstrap_failed else OSError) as caught:
            a.cli.main(a.qa_ref)
        assert caught.value is (primary if bootstrap_failed else publication_error)

    if caller_handling_error:
        try:
            raise LookupError("unrelated caller exception")
        except LookupError:
            invoke()
    else:
        invoke()
    if bootstrap_failed:
        assert any(
            "publication also failed: OSError: synthetic result write failure" in note for note in primary.__notes__
        )
    assert a.calls == ["cv2", "bootstrap"] and not a.cli.RESULT.exists()
