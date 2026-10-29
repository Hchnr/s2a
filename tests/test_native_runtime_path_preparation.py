"""Exercise the preparation CLI with synthetic files, never simulator libraries.

The file-snapshot implementation is replaced at its three entry points. Historical
source hashes and QA files are real temporary files; system-library references
resolve through an explicit temporary-file table, without reading /usr libraries.
"""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_native_runtime_path_snapshot.py"


@pytest.fixture
def preparation(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    safe = tmp_path / "safe_source"
    old = tmp_path / "historical_declaration_only"
    repo.mkdir()
    calls = dict(inspect=[], prepare=[], verify=[], checked=[])
    stub = ModuleType("s2a.native_runtime_path_snapshot")
    stub.REPO, stub.SAFE_SOURCE_ROOT, stub.OLD_SOURCE_ROOT = repo, safe, old

    def forbidden(*args, **kwargs):
        pytest.fail("Unconfigured snapshot operation")

    for name in ("inspect_runtime_sources", "prepare_runtime_snapshot", "verify_runtime_snapshot"):
        setattr(stub, name, forbidden)
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    spec = importlib.util.spec_from_file_location("_synthetic_runtime_preparation", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "NATIVE_PYTHON", Path(sys.executable))
    monkeypatch.chdir(repo)
    for key, value in cli.CPU_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)

    def write(path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
        return cli.reference(path)

    for relative in cli.SOURCE_PATHS:
        write(repo / relative, f"synthetic source: {relative}\n".encode())
    histories = {"libero": [], "robosuite": []}
    source_hashes = {}
    installed = repo / "outputs/libero_env311/lib/python3.11/site-packages/robosuite"
    for family, count in (("libero", 45), ("robosuite", 120)):
        for index in range(count):
            relative = Path(f"LIBERO/libero/libero/source_{index:03}.py")
            actual = safe / relative if family == "libero" else installed / f"source_{index:03}.py"
            declared = old / relative if family == "libero" else actual
            ref = write(actual, f"{family} historical source {index}\n".encode())
            source_hashes[str(declared)] = ref["sha256"]
            histories[family].append((declared, actual))
    original = dict(
        runtime_config=dict(task_suite="libero_goal", task_id=8, seed=42, refresh_controller_after_restore=True),
        simulator_source_sha256=source_hashes,
    )
    manifest_path = repo / "inputs/original_manifest.json"
    virtual_files = {}
    support = []
    for index in range(18):
        backing = tmp_path / "system_files" / f"library_{index:02}.so"
        ref = write(backing, f"synthetic library bytes {index}".encode())
        declared = Path("/usr/lib/x86_64-linux-gnu") / backing.name
        virtual_files[str(declared)] = backing
        support.append(dict(role="system_library", identity="current_only", ref={**ref, "path": str(declared)}))
    for role, path in (
        ("task_bddl", safe / "LIBERO/libero/libero/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl"),
        ("default_osc_json", installed / "controllers/config/osc_pose.json"),
    ):
        support.append(dict(role=role, identity="current_only", ref=write(path, f"{role}\n".encode())))
    config = dict(
        schema="s2a.native-runtime-path-snapshot-inputs.v1",
        original_manifest_ref=write(manifest_path, original),
        original_config_ref=write(safe / "config/config.yaml", {"synthetic": "configuration"}),
        osmesa_ref=write(safe / "systemlibs/libOSMesa.so", b"not an ELF library"),
        current_only_refs=support,
        expected_historical_libero_sources=45,
        expected_historical_robosuite_sources=120,
        output_dir=str(repo / "outputs/native_runtime_path_snapshot_20261029"),
        max_files=10000,
        max_bytes=2**30,
    )
    write(cli.CONFIG, config)
    real_checked = cli.checked

    def checked(ref):
        calls["checked"].append(copy.deepcopy(ref))
        if isinstance(ref, dict) and ref.get("path") in virtual_files:
            assert set(ref) == {"path", "sha256"}
            cli.require(cli.reference(virtual_files[ref["path"]])["sha256"] == ref["sha256"], "File path/SHA mismatch")
            return Path(ref["path"])
        return real_checked(ref)

    monkeypatch.setattr(cli, "checked", checked)
    inspected = dict(copied_file_count=48, copied_bytes=4321, historical_libero_file_count=45, license_included=True)

    def inspect(*args, **kwargs):
        calls["inspect"].append((args, kwargs))
        assert args == tuple(config[key] for key in ("original_manifest_ref", "original_config_ref", "osmesa_ref"))
        assert kwargs == dict(max_files=10000, max_bytes=2**30)
        return copy.deepcopy(inspected)

    def prepare(*args, **kwargs):
        calls["prepare"].append((args, kwargs))
        destination = Path(args[3])
        destination.mkdir()
        return write(destination / "manifest.json", {"synthetic_snapshot": True})

    snapshot = dict(
        file_snapshot_verified=True, runtime_loaded=False, bootstrap_verified=False, control_qualified=False
    )

    def verify(ref):
        calls["verify"].append(ref)
        real_checked(ref)
        return copy.deepcopy(snapshot)

    monkeypatch.setattr(cli, "inspect_runtime_sources", inspect)
    monkeypatch.setattr(cli, "prepare_runtime_snapshot", prepare)
    monkeypatch.setattr(cli, "verify_runtime_snapshot", verify)
    qa = dict(
        complete=True,
        passed=True,
        actual_wait=True,
        returncode=0,
        signal_sent=False,
        automatic_retry=False,
        environment=dict(cli.CPU_ENVIRONMENT),
        python_version="3.11.15",
        python_executable=str(cli.NATIVE_PYTHON),
        command=[str(cli.NATIVE_PYTHON), "-m", "pytest", "-q", *cli.QA_TESTS, f"--basetemp={repo}/outputs/pytest_qa"],
        source_sha256={path: cli.reference(repo / path)["sha256"] for path in cli.SOURCE_PATHS},
        log_ref=write(repo / "qa.log", b"synthetic successful pytest process\n"),
    )
    qa_path = repo / "qa.json"
    return SimpleNamespace(
        cli=cli,
        repo=repo,
        old=old,
        calls=calls,
        config=config,
        original=original,
        histories=histories,
        manifest_path=manifest_path,
        qa=qa,
        qa_path=qa_path,
        qa_ref=write(qa_path, qa),
        write=write,
        snapshot=snapshot,
        inspected=inspected,
    )


def assert_unqualified(result):
    assert result["runtime_loaded"] is result["bootstrap_verified"] is result["control_qualified"] is False
    assert result["automatic_retry"] is False
    assert all(
        result[key] == 0
        for key in (
            "response_evaluations",
            "numerical_inner_steps",
            "physics_steps",
            "world_model_calls",
            "fresh_cases",
        )
    )


def test_inspection_hashes_every_historical_source_without_promoting_current_dependencies(preparation):
    f = preparation
    inspected, refs = f.cli.inspect_inputs(f.config)
    expected = {str(actual) for group in f.histories.values() for _, actual in group}
    assert len(expected) == 165
    assert expected <= {ref["path"] for ref in refs}
    assert expected <= {ref["path"] for ref in f.calls["checked"]}
    assert inspected["historical_libero_file_count"] == 45
    assert len(f.calls["inspect"]) == 1
    assert all(item["identity"] == "current_only" for item in f.config["current_only_refs"])
    assert not f.old.exists()
    assert f.calls["prepare"] == f.calls["verify"] == []


@pytest.mark.parametrize("family", ["libero", "robosuite"])
@pytest.mark.parametrize("fault", ["missing", "changed"])
def test_missing_or_changed_historical_tail_cannot_be_hidden_by_other_matches(preparation, family, fault):
    f = preparation
    declared, actual = f.histories[family][-1]
    if fault == "missing":
        del f.original["simulator_source_sha256"][str(declared)]
        f.config["original_manifest_ref"] = f.write(f.manifest_path, f.original)
    else:
        actual.write_bytes(b"modified since historical registration")
    with pytest.raises(ValueError, match="historical|SHA mismatch"):
        f.cli.inspect_inputs(f.config)
    assert f.calls["inspect"] == []


@pytest.mark.parametrize("fault", ["historical_label", "missing_library", "duplicate_library", "missing_bddl"])
def test_current_dependencies_require_distinct_complete_current_only_refs(preparation, fault):
    f = preparation
    support = f.config["current_only_refs"]
    if fault == "historical_label":
        support[0]["identity"] = "historical_match"
    elif fault == "missing_library":
        support.pop(0)
    elif fault == "duplicate_library":
        support[1] = copy.deepcopy(support[0])
    else:
        support[:] = [item for item in support if item["role"] != "task_bddl"]
    with pytest.raises(ValueError, match="current-only dependencies"):
        f.cli.inspect_inputs(f.config)
    assert f.calls["inspect"] == []


@pytest.mark.parametrize("fault", ["missing_wait", "wait_false", "gpu", "library_path", "retry", "failed_exit"])
def test_qa_requires_actual_successful_cpu_wait_without_retry(preparation, fault):
    f = preparation
    if fault == "missing_wait":
        del f.qa["actual_wait"]
    elif fault == "wait_false":
        f.qa["actual_wait"] = False
    elif fault == "gpu":
        f.qa["environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    elif fault == "library_path":
        f.qa["environment"]["LD_LIBRARY_PATH"] = "/unregistered"
    elif fault == "retry":
        f.qa["automatic_retry"] = True
    else:
        f.qa["returncode"] = 1
    with pytest.raises((KeyError, ValueError)):
        f.cli.verify_native_qa(f.write(f.qa_path, f.qa))
    assert f.calls["prepare"] == []


def test_qa_cannot_omit_any_source_or_either_executed_test(preparation):
    f = preparation
    for path in f.cli.SOURCE_PATHS:
        qa = copy.deepcopy(f.qa)
        del qa["source_sha256"][path]
        with pytest.raises(ValueError, match="Exact final preparation sources"):
            f.cli.verify_native_qa(f.write(f.qa_path, qa))
    for path in f.cli.QA_TESTS:
        qa = copy.deepcopy(f.qa)
        qa["command"].remove(path)
        with pytest.raises(ValueError, match="test modules executed"):
            f.cli.verify_native_qa(f.write(f.qa_path, qa))


@pytest.mark.parametrize("fault", ["string", "collect_only", "test_substring", "outside_basetemp"])
def test_qa_command_must_execute_the_exact_two_tests_in_repository(preparation, fault):
    f = preparation
    if fault == "string":
        f.qa["command"] = " ".join(f.qa["command"])
    elif fault == "collect_only":
        f.qa["command"].insert(3, "--collect-only")
    elif fault == "test_substring":
        f.qa["command"][4] += ".not_executed"
    else:
        f.qa["command"][-1] = f"--basetemp={f.repo.parent}/outside_qa"
    with pytest.raises(ValueError):
        f.cli.verify_native_qa(f.write(f.qa_path, f.qa))


def test_source_changed_after_qa_is_rejected_before_inspection(preparation):
    f = preparation
    (f.repo / f.cli.SOURCE_PATHS[0]).write_bytes(b"untested source revision")
    with pytest.raises(ValueError, match="SHA mismatch"):
        f.cli.main(f.qa_ref, dry=True)
    assert f.calls["inspect"] == f.calls["prepare"] == []
    assert not f.cli.RESULT.parent.exists()


@pytest.mark.parametrize("dry", [True, False])
def test_inspection_time_source_drift_blocks_both_dry_and_preparation(preparation, monkeypatch, dry):
    f = preparation
    original_inspect = f.cli.inspect_runtime_sources

    def changing_source(*args, **kwargs):
        result = original_inspect(*args, **kwargs)
        (f.repo / f.cli.SOURCE_PATHS[0]).write_bytes(b"changed during source inspection")
        return result

    monkeypatch.setattr(f.cli, "inspect_runtime_sources", changing_source)
    with pytest.raises(ValueError, match="SHA mismatch"):
        f.cli.main(f.qa_ref, dry=dry)
    assert len(f.calls["inspect"]) == 1
    assert f.calls["prepare"] == f.calls["verify"] == []
    assert not f.cli.RESULT.parent.exists()
    assert not Path(f.config["output_dir"]).exists()


def test_dry_runs_real_admission_without_preparing_or_writing_any_directory(preparation, capsys):
    f = preparation
    before = {path.relative_to(f.repo): path.read_bytes() for path in f.repo.rglob("*") if path.is_file()}
    before_dirs = {path.relative_to(f.repo) for path in f.repo.rglob("*") if path.is_dir()}
    f.cli.main(f.qa_ref, dry=True)
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is output["passed"] is True
    assert output["new_responses"] == output["physics_steps"] == 0
    assert len(f.calls["inspect"]) == 1
    assert f.calls["prepare"] == f.calls["verify"] == []
    assert {path.relative_to(f.repo): path.read_bytes() for path in f.repo.rglob("*") if path.is_file()} == before
    assert {path.relative_to(f.repo) for path in f.repo.rglob("*") if path.is_dir()} == before_dirs


def test_success_records_file_preparation_without_runtime_or_control_qualification(preparation):
    f = preparation
    f.cli.main(f.qa_ref)
    result = json.loads(f.cli.RESULT.read_bytes())
    assert result["complete"] is result["passed"] is True
    assert len(f.calls["prepare"]) == len(f.calls["verify"]) == 1
    assert result["current_only_refs"] == f.config["current_only_refs"]
    assert result["source_refs_before"] == result["source_refs_unchanged"]
    assert result["native_qa_ref"] == f.qa_ref
    assert_unqualified(result)


def test_mid_copy_failure_retains_partial_files_and_failure_receipt_without_retry(preparation, monkeypatch):
    f = preparation
    partial = Path(f.config["output_dir"]) / "partial.bin"

    def interrupted(*args, **kwargs):
        f.calls["prepare"].append((args, kwargs))
        f.write(partial, b"partial copy retained")
        raise OSError("synthetic copy interrupted")

    monkeypatch.setattr(f.cli, "prepare_runtime_snapshot", interrupted)
    with pytest.raises(OSError, match="copy interrupted"):
        f.cli.main(f.qa_ref)
    receipt = f.cli.RESULT.read_bytes()
    result = json.loads(receipt)
    assert result["complete"] is result["passed"] is False
    assert "copy interrupted" in result["error"]
    assert_unqualified(result)
    with pytest.raises(ValueError, match="preserve failures without retry"):
        f.cli.main(f.qa_ref)
    assert f.cli.RESULT.read_bytes() == receipt
    assert partial.read_bytes() == b"partial copy retained"
    assert len(f.calls["prepare"]) == 1 and not f.calls["verify"]


@pytest.mark.parametrize("field", ["runtime_loaded", "bootstrap_verified", "control_qualified"])
def test_snapshot_verifier_cannot_upgrade_file_preparation_qualification(preparation, field):
    f = preparation
    f.snapshot[field] = True
    with pytest.raises(ValueError, match="File preparation alone grants no"):
        f.cli.main(f.qa_ref)
    result = json.loads(f.cli.RESULT.read_bytes())
    assert result["complete"] is result["passed"] is False
    assert_unqualified(result)


def test_post_preparation_source_drift_cannot_be_rehashed_into_success(preparation, monkeypatch):
    f = preparation
    original_prepare = f.cli.prepare_runtime_snapshot

    def changing_source(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        (f.repo / f.cli.SOURCE_PATHS[0]).write_bytes(b"changed after QA")
        return result

    monkeypatch.setattr(f.cli, "prepare_runtime_snapshot", changing_source)
    with pytest.raises(ValueError, match="SHA mismatch"):
        f.cli.main(f.qa_ref)
    result = json.loads(f.cli.RESULT.read_bytes())
    assert result["complete"] is result["passed"] is False
    original_digest = f.qa["source_sha256"][f.cli.SOURCE_PATHS[0]]
    assert any(ref["sha256"] == original_digest for ref in result["source_refs_before"])
    assert_unqualified(result)


def test_symlinked_outputs_ancestor_is_rejected_before_any_external_write(preparation):
    f = preparation
    outputs = f.repo / "outputs"
    outside = f.repo.parent / "external_outputs"
    outputs.rename(outside)
    outputs.symlink_to(outside, target_is_directory=True)
    before = {path.relative_to(outside): path.read_bytes() for path in outside.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="symlinks"):
        f.cli.main(f.qa_ref)
    assert f.calls["inspect"] == f.calls["prepare"] == []
    assert {path.relative_to(outside): path.read_bytes() for path in outside.rglob("*") if path.is_file()} == before
    assert not (outside / f.cli.RESULT.parent.name).exists()


@pytest.mark.parametrize("location", ["repo_root", "outside", "traversal"])
def test_result_path_cannot_escape_the_registered_outputs_area(preparation, monkeypatch, location):
    f = preparation
    target = {
        "repo_root": f.repo / "result.json",
        "outside": f.repo.parent / "external/result.json",
        "traversal": f.repo / "outputs/../escape/result.json",
    }[location]
    monkeypatch.setattr(f.cli, "RESULT", target)
    with pytest.raises(ValueError, match="outputs|traversal"):
        f.cli.main(f.qa_ref)
    assert not target.exists()
    assert f.calls["inspect"] == f.calls["prepare"] == []


def test_json_read_rechecks_bytes_after_reference_validation(preparation, monkeypatch):
    f = preparation
    real_checked = f.cli.checked

    def racing_checked(ref):
        path = real_checked(ref)
        if path == f.qa_path:
            path.write_bytes(b'{"complete": true, "changed": true}')
        return path

    monkeypatch.setattr(f.cli, "checked", racing_checked)
    assert hashlib.sha256(f.qa_path.read_bytes()).hexdigest() == f.qa_ref["sha256"]
    with pytest.raises(ValueError, match="JSON changed while reading"):
        f.cli.verify_native_qa(f.qa_ref)


def test_correctly_hashed_json_cannot_introduce_nonfinite_numbers(preparation):
    f = preparation
    ref = f.write(f.repo / "overflow.json", b'{"max_bytes": 1e999}')
    with pytest.raises(ValueError, match="Finite JSON number required"):
        f.cli.read(ref)
