"""W2 interface, no-retry and finite-template tests with synthetic delegates."""

import ast
import importlib
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

MAIN = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(MAIN / "scripts"))


def ref(name):
    return dict(path=str(name), sha256="a" * 64)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream)
    from s2a.durable_stage_supervisor import reference

    return reference(path)


def test_finite_templates_match_actual_entry_asts():
    path = (
        MAIN
        / "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/runtime_w2_fit_20261024/preparation/build_entries.py"
    )
    spec = importlib.util.spec_from_file_location("_s2a047_templates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, expected in module.build().items():
        assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse((MAIN / "scripts" / name).read_text()))


@pytest.fixture
def scene(tmp_path, monkeypatch):
    assert tmp_path.is_relative_to(MAIN / "outputs")
    worker = importlib.import_module("execute_wam_clean_training_w2_fit")
    registration = {
        key: ref(key)
        for key in (
            "prepared_ref",
            "execution_registration_ref",
            "d0_collection_ref",
            "d1_collection_ref",
            "d0_capture_audit_ref",
            "d1_capture_audit_ref",
            "d1_owner_ref",
            "d1_handoff_owner_ref",
        )
    }
    registration.update(
        root=str(tmp_path / "control"),
        fit_output_dir=str(tmp_path / "fit/W2"),
        verification_output=str(tmp_path / "review/W2_verified.json"),
        runtime=dict(synthetic=True),
    )
    registered = ref(tmp_path / "admitted.json")
    old_verifier = ref("original_source_verifier")
    monkeypatch.setattr(worker, "_admit", lambda r: (registration, worker.Evidence(), old_verifier))
    calls = []

    @contextmanager
    def proof(ref):
        value = SimpleNamespace(
            registration_ref=ref,
            source_verifier_ref=old_verifier,
            callback_corrected_count=1,
            source_guard_completed=False,
        )
        try:
            yield value
        finally:
            value.source_guard_completed = True

    helper = SimpleNamespace(execution_source_proof=proof)
    contract_calls = []

    def context(*args, **kwargs):
        contract_calls.append((args, kwargs))
        return SimpleNamespace(
            stage="W2", rounds=("D0", "D1"), expected_rows=840000, evidence=SimpleNamespace(finish=lambda: None)
        )

    contract = SimpleNamespace(load_context=context)

    def fit(*args, **kwargs):
        calls.append(("fit", args, kwargs))
        directory = Path(kwargs["output_dir"])
        directory.mkdir(parents=True, exist_ok=False)
        params = dump(directory / "parameters.json", dict(synthetic=True))
        return dump(
            directory / "result.json",
            dict(
                schema_version="s2a.clean-training-fit.v1",
                complete=True,
                passed=True,
                stage="W2",
                prepared_ref=args[0],
                execution_registration_ref=args[1],
                collection_refs=kwargs["collection_refs"],
                capture_audit_refs=kwargs["capture_audit_refs"],
                rounds=["D0", "D1"],
                producer_fits=1,
                fit_attempts=1,
                actual_rows=840000,
                params_ref=params,
            ),
        )

    def independent(*args, **kwargs):
        calls.append(("qr", args, kwargs))
        produced = json.loads(Path(args[2]["path"]).read_text())
        return dump(
            Path(kwargs["output"]),
            dict(
                schema_version="s2a.clean-training-fit-verification.v1",
                complete=True,
                passed=True,
                stage="W2",
                prepared_ref=args[0],
                execution_registration_ref=args[1],
                fit_ref=args[2],
                params_ref=produced["params_ref"],
                actual_rows=840000,
                independent_reconstructions=1,
                full_training_row_reconstruction=True,
            ),
        )

    monkeypatch.setattr(
        worker,
        "_load_modules",
        lambda role, r: (
            fit if role == "producer" else independent,
            helper,
            contract,
            ref("original_math_source"),
            ref("original_source_helper"),
        ),
    )
    return SimpleNamespace(
        worker=worker,
        registration=registration,
        registered=registered,
        calls=calls,
        context_calls=contract_calls,
        contract=contract,
    )


@pytest.mark.parametrize("role", ["producer", "independent"])
def test_two_round_dry_context_and_840k_rows(scene, role):
    result = scene.worker.execute(scene.registered, role=role, dry_run=True)
    assert result["complete"] is result["passed"] is True and result["expected_rows"] == 840000
    assert (
        result["runtime"] == scene.registration["runtime"]
        and result["d1_owner_ref"] == scene.registration["d1_owner_ref"]
    )
    assert not scene.calls and result["producer_fits"] == result["independent_reconstructions"] == 0
    args, kw = scene.context_calls[0]
    assert kw["stage"] == "W2" and kw["collection_refs"] == [
        scene.registration["d0_collection_ref"],
        scene.registration["d1_collection_ref"],
    ]
    assert kw["capture_audit_refs"] == [
        scene.registration["d0_capture_audit_ref"],
        scene.registration["d1_capture_audit_ref"],
    ]


def test_actual_wrapper_interface_passes_two_rounds_then_binds_independent_result(scene):
    owner = importlib.import_module("run_wam_clean_training_w2_fit")
    first = scene.worker.execute(scene.registered, role="producer")
    assert json.loads(Path(first["path"]).read_text())["producer_fits"] == 1
    owner.check_worker(scene.registration, scene.registered, "producer")
    second = scene.worker.execute(scene.registered, role="independent")
    assert json.loads(Path(second["path"]).read_text())["independent_reconstructions"] == 1
    owner.check_worker(scene.registration, scene.registered, "independent")
    assert [call[0] for call in scene.calls] == ["fit", "qr"]
    _, _, kw = scene.calls[0]
    assert kw["stage"] == "W2" and len(kw["collection_refs"]) == len(kw["capture_audit_refs"]) == 2
    assert scene.calls[1][2]["stage"] == "W2"
    for role in ("producer", "independent"):
        with pytest.raises(ValueError, match="already exists"):
            scene.worker.execute(scene.registered, role=role)
    assert len(scene.calls) == 2


def test_wrong_dry_rounds_rejected_before_math(scene):
    scene.contract.load_context = lambda *args, **kw: SimpleNamespace(
        stage="W2", rounds=("D0",), expected_rows=840000, evidence=SimpleNamespace(finish=lambda: None)
    )
    with pytest.raises(ValueError, match="dry context"):
        scene.worker.execute(scene.registered, role="producer", dry_run=True)
    assert not scene.calls


def test_owner_rejects_changed_d1_provenance(scene):
    owner = importlib.import_module("run_wam_clean_training_w2_fit")
    output = scene.worker.execute(scene.registered, role="producer")
    p = Path(output["path"])
    value = json.loads(p.read_text())
    value["d1_owner_ref"] = ref("other_d1")
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="D1 provenance"):
        owner.check_worker(scene.registration, scene.registered, "producer")


def test_w2_owner_routes_only_the_qualified_sim_interpreter():
    owner = importlib.import_module("run_wam_clean_training_w2_fit")
    for role in ("producer", "independent"):
        command = owner.worker_argv(ref("registered"), role)
        assert (
            command[0] == str(MAIN / "outputs/libero_env311/bin/python")
            and command[1] == "scripts/execute_wam_clean_training_w2_fit.py"
        )


@pytest.mark.parametrize("fault", ["none", "producer_exit7"])
def test_w2_supervisor_real_serial_cpu_children(tmp_path, monkeypatch, fault):
    import os
    import sys

    from s2a import clean_training_w2_admission as admission
    from s2a import durable_stage_supervisor as durable

    assert tmp_path.is_relative_to(MAIN / "outputs")
    source = MAIN / "tests/test_clean_training_postcapture_owner.py"
    spec = importlib.util.spec_from_file_location("_s2a047_original_cpu_fixture", source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    tiny = (
        old.TINY_WORKER.replace("W1", "W2")
        .replace("420000", "840000")
        .replace('rounds=["D0"]', 'rounds=["D0", "D1"]')
        .replace("s2a.clean-training-postcapture-", "s2a.clean-training-w2-runtime-fit-")
    )
    needle = "delegate_result_ref=delegate_ref, synthetic=True)"
    assert tiny.count(needle) == 1
    tiny = tiny.replace(
        needle,
        'delegate_result_ref=delegate_ref, d1_owner_ref=registration["d1_owner_ref"], d1_handoff_owner_ref=registration["d1_handoff_owner_ref"], runtime=registration["runtime"], synthetic=True)',
    )
    root = tmp_path / "synthetic047"
    root.mkdir()
    prepared = dump(
        root / "prepared.json", dict(resources=dict(fit_timeout_s=3, independent_fit_timeout_s=3), synthetic=True)
    )
    registered = dump(root / "admitted.json", dict(synthetic=True))
    execution = dump(root / "original_registration.json", dict(synthetic=True))
    launch = dump(root / "fake_detached.json", dict(synthetic=True))
    registration = dict(
        root=str(root),
        prepared_ref=prepared,
        execution_registration_ref=execution,
        d0_owner_ref=ref("synthetic_d0"),
        d1_owner_ref=ref("synthetic_d1"),
        d1_handoff_owner_ref=ref("synthetic_046"),
        failure_ref=ref("synthetic_failure"),
        runtime=dict(synthetic=True),
        fit_output_dir=str(root / "fit/W2"),
        verification_output=str(root / "review/W2_verified.json"),
    )
    config = dump(
        root / "tiny_config.json",
        dict(
            registration=registration,
            repo=str(MAIN),
            fault=fault,
            repair_ref=registered,
            synthetic_verifier_ref=ref("synthetic_source_verifier"),
        ),
    )
    worker = root / "tiny.py"
    worker.write_text(tiny)
    owner = importlib.import_module("run_wam_clean_training_w2_fit")
    registration["source_sha256"] = {str(p): durable.sha(p) for p in (MAIN / owner.ENTRY, MAIN / owner.WORKER, worker)}
    monkeypatch.setattr(admission, "admit", lambda r: registration)
    monkeypatch.setattr(owner, "accept_detached_launch", lambda *args, **kwargs: launch)
    monkeypatch.setattr(owner, "worker_argv", lambda r, role: [sys.executable, str(worker), config["path"], role])
    for name in durable.ENV_KEYS:
        monkeypatch.setenv(name, durable.cpu_environment()[name])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["LD_LIBRARY_PATH"] == ""
    if fault == "none":
        result = owner.execute(registered, mode="run")
        assert durable.read(result)["passed"] is True
        expected = ("producer", "independent")
    else:
        with pytest.raises(Exception):
            owner.execute(registered, mode="run")
        assert json.loads((root / "supervisor/result.json").read_text())["passed"] is False
        assert not (root / "supervisor/independent").exists()
        expected = ("producer",)
    for role in expected:
        exited = json.loads((root / f"supervisor/{role}/exit.json").read_text())
        release = json.loads((root / f"supervisor/{role}/release.json").read_text())
        assert exited["actual_wait"] is release["reaped"] is release["actual_wait"] is True
        assert release["signal_sent"] is False
        assert exited["returncode"] == release["returncode"] == (7 if fault == "producer_exit7" else 0)
