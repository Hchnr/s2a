"""Aggregation must not read labels while capture or lifecycle is incomplete."""

import copy
import json

import pytest

from s2a.artifacts import REPO
from s2a.clean_training_aggregation import RIDGE, ROWS, admit_aggregation, normal_exit
from s2a.clean_training_runtime import reference


def policy():
    return dict(
        schema_version="s2a.clean-aggregation-fit-registration.v1",
        rows=ROWS,
        ridge=RIDGE,
        producer_fits=1,
        automatic_retry=False,
        d2_owner_ref="owner",
        previous_fit_ref="old",
        previous_verification_ref="verified",
    )


def lifecycle():
    entry = dict(launch="launch", exit="exit", release="release")
    records = dict(
        launch=dict(child=dict(pid=123, start_ticks=999)),
        exit=dict(returncode=0, actual_wait=True, automatic_retry=False),
        release=dict(
            pid=123,
            start_ticks=999,
            reaped=True,
            returncode=0,
            signal_sent=False,
            original_error=None,
            only_owned_child_signalled=True,
        ),
    )
    return entry, records


@pytest.mark.parametrize("fault", ["wait", "reap", "signal", "pid", "birth"])
def test_receipt_identity_and_normal_exit_are_required(fault):
    entry, records = lifecycle()
    if fault == "wait":
        records["exit"]["actual_wait"] = False
    elif fault == "reap":
        records["release"]["reaped"] = False
    elif fault == "signal":
        records["release"]["signal_sent"] = True
    elif fault == "pid":
        records["release"]["pid"] += 1
    else:
        records["release"]["start_ticks"] += 1
    with pytest.raises(ValueError):
        normal_exit(entry, records.__getitem__)


def test_real_producer_rejects_live_capture_before_label_or_fit(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import fit_wam_clean_aggregation as producer

    path = tmp_path / "live_owner.json"
    path.write_text(json.dumps(dict(schema_version="s2a.clean-d2-owner.v1", complete=False, passed=False)))
    spec = policy()
    spec["d2_owner_ref"] = reference(path)
    monkeypatch.setattr(producer.np, "load", lambda *a, **k: pytest.fail("opened labels before D2 exit"))
    monkeypatch.setattr(
        producer, "fit_clean_residual_streaming", lambda *a, **k: pytest.fail("fit began before D2 exit")
    )
    with pytest.raises(ValueError, match="full D2 owner"):
        producer.execute(spec, tmp_path)
    assert not (tmp_path / "parameters.json").exists()


@pytest.mark.parametrize(
    "key,value", [("ridge", 0.1), ("rows", 840000), ("producer_fits", 2), ("automatic_retry", True)]
)
def test_policy_cannot_silently_fall_back_to_old_two_round_fit(key, value):
    spec = policy()
    spec[key] = value
    with pytest.raises(ValueError, match="fixed aggregation policy"):
        admit_aggregation(spec, read_json=lambda ref: pytest.fail("invalid policy read evidence"))


def test_full_json_admission_and_round_order():
    entry, docs = lifecycle()
    requests = [
        dict(index=i, name=f"r{i}", anchor_name=f"a{i // 2}", sampling_seed=1000 + i, complete=True) for i in range(300)
    ]
    params = dict(sha256="ca01b9d8a49f2f1585c917f492e52a0a89cdb0b447f6bd2beee050ed0d2fccc0")
    docs["owner"] = dict(
        schema_version="s2a.clean-d2-owner.v1",
        complete=True,
        passed=True,
        owned_commands=[copy.deepcopy(entry), copy.deepcopy(entry)],
        automatic_retry=False,
        fresh_authorized=False,
        gpu_after_ref="gpu",
        collection_ref="D2",
        numerical_ref="audit",
        prepared_ref="prepared",
        registration_ref="registered",
    )
    docs["gpu"] = dict(
        released=True,
        uuid="GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f",
        memory_mib=0,
        compute_apps=[],
        device_minor=6,
        visible_gpu=2,
    )
    for name in ("D0", "D1", "D2"):
        docs[name] = dict(
            complete=True, passed=True, round_name=name, requests=copy.deepcopy(requests), normalization_ref="normal"
        )
    docs["D2"].update(
        schema_version="s2a.clean-d2-capture.v1",
        actual_requests=300,
        actual_denoise_calls=10500,
        actual_head_calls=10500,
        prepared_ref="prepared",
        execution_registration_ref="registered",
        readout_ref=params,
        future_label_reads=0,
        producer_fits=0,
        physics_steps=0,
        fresh_cases=0,
    )
    docs["audit"] = dict(
        schema_version="s2a.clean-d2-capture-audit.v1",
        complete=True,
        passed=True,
        collection_ref="D2",
        actual_requests=300,
        actual_denoise_calls=10500,
        initial_pairs_verified=300,
        prepared_ref="prepared",
        execution_registration_ref="registered",
        readout_ref=params,
        worker_launch_ref="launch",
        worker_exit_ref="exit",
        worker_release_ref="release",
        gpu_after_ref="gpu",
    )
    docs["old"] = dict(
        complete=True,
        passed=True,
        actual_rows=840000,
        params_ref="params",
        collection_refs=["D0", "D1"],
        prepared_ref="labels",
    )
    docs["verified"] = dict(complete=True, passed=True, fit_ref="old", params_ref="params")
    docs["labels"] = dict(training_anchors=[dict(name=f"a{i}") for i in range(150)])
    _, _, refs, _ = admit_aggregation(policy(), read_json=docs.__getitem__)
    assert refs == ["D0", "D1", "D2"]
    docs["D2"]["requests"][0]["sampling_seed"] = 42
    with pytest.raises(ValueError, match="identities differ"):
        admit_aggregation(policy(), read_json=docs.__getitem__)


def test_cpu_owner_reaps_failed_real_child_and_never_starts_qr(tmp_path, monkeypatch):
    import sys

    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import run_wam_clean_aggregation as owner

    command = [sys.executable, "-c", "raise SystemExit(7)"]
    regref = dict(path=str(tmp_path / "registration.json"), sha256="1" * 64)
    dry = dict(passed=True, registration_ref=regref, command=command + ["--dry"])
    (tmp_path / "successful_cli_dry.json").write_text(json.dumps(dry))
    monkeypatch.setattr(owner, "ROOT", tmp_path)
    monkeypatch.setattr(owner, "admit", lambda ref: {})

    def worker_command(spec, ref, *, independent=False):
        assert not independent, "QR launched after producer failure"
        return command

    monkeypatch.setattr(owner, "worker_command", worker_command)
    with pytest.raises(ChildProcessError, match="failed with7"):
        owner.execute(regref)
    folder = tmp_path / "orchestration"
    result = json.loads((folder / "result.json").read_text())
    exit_record = json.loads((folder / "producer/exit.json").read_text())
    release = json.loads((folder / "producer/ownership/release.json").read_text())
    assert result["complete"] is result["passed"] is False
    assert result["available_failure_receipts"]
    assert exit_record["returncode"] == 7 and exit_record["actual_wait"] is True
    assert release["reaped"] is True and release["signal_sent"] is False
    assert not (folder / "independent").exists()
