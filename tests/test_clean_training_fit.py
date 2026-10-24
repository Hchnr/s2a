"""038 synthetic CPU fit/phase/source tests; never open actual training futures."""

import copy
import importlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from execution_proof_fixture import effective_training_metadata, seal_execution, training_seed_metadata
from scipy.spatial.transform import Rotation

from s2a import clean_training_fit_contract as contract
from s2a.artifacts import REPO
from s2a.clean_training_seed_plan import apply_seed_plan
from s2a.geometry import transform
from s2a.native_training import synchronous_labels
from s2a.streaming_ridge_qr import independent_ridge
from s2a.wam_clean_readout import CleanResidualReadout
from s2a.wam_clean_streaming import CleanFitBlock, CleanFitLayout


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("fit_wam_clean_training"), importlib.import_module("verify_wam_clean_training_fit")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return contract.reference(path)


@pytest.fixture
def admission(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("# synthetic admitted source\n")
    snapshot = tmp_path / "snapshot.py"
    snapshot.write_bytes(source.read_bytes())
    sources, snapshots = {str(source): contract.sha256(source)}, {str(snapshot): contract.sha256(snapshot)}
    prepared = dict(
        schema_version=contract.PREPARATION_SCHEMA,
        root=str(tmp_path),
        config=dict(ridge=0.1, feature_terms=[]),
        resources=dict(maximum_fit_block_rows=1400),
        **training_seed_metadata(),
        snapshot_sha256=snapshots,
    )
    prepared_ref = dump(tmp_path / "prepared.json", prepared)
    metadata_ref = dump(
        tmp_path / "metadata.json",
        dict(
            schema_version="s2a.clean-training-preparation-independent.v1",
            complete=True,
            passed=True,
            prepared_ref=prepared_ref,
            metadata_qualified=True,
            execution_ready=False,
            execution_authorized=False,
            consumed_before_after_exact=True,
            verifier_ref=contract.reference(source),
        ),
    )
    registration = dict(
        schema_version=contract.REGISTRATION_SCHEMA,
        complete=True,
        passed=True,
        prepared_ref=prepared_ref,
        automatic_retry=False,
        readiness=dict(ready_for_execution=True),
        metadata_verification_ref=metadata_ref,
        source_sha256=sources,
        snapshot_sha256=snapshots,
    )
    registration = seal_execution(tmp_path, registration)
    registration_ref = dump(tmp_path / "registration.json", registration)
    seed_ref = registration["training_seed_plan_ref"]
    prepared = apply_seed_plan(prepared, prepared_ref, json.loads(Path(seed_ref["path"]).read_text()), seed_ref)
    requests = prepared["training_requests"]
    launch_ref = dump(tmp_path / "launch.json", dict(child=dict(pid=11, start_ticks=22)))
    exit_ref = dump(tmp_path / "exit.json", dict(actual_wait=True, returncode=0))
    release_ref = dump(
        tmp_path / "release.json",
        dict(
            pid=11,
            start_ticks=22,
            reaped=True,
            returncode=0,
            signal_sent=False,
            only_owned_child_signalled=True,
            original_error=None,
        ),
    )
    gpu_ref = dump(tmp_path / "gpu.json", dict(released=True, memory_mib=0, compute_apps=[], uuid=contract.GPU_UUID))
    normal_ref = dump(tmp_path / "normal.json", dict(normalization_id=31, projection_id=0))
    opaque_ref = dump(tmp_path / "opaque_capture.json", dict(synthetic=True))
    collection = dict(
        schema_version=contract.COLLECTION_SCHEMA,
        complete=True,
        passed=True,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        training_seed_plan_ref=seed_ref,
        round_name="D0",
        actual_requests=300,
        actual_denoise_calls=10500,
        future_label_reads=0,
        normalization_ref=normal_ref,
        readout_ref=None,
        requests=[
            dict(
                r,
                **{
                    key: opaque_ref
                    for key in (
                        "compact_ledger_ref",
                        "initial_ref",
                        "request_ref",
                        "response_ref",
                        "trace_ref",
                        "conditioning_ref",
                    )
                },
            )
            for r in requests
        ],
    )
    collection_ref = dump(tmp_path / "collection.json", collection)
    audited = dict(
        schema_version=contract.CAPTURE_AUDIT_SCHEMA,
        complete=True,
        passed=True,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        training_seed_plan_ref=seed_ref,
        collection_ref=collection_ref,
        round_name="D0",
        actual_requests=300,
        actual_denoise_calls=10500,
        worker_launch_ref=launch_ref,
        worker_exit_ref=exit_ref,
        worker_release_ref=release_ref,
        gpu_after_ref=gpu_ref,
        source_sha256=sources,
    )
    audit_ref = dump(tmp_path / "capture_audit.json", audited)
    return SimpleNamespace(
        prepared=prepared,
        registration=registration,
        collection=collection,
        audited=audited,
        prepared_ref=prepared_ref,
        registration_ref=registration_ref,
        collection_ref=collection_ref,
        audit_ref=audit_ref,
        source=source,
        root=tmp_path,
    )


def admit(value, stage="W1", collections=None, audits=None):
    return contract.load_context(
        value.prepared_ref,
        value.registration_ref,
        stage=stage,
        collection_refs=collections or [value.collection_ref],
        capture_audit_refs=audits or [value.audit_ref],
        caller_sources=[value.source],
    )


def test_complete_capture_metadata_admission_opens_no_future_arrays(admission, monkeypatch):
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("metadata admission must not decode futures or capture arrays")
    )
    context = admit(admission)
    assert context.expected_rows == 420000 and context.rounds == ("D0",)
    assert all(Path(path).is_relative_to(REPO / "outputs") for path in context.evidence.inputs)
    original = json.loads(Path(admission.prepared_ref["path"]).read_text())
    assert all(e["sampling_seed"] >= 2**31 for e in original["training_requests"])
    assert context.prepared["training_seed_plan_ref"] == admission.registration["training_seed_plan_ref"]
    assert all(e["request"]["seed"] == e["sampling_seed"] < 2**31 for e in context.prepared["training_requests"])


@pytest.mark.parametrize(
    "bad",
    (
        "metadata_only",
        "audit_failed",
        "wait_missing",
        "signal",
        "birth",
        "gpu_busy",
        "source_changed",
        "missing_request",
        "reorder",
        "future_read",
        "wrong_policy",
    ),
)
def test_phase_and_identity_failures_reject_before_any_future_decode(admission, monkeypatch, bad):
    value = admission
    if bad == "metadata_only":
        value.registration["readiness"]["ready_for_execution"] = False
        value.registration_ref = dump(value.root / "registration.json", value.registration)
    elif bad == "audit_failed":
        value.audited["passed"] = False
        value.audit_ref = dump(value.root / "capture_audit.json", value.audited)
    elif bad in ("wait_missing", "signal", "birth", "gpu_busy"):
        key, change = {
            "wait_missing": ("worker_exit_ref", dict(actual_wait=False)),
            "signal": ("worker_release_ref", dict(signal_sent=True)),
            "birth": ("worker_release_ref", dict(start_ticks=999)),
            "gpu_busy": ("gpu_after_ref", dict(memory_mib=1)),
        }[bad]
        original = value.audited[key]
        payload = json.loads(Path(original["path"]).read_text())
        payload.update(change)
        value.audited[key] = dump(Path(original["path"]), payload)
        value.audit_ref = dump(value.root / "capture_audit.json", value.audited)
    elif bad == "source_changed":
        value.source.write_text("# changed\n")
    else:
        if bad == "missing_request":
            value.collection["requests"].pop()
        elif bad == "reorder":
            value.collection["requests"][0], value.collection["requests"][1] = (
                value.collection["requests"][1],
                value.collection["requests"][0],
            )
        elif bad == "future_read":
            value.collection["future_label_reads"] = 1
        else:
            value.collection["readout_ref"] = value.prepared_ref
        value.collection_ref = dump(value.root / "collection.json", value.collection)
        value.audited["collection_ref"] = value.collection_ref
        value.audit_ref = dump(value.root / "capture_audit.json", value.audited)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("rejected metadata must not open future labels"))
    with pytest.raises(ValueError):
        admit(value)


def test_w2_requires_both_ordered_rounds_and_an_independently_fixed_w1(admission):
    with pytest.raises(ValueError, match="all complete ordered rounds"):
        admit(admission, stage="W2")
    d1 = copy.deepcopy(admission.collection)
    d1["round_name"] = "D1"
    # A presence-only entrypoint or an arbitrary coefficient ref cannot stand
    # in for the actual W1 producer and complete independent verification.
    d1.update(
        readout_ref=admission.prepared_ref,
        w1_fit_ref=admission.prepared_ref,
        w1_verification_ref=admission.prepared_ref,
    )
    d1_ref = dump(admission.root / "d1.json", d1)
    audited = dict(admission.audited, round_name="D1", collection_ref=d1_ref)
    audited_ref = dump(admission.root / "d1_audit.json", audited)
    with pytest.raises(ValueError, match="fixed W1"):
        admit(
            admission,
            stage="W2",
            collections=[admission.collection_ref, d1_ref],
            audits=[admission.audit_ref, audited_ref],
        )


@pytest.fixture
def synthetic_label(tmp_path):
    n, count = 90, 171
    t = np.arange(count, dtype=np.float64)
    source = transform(
        np.column_stack((0.001 * np.sin(t / 9), 0.0003 * t, 0.0001 * t)),
        Rotation.from_euler("xyz", np.column_stack((0.003 * t, -0.002 * t, 0.005 * t))).as_matrix(),
    )
    h = transform([0.02, 0, -0.0036], Rotation.from_euler("z", 0.3).as_matrix())
    site, body = source.copy(), source.copy()
    site[:, :3, :3] = site[:, :3, :3] @ Rotation.from_euler("z", 0.1).as_matrix()
    body[:, :3, 3] += 0.1
    times, opening = 0.25 + 0.05 * t, 0.1 + 0.0002 * t
    labels = synchronous_labels(source, opening, h, times)
    arrays = dict(
        source_pose=source,
        actual_source_site_pose=site,
        actual_eef_body_pose=body,
        opening=opening,
        sim_time=times,
        endpoint_step=np.arange(count),
        **labels,
    )
    trajectory = tmp_path / "synthetic_trajectory.npz"
    np.savez(trajectory, **arrays)
    trajectory_ref = contract.reference(trajectory)
    identity = dict(task_index=0, demo="demo_40", initial_state_sha256="a" * 64)
    case = dict(
        record=dict(identity, anchor_frames=[0, 16, 33, 50, 130]),
        arrays=trajectory_ref,
        array_index={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()},
    )
    case_ref = dump(tmp_path / "synthetic_case.json", case)
    anchor = dict(
        name="task0_demo_40_a130",
        source_identity=identity,
        anchor_frame=130,
        native_steps=n,
        case_ref=case_ref,
        declared_trajectory_ref=trajectory_ref,
        trajectory_ref=trajectory_ref,
        tool_source_from_tcp=h.tolist(),
        current_endpoint=dict(sim_time=float(times[130])),
    )
    normalization = dict(
        domain="franka_panda__parallel_gripper/eef20/observed_local_delta/libero_closed_pad_observed_endpoint_eef10_v1",
        normalization_id=31,
        projection_id=0,
        transform=dict(
            normalization_mask=[True] * 3 + [False] * 125,
            scale=[30.0] * 128,
            bias=[0.0] * 128,
            clip_min=-1.0,
            clip_max=1.0,
        ),
    )
    return SimpleNamespace(evidence=contract.Evidence()), anchor, normalization


def test_independent_reparse_matches_real_schema_terminal_window_without_producer_normalizer(
    scripts, synthetic_label, monkeypatch
):
    producer, checker = scripts
    context, anchor, normalization = synthetic_label
    expected, receipt = producer.load_label(context, anchor, normalization)
    monkeypatch.setattr(
        producer,
        "normalize_active10",
        lambda *a, **k: pytest.fail("independent audit must not call producer normalize"),
    )
    monkeypatch.setattr(
        producer,
        "post_capture_labels",
        lambda *a, **k: pytest.fail("independent audit must not call producer label factory"),
    )
    actual, independent = checker.independent_label(context, anchor, normalization)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32 and actual.shape == (40, 10)
    assert receipt["endpoint_count"] == 41 and independent["all40_measured_labels"]
    assert np.linalg.norm(actual[-1, :3]) > 0  # Terminal measured drift is not padded zero.


def test_both_label_readers_reject_wrong_demo_source_before_binary_read(scripts, synthetic_label, monkeypatch):
    context, anchor, normalization = synthetic_label
    anchor["source_identity"]["demo"] = "demo_16"  # Validation demo cannot be relabeled as training.
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("wrong source must be rejected before opening trajectory")
    )
    for function in (scripts[0].load_label, scripts[1].independent_label):
        with pytest.raises(ValueError, match="source"):
            function(context, anchor, normalization)


def test_point_rows_keep_d1_raw_c0_and_independently_recompute_fixed_w1(scripts):
    producer, checker = scripts
    sample, raw = np.full((40, 10), 0.75, np.float32), np.full((40, 10), 0.5, np.float32)
    x0 = np.full((40, 10), 0.2, np.float32)
    coefficients = np.zeros((1291, 10))
    coefficients[-1] = 0.125
    head = CleanResidualReadout(coefficients, np.zeros(1290), 1.0, 0.1)
    arrays = dict(
        sample_active10_fp32=sample,
        raw_velocity_active10=raw,
        deployed_velocity_active10=raw - 0.25,
        c0_active10=sample.astype(np.float64) - 0.5 * raw,
        hidden=np.zeros((40, 1280), np.float32),
        clean_residual_active10=np.full((40, 10), 0.125),
    )
    first = producer.point_rows(x0, dict(sigma_actual=0.5), arrays, head=head)
    second = checker.independent_point_rows(
        x0, dict(sigma_actual=0.5), arrays, head=(coefficients, np.zeros(1290), 1.0)
    )
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first[1], x0.astype(np.float64) - 0.5)
    arrays["c0_active10"] = sample.astype(np.float64) - 0.5 * arrays["deployed_velocity_active10"]
    for function, param in (
        (producer.point_rows, head),
        (checker.independent_point_rows, (coefficients, np.zeros(1290), 1.0)),
    ):
        with pytest.raises(ValueError, match="C0"):
            function(x0, dict(sigma_actual=0.5), arrays, head=param)


def test_w2_stream_uses_every_node_and_equal_round_mass_with_bounded_blocks(scripts, monkeypatch, tmp_path):
    producer = scripts[0]
    x0 = np.full((40, 10), 0.2, np.float32)
    original = training_seed_metadata(anchors=[dict(name=f"a{i}") for i in range(150)])
    effective = effective_training_metadata(tmp_path, original)
    requests = effective["training_requests"][:1]
    request = requests[0]
    seed = request["sampling_seed"]
    head = CleanResidualReadout(np.zeros((1291, 10)), np.zeros(1290), 1.0, 0.1)
    head_ref = dict(path="synthetic_w1", sha256="a" * 64)
    objects = {
        "normal": dict(domain="unused", normalization_id=31, projection_id=0, transform={}),
        "synthetic_w1": head.to_parameters(),
    }
    collections = []
    for round_name in ("D0", "D1"):
        identity = dict(
            request_name=request["name"],
            anchor_id=request["anchor_name"],
            noise_seed=seed,
            purpose="training",
            round_name=round_name,
            readout_sha256=None if round_name == "D0" else head_ref["sha256"],
        )
        objects[f"{round_name}_ledger"] = dict(request_ref=f"{round_name}_header", points=[])
        objects[f"{round_name}_header"] = dict(request=identity)
        collections.append(
            dict(
                normalization_ref="normal",
                readout_ref=None if round_name == "D0" else head_ref,
                requests=[dict(requests[0], compact_ledger_ref=f"{round_name}_ledger")],
            )
        )

    class FakeEvidence:
        def json(self, ref):
            return objects[ref["path"] if isinstance(ref, dict) else ref]

        def path(self, ref):
            pytest.fail("this synthetic stream has no binary input to open")

    context = SimpleNamespace(
        rounds=("D0", "D1"),
        collections=collections,
        expected_rows=2800,
        prepared=dict(effective, training_requests=requests),
        evidence=FakeEvidence(),
    )
    labels = {f"a{i}": (x0, {}) for i in range(150)}

    def points(ref, *, nodes=35):
        for step in range(nodes):
            sample = np.full((40, 10), step / 128, np.float32)
            raw = np.full((40, 10), 0.25, np.float32)
            yield (
                dict(step_index=step, request_name=request["name"], noise_seed=seed, sigma_actual=0.5),
                dict(
                    sample_active10_fp32=sample,
                    raw_velocity_active10=raw,
                    c0_active10=sample.astype(np.float64) - 0.5 * raw,
                    hidden=np.full((40, 1280), step, np.float32),
                    clean_residual_active10=np.zeros((40, 10)),
                ),
            )

    monkeypatch.setattr(producer, "iter_compact_points", points)
    factory, layout = producer.block_factory(context, labels)
    blocks = list(factory())
    assert len(blocks) == 2 and all(b.features.shape == (1400, 1290) for b in blocks)
    assert all(b.layout == layout and np.isclose(b.weights.sum(), 0.5) for b in blocks)
    assert all(np.all(b.weights == 1 / 2800) for b in blocks)
    np.testing.assert_array_equal(blocks[0].features[:40, :1280], np.zeros((40, 1280)))
    np.testing.assert_array_equal(blocks[0].features[-40:, :1280], np.full((40, 1280), 34))
    monkeypatch.setattr(producer, "iter_compact_points", lambda ref: points(ref, nodes=34))
    with pytest.raises(ValueError, match="all35"):
        list(factory())


def test_production_single_solve_matches_independent_full_row_qr_on_small_synthetic_fixture(scripts):
    rng = np.random.default_rng(42)
    features, targets = rng.normal(size=(36, 3)), rng.normal(size=(36, 10))
    layout = CleanFitLayout(("hidden_0", "hidden_1", "c0_0"), "active10")

    def factory():
        for start in range(0, 36, 9):
            yield CleanFitBlock(features[start : start + 9], targets[start : start + 9], np.full(9, 1 / 36), layout)

    produced = scripts[0].solve_blocks(factory, layout, expected_rows=36)
    independent = independent_ridge(
        lambda: ((b.features, b.target_clean_residual, b.weights) for b in factory()),
        expected_rows=36,
        feature_columns=layout.feature_columns,
        target_columns=10,
    )
    np.testing.assert_allclose(produced.readout.coefficients, independent.coefficients, rtol=1e-9, atol=1e-11)
    assert produced.replay_sha256 == independent.replay_sha256 and produced.rows == independent.rows == 36


def test_real_cli_refuses_metadata_only_registration_without_creating_fit_output(admission):
    value = admission
    value.registration["readiness"]["ready_for_execution"] = False
    value.registration_ref = dump(value.root / "registration.json", value.registration)
    output = value.root / "forbidden_fit"
    command = [
        str(REPO / ".venv/bin/python"),
        "scripts/fit_wam_clean_training.py",
        "--prepared",
        value.prepared_ref["path"],
        "--prepared-sha256",
        value.prepared_ref["sha256"],
        "--execution-registration",
        value.registration_ref["path"],
        "--execution-registration-sha256",
        value.registration_ref["sha256"],
        "--stage",
        "W1",
        "--collection",
        value.collection_ref["path"],
        value.collection_ref["sha256"],
        "--capture-audit",
        value.audit_ref["path"],
        value.audit_ref["sha256"],
        "--output-dir",
        str(output),
        "--seed",
        "42",
    ]
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
    )
    result = subprocess.run(
        command,
        cwd=REPO,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0 and "metadata-only preparation cannot admit a fit" in result.stdout
    assert not output.exists()
