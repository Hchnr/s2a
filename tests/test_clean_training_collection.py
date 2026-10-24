"""Admission/initial-condition invariants without a model or real labels."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from execution_proof_fixture import effective_training_metadata, seal_execution, training_seed_metadata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_wam_clean_training import admit_round, normalization_spec, same_initial  # noqa: E402

from s2a.clean_training_runtime import (  # noqa: E402
    GPU,
    admit_execution,
    read,
    reference,
    validate_current_request,
)
from s2a.wam_readout import LIBERO_DOMAIN  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value))
    return reference(path)


def fixture(tmp_path):
    image = tmp_path / "current.png"
    image.write_bytes(b"synthetic current image")
    image_ref = reference(image)
    template = dict(
        name="historical",
        seed=42,
        vision_path=str(image),
        prompt="task",
        action_chunk_size=40,
        fps=20,
        guidance=1.0,
        image_size="640,960",
        model_mode="wam",
        num_frames=41,
        num_steps=35,
        shift=5.0,
        extra=dict(
            action128=dict(
                action_dim_mask=[True] * 10 + [False] * 118,
                action_domain=LIBERO_DOMAIN,
                action_fps=20,
                camera_layout=[],
                return_video=False,
                selected_roles=[],
                task_title="task",
            )
        ),
    )
    anchors = [dict(name=f"a{i}", image_ref=image_ref, task="task") for i in range(150)]
    original = dict(
        **training_seed_metadata(anchors=anchors, request_template=template),
        zero_identity_cases=[dict(request=template)],
    )
    return (
        effective_training_metadata(tmp_path, original),
        template,
        image_ref,
    )


def admit(prepared, **changes):
    args = dict(
        round_name="D0",
        d0_audit_ref=None,
        w1_fit_ref=None,
        w1_verification_ref=None,
        prepared_ref=dict(path="prepared", sha256="a" * 64),
        registration_ref=dict(path="reg", sha256="b" * 64),
    )
    return admit_round(prepared, **dict(args, **changes))


def test_all_300_requests_admitted_without_future_archive_decode(tmp_path, monkeypatch):
    prepared, _, _ = fixture(tmp_path)
    original = json.loads((tmp_path / "original_seed_metadata.json").read_text())
    assert len(original["training_anchors"]) == len(prepared["training_anchors"]) == 150
    assert len(original["training_requests"]) == len(prepared["training_requests"]) == 300
    assert all(e["sampling_seed"] >= 2**31 for e in original["training_requests"])
    assert all(e["sampling_seed"] < 2**31 for e in prepared["training_requests"])
    assert "training_seed_plan_ref" not in original

    def deny(*args, **kwargs):
        raise AssertionError("metadata admission must never decode future arrays")

    monkeypatch.setattr(np, "load", deny)
    assert admit(prepared) == (None, None, None)


@pytest.mark.parametrize("field", ["future_state", "expert_command", "task_plan", "objects"])
def test_privileged_top_level_fields_rejected(tmp_path, field):
    prepared, _, _ = fixture(tmp_path)
    prepared["training_requests"][2]["request"][field] = [1]
    with pytest.raises(ValueError, match="template"):
        admit(prepared)


def test_privileged_nested_conditioning_rejected_even_when_template_has_it(tmp_path):
    _, template, image = fixture(tmp_path)
    bad = copy.deepcopy(template)
    bad["extra"]["action128"]["future_action"] = [1]
    with pytest.raises(ValueError, match="privileged"):
        validate_current_request(bad, name=bad["name"], seed=42, image_ref=image, task="task", template=bad)


@pytest.mark.parametrize("fault", ["evaluation_seed", "duplicate_seed", "count", "order", "image", "task", "nfe"])
def test_actual_noise_full_denominator_source_and_protocol_enforced(tmp_path, fault):
    prepared, _, _ = fixture(tmp_path)
    if fault == "evaluation_seed":
        prepared["noise_roster"]["training_seeds"][0] = 42
        prepared["training_requests"][0]["sampling_seed"] = 42
    elif fault == "duplicate_seed":
        prepared["training_requests"][1]["sampling_seed"] = prepared["training_requests"][0]["sampling_seed"]
    elif fault == "count":
        prepared["training_requests"].pop()
    elif fault == "order":
        prepared["training_requests"][0]["index"] = 1
    elif fault == "image":
        (tmp_path / "current.png").write_bytes(b"changed")
    elif fault == "task":
        prepared["training_requests"][1]["request"]["prompt"] = "another task"
    else:
        prepared["training_requests"][1]["request"]["num_steps"] = 34
    with pytest.raises(ValueError):
        admit(prepared)


def test_raw_cannot_use_head_and_w1_cannot_skip_full_d0_audit(tmp_path):
    prepared, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="raw"):
        admit(prepared, w1_fit_ref={})
    with pytest.raises(ValueError, match="prerequisites"):
        admit(prepared, round_name="D1")
    failed = save(tmp_path / "failed.json", dict(schema_version="s2a.clean-training-capture-audit.v1", passed=False))
    with pytest.raises(ValueError, match="full normally"):
        admit(prepared, round_name="D1", d0_audit_ref=failed, w1_fit_ref={}, w1_verification_ref={})


@pytest.mark.parametrize("fault", [None, "noise", "mask", "prefix", "tokens", "seed", "dtype"])
def test_initial_pair_requires_exact_actual_noise_and_all_conditioning(tmp_path, fault):
    arrays = dict(
        initial_noise=np.arange(16, dtype=np.float32),
        condition_mask=np.zeros(16, np.float32),
        first_condition_latent=np.ones((1, 2, 1, 2, 2), np.float32),
    )

    def initial(name, values, **updates):
        path = tmp_path / f"{name}.npz"
        np.savez(path, **values)
        return save(
            tmp_path / f"{name}.json",
            dict(
                arrays=reference(path),
                sampling_seed=101,
                caption="current",
                text_tokens=[[1, 2]],
                inference_only_current_image=True,
                **updates,
            ),
        )

    left = initial("left", arrays)
    values = {key: value.copy() for key, value in arrays.items()}
    meta = dict(sampling_seed=101, caption="current", text_tokens=[[1, 2]], inference_only_current_image=True)
    if fault in ("noise", "mask", "prefix"):
        key = dict(noise="initial_noise", mask="condition_mask", prefix="first_condition_latent")[fault]
        values[key].reshape(-1)[0] += 1
    elif fault == "tokens":
        meta["text_tokens"] = [[1, 3]]
    elif fault == "seed":
        meta["sampling_seed"] = 102
    elif fault == "dtype":
        values["initial_noise"] = values["initial_noise"].astype(np.float64)
    path = tmp_path / "right.npz"
    np.savez(path, **values)
    right = save(tmp_path / "right.json", dict(arrays=reference(path), **meta))
    if fault is None:
        same_initial(left, right)
    else:
        with pytest.raises(ValueError, match="initial"):
            same_initial(left, right)


def test_metadata_only_preparation_never_admits_execution(tmp_path):
    prepared = save(tmp_path / "prepared.json", dict(schema_version="s2a.clean-training-preparation.v1"))
    metadata = save(tmp_path / "metadata.json", dict(complete=True, passed=True, prepared_ref=prepared))
    registration = save(
        tmp_path / "reg.json",
        dict(
            schema_version="s2a.clean-training-execution-registration.v1",
            prepared_ref=prepared,
            complete=True,
            passed=True,
            readiness=dict(ready_for_execution=False),
            automatic_retry=False,
            metadata_verification_ref=metadata,
        ),
    )
    with pytest.raises(ValueError, match="execution registration"):
        admit_execution(prepared, registration, required_sources=[])


def test_binary_future_cannot_be_opened_as_admission_json(tmp_path):
    future = tmp_path / "future.npz"
    future.write_bytes(b"opaque training labels")
    with pytest.raises(ValueError, match="binary"):
        read(reference(future))


@pytest.mark.parametrize("fault", [None, "domain", "projection", "dict_instead_of_published_list", "missing_transform"])
def test_published_transform_list_is_indexed_by_normalization31(fault):
    transform = dict(
        normalization_mask=[True] * 10 + [False] * 118, scale=[1.0] * 128, bias=[0.0] * 128, clip_min=-1.0, clip_max=1.0
    )
    spec = dict(
        registry=dict(domains=["other"] * 31 + [LIBERO_DOMAIN], domain_projection_ids=[1] * 31 + [0]),
        transforms=[None] * 31 + [transform],
    )
    if fault == "domain":
        spec["registry"]["domains"][31] = "wrong domain"
    elif fault == "projection":
        spec["registry"]["domain_projection_ids"][31] = 31
    elif fault == "dict_instead_of_published_list":
        spec["transforms"] = {LIBERO_DOMAIN: transform}
    elif fault == "missing_transform":
        transform.pop("clip_min")
    if fault is None:
        normalized = normalization_spec(spec)
        assert normalized["transform"] == transform
        assert normalized["normalization_id"] == 31 and normalized["projection_id"] == 0
        normalized["transform"]["bias"][0] = 10.0
        assert transform["bias"][0] == 0.0
    else:
        with pytest.raises(ValueError):
            normalization_spec(spec)


@pytest.mark.parametrize(
    "fault", [None, "generic_true", "execution_ready", "changed_verifier", "unfrozen_verifier", "small_metadata"]
)
def test_execution_admission_binds_independent_metadata_code_and_checkpoint_bytes(tmp_path, fault):
    source = tmp_path / "worker.py"
    source.write_text("# synthetic execution worker\n")
    verifier = tmp_path / "independent.py"
    verifier.write_text("# independent metadata fixture\n")
    frozen = {str(source): reference(source)["sha256"]}
    small = []
    for i in range(3):
        path = tmp_path / f"metadata{i}"
        path.write_bytes(b"small checkpoint metadata")
        small.append(path)
    small_map = {str(path): reference(path)["sha256"] for path in small}
    qualified = save(
        tmp_path / "qualified.json",
        dict(snapshot_sha256=frozen, input_sha256=small_map, checkpoint_small_metadata_sha256=small_map),
    )
    # This synthetic gate exercises real byte/SHA admission. It never prepares
    # a model and does not assert any source/model qualification for the fixture.
    prep = save(
        tmp_path / "prepared.json",
        dict(
            schema_version="s2a.clean-training-preparation.v1",
            root=str(
                Path(__file__).resolve().parents[1] / "outputs/wam_stability_20261021/clean_residual_training_20261023"
            ),
            qualified_model_source_sha256=frozen,
            source_sha256=frozen,
            snapshot_sha256=frozen,
            input_sha256=frozen,
            prerequisite_refs=dict(bridge_prepared=qualified),
            config=dict(gpu_uuid=GPU),
            weight_files={
                str(path): dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns) for path in small
            },
            **training_seed_metadata(),
        ),
    )
    metadata = dict(
        schema_version="s2a.clean-training-preparation-independent.v1",
        complete=True,
        passed=True,
        metadata_qualified=True,
        execution_ready=False,
        execution_authorized=False,
        consumed_before_after_exact=True,
        prepared_ref=prep,
        verifier_ref=reference(verifier),
    )
    if fault == "generic_true":
        metadata["schema_version"] = "s2a.producer.v1"
    elif fault == "execution_ready":
        metadata["execution_ready"] = True
    mref = save(tmp_path / "metadata.json", metadata)
    sources = dict(frozen, **{str(verifier): reference(verifier)["sha256"]})
    if fault == "unfrozen_verifier":
        sources.pop(str(verifier))
    reg = save(
        tmp_path / "registration.json",
        seal_execution(
            tmp_path,
            dict(
                schema_version="s2a.clean-training-execution-registration.v1",
                prepared_ref=prep,
                complete=True,
                passed=True,
                readiness=dict(ready_for_execution=True),
                automatic_retry=False,
                metadata_verification_ref=mref,
                fresh_performance_authorized=False,
                source_sha256=sources,
                snapshot_sha256=frozen,
            ),
        ),
    )
    if fault == "changed_verifier":
        verifier.write_text("# modified after the metadata audit\n")
    elif fault == "small_metadata":
        small[0].write_bytes(b"changed checkpoint metadata")
    if fault is None:
        value, _, admitted = admit_execution(prep, reg, required_sources=[source])
        assert value["config"]["gpu_uuid"] == GPU and all(admitted[k] == v for k, v in sources.items())
        assert all(e["request"]["seed"] == e["sampling_seed"] < 2**31 for e in value["training_requests"])
        assert all(e["sampling_seed"] >= 2**31 for e in read(prep)["training_requests"])
    else:
        with pytest.raises(ValueError):
            admit_execution(prep, reg, required_sources=[source])
