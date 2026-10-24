"""State-sampler metadata/zero/W2 gates without model or validation labels."""

import copy
import importlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from s2a.artifacts import REPO
from s2a.clean_training_runtime import GPU, reference
from s2a.wam_readout import LIBERO_DOMAIN


@pytest.fixture
def sampler(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("sample_wam_clean_training_state")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return reference(path)


@pytest.fixture
def inputs(tmp_path):
    image = tmp_path / "current.png"
    image.write_bytes(b"synthetic current image")
    image_ref = reference(image)
    values = np.zeros((40, 128)).tolist()
    response = dict(outputs=[dict(content=dict(action=values, action128=dict(normalized_action=values)))])
    response_ref = dump(tmp_path / "historical_response.json", response)
    template = dict(
        name="historical_other_episode",
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
    zeros = [
        dict(
            name=f"task{i}_original",
            request=copy.deepcopy(template),
            image_ref=image_ref,
            expected_response_ref=response_ref,
        )
        for i in range(3)
    ]
    qualified = dict(
        generations=[
            dict(
                index=i,
                variant="raw",
                case_id=c["name"],
                sampling_seed=42,
                complete=True,
                historical_output_exact=True,
                response_ref=response_ref,
            )
            for i, c in enumerate(zeros)
        ]
    )
    anchors, generations = [], []
    for episode in (1254, 1126, 868, 1139, 1209, 1176):
        for index in range(4):
            name = f"ep{episode}_a{index}"
            anchors.append(
                dict(
                    name=name,
                    episode=episode,
                    image_ref=image_ref,
                    task="task",
                    request_template=copy.deepcopy(template),
                    future_labels_parsed=False,
                )
            )
            for seed in (42, 43):
                for variant in ("raw", "w2"):
                    entry_name = f"{name}_{variant}_s{seed}"
                    request = dict(copy.deepcopy(template), name=entry_name, seed=seed)
                    generations.append(
                        dict(
                            index=len(generations),
                            name=entry_name,
                            anchor_name=name,
                            variant=variant,
                            sampling_seed=seed,
                            request=request,
                        )
                    )
    return dict(zero_identity_cases=zeros, validation_anchors=anchors, validation_generations=generations), qualified


def test_exact99_roster_preserves_zero_historical_alias_and_paired_order(sampler, inputs, monkeypatch):
    prepared, qualification = inputs
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("roster must not open any arrays"))
    entries = sampler.sampling_entries(prepared, qualification)
    assert len(entries) == len({entry["name"] for entry in entries}) == 99
    assert [e["index"] for e in entries] == list(range(99))
    assert [e["variant"] for e in entries] == ["zero"] * 3 + ["raw", "w2"] * 48
    assert entries[0]["request"]["name"] == "historical_other_episode"
    assert entries[0]["name"] == "zero_task0_original" and entries[0]["qualified_generation_index"] == 0
    assert [e["validation_index"] for e in entries[3:]] == list(range(96))
    for raw, w2 in zip(entries[3::2], entries[4::2], strict=True):
        assert raw["pair_id"] == w2["pair_id"] and raw["sampling_seed"] == w2["sampling_seed"]
        assert w2["paired_raw_generation_index"] == raw["index"]
        assert raw["image_ref"] == w2["image_ref"]


@pytest.mark.parametrize("bad", ("missing", "training_seed", "reorder", "wrong_anchor", "future_field", "zero_source"))
def test_incomplete_reordered_or_privileged_state_roster_fails_without_decoding(sampler, inputs, monkeypatch, bad):
    prepared, qualification = inputs
    if bad == "missing":
        prepared["validation_generations"].pop()
    elif bad == "training_seed":
        prepared["validation_generations"][0]["sampling_seed"] = 101
    elif bad == "reorder":
        prepared["validation_generations"][0]["variant"] = "w2"
    elif bad == "wrong_anchor":
        prepared["validation_generations"][0]["anchor_name"] = prepared["validation_anchors"][1]["name"]
    elif bad == "future_field":
        prepared["validation_generations"][0]["request"]["future_state"] = [1]
    else:
        qualification["generations"][0]["case_id"] = "different_source"
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("rejected roster must not open future archives"))
    with pytest.raises(ValueError):
        sampler.sampling_entries(prepared, qualification)


def test_w2_requires_normally_owned_full_fit_and_independent_reconstruction(sampler, inputs, monkeypatch):
    prepared, qualification = inputs
    pref, reg, fit_ref, check_ref = ({"path": name, "sha256": "a" * 64} for name in ("prepared", "reg", "fit", "check"))
    prepared["prerequisite_refs"] = dict(bridge_qualification=dict(path="qualified", sha256="b" * 64))
    fit = dict(
        rounds=["D0", "D1"],
        actual_rows=840000,
        collection_refs=[1, 2],
        capture_audit_refs=[3, 4],
        fit_metadata=dict(replay_sha256=dict(features="x", weights="w", target_clean_residual="y")),
    )
    verified = dict(
        actual_rows=840000,
        independent_reconstructions=1,
        full_training_row_reconstruction=True,
        qr_metadata=dict(replay_sha256=dict(features="x", weights="w", targets="y")),
    )
    qualification.update(complete=True, passed=True, bridge_qualified=True)
    mapping = {"fit": fit, "check": verified, "qualified": qualification}
    monkeypatch.setattr(sampler, "read", lambda ref: mapping[ref["path"]])
    monkeypatch.setattr(sampler, "prior", lambda *a: dict(producer_ref=fit_ref, verification_ref=check_ref))
    monkeypatch.setattr(sampler, "trained_head", lambda *a, **k: ("synthetic_head", "synthetic_params"))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("fit admission must not open future arrays"))
    result = sampler.admit_sampling(prepared, pref, reg, fit_ref, check_ref)
    assert result[0] == "synthetic_head" and len(result[-1]) == 99
    verified["actual_rows"] = 420000
    with pytest.raises(ValueError, match="840000"):
        sampler.admit_sampling(prepared, pref, reg, fit_ref, check_ref)
    verified["actual_rows"] = 840000
    monkeypatch.setattr(sampler, "prior", lambda *a: dict(producer_ref=pref, verification_ref=check_ref))
    with pytest.raises(ValueError, match="exact normally exited"):
        sampler.admit_sampling(prepared, pref, reg, fit_ref, check_ref)


def test_zero_response_and_source_bytes_fail_exactly_including_signed_zero(sampler, inputs):
    prepared, qualification = inputs
    entries = sampler.sampling_entries(prepared, qualification)
    response = json.loads(Path(entries[0]["expected_response_ref"]["path"]).read_text())
    sampler.require_zero_response(response, entries[0], qualification["generations"][0])
    response["outputs"][0]["content"]["action"][0][0] = 1e-10
    with pytest.raises(ValueError, match="zero head"):
        sampler.require_zero_response(response, entries[0], qualification["generations"][0])
    left = np.zeros(1, np.float32)
    right = np.full(1, -0.0, np.float32)
    import hashlib

    expected = dict(sample=dict(snapshot_sha256=hashlib.sha256(left.tobytes()).hexdigest()))
    changed = dict(sample=dict(snapshot_sha256=hashlib.sha256(right.tobytes()).hexdigest()))
    assert np.array_equal(left, right)
    with pytest.raises(ValueError, match="source array byte"):
        sampler.require_zero_sources(dict(source_arrays=changed), expected)


def test_historical_gold_hashes_all35_saved_model_points_and_refuses_partial(sampler, tmp_path):
    points = []
    for step in range(35):
        arrays = {f"field{i}": np.full((2, 2), step + i, np.float32) for i in range(18)}
        path = tmp_path / f"point{step:02d}.npz"
        np.savez(path, **arrays)
        points.append(
            dump(
                path.with_suffix(".json"),
                dict(
                    step_index=step,
                    request_name="raw_original",
                    arrays=reference(path),
                    array_index={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()},
                    original_dtypes={k: "torch.float32" for k in arrays},
                ),
            )
        )
    generation = dict(variant="raw", name="raw_original", points=points)
    gold = sampler.historical_source_summaries(generation)
    assert len(gold) == 35 and all(len(point) == 18 for point in gold)
    assert gold[0]["field0"]["snapshot_sha256"] != gold[-1]["field0"]["snapshot_sha256"]
    generation["points"] = points[:-1]
    with pytest.raises(ValueError, match="all35"):
        sampler.historical_source_summaries(generation)


def test_real_entrypoint_rejects_metadata_only_registration_before_model_or_output(sampler, tmp_path):
    prepared_ref = dump(tmp_path / "prepared.json", dict(schema_version="s2a.clean-training-preparation.v1"))
    reg_ref = dump(
        tmp_path / "registration.json",
        dict(
            schema_version="s2a.clean-training-execution-registration.v1",
            prepared_ref=prepared_ref,
            complete=True,
            passed=True,
            automatic_retry=False,
            readiness=dict(ready_for_execution=False),
        ),
    )
    unused = dump(tmp_path / "unused.json", dict(synthetic=True))
    command = [str(REPO / ".venv/bin/python"), "scripts/sample_wam_clean_training_state.py"]
    for flag, ref in (
        ("prepared", prepared_ref),
        ("execution-registration", reg_ref),
        ("w2-fit", unused),
        ("w2-verification", unused),
    ):
        command += ["--" + flag, ref["path"], "--" + flag + "-sha256", ref["sha256"]]
    command += ["--gpu-uuid", GPU, "--seed", "42"]
    result = subprocess.run(
        command,
        cwd=REPO,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0 and "complete independent execution registration" in result.stdout
    assert not (tmp_path / "state_sampling").exists()
