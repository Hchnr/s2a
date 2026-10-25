"""054 paired RNG identity and rejection of privileged/future conditioning."""

import copy

import numpy as np
import pytest

from s2a.clean_training_runtime import reference
from s2a.synchronous_state_sampling import paired_entries
from s2a.wam_readout import LIBERO_DOMAIN


def fixture(tmp_path):
    image = tmp_path / "current.png"
    image.write_bytes(b"current-only bytes for request-hash validation")
    gold = tmp_path / "gold.json"
    gold.write_text("{}")
    template = {
        "action_chunk_size": 40,
        "domain_name": "franka_panda__parallel_gripper",
        "fps": 20,
        "guidance": 1.0,
        "image_size": 960,
        "model_mode": "wam",
        "name": "zero0",
        "num_frames": 41,
        "num_steps": 35,
        "prompt": "task",
        "seed": 42,
        "shift": 5.0,
        "vision_path": str(image),
        "extra": {
            "action128": {
                "action_dim_mask": [True] * 10 + [False] * 118,
                "action_domain": LIBERO_DOMAIN,
                "action_fps": 20,
                "camera_layout": [["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
                "return_video": False,
                "selected_roles": ["auxiliary_0", "left_wrist"],
                "task_title": "task",
            }
        },
    }
    anchors = [
        {
            "name": f"anchor{i}",
            "task": "task",
            "anchor_frame": i,
            "image_ref": reference(image),
            "current_time_s": 0.25 + i * 0.05,
            "current_sim_state_sha256": str(i),
            "measured_state128": [0.0] * 128,
            "state_mask128": (np.arange(128) < 10).tolist(),
            "initial_sync_certified": True,
        }
        for i in range(24)
    ]
    zero, qualified = [], {"generations": []}
    for i in range(3):
        request = copy.deepcopy(template)
        request["name"] = f"zero{i}"
        zero.append(
            {
                "name": f"case{i}",
                "request": request,
                "image_ref": reference(image),
                "expected_response_ref": reference(gold),
            }
        )
        qualified["generations"].append(
            {"variant": "raw", "case_id": f"case{i}", "historical_output_exact": True, "response_ref": reference(gold)}
        )
    return anchors, template, zero, qualified


def test_full99_matrix_pairs_same_current_image_task_seed(tmp_path):
    values = fixture(tmp_path)
    entries = paired_entries(*values)
    assert [e["variant"] for e in entries[:3]] == ["zero"] * 3
    assert len(entries) == 99
    assert [e["index"] for e in entries] == list(range(99))
    for i in range(3, 99, 2):
        raw, probe = entries[i : i + 2]
        assert raw["variant"] == "raw" and probe["variant"] == "probe052"
        assert probe["paired_raw_generation_index"] == raw["index"]
        assert raw["pair_id"] == probe["pair_id"]
        assert raw["sampling_seed"] == probe["sampling_seed"] in (42, 43)
        a, b = copy.deepcopy(raw["request"]), copy.deepcopy(probe["request"])
        a.pop("name")
        b.pop("name")
        assert a == b


@pytest.mark.parametrize(
    "violation", ["future", "command", "predicate", "sync", "duplicate", "missing", "mask", "nonfinite"]
)
def test_current_input_boundary_or_denominator_violation_rejected(tmp_path, violation):
    values = fixture(tmp_path)
    anchors = values[0]
    if violation in ["future", "command", "predicate"]:
        anchors[0][{"future": "future_state", "command": "expert_action", "predicate": "success"}[violation]] = []
    elif violation == "sync":
        anchors[0]["initial_sync_certified"] = False
    elif violation == "duplicate":
        anchors[1]["name"] = anchors[0]["name"]
    elif violation == "missing":
        anchors.pop()
    elif violation == "mask":
        anchors[0]["state_mask128"][10] = True
    else:
        anchors[0]["measured_state128"][0] = float("nan")
    with pytest.raises(ValueError):
        paired_entries(*values)


@pytest.mark.parametrize("violation", ["privileged", "sampling_steps", "guidance", "domain"])
def test_model_conditioning_or_sampling_contract_change_rejected(tmp_path, violation):
    values = fixture(tmp_path)
    template = values[1]
    if violation == "privileged":
        template["extra"]["action128"]["future_action"] = []
    elif violation == "sampling_steps":
        template["num_steps"] = 70
    elif violation == "guidance":
        template["guidance"] = 3
    else:
        template["extra"]["action128"]["action_domain"] = "other"
    with pytest.raises(ValueError):
        paired_entries(*values)


def head_evidence():
    params = {
        "schema_version": "s2a.clean-residual-readout.v1",
        "coefficients": np.zeros((1291, 10)).tolist(),
        "center": [0.0] * 1290,
        "scale": 1.0,
        "ridge": 0.1 / 1290,
    }
    produced = {
        "schema_version": "s2a.clean-regularization-probe.v1",
        "passed": True,
        "complete": True,
        "params_ref": "parameters",
        "producer_fits": 1,
        "fit_metadata": {
            "rows": 840000,
            "replay_sha256": {"features": "x", "target_clean_residual": "y", "weights": "w"},
        },
    }
    audit = {
        "schema_version": "s2a.clean-regularization-probe-qr.v1",
        "passed": True,
        "complete": True,
        "producer_ref": "producer",
        "params_ref": "parameters",
        "independent_reconstructions": 1,
        "rtol": 1e-9,
        "atol": 1e-11,
        "comparisons": {k: {"passed": True} for k in ["coefficients", "center", "scale"]},
        "qr_metadata": {"rows": 840000, "replay_sha256": {"features": "x", "targets": "y", "weights": "w"}},
    }
    return {"producer": produced, "audit": audit, "parameters": params}


@pytest.mark.parametrize(
    "violation", ["none", "old_identity", "old_ridge", "partial", "missing_comparison", "tolerance", "target", "params"]
)
def test_candidate_requires_own_full_fit_and_independent_qr(monkeypatch, violation):
    import s2a.synchronous_state_sampling as sampling

    values = head_evidence()
    if violation == "old_identity":
        values["producer"]["schema_version"] = "s2a.clean-training-fit.v1"
    elif violation == "old_ridge":
        values["parameters"]["ridge"] = 0.1
    elif violation == "partial":
        values["audit"]["qr_metadata"]["rows"] = 420000
    elif violation == "missing_comparison":
        values["audit"]["comparisons"] = {}
    elif violation == "tolerance":
        values["audit"]["rtol"] = 1e-6
    elif violation == "target":
        values["audit"]["qr_metadata"]["replay_sha256"]["targets"] = "other"
    elif violation == "params":
        values["audit"]["params_ref"] = "other"
    monkeypatch.setattr(sampling, "read", lambda ref: values[ref])
    if violation == "none":
        head, ref = sampling.candidate_head("producer", "audit")
        assert ref == "parameters" and head.ridge == 0.1 / 1290
    else:
        with pytest.raises(ValueError):
            sampling.candidate_head("producer", "audit")


@pytest.mark.parametrize("change", [False, True])
def test_sampler_body_proof_rejects_unregistered_numeric_change(tmp_path, monkeypatch, change):
    import verify_wam_synchronous_state_metadata as metadata

    from s2a.artifacts import REPO

    folder = tmp_path / "scripts"
    folder.mkdir()
    old = (REPO / "scripts/sample_wam_clean_training_state.py").read_text()
    new = (REPO / "scripts/sample_wam_synchronous_state.py").read_text()
    if change:
        assert 'counter.scope(entry["name"], 35)' in new
        new = new.replace('counter.scope(entry["name"], 35)', 'counter.scope(entry["name"], 34)')
    (folder / "sample_wam_clean_training_state.py").write_text(old)
    (folder / "sample_wam_synchronous_state.py").write_text(new)
    monkeypatch.setattr(metadata, "REPO", tmp_path)
    if change:
        with pytest.raises(ValueError):
            metadata.worker_body_equivalent()
    else:
        assert metadata.worker_body_equivalent()["passed"]
