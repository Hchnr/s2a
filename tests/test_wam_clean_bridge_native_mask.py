"""Native zero-padding mask semantics and scoped read-only039 verifier routing."""

import copy
import importlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from s2a.artifacts import REPO


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("verify_wam_clean_bridge_native_mask")


def initial_fixture():
    shape = (1, 48, 11, 60, 40)
    current = np.full((1, 48, 1, 60, 40), 0.25, np.float32)
    vision = np.full(shape, np.float32(0.5003), np.float32)
    vision[:, :, :1] = current
    actions = np.zeros((40, 128), np.float32)
    actions[:, :10] = np.float32(0.7501)
    initial = np.concatenate((vision.reshape(-1), actions.reshape(-1)))
    vision_mask = np.zeros(shape, np.float32)
    vision_mask[:, :, :1] = 1
    action_mask = np.ones((40, 128), np.float32)
    action_mask[:, :10] = 0
    arrays = dict(
        initial_noise=initial,
        condition_mask=np.concatenate((vision_mask.reshape(-1), action_mask.reshape(-1))),
        first_condition_latent=current,
    )
    generation = dict(name="raw", sampling_seed=42, variant="raw", index=0)
    conditioning = dict(caption="current-image task only")
    metadata = dict(
        name="raw",
        sampling_seed=42,
        variant="raw",
        generation_index=0,
        inference_only_current_image=True,
        caption=conditioning["caption"],
    )
    first = dict(sample_fp32_flat=initial.copy(), sample_full128_fp32=actions.copy())
    return metadata, arrays, generation, conditioning, first


def test_native_inactive_ones_are_zero_padding_and_original_all_action_zero_assumption_rejects_them(audit):
    fixture = initial_fixture()
    original_bytes = {key: array.tobytes() for key, array in fixture[1].items()}
    with pytest.raises(ValueError, match="privileged"):
        audit.original.verify_initial(*fixture)
    audit.verify_initial(*fixture)
    assert original_bytes == {key: array.tobytes() for key, array in fixture[1].items()}
    mask = fixture[1]["condition_mask"][-5120:].reshape(40, 128)
    assert not mask[:, :10].any() and mask[:, 10:].all()
    assert not fixture[1]["initial_noise"][-5120:].reshape(40, 128)[:, 10:].any()


@pytest.mark.parametrize(
    "fault", ["active", "future_vision", "missing_current", "missing_padding", "nonbinary", "negative_zero"]
)
def test_complete_mask_pattern_rejects_privileged_or_missing_conditioning(audit, fault):
    metadata, arrays, generation, conditioning, first = initial_fixture()
    vision = arrays["condition_mask"][:-5120].reshape(1, 48, 11, 60, 40)
    action = arrays["condition_mask"][-5120:].reshape(40, 128)
    if fault == "active":
        action[0, 0] = 1
    elif fault == "future_vision":
        vision[0, 0, 1, 0, 0] = 1
    elif fault == "missing_current":
        vision[0, 0, 0, 0, 0] = 0
    elif fault == "missing_padding":
        action[0, 10] = 0
    elif fault == "nonbinary":
        action[0, 10] = 0.5
    else:
        action[0, 0] = -0.0
    with pytest.raises(ValueError, match="native mask"):
        audit.verify_initial(metadata, arrays, generation, conditioning, first)


def test_future_vision_with_all_action_zero_was_previously_accepted_but_is_now_rejected(audit):
    fixture = initial_fixture()
    mask = fixture[1]["condition_mask"]
    mask[-5120:] = 0
    mask[:-5120].reshape(1, 48, 11, 60, 40)[0, 0, 1, 0, 0] = 1
    audit.original.verify_initial(*fixture)
    with pytest.raises(ValueError, match="native mask"):
        audit.verify_initial(*fixture)


@pytest.mark.parametrize(
    "fault",
    [
        "initial_inactive",
        "first_inactive",
        "current_latent",
        "first_flat",
        "first_action",
        "packed_bf16_sample",
        "fp16_initial",
        "fp16_mask",
        "current_shape",
        "flat_shape",
        "missing_array",
        "nonfinite",
    ],
)
def test_actual_fp32_initial_sample_current_latent_and_zero_clamp_are_bound(audit, fault):
    metadata, arrays, generation, conditioning, first = initial_fixture()
    if fault == "initial_inactive":
        arrays["initial_noise"][-5120:].reshape(40, 128)[0, 10] = 1
        first["sample_fp32_flat"] = arrays["initial_noise"].copy()
        first["sample_full128_fp32"][0, 10] = 1
    elif fault == "first_inactive":
        first["sample_full128_fp32"][0, 10] = 1
    elif fault == "current_latent":
        arrays["first_condition_latent"][0, 0, 0, 0, 0] = 0.5
    elif fault == "first_flat":
        first["sample_fp32_flat"][0] = 0.5
    elif fault == "first_action":
        first["sample_full128_fp32"][0, 0] = 1
    elif fault == "packed_bf16_sample":
        # A packed BF16 snapshot expanded to FP32 still is not the true FP32 sampler state.
        arrays["initial_noise"] = audit.original.bf16(arrays["initial_noise"])
    elif fault == "fp16_initial":
        arrays["initial_noise"] = arrays["initial_noise"].astype(np.float16)
    elif fault == "fp16_mask":
        arrays["condition_mask"] = arrays["condition_mask"].astype(np.float16)
    elif fault == "current_shape":
        arrays["first_condition_latent"] = arrays["first_condition_latent"].reshape(1, 48, 60, 1, 40)
    elif fault == "flat_shape":
        arrays["initial_noise"] = arrays["initial_noise"][None]
    elif fault == "missing_array":
        first.pop("sample_full128_fp32")
    else:
        arrays["first_condition_latent"][0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        audit.verify_initial(metadata, arrays, generation, conditioning, first)


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "other"),
        ("sampling_seed", 43),
        ("variant", "zero"),
        ("generation_index", 1),
        ("inference_only_current_image", False),
        ("caption", "future supplied"),
    ],
)
def test_original_request_metadata_bindings_remain_required(audit, field, value):
    metadata, arrays, generation, conditioning, first = initial_fixture()
    metadata[field] = value
    with pytest.raises(ValueError, match="request binding"):
        audit.verify_initial(metadata, arrays, generation, conditioning, first)


def routed_args(audit, tmp_path, monkeypatch):
    root = tmp_path / "new_audit"
    monkeypatch.setattr(audit, "AUDIT_ROOT", root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    return Namespace(
        prepared="old-prepared.json",
        prepared_sha256="a" * 64,
        qualification="old-result.json",
        output=str(root / "review/verified.json"),
        seed=42,
    )


def test_scoped_wrapper_reuses_original_verify_with_deepcopied_output_only_root(audit, tmp_path, monkeypatch):
    args = routed_args(audit, tmp_path, monkeypatch)
    source = dict(
        root=str(tmp_path / "original"),
        config={"cosmos_source_root": "frozen-model", "nested": [1]},
        source_sha256={"frozen": "b" * 64},
        input_sha256={"old": "c" * 64},
    )
    before = copy.deepcopy(source)
    reads = []

    def read_prepared(*values, **kwargs):
        reads.append((values, kwargs))
        return source

    mathematical_functions = {
        name: getattr(audit.original, name)
        for name in (
            "verify_point",
            "verify_trace",
            "verify_pair",
            "verify_imports",
            "verify_request",
            "verify_conditioning",
        )
    }

    def original_pipeline(actual_args):
        assert actual_args is args and audit.original.verify_initial is audit.verify_initial
        actual = audit.original.read_prepared("original-ref")
        assert actual["root"] == str(audit.AUDIT_ROOT.resolve())
        assert actual["config"] == before["config"] and actual["input_sha256"] == before["input_sha256"]
        actual["config"]["nested"].append(2)
        assert audit.original.read_prepared("original-ref")["config"] == before["config"]
        assert all(getattr(audit.original, name) is value for name, value in mathematical_functions.items())
        audit.original.verify_initial(*initial_fixture())
        path = Path(actual_args.output)
        path.parent.mkdir(parents=True, exist_ok=False)
        audit.original.publish(path, dict(passed=True, complete=True, model_training_authorized=False))

    monkeypatch.setattr(audit.original, "read_prepared", read_prepared)
    monkeypatch.setattr(audit.original, "verify", original_pipeline)
    originals = {name: getattr(audit.original, name) for name in ("read_prepared", "verify_initial", "publish")}
    audit.verify(args)
    assert len(reads) == 2 and source == before
    assert all(getattr(audit.original, name) is value for name, value in originals.items())
    receipt = json.loads(Path(args.output).read_text())
    assert receipt["passed"] and receipt["model_training_authorized"] is False
    assert receipt["native_initial_mask_audit"]["original_prepared_root"] == before["root"]
    assert receipt["native_initial_mask_audit"]["output_only_in_memory_root_routing"] is True
    assert receipt["native_initial_mask_audit"]["new_model_initializations"] == 0
    assert not Path(before["root"]).exists()


def test_scoped_globals_restored_when_legacy_math_rejects(audit, tmp_path, monkeypatch):
    args = routed_args(audit, tmp_path, monkeypatch)
    originals = {name: getattr(audit.original, name) for name in ("read_prepared", "verify_initial", "publish")}

    def reject(_):
        raise ValueError("unchanged independent math rejected")

    monkeypatch.setattr(audit.original, "verify", reject)
    with pytest.raises(ValueError, match="independent math"):
        audit.verify(args)
    assert all(getattr(audit.original, name) is value for name, value in originals.items())
    assert not Path(args.output).exists()


@pytest.mark.parametrize("kind", ["original", "outside", "existing"])
def test_new_receipt_cannot_write_old_external_or_existing_root(audit, tmp_path, monkeypatch, kind):
    args = routed_args(audit, tmp_path, monkeypatch)
    if kind == "original":
        monkeypatch.setattr(audit.original, "read_prepared", lambda *_: {"root": str(audit.AUDIT_ROOT)})
        monkeypatch.setattr(audit.original, "verify", lambda _: audit.original.read_prepared("ref"))
    elif kind == "outside":
        args.output = str(tmp_path.parent / "elsewhere/review/verified.json")
    else:
        Path(args.output).parent.mkdir(parents=True)
    with pytest.raises(ValueError):
        audit.verify(args)
