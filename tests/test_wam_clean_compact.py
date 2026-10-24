"""Online-to-disk witness checks, with exact binary arithmetic examples.

These tests use no model, training labels, or future observations. The small
vision geometry is a protocol fixture, not Cosmos runtime qualification.
"""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from s2a.wam_clean_capture import CleanCapturePoint
from s2a.wam_clean_compact import CompactCaptureSink, CompactRequest, iter_compact_points, load_compact_point

VISION = (1, 1, 2, 2, 2)
GRID = tuple((float(np.float32(0.875 - i / 128)), 1000 - i) for i in range(35))


def request(*, readout=None, purpose="training", seed=101):
    return CompactRequest(
        "native-anchor0", seed, purpose, "raw" if readout is None else "w1", "task0/demo2/a0", readout, "a" * 64
    )


def point(step=0, *, adapted=False, raw_dtype="torch.bfloat16", sigma=None):
    sigma = GRID[step][0] if sigma is None else sigma
    sample = np.zeros((40, 128), np.float32)
    sample[:, :10] = np.float32(1.0003)  # Deliberately distinct from the BF16 packed input.
    raw = np.zeros_like(sample)
    raw[:, :10] = 1
    deployed = raw.copy()
    deployed[:, :10] = 0.875 if adapted else 1
    packed = raw.copy()
    c0 = sample.astype(np.float64) - sigma * raw.astype(np.float64)
    residual = np.full((40, 10), sigma / 8 if adapted else 0, np.float64)
    delta = np.full((40, 10), -0.125 if adapted else 0, np.float64)
    arrays = dict(
        packed_vision=np.full(VISION, 1.5, np.float32),
        packed_action=packed,
        time_vision=np.full(2, GRID[step][1], np.float32),
        time_action=np.full(40, GRID[step][1], np.float32),
        condition_vision=np.zeros((2, 1, 1), np.float32),
        condition_action=np.zeros((40, 1), np.float32),
        sample_fp32_flat=np.concatenate((np.full(8, 1.5, np.float32), sample.reshape(-1))),
        hidden=np.full((40, 1280), 0.25, np.float32),
        projection_ids=np.zeros(40, np.int64),
        head_base_full128=np.ones((40, 128), np.float32),
        sample_full128_fp32=sample,
        raw_velocity_full128=raw,
        c0_full128=c0,
        raw_velocity_flat=np.concatenate((np.full(8, 0.5, np.float32), raw.reshape(-1))),
        deployed_velocity_flat=np.concatenate((np.full(8, 0.5, np.float32), deployed.reshape(-1))),
        deployed_velocity_full128=deployed,
        clean_residual_active10=residual,
        velocity_delta_active10=delta,
    )
    dtypes = {name: "torch.float32" for name in arrays}
    for name in ("packed_vision", "packed_action", "hidden", "head_base_full128"):
        dtypes[name] = "torch.bfloat16"
    for name in ("raw_velocity_full128", "raw_velocity_flat", "deployed_velocity_flat", "deployed_velocity_full128"):
        dtypes[name] = raw_dtype
    for name in ("c0_full128", "clean_residual_active10", "velocity_delta_active10"):
        dtypes[name] = "numpy.float64"
    dtypes["projection_ids"] = "torch.int64"
    metadata = dict(
        request_name="native-anchor0",
        noise_seed=101,
        purpose="training",
        step_index=step,
        sigma_actual=sigma,
        network_timestep=GRID[step][1],
        action_offset=8,
        action_dim_mask=(True,) * 10 + (False,) * 118,
        feature_terms=(),
        original_dtypes=tuple(sorted(dtypes.items())),
        clean_semantics="ideal FP64 sample_FP32 - actual_sigma * raw_velocity; not scheduler x0",
        returned_original_velocity_object=not adapted,
    )
    return CleanCapturePoint(metadata, arrays)


def changed(original, *, meta=None, arrays=None):
    return CleanCapturePoint(dict(original.metadata, **(meta or {})), dict(original.arrays, **(arrays or {})))


def sink(tmp_path, *, readout=None, grid=GRID):
    return CompactCaptureSink(
        tmp_path / "capture", request(readout=readout), expected_vision_shape=VISION, expected_grid=grid
    )


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("adapted", [False, True])
@pytest.mark.parametrize("dtype", ["torch.bfloat16", "torch.float32"])
def test_actual_fp32_sample_and_exact_binary_residual_survive_compact_replay(tmp_path, adapted, dtype):
    writer = sink(tmp_path, readout="b" * 64 if adapted else None)
    original = point(adapted=adapted, raw_dtype=dtype)
    before = {name: value.tobytes() for name, value in original.arrays.items()}
    saved = writer(original)
    metadata, arrays = load_compact_point(saved)
    assert metadata["sigma_actual"] == 0.875
    np.testing.assert_array_equal(arrays["sample_active10_fp32"], np.full((40, 10), np.float32(1.0003)))
    np.testing.assert_array_equal(arrays["c0_active10"], np.full((40, 10), float(np.float32(1.0003)) - 0.875))
    np.testing.assert_array_equal(arrays["deployed_velocity_active10"], np.full((40, 10), 0.875 if adapted else 1))
    np.testing.assert_array_equal(arrays["velocity_delta_active10"], np.full((40, 10), -0.125 if adapted else 0))
    assert all(not array.flags.writeable for array in arrays.values())
    assert before == {name: value.tobytes() for name, value in original.arrays.items()}
    assert "packed_vision" not in arrays and "sample_fp32_flat" not in arrays
    record = json.loads((tmp_path / "capture/step00.json").read_text())
    assert set(record["source_arrays"]) == set(original.arrays)
    assert record["source_arrays"]["packed_vision"]["original_dtype"] == "torch.bfloat16"
    assert record["source_arrays"]["packed_vision"]["storage_dtype"] == np.dtype(np.float32).str
    assert record["omitted_vision_reconstructible"] is False


def test_zero_head_with_explicit_parameter_identity_keeps_raw_bits_and_positive_zero_delta(tmp_path):
    writer = sink(tmp_path, readout="0" * 64)
    _, arrays = load_compact_point(writer(point()))
    assert not np.any(arrays["clean_residual_active10"])
    assert not np.signbit(arrays["velocity_delta_active10"]).any()
    assert arrays["raw_velocity_active10"].tobytes() == arrays["deployed_velocity_active10"].tobytes()


def test_direct_bfloat16_midpoint_gold_rejects_fp32_double_rounding(tmp_path):
    grid = ((1.0, GRID[0][1]), *GRID[1:])
    original = point(adapted=True, sigma=1.0)
    residual = np.full((40, 10), -(2**-8 + 2**-30), np.float64)
    deployed = original.arrays["deployed_velocity_full128"].copy()
    deployed[:, :10] = 1 + 2**-7
    flat = original.arrays["deployed_velocity_flat"].copy()
    flat[8:] = deployed.reshape(-1)
    exact = changed(
        original,
        arrays=dict(
            clean_residual_active10=residual,
            velocity_delta_active10=-residual,
            deployed_velocity_full128=deployed,
            deployed_velocity_flat=flat,
        ),
    )
    writer = sink(tmp_path, readout="b" * 64, grid=grid)
    _, saved = load_compact_point(writer(exact))
    np.testing.assert_array_equal(saved["deployed_velocity_active10"], np.full((40, 10), 1 + 2**-7))
    double_rounded_dir = tmp_path / "wrong"
    wrong_writer = CompactCaptureSink(
        double_rounded_dir, request(readout="b" * 64), expected_vision_shape=VISION, expected_grid=grid
    )
    deployed[:, :10] = 1  # FP32 intermediate lands exactly on a BF16 even midpoint.
    flat[8:] = deployed.reshape(-1)
    with pytest.raises(ValueError, match="rounding"):
        wrong_writer(changed(exact, arrays=dict(deployed_velocity_full128=deployed, deployed_velocity_flat=flat)))


@pytest.mark.parametrize(
    "fault", ["vision", "inactive_bits", "flat_action", "projection", "packed_action", "c0", "delta", "nonfinite"]
)
def test_inconsistent_online_witnesses_rejected_without_any_point_artifact(tmp_path, fault):
    original = point(adapted=True)
    arrays = {name: value.copy() for name, value in original.arrays.items()}
    if fault == "vision":
        arrays["deployed_velocity_flat"][0] = 0.75
    elif fault == "inactive_bits":
        arrays["deployed_velocity_full128"][0, 10] = -0.0
        arrays["deployed_velocity_flat"][8:] = arrays["deployed_velocity_full128"].reshape(-1)
    elif fault == "flat_action":
        arrays["sample_fp32_flat"][8] += 0.25
    elif fault == "projection":
        arrays["projection_ids"][0] = 1
    elif fault == "packed_action":
        arrays["packed_action"][0, 0] = 2
    elif fault == "c0":
        arrays["c0_full128"] = arrays["packed_action"].astype(np.float64) - 0.875 * arrays["raw_velocity_full128"]
    elif fault == "delta":
        arrays["velocity_delta_active10"] *= -1
    else:
        arrays["hidden"][0, 0] = np.nan
    writer = sink(tmp_path, readout="b" * 64)
    with pytest.raises(ValueError):
        writer(CleanCapturePoint(original.metadata, arrays))
    assert sorted(path.name for path in writer.directory.iterdir()) == ["request.json"]
    with pytest.raises(ValueError, match="failed"):
        writer.finish()


@pytest.mark.parametrize(
    "fault",
    ["sigma", "time", "mask", "vision_shape", "hidden_shape", "dtype", "future_meta", "future_array", "request"],
)
def test_registered_source_grid_shape_dtype_and_label_free_schema_are_enforced(tmp_path, fault):
    original = point()
    arrays, meta = {}, {}
    if fault == "sigma":
        # Still positive, but belongs to another node; positivity alone is insufficient.
        meta["sigma_actual"] = GRID[1][0]
    elif fault == "time":
        meta["network_timestep"] = original.metadata["network_timestep"] + 1
    elif fault == "mask":
        meta["action_dim_mask"] = (True,) * 10
    elif fault == "vision_shape":
        arrays["packed_vision"] = original.arrays["packed_vision"].reshape(1, 1, 1, 2, 4)
    elif fault == "hidden_shape":
        arrays["hidden"] = np.zeros((39, 1280), np.float32)
    elif fault == "dtype":
        arrays["sample_full128_fp32"] = original.arrays["sample_full128_fp32"].astype(np.float64)
    elif fault == "future_meta":
        meta["future_state"] = [1, 2, 3]
    elif fault == "future_array":
        arrays["target"] = np.zeros((40, 10))
    else:
        meta["noise_seed"] = 102
    writer = sink(tmp_path)
    with pytest.raises(ValueError):
        writer(changed(original, meta=meta, arrays=arrays))


def test_35node_ledger_has_complete_replay_and_rejects_duplicate_finish_or_directory(tmp_path):
    writer = sink(tmp_path, readout="b" * 64)
    for step in range(35):
        writer(point(step, adapted=True))
    reference = writer.finish()
    rows = 0
    for step, (metadata, arrays) in enumerate(iter_compact_points(reference)):
        assert metadata["step_index"] == step
        rows += len(arrays["hidden"])
        assert set(arrays) == {
            "sample_active10_fp32",
            "raw_velocity_active10",
            "deployed_velocity_active10",
            "c0_active10",
            "hidden",
            "projection_ids",
            "clean_residual_active10",
            "velocity_delta_active10",
            "time_vision",
            "time_action",
            "condition_vision",
            "condition_action",
            "action_dim_mask",
        }
    assert rows == 1400
    with pytest.raises(ValueError, match="closed"):
        writer.finish()
    with pytest.raises(FileExistsError):
        sink(tmp_path)


@pytest.mark.parametrize("step", [0, 2])
def test_duplicate_and_skipped_node_poison_request_instead_of_issuing_complete_ledger(tmp_path, step):
    writer = sink(tmp_path)
    writer(point())
    with pytest.raises(ValueError, match="out-of-order"):
        writer(point(step))
    assert not (writer.directory / "ledger.json").exists()
    with pytest.raises(ValueError, match="failed"):
        writer(point(1))


def test_partial_finish_fails_and_no_labels_are_needed_or_accepted(tmp_path):
    writer = sink(tmp_path)
    writer(point())
    with pytest.raises(ValueError, match="partial"):
        writer.finish()
    with pytest.raises(TypeError):
        request_with_labels = dict(
            request_name="x",
            noise_seed=101,
            purpose="training",
            round_name="raw",
            anchor_id="a",
            readout_sha256=None,
            current_image_sha256="a" * 64,
            labels=[1],
        )
        CompactRequest(**request_with_labels)


def test_offline_array_change_is_rejected_even_if_npz_file_hash_is_rewritten(tmp_path):
    writer = sink(tmp_path)
    saved = writer(point())
    metadata_path = writer.directory / "step00.json"
    record = json.loads(metadata_path.read_text())
    array_path = writer.directory / "step00.npz"
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["hidden"][0, 0] = 0.5
    with array_path.open("wb") as stream:
        np.savez(stream, **arrays)
    record["arrays_ref"] = ref(array_path)
    metadata_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="SHA differs"):
        load_compact_point(saved)
    with pytest.raises(ValueError, match="shape/dtype/SHA"):
        load_compact_point(ref(metadata_path))


def test_request_and_grid_cannot_silently_admit_eval_noise_or_nominal_sigma(tmp_path):
    with pytest.raises(ValueError, match="training noise"):
        replace(request(), noise_seed=42)
    with pytest.raises(ValueError, match="training noise"):
        replace(request(), purpose="evaluation")
    with pytest.raises(ValueError, match="FP32 sigma"):
        sink(tmp_path, grid=((0.8, 1000), *GRID[1:]))
    with pytest.raises(ValueError, match="35"):
        sink(tmp_path, grid=GRID[:3])
