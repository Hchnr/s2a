"""024 real tensor boundaries and fail-closed preparation, with no model/GPU."""

import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_wam_current_numerics.py"
    spec = importlib.util.spec_from_file_location("current_numerics_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_fails_closed_cpu_and_physical_gpu6(runner, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    runner.environment()
    with pytest.raises(ValueError, match="CUDA6"):
        runner.environment(run=True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    runner.environment(run=True)
    for cuda, ld in (("0", ""), ("4,5,6,7", ""), ("6", "/tmp")):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda)
        monkeypatch.setenv("LD_LIBRARY_PATH", ld)
        with pytest.raises(ValueError, match="CUDA6"):
            runner.environment(run=True)


def test_future_arrays_uses_anchor_complete_real_futures_and_valid_mask(runner, tmp_path):
    path = tmp_path / "future.npz"
    arrays = dict(
        auxiliary_0=np.arange(43, dtype=np.uint8)[:, None, None, None],
        left_wrist=np.arange(43, dtype=np.uint8)[:, None, None, None],
        actions=np.zeros((43, 128)),
        action_valid=np.ones(43, dtype=bool),
        action_mask=np.broadcast_to(runner.ACTIVE, (43, 128)),
    )
    np.savez(path, **arrays)
    sample = dict(arrays_path=str(path), anchor_frame=2)
    future = runner.future_arrays(sample)
    assert future["auxiliary_0"][:, 0, 0, 0].tolist() == list(range(2, 43))
    assert future["physical_action"].shape == (40, 128)
    with pytest.raises(ValueError, match="complete true41"):
        runner.future_arrays(dict(sample, anchor_frame=3))
    arrays["action_valid"][15] = False
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="complete true41"):
        runner.future_arrays(sample)


def test_exact_gate_checks_dtype_shape_and_every_value(runner):
    x = np.asarray([0.1], dtype=np.float32)
    runner.exact("same", x, x.copy())
    for changed in (x.astype(np.float64), x[:, None], np.nextafter(x, np.float32(1))):
        with pytest.raises(ValueError, match="prerequisite differs"):
            runner.exact("changed", x, changed)


def test_correct_task_request_cannot_borrow_another_task_or_mask(runner):
    sample = dict(task="put wine in rack", image_path="image.png")
    request = dict(
        prompt=sample["task"],
        vision_path=sample["image_path"],
        seed=42,
        action_chunk_size=40,
        num_frames=41,
        guidance=1.0,
        shift=5.0,
        num_steps=35,
        fps=20,
        model_mode="wam",
        extra=dict(action128=dict(task_title=sample["task"], action_dim_mask=runner.ACTIVE.copy())),
    )
    runner.validate_request(request, sample)
    request["extra"]["action128"]["task_title"] = "put bowl on plate"
    with pytest.raises(ValueError, match="correct-task"):
        runner.validate_request(request, sample)
    request["extra"]["action128"]["task_title"] = sample["task"]
    request["extra"]["action128"]["action_dim_mask"][10] = True
    with pytest.raises(ValueError, match="correct-task"):
        runner.validate_request(request, sample)


def test_paired_rng_restore_includes_python_numpy_torch_and_full_device_state(runner, monkeypatch, tmp_path):
    gpu_state = [torch.arange(8, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [x.clone() for x in gpu_state])

    def set_gpu(values):
        gpu_state[:] = [x.clone() for x in values]

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", set_gpu)
    state = runner.rng_state(torch)
    first = (random.random(), np.random.random(), torch.rand(2))
    gpu_state[0].add_(1)
    runner.restore_rng(torch, state)
    second = (random.random(), np.random.random(), torch.rand(2))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])
    assert torch.equal(gpu_state[0], state["torch_cuda"][0])
    receipt = runner.save_rng(tmp_path, "rng", state)
    metadata = runner.read(receipt["path"])
    with np.load(metadata["arrays"]["path"], allow_pickle=False) as evidence:
        np.testing.assert_array_equal(evidence["numpy_keys"], state["numpy"][1])
        np.testing.assert_array_equal(evidence["torch_cpu"], state["torch_cpu"].numpy())
        np.testing.assert_array_equal(evidence["torch_cuda_0"], state["torch_cuda"][0].numpy())


def test_tensor_dump_saves_bfloat16_losslessly_and_rejects_unknown_metadata(runner):
    @dataclass
    class Payload:
        tokens: list
        caption: str

    original = torch.tensor([1, 1.0078125], dtype=torch.bfloat16)
    dump = runner.TensorDump()
    tree = dump.tree("actual", Payload([original], "task"))
    assert tree["tokens"][0] == dict(npz_key="actual.tokens.0", original_dtype="torch.bfloat16")
    np.testing.assert_array_equal(dump.arrays["actual.tokens.0"], original.float().numpy())
    assert dump.arrays["actual.tokens.0"].dtype == np.float32
    with pytest.raises(TypeError, match="unrecorded actual metadata"):
        dump.tree("bad", object())


@dataclass
class Modality:
    tokens: list
    condition_mask: list
    loss_valid_mask: list


@dataclass
class Packed:
    vision: Modality
    action: Modality


def simple_packed():
    return Packed(
        vision=Modality(
            [torch.zeros(1, 2, 1, 1)], [torch.tensor([[[1.0]], [[0.0]]])], [torch.ones(2, 1, 1, dtype=torch.bool)]
        ),
        action=Modality(
            [torch.zeros(2, 128)], [torch.zeros(2, 1)], [torch.tensor([True] * 10 + [False] * 118).expand(2, -1)]
        ),
    )


def test_actual_denoise_budget_cannot_overrun_repeat_or_run_outside_boundary(runner):
    observed = []

    def original(**kwargs):
        observed.append(kwargs)
        return dict(preds_vision=[torch.zeros(1, 2, 1, 1)], preds_action=[torch.zeros(2, 128)])

    labels = [
        f"{task}:{sigma}:{branch}" for task in range(3) for sigma in (0.2, 0.5, 0.8) for branch in ("A", "B", "C")
    ]
    budget = runner.DenoiseBudget(original, labels)
    with pytest.raises(ValueError, match="outside"):
        budget(data_batch_packed=simple_packed())
    for label in labels:
        budget.arm(label)
        budget(data_batch_packed=simple_packed())
    assert len(observed) == len(budget.calls) == 27
    assert all(c["completed"] for c in budget.calls)
    with pytest.raises(ValueError, match="exhausted27"):
        budget.arm(labels[0])
    assert len(observed) == 27


def test_inactive_input_rejected_before_wrapper_or_forward(runner):
    calls = []
    budget = runner.DenoiseBudget(lambda **kwargs: calls.append(kwargs), ["A"])
    packed = simple_packed()
    packed.action.tokens[0][0, 10] = 0.1
    budget.arm("A")
    with pytest.raises(ValueError, match="before the mask wrapper"):
        budget(data_batch_packed=packed)
    assert not calls


def test_flow_checks_model_precision_cast_and_masked_target(runner):
    clean = SimpleNamespace(
        x0_tokens_vision=[torch.ones(1, 2, 1, 1)],
        x0_tokens_action=[torch.ones(2, 128)],
    )
    packed = simple_packed()
    noised = SimpleNamespace()
    for kind in ("vision", "action"):
        x0 = getattr(clean, f"x0_tokens_{kind}")[0]
        sigma = torch.full_like(getattr(packed, kind).condition_mask[0], 0.2)
        epsilon = torch.full_like(x0, 2)
        xt = (epsilon * sigma + x0 * (1 - sigma)).bfloat16()
        target = epsilon - x0
        if kind == "action":
            valid = packed.action.loss_valid_mask[0]
            xt, target = xt.masked_fill(~valid, 0), target.masked_fill(~valid, 0)
        for key, value in (("epsilon", epsilon), ("xt_tokens", xt), ("vt_target", target), ("sigmas", sigma)):
            setattr(noised, f"{key}_{kind}", [value])
    tol = dict(flow_rtol=1e-6, flow_atol=1e-7)
    runner.check_flow(clean, noised, packed, tol)
    noised.vt_target_action[0][0, 0] += 1e-3
    with pytest.raises(ValueError, match="action flow target"):
        runner.check_flow(clean, noised, packed, tol)


def test_loss_uses_explicit_active_denominator_and_rejects_extra_total(runner):
    dump = runner.TensorDump()
    for kind in ("vision", "action"):
        dump.arrays.update(
            {
                f"pred_{kind}": np.full((2, 2), 2, dtype=np.float32),
                f"noised.vt_target_{kind}.0": np.zeros((2, 2), dtype=np.float32),
                f"condition_{kind}": np.asarray([[1], [0]], dtype=np.float32),
                f"valid_{kind}": np.asarray([[True, True], [True, False]]),
                f"time_weight_{kind}": np.asarray([1], dtype=np.float32),
            }
        )
    losses = dict(flow_matching_loss_vision=4, flow_matching_loss_action=4)
    settings, tol = dict(vision_weight=2, action_weight=3), dict(loss_rtol=1e-5, loss_atol=1e-6)
    result = runner.validate_losses(dump, 20, losses, settings, tol)
    assert result["modalities"]["vision"]["denominator"] == 1
    assert result["total"] == 20
    with pytest.raises(ValueError, match="total objective"):
        runner.validate_losses(dump, 21, losses, settings, tol)
    with pytest.raises(ValueError, match="vision loss"):
        runner.validate_losses(dump, 20, dict(losses, flow_matching_loss_vision=1), settings, tol)


def test_eval_requirement_checks_nested_submodules(runner):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.eval()
    runner.require_eval(model)
    model[1].train()
    with pytest.raises(ValueError, match="every submodule"):
        runner.require_eval(model)


def test_source_and_array_bundles_are_immutable_and_finite(runner, tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("fixed")
    runner.check_hashes({str(source): runner.sha256(source)})
    source.write_text("changed")
    with pytest.raises(ValueError, match="frozen input changed"):
        runner.check_hashes({str(source): "0" * 64})
    runner.bundle(tmp_path, "evidence", dict(x=np.asarray([1])), dict(complete=True))
    with pytest.raises(FileExistsError):
        runner.bundle(tmp_path, "evidence", dict(x=np.asarray([2])), {})
