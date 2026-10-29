"""Saved synthetic ledger arithmetic only, without a solver or simulator."""

import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("diagnose_prefix_trust_saved_linearization")


class Archive:
    """A prevalidated affine q fixture, represented as six-component TCP errors."""

    def __init__(self, q_function=None):
        self.trace, self.ledger, self.phases, self.seen = [], [], [], {}
        self.bias = np.full(24, 0.3)
        self.bias[18:] = 1.2
        self.matrix = np.arange(24 * 72, dtype=float).reshape(24, 72) / 100000
        self.q_function = q_function or (lambda x: self.bias + self.matrix @ x)
        self.stage = None
        self.max_evaluations = 4000
        self.request(np.zeros((12, 6)), "warm")

    def request(self, commands, role, *, axis=None, delta=None, rejected=False):
        commands = np.asarray(commands, dtype=np.float64).copy()
        key, before = commands.tobytes().hex(), len(self.trace)
        response, new = None, None
        if rejected:
            assert key not in self.seen
            outcome = "budget_rejected"
            self.max_evaluations = before
        elif key in self.seen:
            response, outcome = self.seen[key], "cache_hit"
        else:
            q = self.q_function(commands.ravel()).reshape(12, 2)
            assert np.all(q >= 0)
            errors = np.zeros((12, 6))
            errors[:, 0], errors[:, 3] = np.sqrt(q[:, 0]) * 0.01, np.sqrt(q[:, 1]) * 0.05
            failed = np.flatnonzero(
                (np.linalg.norm(errors[:, :3], axis=1) > 0.01) | (np.linalg.norm(errors[:, 3:], axis=1) > 0.05)
            )
            prefix = int(failed[0]) if len(failed) else 12
            response = new = len(self.trace)
            outcome = "new_response"
            self.seen[key] = response
            self.trace.append(
                dict(
                    commands=commands,
                    tcp_errors=errors,
                    prefix_length=prefix,
                    joint_feasible=True,
                    feasible=prefix == 12,
                    safety_min=0.1,
                )
            )
        index = len(self.ledger)
        self.ledger.append(
            dict(
                request_index=index,
                stage_index=self.stage,
                role=role,
                command_key=key,
                response_index=response,
                new_response_index=new,
                outcome=outcome,
                response_evaluations_before=before,
                response_evaluations_after=len(self.trace),
                axis=axis,
                nominal_delta=delta,
            )
        )
        return index

    def begin(self, *, backward=False):
        self.stage = len(self.phases)
        upper = np.full((12, 6), 0.01)
        if backward:
            upper[0, 0] = 0.0
        self.phases.append(
            dict(
                stage_index=self.stage,
                requests_before=len(self.ledger),
                prefix_before=9,
                target_cursor=9,
                radius=0.01,
                stage_lower_command=np.full((12, 6), -0.01),
                stage_upper_command=upper,
                anchor_before_index=0,
                initial_commands=np.zeros((12, 6)),
                callback_count=0,
            )
        )
        self.request(np.zeros((12, 6)), "stage_anchor")

    def group(self, commands, *, role="gradient", reject_axis=None):
        commands = np.asarray(commands).copy()
        index = self.request(commands, role + "_base")
        upper = self.phases[-1]["stage_upper_command"].ravel()
        for axis in range(72):
            delta = 1e-4 if commands.ravel()[axis] + 1e-4 <= upper[axis] else -1e-4
            probe = commands.copy()
            probe.ravel()[axis] += delta
            self.request(probe, role + "_probe", axis=axis, delta=delta, rejected=axis == reject_axis)
            if axis == reject_axis:
                break
        return index

    def callback(self, commands):
        index = self.request(commands, "callback")
        self.phases[-1]["callback_count"] += 1
        self.phases[-1]["last_callback_commands"] = np.asarray(commands).copy()
        return index

    def end(self):
        self.phases[-1]["requests_after"] = len(self.ledger)

    def producer(self):
        return dict(
            complete=True,
            passed=True,
            case_name="candidate",
            response_evaluations=len(self.trace),
            answer=dict(
                horizon=12,
                finite_difference_step=1e-4,
                evaluated_candidates=self.trace,
                request_ledger=self.ledger,
                phases=self.phases,
                response_evaluations=len(self.trace),
                lower_command=-np.ones((12, 6)),
                upper_command=np.ones((12, 6)),
                max_evaluations=self.max_evaluations,
            ),
        )


def point(value):
    command = np.zeros((12, 6))
    command[0, 0] = value
    return command


def simple_archive():
    archive = Archive()
    archive.begin()
    archive.group(point(0))
    archive.callback(point(0.001))
    archive.end()
    return archive


def test_affine_q_has_zero_residual_and_latest_model_uses_ledger_not_previous_callback(diagnostic):
    archive = Archive()
    archive.begin()
    first = archive.group(point(0))
    archive.callback(point(0.001))
    second = archive.group(point(0.002), role="jacobian")  # Neither the anchor nor the previous callback.
    archive.callback(point(0.003))
    future = archive.group(point(0.004))
    archive.callback(point(0.004))
    archive.end()
    result = diagnostic.diagnose_saved_linearization(archive.producer())
    callbacks = result["callbacks"]
    assert [row["latest_complete_fd"]["base_ledger_index"] for row in callbacks] == [first, second, future]
    assert [row["stage_anchor_fd"]["base_ledger_index"] for row in callbacks] == [first] * 3
    assert result["summary"]["zero_step_count"] == 1
    assert result["summary"]["nonzero_step_count"] == 2
    for row in callbacks:
        for name in ("latest_complete_fd", "stage_anchor_fd"):
            model = row[name]
            assert np.asarray(model["q_linear"]).shape == (24,)
            np.testing.assert_allclose(model["residual"], 0, atol=2e-12, rtol=0)
    # The final callback exactly repeats its latest base, and stays in the archive.
    assert callbacks[-1]["latest_complete_fd"]["zero_step"]
    assert callbacks[-1]["latest_complete_fd"]["command_bytes_equal"]


def test_same_stage_causality_backward_difference_and_noncontiguous_cached_probes(diagnostic):
    archive = simple_archive()
    archive.begin(backward=True)
    before_group = archive.callback(point(-0.001))
    backward = archive.group(point(0), role="jacobian")
    archive.callback(point(-0.002))
    archive.end()
    result = diagnostic.diagnose_saved_linearization(archive.producer())
    missing = next(row for row in result["callbacks"] if row["ledger_index"] == before_group)
    assert missing["latest_complete_fd"] is missing["stage_anchor_fd"] is None
    assert len(result["data_gaps"]) == 2  # No reuse of stage0 and no use of the future group.
    group = result["groups"][-1]
    assert group["base_ledger_index"] == backward
    assert group["nominal_deltas"][0] == -1e-4 and group["nominal_deltas"][1:] == [1e-4] * 71
    assert group["probe_response_indices"][1:] == result["groups"][0]["probe_response_indices"][1:]
    assert group["probe_response_indices"][0] > group["probe_response_indices"][1]
    np.testing.assert_allclose(result["callbacks"][-1]["latest_complete_fd"]["residual"], 0, atol=2e-12, rtol=0)


def test_budget_truncated_group_is_retained_but_not_used_as_a_model(diagnostic):
    archive = simple_archive()
    archive.group(point(0.002), reject_axis=3)
    archive.end()
    result = diagnostic.diagnose_saved_linearization(archive.producer())
    assert result["complete_fd_group_count"] == result["incomplete_fd_group_count"] == 1
    incomplete = result["groups"][-1]
    assert incomplete["incomplete_reason"] == "probe_budget_rejected"
    assert incomplete["probe_response_indices"][-1] is None and "jacobian" not in incomplete
    assert result["callbacks"][0]["latest_complete_fd"]["fd_group_index"] == 0


@pytest.mark.parametrize("change", ["axis", "delta", "response", "signed_zero", "stage", "truncation"])
def test_fd_metadata_or_command_tampering_is_rejected(diagnostic, change):
    producer = deepcopy(simple_archive().producer())
    answer = producer["answer"]
    ledger = answer["request_ledger"]
    probe = ledger[3]
    assert probe["role"] == "gradient_probe" and probe["axis"] == 0
    if change == "axis":
        probe["axis"] = 1
    elif change == "delta":
        probe["nominal_delta"] = -1e-4
    elif change == "response":
        probe["response_index"] = 0
    elif change == "signed_zero":
        command = answer["evaluated_candidates"][probe["response_index"]]["commands"]
        command[0, 1] = -0.0
        probe["command_key"] = command.tobytes().hex()
    elif change == "stage":
        probe["stage_index"] = 1
    else:
        # A callback cannot quietly stand in for a missing last derivative probe.
        ledger[74]["role"] = "objective"
        ledger[74]["axis"] = ledger[74]["nominal_delta"] = None
    with pytest.raises(ValueError):
        diagnostic.diagnose_saved_linearization(producer)


def test_negative_linear_predictions_remain_unclipped(diagnostic):
    bias = np.full(24, 0.3)
    bias[18:] = 1.2
    archive = Archive(lambda x: bias * np.exp(-1000 * x[0]))
    archive.begin()
    archive.group(point(0))
    archive.callback(point(0.003))
    archive.end()
    model = diagnostic.diagnose_saved_linearization(archive.producer())["callbacks"][0]["latest_complete_fd"]
    assert np.all(model["q_linear"] < 0)
    assert model["linear_negative_component_count"] == 24
    np.testing.assert_array_equal(model["residual"], model["q_actual"] - model["q_linear"])


def test_cli_interface_binds_sources_and_writes_exclusively_without_qualification(diagnostic, monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(diagnostic, "REPO", tmp_path)
    script = tmp_path / "diagnostic.py"
    script.write_text("synthetic source binding\n")
    plan = tmp_path / "plan.md"
    plan.write_text("synthetic arithmetic-only plan\n")
    monkeypatch.setattr(diagnostic, "__file__", str(script))
    monkeypatch.setattr(diagnostic, "PLAN", plan)
    source = tmp_path / "producer.json"
    source.write_text(json.dumps(simple_archive().producer(), default=lambda value: value.tolist()))
    output = tmp_path / "result.json"
    ref = diagnostic.reference(source)
    diagnostic.main(source, ref["sha256"], output)
    result = json.loads(output.read_text())
    assert result["input_ref"] == ref and result["source_ref"] == diagnostic.reference(script)
    assert result["plan_ref"] == diagnostic.reference(plan)
    assert not result["independent_numerical_qualified"] and not result["control_qualified"]
    assert all(
        result[key] == 0
        for key in (
            "new_responses",
            "query_count",
            "numerical_inner_steps",
            "physics_steps",
            "model_calls",
            "world_model_calls",
            "fresh_cases",
        )
    )
    with pytest.raises(ValueError, match="Exclusive"):
        diagnostic.main(source, ref["sha256"], output)
    with pytest.raises(ValueError, match="Input SHA"):
        diagnostic.main(source, "0" * 64, tmp_path / "wrong.json")
    assert not (tmp_path / "wrong.json").exists()
