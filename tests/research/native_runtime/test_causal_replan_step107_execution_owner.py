import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import run_causal_replan_step107_execution as owner


def test_worker_adapter_enforces_registered_physical_limit(monkeypatch):
    from s2a.research.native_runtime import reset_bounded_runtime as bounded
    from scripts.simulation.runtime import reset_bowl_reference_tail as tail

    for name in ("OUTPUT", "load_inputs", "input_receipt", "source_refs"):
        monkeypatch.setattr(owner.original.previous, name, getattr(owner.original.previous, name))
    monkeypatch.setattr(owner.original, "OUTPUT", owner.original.OUTPUT)
    for name in ("run_producer", "run_checker"):
        monkeypatch.setattr(tail, name, getattr(tail, name))
    base = bounded.ResetBoundedRuntime
    monkeypatch.setattr(base, "run_feedback", lambda _self, *args, max_control_steps, **kwargs: max_control_steps)
    monkeypatch.setattr(bounded, "ResetBoundedRuntime", base)
    captured = []
    monkeypatch.setattr(owner.continuation, "run_producer", lambda *args, **kwargs: captured.append(kwargs))
    prior = dict(previous_88_producer="prior88", capture88="capture88", reference88="raw88", envelope88="env88")
    inputs = dict(
        base=dict(source=prior, producer="actual_prior"), capture="capture107", raw="raw107", envelope="env107"
    )
    owner.bind_worker(inputs)
    adapter = bounded.ResetBoundedRuntime.__new__(bounded.ResetBoundedRuntime)
    assert adapter.run_feedback(max_control_steps=88) == 147
    with pytest.raises(ValueError, match="original bound"):
        adapter.run_feedback(max_control_steps=56)
    tail.run_producer()
    assert captured[0]["prior_producer"] == "actual_prior"
    assert captured[0]["load_new_reference"]() == ("raw107", "env107")


def test_checker_import_keeps_both_policies_and_simulator_unloaded():
    code = (
        "import sys; from scripts.simulation.runtime import run_causal_replan_step107_execution as o; "
        "o.original.previous.isolation.checker_isolation(); o.original.previous.original.require_no_simulator_import(); "
        "assert 's2a.research.joint_control.preview.causal_reference_replan' not in sys.modules"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_summary_checks_the_actual_capture_and_stop(monkeypatch):
    def row(step):
        return dict(
            step=step,
            actual_inner=[{}] * 25,
            tcp_error=[0.0] * 6,
            pose_caps_satisfied=True,
            minimum_inner_margin_rad=0.4,
            target_grasp=False,
            task_success=False,
        )

    rows = [row(step) for step in range(107)]
    monkeypatch.setattr(owner.continuation, "prefix_periods", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(owner.tail, "tail_periods", lambda *_args: [])
    stop = dict(
        reason="requested_observation",
        at_step=107,
        next_action_dispatched=False,
        trigger=dict(replan=True, current_step=107),
    )
    candidate = dict(
        quality_refiner=dict(
            quality_response_evaluations=0,
            baseline_response_evaluations=0,
            endpoint_queries_charged=0,
            transition_steps_charged=0,
        ),
        gripper_proposals=dict(candidate_sequences=0),
    )
    runtime = dict(
        complete=True,
        passed=True,
        cleanup_complete=True,
        reset_observation_verified=True,
        reset_image_match=True,
        previous_saved_command=None,
        continuation=dict(
            prefix107=dict(new_reference=dict(stop=stop)),
            replanned=dict(
                stop=dict(reason="quality_no_witness", at_step=107, next_action_dispatched=False),
                generated_periods=0,
                complete=False,
            ),
        ),
        control_steps_attempted=107,
        control_steps_completed=107,
        newly_selected_commands=107,
        counters=dict(
            mj_step=dict(attempted=2675, completed=2675), model_construct=dict(completed=3), context=dict(completed=3)
        ),
        segment_receipts=[candidate],
        replan_capture=dict(
            snapshot=dict(state_sha256=owner.FIXED["state_sha256"]),
            capture=dict(canvas_png=dict(sha256=owner.FIXED["image_sha256"])),
        ),
    )
    branch = dict(complete=True, passed=True, runtime=runtime)
    counts = owner.summarize_branch(branch, "producer", owner.FIXED)
    assert counts["control_steps"] == 107 and counts["new_reference_steps"] == 0
    runtime["replan_capture"]["snapshot"]["state_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="causal trigger"):
        owner.summarize_branch(branch, "producer", owner.FIXED)
