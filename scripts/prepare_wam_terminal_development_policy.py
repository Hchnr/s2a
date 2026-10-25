"""Bind the fixed061 policy to qualified current-only tasks/templates; no model."""

import os
from pathlib import Path

from verify_wam_clean_training_capture import Evidence, require

from s2a.clean_training_runtime import publish, reference
from s2a.terminal_development_gate import ROOT, admit_state
from s2a.terminal_development_policy import DevelopmentPolicy, case_order


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    e = Evidence()
    _, admission = admit_state(reference(ROOT / "orchestration/result.json"), read_json=e.json)
    prepared = e.json(admission["state_prepared_ref"])
    tasks = []
    for index in range(3):
        zeros = [z for z in prepared["zero_cases"] if z["name"].startswith(f"task{index}_")]
        current = [a for a in prepared["current_inputs"] if a["name"].startswith(f"task{index}_")]
        require(len(zeros) == 1 and len(current) == 8, "Exact two-demo/four-anchor task metadata required")
        task = zeros[0]["request"]["prompt"]
        require(all(a["task"] == task for a in current), "Task/template identity differs")
        tasks.append(dict(task_index=index, task=task))
    manifest = dict(
        root=str((Path("outputs/terminal_development_20261025")).resolve()),
        zero_cases=[dict(request=z["request"]) for z in prepared["zero_cases"]],
        development_tasks=tasks,
    )
    policy = DevelopmentPolicy(manifest)
    require(policy.order == case_order() and len(policy.order) == 12, "Actual061 policy roster differs")
    e.finish()
    root = Path("outputs/terminal_development_preparation_20261025")
    root.mkdir(exist_ok=True)
    publish(
        root / "policy_metadata.json",
        dict(
            passed=True,
            complete=True,
            admission=admission,
            policy_manifest=manifest,
            case_order=case_order(),
            expected_requests=900,
            expected_physics_steps=7200,
            execution_authorized=False,
            fresh_authorized=False,
            input_sha256=e.inputs,
            source_refs=[reference(Path(__file__)), reference(Path("src/s2a/terminal_development_policy.py"))],
            model_initializations=0,
            new_model_calls=0,
            physics_steps=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
