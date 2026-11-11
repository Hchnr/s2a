"""Bind cross-task replan feedback to completed histories and prior replays."""

import copy
import hashlib
import subprocess
from pathlib import Path

import numpy as np

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_replan_quality_feedback as previous
from scripts.simulation.runtime import run_replan_reference_switch as switch
from scripts.simulation.runtime import run_task2_shared_shift_feedback as task2

REPO = prefix.REPO
CONFIG = REPO / "configs/tracking/joint_control/replan_shared_shift_feedback_20261111.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/replan_shared_shift_feedback_20261111.md"
RESTORATION = {
    "producer": {
        "path": str(REPO / "outputs/replan_restoration_feedback_20261110/producer/result.json"),
        "sha256": "056e6de6d3f681acc97069e9db8f543f42bb8c54821bef75793f58217bdffffd",
    },
    "checker": {
        "path": str(REPO / "outputs/replan_restoration_feedback_20261110/checker/result.json"),
        "sha256": "4a7bbcab1ccee3ca2711fbb2b0185d136996f01a2d7411fb3ed2aed6ea27fb42",
    },
}
TASK2 = {
    "producer": {
        "path": str(REPO / "outputs/task2_shared_shift_feedback_20261111/producer/result.json"),
        "sha256": "5c8fbc65c947aabf5381df51b06c1443a7c0c3bc1cdf780f7d4b23b581fbd824",
    },
    "checker": {
        "path": str(REPO / "outputs/task2_shared_shift_feedback_20261111/checker/result.json"),
        "sha256": "85ceb3299738b6cac47289984e9099423567e66b89d979d7ce160510fdae28dd",
    },
}
CASES = (2, 6)
LEGACY_CHANGES = {
    "src/s2a/research/joint_control/preview/refined_feedback.py": (
        "1c1e87ae7d1f69a09b758a6943e962cbea4789cf",
        "44b16c9525768f54df8ec6d6592769ce2de22fe08e687ffd9114d1ac9686f88a",
        "5ca9620f8bdd62924984224dc272c746d83d74c10ea4bd0cede8d8bf6b54fe08",
    ),
    "tests/research/joint_control/preview/test_refined_feedback.py": (
        "1c1e87ae7d1f69a09b758a6943e962cbea4789cf",
        "66e8d0bb14b5407691788b88e58fe027588e11c3208900b186928b08638e334b",
        "43853268c80aea0ce4826dccfe76943a23b93ef67b999ba12a54d39ca442fa8c",
    ),
    "src/s2a/research/native_runtime/development_prefix_runtime.py": (
        "886896a412f92f8955e9927d511727c3b8a4e075",
        "7ab18db4dbf778f13659931815e35895c222273ccfbdc5e62e2bdfd25deff517",
        "efac069505e92117766472fd474e0bbc19b929ae1a50a54b13ffcb3f7455886c",
    ),
    "tests/research/native_runtime/test_development_prefix_replay.py": (
        "886896a412f92f8955e9927d511727c3b8a4e075",
        "bb07baa6527c8aa2a536c8f61160d565eab5ae900750ffd1645e149b5e10e0b1",
        "e277668d6369a6352361616c9d718111a0bf6c3e5f5b3fe743211df25db6577c",
    ),
}


def read_config():
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    expected = dict(task2.read_config()[1])
    expected.update(
        schema="s2a.replan-shared-shift-feedback.v1",
        case_indices=list(CASES),
        branch="shared_shift_quality_new",
        prefix_steps=16,
        reference_chunk_index=2,
        reference_cursor=0,
        capture_final_prefix_inner=True,
        max_control_steps_per_phase=48,
        max_native_steps_per_phase=1200,
    )
    prefix.require(config == expected, "Exact cross-task shared-budget feedback registration")
    return ref, config


def validate_legacy_sources(recorded, current):
    """Verify the exact historical bytes and enumerate known later changes."""
    prefix.require(len(recorded) == len(current), "The historical source list must keep its order and length")
    found = []
    for before, now in zip(recorded, current, strict=True):
        prefix.require(before["path"] == now["path"], "The historical source paths must retain their identities")
        if before == now:
            continue
        relative = str(Path(before["path"]).relative_to(REPO))
        prefix.require(relative in LEGACY_CHANGES, "Only explicitly recorded historical source changes are permitted")
        commit, old_sha, current_sha = LEGACY_CHANGES[relative]
        prefix.require(
            before["sha256"] == old_sha and now["sha256"] == current_sha,
            "Exact historical and current source fingerprints required",
        )
        content = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO)
        prefix.require(hashlib.sha256(content).hexdigest() == old_sha, "Historical Git bytes match the old receipt")
        found.append(relative)
    prefix.require(set(found) == set(LEGACY_CHANGES), "All four known historical source changes must be present")
    return found


def source_refs():
    return (
        previous.source_refs(restore=True)
        + task2.source_refs()
        + [
            prefix.reference(path)
            for path in (
                Path(__file__),
                CONFIG,
                PLAN,
                REPO / "tests/research/native_runtime/test_replan_shared_shift_inputs.py",
            )
        ]
    )


def policy_case(case):
    """Give the candidate only generated targets and completed controller history."""
    selected = switch.selected_case(case, "new")
    history = selected["completed_history"]
    prefix.require(
        selected["spec"]["case_index"] in CASES
        and selected["spec"]["prefix_steps"] == 16
        and selected["intervention_spec"]["cursor"] == 0
        and selected["targets"].shape == (40, 4, 4)
        and history.completed_periods == 16
        and history.last_plan.shape == (8, 6)
        and np.array_equal(history.last_plan[0], history.previous_command[:6]),
        "Causal step-16 history and already generated reference",
    )
    return {
        "spec": {"prefix_steps": 16},
        "intervention_spec": {"cursor": 0},
        **{
            key: selected[key]
            for key in ("targets", "anchor", "tool", "controller", "previous_evidence", "completed_history")
        },
    }


def load_inputs():
    config_ref, config = read_config()
    inputs = previous.load_restoration_inputs()
    original_sources, assets = copy.deepcopy(inputs["source_sha256"]), copy.deepcopy(inputs["asset_refs"])
    refs = inputs["source_sha256"]
    refs[config_ref["path"]] = config_ref["sha256"]
    for phase, ref in RESTORATION.items():
        result = prefix.read(ref, refs)
        prefix.require(
            result["phase"] == phase
            and result["complete"] is result["passed"] is result["sources_unchanged"] is True
            and len(result["branch_refs"]) == len(CASES),
            "Both previous restoration branches and replays must be closed",
        )
        registration = prefix.read(result["registration_ref"], refs)
        prefix.require(
            registration["source_sha256"] == original_sources and registration["asset_refs"] == assets,
            "The previous restoration used the same source-pinned development inputs",
        )
        validate_legacy_sources(registration["source_refs"], previous.source_refs(restore=True))
        seen = []
        for branch_ref in result["branch_refs"]:
            item = prefix.read(branch_ref, refs)
            prefix.require(
                item["phase"] == phase
                and item["branch"] == "restoration_quality_new"
                and item["complete"] is item["passed"] is True
                and (phase != "checker" or item["restoration_module_imported"] is False),
                "A matching completed restoration branch is required",
            )
            seen.append(item["case_index"])
        prefix.require(tuple(seen) == CASES, "Both earlier task identities and order are fixed")
    for phase, ref in TASK2.items():
        result = prefix.read(ref, refs)
        prefix.require(
            result["phase"] == phase
            and result["complete"] is result["passed"] is result["sources_unchanged"] is True
            and result["control_qualified"] is False
            and len(result["branch_refs"]) == len(task2.BRANCHES),
            "The prior task2 shared-shift feedback and replay must be closed",
        )
        registration = prefix.read(result["registration_ref"], refs)
        prefix.require(registration["source_refs"] == task2.source_refs(), "The same audited task2 candidate")
        for identity, branch_ref in zip(task2.BRANCHES, result["branch_refs"], strict=True):
            item = prefix.read(branch_ref, refs)
            prefix.require(
                (item["case"]["case_index"], item["branch"]) == identity
                and item["complete"] is item["passed"] is True
                and (phase != "checker" or item["candidate_modules_unloaded"] is True),
                "The two previous task2 branches retain their identities",
            )
    prefix.require(tuple(case["spec"]["case_index"] for case in inputs["cases"]) == CASES, "Both replan cases")
    for case in inputs["cases"]:
        policy_case(case)
    inputs.update(replan_shared_config_ref=config_ref, replan_shared_config=config)
    return inputs


def input_receipt(inputs):
    return dict(
        replan_shared_config_ref=inputs["replan_shared_config_ref"],
        restoration_refs=RESTORATION,
        task2_refs=TASK2,
        legacy_source_changes={path: list(item) for path, item in LEGACY_CHANGES.items()},
        asset_refs=inputs["asset_refs"],
        source_sha256=inputs["source_sha256"],
    )
