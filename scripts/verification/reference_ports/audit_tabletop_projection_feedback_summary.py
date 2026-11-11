"""Correct replay-summary semantics without rerunning the physical experiment."""

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime.run_tabletop_projection_feedback_probe import check_pair

ROOT = prefix.REPO
OWNER = ROOT / "outputs/tabletop_projection_feedback_probe_20261111/result.json"
OWNER_SHA = "a3598795456067aeb66b6d23ec3a7cc26adcda07c690286cd12383f8c5069466"
OUTPUT = ROOT / "outputs/tabletop_projection_feedback_summary_audit_20261111/result.json"


def replay_reference_shift(produced, replayed):
    """Verify the replayed target and calculate its offset from the original."""
    initial = np.asarray(produced["decision"]["original_target_tcp"], dtype=float)
    selected = np.asarray(produced["decision"]["selected_target_tcp"], dtype=float)
    saved_replay = np.asarray(replayed["decision"]["selected_target_tcp"], dtype=float)
    prefix.require(
        initial.shape == selected.shape == saved_replay.shape == (4, 4), "Three complete target poses required"
    )
    np.testing.assert_allclose(saved_replay, selected, atol=1e-12, rtol=0)
    shift = float(np.linalg.norm(selected[:3, 3] - initial[:3, 3]))
    np.testing.assert_allclose(shift, produced["decision"]["reference_translation_m"], atol=1e-12, rtol=0)
    return shift


def main():
    prefix.environment()
    owner_ref = {"path": str(OWNER), "sha256": OWNER_SHA}
    owner = prefix.read(owner_ref)
    prefix.require(
        owner["complete"] is owner["passed"] is True
        and owner["control_qualified"] is False
        and owner["total_native_steps"] == 19050
        and owner["newly_selected_commands"] == 21,
        "The complete original bounded feedback result is required",
    )
    corrections = {}
    for mode in ("original", "nominal", "pose_bounded"):
        producer = prefix.read(owner["producer_refs"][mode])
        checker = prefix.read(owner["checker_refs"][mode])
        check_pair(producer, checker)
        original = producer["runtime"]["continuation"]["probe"]["periods"]
        replayed = checker["runtime"]["continuation"]["probe"]["periods"]
        shifts = [replay_reference_shift(a, b) for a, b in zip(original, replayed, strict=True)]
        maximum = max(shifts, default=0.0)
        np.testing.assert_allclose(
            maximum, owner["summaries"][mode]["producer"]["max_reference_translation_m"], atol=1e-12, rtol=0
        )
        prefix.require(
            owner["summaries"][mode]["checker"]["max_reference_translation_m"] == 0.0
            and checker["runtime"]["newly_selected_commands"] == 0,
            "The historical checker summary must be identified without claiming a policy rerun",
        )
        corrections[mode] = {
            "saved_checker_summary_max_reference_translation_m": 0.0,
            "saved_checker_summary_zero_means": "producer-only decision field absent; not a recomputed zero shift",
            "replayed_selected_reference_max_shift_m": maximum,
            "checker_recomputed_policy": False,
            "producer_stop_independently_rechecked": False,
            "period_shifts_m": shifts,
        }
    print(
        publish(
            OUTPUT,
            {
                "complete": True,
                "passed": True,
                "control_qualified": False,
                "owner_ref": owner_ref,
                "source_ref": prefix.reference(__file__),
                "corrections": corrections,
                "native_models": 0,
                "physics_steps": 0,
                "world_model_calls": 0,
                "fresh_cases": 0,
            },
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
