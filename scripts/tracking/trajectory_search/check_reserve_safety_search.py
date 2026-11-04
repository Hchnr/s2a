"""Registered safety-only feedback arm of the shared full native search audit."""

from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO
from scripts.tracking.trajectory_search.check_reserve_entry_search import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            "safety_feedback",
            config_path=REPO / "configs/tracking/trajectory_search/reserve_safety_search_20261102.json",
            plan_path=REPO / "plans/v0.0.1/tracking/trajectory_search/reserve_safety_search_20261102.md",
            qa_path=REPO / "outputs/qa/reserve_safety_search_20261102/result.json",
            output_root=REPO / "outputs/reserve_safety_search_20261102",
            aligned_floor=True,
            safety_feedback=True,
        )
    )
