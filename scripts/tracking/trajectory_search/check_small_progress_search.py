"""Explicit two-field progress-floor variant of the finite native search audit."""

from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO
from scripts.tracking.trajectory_search.check_reserve_entry_search import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            "small_progress",
            config_path=REPO / "configs/tracking/trajectory_search/small_progress_search_20261102.json",
            plan_path=REPO / "plans/v0.0.1/tracking/trajectory_search/small_progress_search_20261102.md",
            qa_path=REPO / "outputs/qa/small_progress_search_20261102/result.json",
            output_root=REPO / "outputs/small_progress_search_20261102",
            aligned_floor=True,
            safety_feedback=True,
            progress_floor=1e-5,
        )
    )
