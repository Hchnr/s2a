"""Explicit registered aligned-floor arm of the shared finite search protocol."""

from s2a.research.trajectory_search.restored_anchor_sqp_evidence import REPO
from scripts.tracking.trajectory_search.check_reserve_entry_search import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            "aligned",
            config_path=REPO / "configs/tracking/trajectory_search/aligned_reserve_search_20261102.json",
            plan_path=REPO / "plans/v0.0.1/tracking/trajectory_search/aligned_reserve_search_20261102.md",
            qa_path=REPO / "outputs/qa/aligned_reserve_search_20261102/result.json",
            output_root=REPO / "outputs/aligned_reserve_search_20261102",
            aligned_floor=True,
        )
    )
