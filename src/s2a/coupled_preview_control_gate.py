"""Require the complete fixed preview physical qualification before WAM use."""

from pathlib import Path
from types import SimpleNamespace

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, require
from s2a.owned_receipt_contract import normal_receipts

RUN = REPO / "outputs/coupled_preview_control_20261026"
OWNER = REPO / "outputs/coupled_preview_continuation_20261026"
REPAIR = REPO / "outputs/coupled_preview_verification_repair_20261026"
CONFIG = REPO / "configs/coupled_preview_control_registered_20261026.json"


def admit_preview_control(closure_ref):
    """Read-only admission; a prefix, performance failure, or live owner cannot pass.

    Recheck the actual closure's complete evidence roster, including the original
    failure and checker-only repair. This never retries or collects a trajectory.
    New WAM execution still needs its own source/service/UUID registration.
    """
    require(Path(closure_ref["path"]).resolve() == RUN / "closure_verified.json", "Fixed preview closure required")
    closure = read(closure_ref)
    require(
        all(
            closure.get(k) is True
            for k in ("complete", "owner_normal_wait_verified", "mechanical_qualification", "performance_qualified")
        )
        and closure["verified_cases"] == 18
        and closure["physics_steps"] == 4907,
        "Complete18/4907 performance-qualified preview closure required",
    )
    sources = closure["source_sha256"]
    require(bool(sources), "Complete physical evidence roster required")
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))

    def bound_json(path):
        path = str(Path(path).resolve())
        require(path in sources, "Required evidence absent from physical closure")
        return read(dict(path=path, sha256=sources[path]))

    evidence = SimpleNamespace(json=bound_json)

    def completed(directory):
        child = normal_receipts(
            evidence, directory / "launch.json", directory / "exit.json", directory / "release.json", format="durable"
        )["child"]
        proc = Path(f"/proc/{child['pid']}/stat")
        require(
            not proc.exists() or int(proc.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"],
            "Physical child still exists with its original identity",
        )

    require(Path(closure["summary_ref"]["path"]).resolve() == RUN / "regression_summary.json", "Fixed summary required")
    summary = read(closure["summary_ref"])
    require(summary == bound_json(RUN / "regression_summary.json"), "Summary/closure binding differs")
    config = bound_json(CONFIG)
    require(
        summary["complete"]
        is summary["performance_qualified"]
        is summary["execution_and_independent_checks_passed"]
        is True
        and summary["candidate_stable"]
        == summary["candidate_trials"]
        == len(summary["cases"])
        == len(config["profiles"])
        == 18
        and summary["physics_steps"] == 4907,
        "Complete fixed18 performance summary required",
    )
    for i, (case, profile) in enumerate(zip(summary["cases"], config["profiles"], strict=True)):
        proof = bound_json(RUN / f"case{i:02d}/independent_verified.json")
        control = proof["control"]
        tracking = control["tracking"]["admitted"]
        require(
            proof["complete"] is proof["mechanical_labels_images_passed"] is proof["performance_qualified"] is True
            and case["case"] == i
            and case["record"] == profile["record"]["name"]
            and case["physics_steps"] == control["steps"] == profile["physics_steps"]
            and case["outcome"] == control["outcome"]
            and case["tracking"] == tracking
            and control["outcome"]["stable"] is True
            and tracking["position_rmse_m"] <= 0.01
            and tracking["so3_rmse_rad"] <= 0.05,
            "A physical case fails the original performance gate",
        )
        completed(RUN / "orchestration" / f"case{i:02d}")
        completed(REPAIR / "case00" if i == 0 else RUN / "verification" / f"case{i:02d}")
    completed(OWNER / "owner_wait")
    repair = bound_json(REPAIR / "verified.json")
    require(
        repair["complete"] is repair["reuses_original_collection"] is repair["actual_owned_wait_recorded"] is True,
        "Checker-only repair evidence required",
    )
    return dict(
        closure_ref=closure_ref,
        summary_ref=closure["summary_ref"],
        control_config_ref=dict(path=str(CONFIG), sha256=sources[str(CONFIG)]),
        physical_control_qualified=True,
        wam_execution_authorized=False,
        fresh_authorized=False,
    )
