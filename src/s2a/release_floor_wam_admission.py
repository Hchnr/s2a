"""Bind the new WAM package to its own qualified release policy and actual replay."""

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.release_floor_control_gate import admit_release_floor_control


def admit_tracking_control(evidence):
    proof_ref = reference(REPO / "outputs/release_floor_wam_admission_20261028/verified.json")
    proof = read(proof_ref)
    require(proof["complete"] is proof["passed"] is True, "Completed release policy admission")
    expected = proof["admission"]["evidence"]
    require(evidence == expected["guarded_evidence"]["tracking_evidence"], "Own release policy qualification required")
    admission = admit_release_floor_control(expected)
    require(admission == proof["admission"], "Frozen release policy admission changed")
    sources = dict(admission["source_sha256"])
    decisions_ref = reference(REPO / "outputs/gripper_release_floor_decisions_fast_20261028/verified.json")
    decisions = read(decisions_ref)
    require(
        decisions["complete"] is decisions["passed"] is True and decisions["periods"] == 600,
        "Independent complete release decisions",
    )
    refs = [proof_ref, proof["source_ref"], decisions_ref, decisions["registration_ref"]]
    refs += read(decisions["registration_ref"])["source_refs"]
    build_ref = reference(REPO / "outputs/velocity_scalar_audit_native_20261028/build.json")
    build = read(build_ref)
    refs += [
        build_ref,
        build["source_ref"],
        build["binary_ref"],
        reference(REPO / "scripts/velocity_plan_compiled_audit.py"),
    ]
    for ref in refs:
        checked(ref)
        require(
            ref["path"] not in sources or sources[ref["path"]] == ref["sha256"],
            "Consistent release runtime/review source versions",
        )
        sources[ref["path"]] = ref["sha256"]
    return dict(
        evidence=evidence,
        source_sha256=sources,
        scope="Own release-floor control/decision admission only; no WAM or fresh success",
    )
