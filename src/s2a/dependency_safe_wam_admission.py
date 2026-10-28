"""Bind the separate WAM version to completed guarded control qualification."""

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.dependency_safe_control_gate import admit_dependency_safe_control


def admit_tracking_control(evidence):
    proof_ref = reference(REPO / "outputs/dependency_safe_wam_admission_20261027/verified.json")
    proof = read(proof_ref)
    require(proof["complete"] is proof["passed"] is True, "Completed guarded WAM admission")
    expected = proof["admission"]["evidence"]
    require(evidence == expected["tracking_evidence"], "Guarded qualification references required")
    admission = admit_dependency_safe_control(expected)
    require(admission == proof["admission"], "Guarded evidence/source closure unchanged")
    sources = dict(admission["source_sha256"])
    refs = [proof_ref, proof["source_ref"], expected["recovery_owner_ref"]]
    owner = read(expected["recovery_owner_ref"])
    refs.extend([owner["result_ref"], owner["registration_ref"], *owner["owned_command"].values()])
    refs.append(read(owner["result_ref"])["dispatch_ref"])
    for ref in refs:
        checked(ref)
        require(ref["path"] not in sources or sources[ref["path"]] == ref["sha256"], "Consistent recovery lineage")
        sources[ref["path"]] = ref["sha256"]
    return dict(
        evidence=evidence,
        source_sha256=sources,
        scope="Guarded control and recovered actual dispatch admission; no WAM or fresh task success",
    )
