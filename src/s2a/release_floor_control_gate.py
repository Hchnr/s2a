"""Bind release-floor WAM admission to its own full control and physical evidence."""

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require
from s2a.dependency_safe_control_gate import admit_dependency_safe_control


def admit_release_floor_control(evidence):
    require(
        set(evidence) == {"guarded_evidence", "release_probe_owner_ref"}, "Exact release policy admission references"
    )
    admission = admit_dependency_safe_control(evidence["guarded_evidence"])
    tracking = evidence["guarded_evidence"]["tracking_evidence"]
    qualification = read(tracking["qualification_ref"])
    require(qualification.get("release_reference_minimum") == 0.35, "Explicit release-floor qualification required")
    clarification = read(qualification["metadata_clarification_ref"])
    require(
        clarification["registration_ref"] == read(qualification["owner_ref"])["registration_ref"],
        "Release metadata describes actual qualified implementation",
    )
    sources = dict(admission["source_sha256"])
    for module in ("gripper_release_floor", "loaded_velocity_release_floor"):
        ref = reference(REPO / f"src/s2a/{module}.py")
        require(
            sources.get(ref["path"]) == ref["sha256"], "Release implementation actually included in full qualification"
        )
    owner = read(evidence["release_probe_owner_ref"])
    require(owner["complete"] is owner["passed"] is True, "Closed independent release replay owner")
    owned = owner["owned_command"]
    exit_record, release = read(owned["exit"]), read(owned["release"])
    require(
        exit_record["actual_wait"] is release["reaped"] is True
        and exit_record["returncode"] == release["returncode"] == 0,
        "Actual release replay wait0/reap",
    )
    require(
        read(owned["launch"])["command"][-1] == "scripts/check_gripper_release_floor_pair.py",
        "Actual independent600 replay command",
    )
    replay = read(owner["result_ref"])
    require(
        replay["complete"] is replay["passed"] is True and replay["physics_steps"] == 600, "Independent full600 replay"
    )
    require(
        replay["maximum_state_difference"] <= 1e-8 and replay["changed_gripper_commands"] > 0,
        "Actual intervention and independently reproduced native states",
    )
    producer_ref = read(replay["registration_ref"])["producer_ref"]
    producer = read(producer_ref)
    require(
        producer["complete"] is producer["passed"] is True and producer["physics_steps"] == 1200,
        "Full paired600 plus600 experiment",
    )
    candidate = next(branch for branch in producer["branches"] if branch["branch"] == "candidate")
    require(
        candidate["release_reference_minimum"] == 0.35 and candidate["actual_command_changed"] is True,
        "Actual registered release policy dispatched",
    )
    refs = [
        evidence["release_probe_owner_ref"],
        owner["result_ref"],
        owner["registration_ref"],
        *owned.values(),
        producer_ref,
        replay["registration_ref"],
        producer["registration_ref"],
        qualification["metadata_clarification_ref"],
    ]
    refs.extend(candidate[k] for k in ("states_ref", "steps_ref"))
    for path, digest in read(owner["registration_ref"])["source_sha256"].items():
        require(
            path not in sources or sources[path] == digest, "Same qualified and physically replayed source versions"
        )
        sources[path] = digest
    for ref in refs:
        checked(ref)
        require(
            ref["path"] not in sources or sources[ref["path"]] == ref["sha256"], "Consistent release evidence binding"
        )
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    return dict(
        evidence=evidence,
        source_sha256=sources,
        scope="Own full control qualification and independent release replay only; no WAM or fresh task success implied",
    )
