"""Require complete correct-reference and actual-dispatch evidence before WAM."""

from s2a.clean_training_runtime import checked, read, require


def _closed_stage(ref):
    stage = read(ref)
    require(stage["complete"] is stage["passed"] is True, "Successful owned stage")
    exit, release = read(stage["exit_ref"]), read(stage["release_ref"])
    require(
        exit["actual_wait"] is release["reaped"] is True and exit["returncode"] == release["returncode"] == 0,
        "Actual wait0/reap",
    )


def admit_tracking_control(evidence):
    require(
        set(evidence) == {"qualification_ref", "closure_owner_ref", "transaction_owner_ref"},
        "Exact tracking admission references",
    )
    qualification = read(evidence["qualification_ref"])
    owner = read(evidence["closure_owner_ref"])
    require(
        owner["complete"] is owner["passed"] is True and owner["verified_ref"] == evidence["qualification_ref"],
        "Complete qualification owner binding",
    )
    _closed_stage(owner["stage_ref"])
    require(
        qualification["complete"] is qualification["passed"] is qualification["performance_qualified"] is True,
        "Correct-reference performance qualification",
    )
    require(
        qualification["qualified_cases"] == 18
        and qualification["candidate_physics_steps"] == qualification["replay_physics_steps"] == 4907,
        "Original18 budgets",
    )
    cases = qualification["cases"]
    require(len(cases) == 18 and {c["case_index"] for c in cases} == set(range(18)), "Exact original18 cases")
    for case in cases:
        v = read(case["verified_ref"])
        require(v["complete"] is v["passed"] is v["performance_qualified"] is True, "Independent case qualification")
        require(
            v["outcome"]["stable"] and v["position_rmse_m"] <= 0.01 and v["so3_rmse_rad"] <= 0.05,
            "Original case thresholds",
        )
        for kind in ("robot_inputs", "labels_images"):
            a = read(case["audit_refs"][kind])
            require(
                a["complete"] is a["passed"] is True and a["result_ref"] == v["result_ref"],
                "Same actual candidate audit",
            )
            if kind == "labels_images":
                require(
                    a["replay_pixels_compared"] is True and a["replay_ref"] == case["verified_ref"],
                    "Actual native camera replay",
                )
    transaction = read(evidence["transaction_owner_ref"])
    require(transaction["complete"] is transaction["passed"] is True, "Actual optimization dispatch qualification")
    for stage in transaction["stages"]:
        _closed_stage(stage["stage_ref"])
    parity = read(transaction["parity_ref"])
    require(
        parity["complete"] is parity["passed"] is True and parity["active_periods"] > 0,
        "Optimization route actually exercised",
    )
    sources = dict(qualification["source_sha256"])
    for path, digest in read(transaction["registration_ref"])["source_sha256"].items():
        require(path not in sources or sources[path] == digest, "Consistent source versions")
        sources[path] = digest
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    return dict(
        evidence=evidence,
        source_sha256=sources,
        scope="Correct-reference and dispatch admission only; no WAM or fresh success implied",
    )
