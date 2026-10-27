"""Gate056 aggregate supervision on completed capture and real lifecycle receipts.

This module reads JSON only. Label decoding belongs strictly after admission.
"""

from s2a.clean_training_runtime import read, require

ROWS = 1260000
RIDGE = 0.1 / 1290


def normal_exit(entry, read_json):
    launch, exit_record, release = (read_json(entry[k]) for k in ("launch", "exit", "release"))
    require(
        exit_record == dict(returncode=0, actual_wait=True, automatic_retry=False),
        "056 aggregation requires actual wait0",
    )
    require(
        release["reaped"] is True
        and release["returncode"] == 0
        and release["signal_sent"] is False
        and release["original_error"] is None
        and release["only_owned_child_signalled"] is True,
        "056 aggregation requires normal reap/no-signal",
    )
    require(
        all(launch["child"][k] == release[k] for k in ("pid", "start_ticks")),
        "056 aggregation process identity differs",
    )


def admit_aggregation(spec, *, read_json=read):
    require(
        spec["schema_version"] == "s2a.clean-aggregation-fit-registration.v1"
        and spec["rows"] == ROWS
        and spec["ridge"] == RIDGE
        and spec["producer_fits"] == 1
        and spec["automatic_retry"] is False,
        "056 fixed aggregation policy differs",
    )
    owner = read_json(spec["d2_owner_ref"])
    require(
        owner["schema_version"] == "s2a.clean-d2-owner.v1" and owner["complete"] is owner["passed"] is True,
        "056 full D2 owner required before labels",
    )
    require(
        len(owner["owned_commands"]) == 2 and owner["automatic_retry"] is owner["fresh_authorized"] is False,
        "056 exact model/auditor lifecycle required",
    )
    for entry in owner["owned_commands"]:
        normal_exit(entry, read_json)
    gpu = read_json(owner["gpu_after_ref"])
    require(
        gpu["released"] is True
        and gpu["uuid"] == "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f"
        and gpu["memory_mib"] == 0
        and gpu["compute_apps"] == []
        and gpu["device_minor"] == 6
        and gpu["visible_gpu"] == 2,
        "056 real GPU release required before labels",
    )
    d2 = read_json(owner["collection_ref"])
    audit = read_json(owner["numerical_ref"])
    require(
        d2["schema_version"] == "s2a.clean-d2-capture.v1"
        and audit["schema_version"] == "s2a.clean-d2-capture-audit.v1"
        and all(v["complete"] is v["passed"] is True for v in (d2, audit))
        and audit["collection_ref"] == owner["collection_ref"],
        "056 independent D2 qualification required",
    )
    require(
        audit["actual_requests"] == d2["actual_requests"] == 300
        and audit["actual_denoise_calls"] == d2["actual_denoise_calls"] == d2["actual_head_calls"] == 10500
        and audit["initial_pairs_verified"] == 300,
        "056 complete D2 numerical denominator",
    )
    require(
        audit["prepared_ref"] == d2["prepared_ref"] == owner["prepared_ref"]
        and audit["execution_registration_ref"] == d2["execution_registration_ref"] == owner["registration_ref"]
        and audit["readout_ref"] == d2["readout_ref"],
        "056 D2 producer/audit/owner bindings differ",
    )
    require(
        d2["readout_ref"]["sha256"] == "ca01b9d8a49f2f1585c917f492e52a0a89cdb0b447f6bd2beee050ed0d2fccc0"
        and all(d2[k] == 0 for k in ("future_label_reads", "producer_fits", "physics_steps", "fresh_cases")),
        "056 fixed label-free collection policy required",
    )
    for field, role in (
        ("worker_launch_ref", "launch"),
        ("worker_exit_ref", "exit"),
        ("worker_release_ref", "release"),
    ):
        require(audit[field] == owner["owned_commands"][0][role], "056 numerical/lifecycle receipts differ")
    require(audit["gpu_after_ref"] == owner["gpu_after_ref"], "056 numerical GPU receipt differs")
    previous = read_json(spec["previous_fit_ref"])
    verified = read_json(spec["previous_verification_ref"])
    require(
        all(v["complete"] is v["passed"] is True for v in (previous, verified))
        and previous["actual_rows"] == 840000
        and verified["fit_ref"] == spec["previous_fit_ref"]
        and verified["params_ref"] == previous["params_ref"],
        "056 original D0/D1 fit qualification required",
    )
    require(len(previous["collection_refs"]) == 2, "056 original two rounds required")
    collection_refs = previous["collection_refs"] + [owner["collection_ref"]]
    collections = [read_json(ref) for ref in collection_refs]
    require([c["round_name"] for c in collections] == ["D0", "D1", "D2"], "056 fixed round order required")
    for collection in collections:
        require(
            collection["complete"] is collection["passed"] is True
            and len(collection["requests"]) == 300
            and collection["normalization_ref"] == collections[0]["normalization_ref"],
            "056 full compatible collection required",
        )
    for entries in zip(*(c["requests"] for c in collections), strict=True):
        require(
            all(
                e["complete"] is True
                and all(e[k] == entries[0][k] for k in ("index", "name", "anchor_name", "sampling_seed"))
                for e in entries
            ),
            "056 training identities differ across rounds",
        )
    prepared = read_json(previous["prepared_ref"])
    require(
        len(prepared["training_anchors"]) == len({a["name"] for a in prepared["training_anchors"]}) == 150,
        "056 full training labels roster required",
    )
    return previous, prepared, collection_refs, collections
