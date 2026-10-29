"""Compare ordered and prefetched audits on real raw/terminal full35 records."""

import copy
import json
import time
from pathlib import Path

import numpy as np
from verify_wam_clean_training_capture import Evidence
from verify_wam_terminal_development_points import audit_points, head_parameters
from verify_wam_terminal_points_prefetch import audit_points as prefetched

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, reference, require


def main():
    root = REPO / "outputs/tracking_preview_wam_development_v4_20261027"
    ev = Evidence()
    prepared = ev.json(reference(root / "prepared.json"))
    run = ev.json(reference(root / "development/orchestration/result.json"))
    service = ev.json(run["service_ref"])
    head = (head_parameters(ev.json(prepared["baseline_params_ref"])), head_parameters(ev.json(run["readout_ref"])))
    records = []
    for index in (0, 75):
        request = service["requests"][index]
        ledger = ev.json(request["compact_ledger_ref"])
        header = ev.json(ledger["request_ref"])
        steps = ev.json(request["trace_ref"])["steps"]
        directory = Path(request["request_ref"]["path"]).parent
        results, maps, durations = [], [], []
        for method in (audit_points, prefetched):
            local = Evidence()
            start = time.monotonic()
            result = method(local, ledger, header, directory, steps, head=head, purpose="evaluation")
            durations.append(time.monotonic() - start)
            local.finish()
            results.append(result)
            maps.append(local.inputs)
        first, second = results
        require(first[1:] == second[1:] and first[1] == 35, "Identical full count/mutations")
        require(first[0].keys() == second[0].keys(), "Identical first-point evidence")
        for key in first[0]:
            if isinstance(first[0][key], np.ndarray):
                require(np.array_equal(first[0][key], second[0][key]), "Identical first sample")
            else:
                require(first[0][key] == second[0][key], "Identical first metadata")
        require(maps[0] == maps[1] and len(maps[0]) == 70, "Same all35 JSON/archive fingerprints")
        for mutation in ("sha", "stream"):
            bad = copy.deepcopy(ledger)
            if mutation == "sha":
                bad["points"][17]["sha256"] = "0" * 64
            else:
                bad["source_stream_sha256"] = "0" * 64
            try:
                prefetched(Evidence(), bad, header, directory, steps, head=head, purpose="evaluation")
            except ValueError:
                pass
            else:
                raise ValueError("Prefetch accepted corrupted " + mutation)
        ev.mapping(maps[0])
        records.append(
            dict(
                request_index=index,
                full_points=35,
                consumed_files=70,
                seconds=durations,
                corrupted_sha_rejected=True,
                corrupted_stream_rejected=True,
            )
        )
    ev.finish()
    result = dict(
        complete=True,
        passed=True,
        records=records,
        source_and_artifact_sha256=ev.inputs,
        source_refs=[
            reference(Path(__file__)),
            reference(REPO / "scripts/verify_wam_terminal_points_prefetch.py"),
            reference(REPO / "scripts/verify_wam_terminal_development_points.py"),
        ],
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        scope="Two real full35 requests only; no full development qualification",
    )
    print(json.dumps(publish(REPO / "outputs/terminal_points_prefetch_check_20261028/verified.json", result)))


if __name__ == "__main__":
    main()
