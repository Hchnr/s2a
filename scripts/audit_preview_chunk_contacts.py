"""Inspect saved contact geometry at all long-branch endpoints without stepping."""

import json
from collections import Counter

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_execution import CONTROL_REF


def main():
    strict_native_environment()
    root = REPO / "outputs/preview_chunk_contacts_20261027"
    root.mkdir(exist_ok=False)
    tracking_ref = reference(REPO / "outputs/preview_chunk_tracking_restored_20261027/verified.json")
    tracking = read(tracking_ref)
    require(tracking["passed"] is True, "Tracking verification required")
    paired = read(tracking["verified_ref"])
    prepared = read(paired["prepared_ref"])
    guard = read(CONTROL_REF)["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from s2a.libero_simulation import make_environment
    from s2a.native_training import state_sha256

    cases = []
    for i, case in enumerate(prepared["cases"]):
        manifest = read(case["manifest_ref"])
        rows_ref = reference(REPO / f"outputs/preview_chunk_branches_20261027/case{i:02d}_h40/steps.json")
        rows = read(rows_ref)
        env = None
        endpoints, count = [], Counter()
        try:
            env, _ = make_environment(manifest["runtime_config"], root / f"case{i:02d}", manifest["record"], 0)
            for j, row in enumerate(rows):
                state = np.asarray(row["sim_state"])
                require(state_sha256(state) == row["state_sha256"], "State SHA")
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                np.testing.assert_allclose(env.sim.get_state().flatten(), state, atol=1e-14, rtol=0)
                contacts = []
                for k in range(env.sim.data.ncon):
                    contact = env.sim.data.contact[k]
                    names = [
                        env.sim.model.geom_id2name(int(g)) or f"unnamed_geom_{g}"
                        for g in (contact.geom1, contact.geom2)
                    ]
                    pair = tuple(sorted(names))
                    count[pair] += 1
                    contacts.append(dict(geoms=names, distance_m=float(contact.dist)))
                endpoints.append(dict(index=j, contacts=contacts))
            cases.append(
                dict(
                    case=case["case"],
                    rows_ref=rows_ref,
                    endpoints=endpoints,
                    pair_contact_counts=[dict(geoms=list(pair), contacts=n) for pair, n in count.items()],
                )
            )
        finally:
            if env is not None:
                env.close()
        print(json.dumps(dict(completed_case=i)), flush=True)
    print(
        json.dumps(
            publish(
                root / "verified.json",
                dict(
                    complete=True,
                    passed=True,
                    tracking_ref=tracking_ref,
                    source_ref=reference(__file__),
                    cases=cases,
                    physics_steps=0,
                    model_calls=0,
                    fresh_cases=0,
                    scope="recomputed endpoint contact geometry only; no historical force or between-endpoint contact claim",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
