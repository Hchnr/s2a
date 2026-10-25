"""056 current-only D2 proposal; cannot initialize or authorize a model."""

import copy
import json

from s2a.artifacts import REPO
from s2a.clean_training_d2 import BUDGET, SCHEMA, validate_roster
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.synchronous_state_sampling import candidate_head

ROOT = REPO / "outputs/clean_on_policy_aggregation_20261024"


def main():
    require(not ROOT.exists(), "056 proposal directory already exists; no overwrite")
    probe_root = REPO / "outputs/clean_regularization_probe_20261024"
    fit_ref = reference(probe_root / "result.json")
    verified_ref = reference(probe_root / "independent_verified.json")
    _, params_ref = candidate_head(fit_ref, verified_ref)
    probe = read(fit_ref)
    old_fit = read(probe["previous_fit_ref"])
    prepared = read(old_fit["prepared_ref"])
    d0_ref = old_fit["collection_refs"][0]
    d0 = read(d0_ref)
    require(
        d0["round_name"] == "D0" and d0["passed"] is d0["complete"] is True and d0["actual_requests"] == 300,
        "Original complete D0 required",
    )
    seed_plan = read(d0["training_seed_plan_ref"])
    anchors = [{k: copy.deepcopy(a[k]) for k in ("name", "task", "image_ref")} for a in prepared["training_anchors"]]
    templates = {e["request"]["prompt"]: copy.deepcopy(e["request"]) for e in prepared["zero_identity_cases"]}
    entries = []
    inputs = [
        fit_ref,
        verified_ref,
        params_ref,
        probe["previous_fit_ref"],
        old_fit["prepared_ref"],
        d0_ref,
        d0["training_seed_plan_ref"],
    ]
    require(len(d0["requests"]) == len(seed_plan["entries"]) == 300, "Full D0/seed plan required")
    for original, seed in zip(d0["requests"], seed_plan["entries"], strict=True):
        require(
            original["complete"] is True
            and all(original[k] == seed[k] for k in ("index", "name", "anchor_name", "sampling_seed")),
            "Actual D0 request/seed mismatch",
        )
        request = read(original["request_ref"])
        require(
            request["seed"] == original["sampling_seed"] and request["name"] == original["name"],
            "D0 request content differs",
        )
        initial = read(original["initial_ref"])
        require(
            initial["sampling_seed"] == original["sampling_seed"] and initial["inference_only_current_image"] is True,
            "D0 current initial identity differs",
        )
        checked(initial["arrays"])
        inputs.extend([original["request_ref"], original["initial_ref"], initial["arrays"]])
        entries.append(
            dict(
                index=original["index"],
                name=original["name"],
                anchor_name=original["anchor_name"],
                sampling_seed=original["sampling_seed"],
                request=request,
                d0_initial_ref=original["initial_ref"],
                d0_request_ref=original["request_ref"],
            )
        )
    value = dict(
        schema_version=SCHEMA,
        anchors=anchors,
        entries=entries,
        noise_roster=dict(training_seeds=[e["sampling_seed"] for e in entries], evaluation_seeds=[42, 43]),
        request_templates=templates,
        candidate_fit_ref=fit_ref,
        candidate_verification_ref=verified_ref,
        candidate_params_ref=params_ref,
        d0_collection_ref=d0_ref,
        training_seed_plan_ref=d0["training_seed_plan_ref"],
        budget=BUDGET,
        execution_authorized=False,
    )
    summary = validate_roster(value)
    inputs.extend(a["image_ref"] for a in anchors)
    for ref in inputs:
        checked(ref)
    ROOT.mkdir()
    manifest_ref = publish(ROOT / "current_manifest.json", value)
    proof = dict(
        complete=True,
        passed=True,
        execution_authorized=False,
        manifest_ref=manifest_ref,
        summary=summary,
        inputs_sha256={ref["path"]: ref["sha256"] for ref in inputs},
        source_refs=[
            reference(REPO / p)
            for p in (
                "scripts/prepare_wam_clean_d2.py",
                "src/s2a/clean_training_d2.py",
                "src/s2a/clean_training_runtime.py",
                "src/s2a/synchronous_state_sampling.py",
                "plans/v0.0.1/056_clean_on_policy_aggregation_20261024.md",
            )
        ],
        future_trajectory_archives_decoded=0,
        model_initializations=0,
        new_fits=0,
        physics_steps=0,
        fresh_cases=0,
    )
    publish(ROOT / "proposal_verified_by_preparer.json", proof)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
