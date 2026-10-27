"""Current-only paired requests and explicit052 head admission for protocol054."""

import copy
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, require, validate_current_request
from s2a.wam_clean_readout import CleanResidualReadout

GPU = "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f"
BUDGET = {
    "requests": 99,
    "denoise_calls": 3465,
    "head_boundary_calls": 3465,
    "model_initializations": 1,
    "fits": 0,
    "physics_steps": 0,
    "fresh_cases": 0,
}
CURRENT_KEYS = {
    "name",
    "task",
    "anchor_frame",
    "image_ref",
    "current_time_s",
    "current_sim_state_sha256",
    "measured_state128",
    "state_mask128",
    "initial_sync_certified",
}


def paired_entries(current_inputs, template, zero_cases, qualification):
    """Return exactly3 historical zero regressions then48 complete raw/probe pairs."""
    require(
        len(current_inputs) == 24 and len({a["name"] for a in current_inputs}) == 24,
        "All24 unique current anchors required",
    )
    require(len(zero_cases) == 3, "All3 exact historical zero regressions required")
    entries = []
    gold = [(i, g) for i, g in enumerate(qualification["generations"]) if g["variant"] == "raw"]
    require(len(gold) == 3 and [i for i, _ in gold] == [0, 1, 2], "Original3 qualified raw cases required")
    for case, (index, original) in zip(zero_cases, gold, strict=True):
        require(
            case["name"] == original["case_id"] and original["historical_output_exact"] is True, "Zero identity differs"
        )
        request = copy.deepcopy(case["request"])
        validate_current_request(
            request,
            name=request["name"],
            seed=42,
            image_ref=case["image_ref"],
            task=request["prompt"],
            template=case["request"],
        )
        entries.append(
            {
                "index": len(entries),
                "name": "zero_" + case["name"],
                "variant": "zero",
                "validation_index": None,
                "anchor_name": None,
                "zero_case_name": case["name"],
                "pair_id": case["name"],
                "sampling_seed": 42,
                "image_ref": case["image_ref"],
                "request": request,
                "qualified_generation_index": index,
                "expected_response_ref": case["expected_response_ref"],
                "qualified_response_ref": original["response_ref"],
            }
        )
    for anchor_index, anchor in enumerate(current_inputs):
        require(
            set(anchor) == CURRENT_KEYS and anchor["initial_sync_certified"] is True,
            "Current-only synchronized schema differs",
        )
        state, mask = np.asarray(anchor["measured_state128"]), np.asarray(anchor["state_mask128"])
        require(
            state.shape == (128,)
            and np.isfinite(state).all()
            and mask.dtype == np.bool_
            and np.array_equal(mask, np.arange(128) < 10)
            and np.isfinite(anchor["current_time_s"]),
            "Current measured state/mask differs",
        )
        for seed in (42, 43):
            raw_index = len(entries)
            for variant in ("raw", "probe052"):
                name = f"{anchor['name']}_{variant}_s{seed}"
                request = copy.deepcopy(template)
                request.update(name=name, seed=seed, vision_path=anchor["image_ref"]["path"], prompt=anchor["task"])
                request["extra"]["action128"]["task_title"] = anchor["task"]
                validate_current_request(
                    request, name=name, seed=seed, image_ref=anchor["image_ref"], task=anchor["task"], template=template
                )
                entries.append(
                    {
                        "index": len(entries),
                        "validation_index": len(entries) - 3,
                        "anchor_index": anchor_index,
                        "name": name,
                        "variant": variant,
                        "anchor_name": anchor["name"],
                        "zero_case_name": None,
                        "pair_id": f"{anchor['name']}_s{seed}",
                        "sampling_seed": seed,
                        "image_ref": anchor["image_ref"],
                        "request": request,
                        "paired_raw_generation_index": raw_index if variant == "probe052" else None,
                    }
                )
    require(len(entries) == len({e["name"] for e in entries}) == 99, "054 generation identity collision")
    return entries


def candidate_head(producer_ref, verified_ref):
    """Admit new052 parameters under their own identity; never impersonate old W2."""
    produced, verified = read(producer_ref), read(verified_ref)
    require(
        produced["schema_version"] == "s2a.clean-regularization-probe.v1"
        and verified["schema_version"] == "s2a.clean-regularization-probe-qr.v1"
        and produced["passed"] is produced["complete"] is verified["passed"] is verified["complete"] is True,
        "Complete052 production and independent QR required",
    )
    params_ref = produced["params_ref"]
    require(
        verified["producer_ref"] == producer_ref
        and verified["params_ref"] == params_ref
        and produced["producer_fits"] == verified["independent_reconstructions"] == 1,
        "052 producer/audit parameter identity differs",
    )
    require(
        verified["rtol"] == 1e-9
        and verified["atol"] == 1e-11
        and set(verified["comparisons"]) == {"coefficients", "center", "scale"}
        and all(v["passed"] is True for v in verified["comparisons"].values()),
        "052 QR tolerance differs",
    )
    require(produced["fit_metadata"]["rows"] == verified["qr_metadata"]["rows"] == 840000, "Full052 fit required")
    a, b = produced["fit_metadata"]["replay_sha256"], verified["qr_metadata"]["replay_sha256"]
    require(
        (a["features"], a["target_clean_residual"], a["weights"]) == (b["features"], b["targets"], b["weights"]),
        "052 replay identities differ",
    )
    params = read(params_ref)
    require(params["ridge"] == 0.1 / 1290 and len(params["center"]) == 1290, "054 fixed052 head differs")
    return CleanResidualReadout.from_parameters(params), params_ref


def admit(prepared_ref, registration_ref, *, required_sources):
    prepared, registered = read(prepared_ref), read(registration_ref)
    require(
        prepared["schema_version"] == "s2a.synchronous-state-preparation.v1"
        and registered["schema_version"] == "s2a.synchronous-state-registration.v1"
        and registered["prepared_ref"] == prepared_ref
        and registered["passed"] is True
        and registered["execution_authorized"] is True
        and registered["automatic_retry"] is False,
        "Independent054 execution registration required",
    )
    root = Path(prepared["root"]).resolve()
    require(
        root == REPO / "outputs/synchronous_state_probe052_20261024" and prepared["budget"] == BUDGET,
        "054 output/budget differs",
    )
    metadata = read(registered["metadata_verification_ref"])
    require(
        metadata["passed"] is True
        and metadata["prepared_ref"] == prepared_ref
        and metadata["execution_authorized"] is False,
        "Independent054 metadata envelope differs",
    )
    for field, filename in [
        ("numerical_verifier_ref", "verify_wam_synchronous_state.py"),
        ("scorer_ref", "score_wam_synchronous_state.py"),
        ("owner_ref", "run_wam_synchronous_state.py"),
    ]:
        ref = registered[field]
        require(
            Path(ref["path"]).resolve() == REPO / "scripts" / filename
            and prepared["source_sha256"].get(ref["path"]) == ref["sha256"],
            "054 final audit/owner source is not bound",
        )
        checked(ref)
    require(registered["budget"] == BUDGET, "054 execution budget differs")
    sources = prepared["source_sha256"]
    require(
        sources.get(metadata["verifier_ref"]["path"]) == metadata["verifier_ref"]["sha256"],
        "054 metadata verifier is not frozen",
    )
    for mapping in [sources, prepared["snapshot_sha256"], prepared["input_sha256"]]:
        for path, sha in mapping.items():
            checked({"path": path, "sha256": sha})
    require(
        all(str(Path(p).resolve()) in sources for p in required_sources), "054 actual worker is outside source closure"
    )
    current = read(prepared["current_inputs_ref"])
    require(
        current["passed"] is current["complete"] is True
        and current["future_archive_refs_exported"] == 0
        and current["expert_commands_exported"] == current["task_predicates_exported"] == 0,
        "Qualified current-only manifest required",
    )
    require(current["current_inputs"] == prepared["current_inputs"], "054 current inputs changed")
    candidate_head(prepared["candidate_fit_ref"], prepared["candidate_verification_ref"])
    qualification = read(prepared["prerequisite_refs"]["bridge_qualification"])
    entries = paired_entries(
        prepared["current_inputs"], prepared["request_template"], prepared["zero_cases"], qualification
    )
    require(entries == prepared["sampling_entries"], "054 complete request matrix changed")
    return prepared, registered, sources


def admit_sampling(prepared, prepared_ref, registration_ref, candidate_fit_ref, candidate_verification_ref):
    require(
        candidate_fit_ref == prepared["candidate_fit_ref"]
        and candidate_verification_ref == prepared["candidate_verification_ref"],
        "054 candidate arguments differ",
    )
    head, params = candidate_head(candidate_fit_ref, candidate_verification_ref)
    qualification_ref = prepared["prerequisite_refs"]["bridge_qualification"]
    qualification = read(qualification_ref)
    require(
        qualification["passed"] is qualification["complete"] is qualification["bridge_qualified"] is True,
        "Original bridge not qualified",
    )
    return head, params, qualification_ref, qualification, prepared["sampling_entries"]
