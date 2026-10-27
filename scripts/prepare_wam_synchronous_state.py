"""Prepare054 current-only matrix and frozen worker sources, without a model."""

import argparse
import copy
import json
from pathlib import Path

from prepare_wam_synchronous_validation import source_closure

from s2a.artifacts import REPO, sha256
from s2a.clean_training_runtime import checked, publish, read, reference
from s2a.synchronous_state_sampling import BUDGET, candidate_head, paired_entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "outputs/synchronous_state_probe052_20261024")
    args = parser.parse_args()
    root = args.output.resolve()
    if root != REPO / "outputs/synchronous_state_probe052_20261024" or root.exists():
        raise ValueError("054 requires its fresh fixed namespace")
    old_path = (
        REPO
        / "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/preparation/prepared.json"
    )
    old = read(reference(old_path))
    inputs_ref = reference(REPO / "outputs/synchronous_validation_20261024/current_inputs.json")
    current = read(inputs_ref)
    if not current["passed"] or not current["complete"] or len(current["current_inputs"]) != 24:
        raise ValueError("Complete053 current-only manifest required")
    fit_ref = reference(REPO / "outputs/clean_regularization_probe_20261024/result.json")
    qr_ref = reference(REPO / "outputs/clean_regularization_probe_20261024/independent_verified.json")
    _, params_ref = candidate_head(fit_ref, qr_ref)
    qualification_ref = old["prerequisite_refs"]["bridge_qualification"]
    qualification = read(qualification_ref)
    template = old["validation_generations"][0]["request"]
    entries = paired_entries(current["current_inputs"], template, old["zero_identity_cases"], qualification)
    seed_ref = reference(old_path.parents[1] / "execution/seed_plan.json")
    seed_plan = read(seed_ref)
    noise = copy.deepcopy(old["noise_roster"])
    noise["training_seeds"] = [e["sampling_seed"] for e in seed_plan["entries"]]
    source_entries = [
        "scripts/prepare_wam_synchronous_state.py",
        "scripts/verify_wam_synchronous_state_metadata.py",
        "scripts/sample_wam_synchronous_state.py",
        "src/s2a/synchronous_state_sampling.py",
        "tests/test_synchronous_state_sampling.py",
        "plans/v0.0.1/054_synchronous_state_probe052_20261024.md",
    ]
    sources = dict(old["qualified_model_source_sha256"])
    new_sources = source_closure(source_entries)
    if any(path in sources and sources[path] != digest for path, digest in new_sources.items()):
        raise ValueError("New closure conflicts with qualified model source")
    sources.update(new_sources)
    for path, digest in sources.items():
        checked({"path": path, "sha256": digest})
    root.mkdir()
    snapshots = {}
    (root / "source").mkdir()
    for index, (name, digest) in enumerate(sorted(sources.items())):
        target = root / "source" / f"{index:04d}_{Path(name).name}"
        with target.open("xb") as stream:
            stream.write(Path(name).read_bytes())
        if sha256(target) != digest:
            raise ValueError("054 source changed while snapshotting")
        snapshots[str(target)] = digest
    refs = [
        inputs_ref,
        current["source_audit_ref"],
        fit_ref,
        qr_ref,
        params_ref,
        seed_ref,
        old["prerequisite_refs"]["bridge_prepared"],
        qualification_ref,
    ]
    refs += [e["image_ref"] for e in entries]
    input_hashes = {ref["path"]: ref["sha256"] for ref in refs}
    for ref in refs:
        checked(ref)
    prepared = {
        "schema_version": "s2a.synchronous-state-preparation.v1",
        "root": str(root),
        "current_inputs_ref": inputs_ref,
        "current_inputs": current["current_inputs"],
        "candidate_fit_ref": fit_ref,
        "candidate_verification_ref": qr_ref,
        "candidate_params_ref": params_ref,
        "prerequisite_refs": {k: old["prerequisite_refs"][k] for k in ["bridge_prepared", "bridge_qualification"]},
        "request_template": template,
        "zero_cases": old["zero_identity_cases"],
        "sampling_entries": entries,
        "model_config": old["model_config"],
        "noise_roster": noise,
        "training_seed_plan_ref": seed_ref,
        "weight_files": old["weight_files"],
        "budget": BUDGET,
        "source_sha256": sources,
        "snapshot_sha256": snapshots,
        "input_sha256": input_hashes,
        "execution_authorized": False,
        "fresh_authorized": False,
        "automatic_retry": False,
    }
    print(json.dumps(publish(root / "prepared.json", prepared)), flush=True)


if __name__ == "__main__":
    main()
