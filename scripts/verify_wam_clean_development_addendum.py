"""Post-run read-only replay with explicit, proven template relocation."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

from s2a.clean_training_seed_plan import apply_seed_plan
from s2a.development_audit_relocation import audit_tree, equivalent_template
from s2a.development_repair_admission import ROOT, admit
from s2a.development_runtime_bridge import extend_sources
from s2a.durable_stage_supervisor import check_sources, identity, read, reference, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "", "CPU only")
    registration_ref = reference(ROOT / "execution/admitted.json")
    registration = admit(registration_ref)
    check_sources(registration["source_sha256"])
    check_sources(registration["snapshot_sha256"])
    migration = read(registration["relocation_ref"])
    template = equivalent_template(migration)
    auditor = ROOT / "source_runtime_py311/verify_wam_clean_training_development.py"
    auditor_ref = reference(auditor)
    require(registration["source_sha256"][str(auditor)] == auditor_ref["sha256"], "original auditor changed")
    tree = audit_tree(auditor.read_text(), template, seed_overlay=True)
    output = args.output.absolute()
    require(output.is_relative_to(ROOT / "addendum") and not output.exists(), "new addendum output required")
    proof = dict(
        registration_ref=registration_ref,
        auditor_ref=auditor_ref,
        relocation_ref=registration["relocation_ref"],
        actual_entrypoint_ref=reference(Path(__file__).resolve()),
        helper_ref=reference(Path(__file__).resolve().parents[1] / "src/s2a/development_audit_relocation.py"),
        transformed_ast_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        identity=identity(os.getpid()),
        original_failure_preserved=True,
        new_model_calls=0,
        new_physics_steps=0,
        fresh_performance_authorized=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    proof_path = output.with_suffix(".provenance.json")
    with proof_path.open("x") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    if not args.run:
        print(json.dumps(dict(dry_run=True, provenance_ref=reference(proof_path))))
        return
    namespace = dict(__name__="s2a_development_addendum_delegate", __file__=str(auditor))
    exec(compile(tree, str(auditor), "exec"), namespace)
    # Explicit post-run source closure; do not impersonate the consumed audit role receipt.
    namespace["current_sources"] = lambda original: extend_sources(original, registration["source_sha256"])
    prepared = read(registration["prepared_ref"])
    run_ref = reference(Path(prepared["root"]) / "development_dependency_repair_20261024/orchestration/result.json")
    failed = Path(prepared["root"]) / "review/DEVELOPMENT_dependency_repair_20261024_verified.json"
    require(read(reference(failed))["passed"] is False, "original failed audit missing")
    call = argparse.Namespace(
        seed=42,
        output=output,
        prepared=registration["prepared_ref"]["path"],
        prepared_sha256=registration["prepared_ref"]["sha256"],
        execution_registration=registration["execution_registration_ref"]["path"],
        execution_registration_sha256=registration["execution_registration_ref"]["sha256"],
        development=run_ref["path"],
        development_sha256=run_ref["sha256"],
    )

    def registered_seed_plan(prepared, reg, ev, refs):
        seed_ref = reg["training_seed_plan_ref"]
        return apply_seed_plan(prepared, refs["prepared"], ev.json(seed_ref), seed_ref)

    namespace["apply_registered_seed_plan"] = registered_seed_plan
    namespace["verify"](call)
    check_sources(registration["source_sha256"])
    check_sources(registration["snapshot_sha256"])
    print(json.dumps(dict(result_ref=reference(output), provenance_ref=reference(proof_path))))


if __name__ == "__main__":
    main()
