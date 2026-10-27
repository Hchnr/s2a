"""Independent054 request/source admission; no model, simulator or validation futures."""

import argparse
import ast
import json
from pathlib import Path

from sample_wam_clean_training_state import historical_source_summaries

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.synchronous_state_sampling import BUDGET, candidate_head, paired_entries


def worker_body_equivalent():
    """Prove the copied sampler math differs only at declared identity/admission seams."""
    original = (REPO / "scripts/sample_wam_clean_training_state.py").read_text()
    expected = original[original.index("def run(") : original.index("\ndef main():")]
    expected = expected.replace("w2_fit_ref", "candidate_fit_ref").replace(
        "w2_verification_ref", "candidate_verification_ref"
    )
    expected = expected.replace("admit_execution(", "admit(").replace('"w2"', '"probe052"')
    expected = expected.replace(
        'REPO / "src/s2a/clean_training_runtime.py",',
        'REPO / "src/s2a/clean_training_runtime.py",\n        REPO / "src/s2a/synchronous_state_sampling.py",',
    )
    expected = expected.replace(
        '"state sampler must use the single038 registered UUID"',
        '"054 sampler must use its single registered allowed UUID"',
    )
    current = ast.parse((REPO / "scripts/sample_wam_synchronous_state.py").read_text())
    actual = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    require(ast.dump(actual) == ast.dump(ast.parse(expected).body[0]), "054 worker math changed beyond declared seams")
    return {
        "passed": True,
        "method": "complete run AST comparison after enumerated admission/identity renames",
        "original_ref": reference(REPO / "scripts/sample_wam_clean_training_state.py"),
        "current_ref": reference(REPO / "scripts/sample_wam_synchronous_state.py"),
    }


def numerical_body_equivalent():
    original = (REPO / "scripts/verify_wam_clean_training_state.py").read_text()
    expected = original[original.index("def verify_generations(") : original.index("\ndef verify(args):")]
    expected = expected.replace('prepared["validation_anchors"]', 'prepared["current_inputs"]')
    expected = expected.replace('prepared["zero_identity_cases"]', 'prepared["zero_cases"]')
    expected = expected.replace("roster = planned_generations(prepared)", 'roster = prepared["sampling_entries"]')
    expected = expected.replace('"w2"', '"probe052"')
    current = ast.parse((REPO / "scripts/verify_wam_synchronous_state.py").read_text())
    actual = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == "verify_generations")
    require(
        ast.dump(actual) == ast.dump(ast.parse(expected).body[0]),
        "054 complete numerical audit changed beyond identity seams",
    )
    return {
        "passed": True,
        "original_ref": reference(REPO / "scripts/verify_wam_clean_training_state.py"),
        "current_ref": reference(REPO / "scripts/verify_wam_synchronous_state.py"),
    }


def verify(pref):
    prepared = read(pref)
    require(
        prepared["schema_version"] == "s2a.synchronous-state-preparation.v1" and prepared["budget"] == BUDGET,
        "054 schema/budget differs",
    )
    require(
        prepared["execution_authorized"] is prepared["fresh_authorized"] is prepared["automatic_retry"] is False,
        "054 preparation cannot authorize execution",
    )
    for mapping in [prepared["source_sha256"], prepared["snapshot_sha256"], prepared["input_sha256"]]:
        for path, digest in mapping.items():
            checked({"path": path, "sha256": digest})
    require(
        all(Path(path).suffix in (".json", ".png") for path in prepared["input_sha256"]),
        "054 model inputs include an unregistered binary or future archive",
    )
    current = read(prepared["current_inputs_ref"])
    audit = read(current["source_audit_ref"])
    require(
        audit["passed"] is audit["complete"] is audit["eligible_for_model_validation"] is True
        and audit["native_stable_cases"] == 6
        and audit["current_model_inputs"] == current["current_inputs"],
        "053 physical/current-input qualification changed",
    )
    require(current["current_inputs"] == prepared["current_inputs"], "054 current-only inputs changed")
    _, params = candidate_head(prepared["candidate_fit_ref"], prepared["candidate_verification_ref"])
    require(params == prepared["candidate_params_ref"], "054 params binding differs")
    qualification = read(prepared["prerequisite_refs"]["bridge_qualification"])
    expected = paired_entries(
        current["current_inputs"], prepared["request_template"], prepared["zero_cases"], qualification
    )
    require(expected == prepared["sampling_entries"], "054 paired matrix differs")
    for i in range(3, 99, 2):
        raw, probe = expected[i : i + 2]
        a, b = dict(raw["request"]), dict(probe["request"])
        a.pop("name")
        b.pop("name")
        require(a == b and probe["paired_raw_generation_index"] == raw["index"], "054 RNG/input pair differs")
    zero_nodes = 0
    for entry in expected[:3]:
        require(
            len(historical_source_summaries(qualification["generations"][entry["qualified_generation_index"]])) == 35,
            "Historical model capture roster changed",
        )
        zero_nodes += 35
    body = worker_body_equivalent()
    numeric_body = numerical_body_equivalent()
    seed_plan = read(prepared["training_seed_plan_ref"])
    require(
        prepared["noise_roster"]["training_seeds"] == [e["sampling_seed"] for e in seed_plan["entries"]]
        and prepared["noise_roster"]["evaluation_seeds"] == [42, 43],
        "054 actual noise roster changed",
    )
    for path, digest in prepared["source_sha256"].items():
        checked({"path": path, "sha256": digest})
    return {
        "schema_version": "s2a.synchronous-state-metadata-verification.v1",
        "complete": True,
        "passed": True,
        "prepared_ref": pref,
        "verifier_ref": reference(Path(__file__)),
        "worker_body_proof": body,
        "numerical_body_proof": numeric_body,
        "paired_requests_verified": 48,
        "zero_capture_nodes_available": zero_nodes,
        "source_entries": len(prepared["source_sha256"]),
        "snapshot_entries": len(prepared["snapshot_sha256"]),
        "new_model_initializations": 0,
        "new_requests": 0,
        "validation_future_archive_reads": 0,
        "execution_authorized": False,
        "fresh_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(reference(args.prepared))
    root = args.prepared.resolve().parent
    print(json.dumps(publish(args.output or root / "metadata_verified.json", result)), flush=True)


if __name__ == "__main__":
    main()
