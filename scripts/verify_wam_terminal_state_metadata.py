"""Independent terminal058 request/source qualification without future labels."""

import argparse
import ast
import copy
import json
from pathlib import Path

from sample_wam_clean_training_state import historical_source_summaries
from verify_wam_clean_training_capture import Evidence, require

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, reference
from s2a.synchronous_state_sampling import paired_entries as baseline_entries
from s2a.terminal_candidate import candidate_head
from s2a.terminal_state_execution import ROOT, select_readout


def body_proofs():
    original = (REPO / "scripts/sample_wam_synchronous_state.py").read_text()
    expected = original[original.index("def run(") : original.index("\ndef main():")]
    expected = expected.replace("candidate_fit_ref, candidate_verification_ref", "candidate_proof_ref")
    expected = expected.replace(
        "        candidate_fit_ref=candidate_fit_ref,\n        candidate_verification_ref=candidate_verification_ref,",
        "        candidate_proof_ref=candidate_proof_ref,",
    )
    expected = expected.replace('"src/s2a/synchronous_state_sampling.py"', '"src/s2a/terminal_state_execution.py"')
    expected = expected.replace(
        'dict(raw=None, zero=zero, w2=head)[entry["variant"]]',
        'select_readout(entry["variant"], zero=zero, candidate=head)',
    )
    expected = expected.replace("probe052", "terminal058").replace('"054 sampler', '"056 aggregate sampler')
    actual = ast.parse((REPO / "scripts/sample_wam_terminal_state.py").read_text())
    worker = next(n for n in actual.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    require(
        ast.dump(worker) == ast.dump(ast.parse(expected).body[0]),
        "Terminal058 complete worker body changed beyond declared seams",
    )
    original = (REPO / "scripts/verify_wam_synchronous_state.py").read_text()
    expected = original[original.index("def verify_generations(") : original.index("\ndef verify(")]
    expected = expected.replace("probe052", "terminal058").replace('"054 ', '"Terminal058 ')
    actual = ast.parse((REPO / "scripts/verify_wam_terminal_state.py").read_text())
    numeric = next(n for n in actual.body if isinstance(n, ast.FunctionDef) and n.name == "verify_generations")
    require(ast.dump(numeric) == ast.dump(ast.parse(expected).body[0]), "Terminal058 complete numerical body changed")

    class Rename(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(node.value.replace("probe052", "terminal058")), node)
            return node

    def functions(path):
        return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}

    baseline = functions(REPO / "src/s2a/synchronous_state_metrics.py")
    metrics = functions(REPO / "src/s2a/terminal_state_metrics.py")
    for name in ("pool", "state_gate", "diagnosis"):
        require(
            ast.dump(Rename().visit(baseline[name])) == ast.dump(metrics[name]),
            "Terminal058 original state metric/gate changed",
        )
    return dict(
        passed=True,
        worker="complete run AST",
        numerical="complete verify_generations AST",
        metrics="complete pool/state_gate/diagnosis AST; candidate identity only",
    )


def verify(prepared_ref):
    e = Evidence()
    p = e.json(prepared_ref)
    budget = dict(
        requests=99,
        denoise_calls=3465,
        head_boundary_calls=3465,
        model_initializations=1,
        fits=0,
        physics_steps=0,
        fresh_cases=0,
    )
    require(
        p["schema_version"] == "s2a.terminal-state-preparation.v1"
        and Path(p["root"]) == ROOT
        and p["budget"] == budget,
        "Terminal058 fixed preparation required",
    )
    require(
        p["execution_authorized"] is p["fresh_authorized"] is p["automatic_retry"] is False,
        "Terminal058 preparation cannot launch a model",
    )
    baseline = e.json(p["baseline_prepared_ref"])
    require(
        Path(p["baseline_prepared_ref"]["path"])
        == REPO / "outputs/synchronous_state_probe052_continuation_20261024/prepared.json",
        "Terminal058 baseline identity differs",
    )
    for key in (
        "model_config",
        "prerequisite_refs",
        "weight_files",
        "current_inputs_ref",
        "current_inputs",
        "request_template",
        "zero_cases",
        "training_seed_plan_ref",
        "noise_roster",
    ):
        require(p[key] == baseline[key], "Terminal058 fixed baseline field changed: " + key)
    require(
        all(p["source_sha256"].get(k) == v for k, v in baseline["source_sha256"].items()),
        "Terminal058 changed a frozen prerequisite source",
    )
    for mapping in (p["source_sha256"], p["snapshot_sha256"], p["input_sha256"]):
        require(bool(mapping), "Terminal058 empty source/input closure")
        for path, digest in mapping.items():
            e.path(dict(path=path, sha256=digest))
    require(
        all(Path(path).suffix in (".json", ".png") for path in p["input_sha256"]),
        "Terminal058 online input map includes noncurrent binary archives",
    )
    _, params_ref = candidate_head(p["candidate_proof_ref"], read_json=e.json)
    require(params_ref == p["candidate_params_ref"], "Terminal058 candidate parameters differ")
    current = e.json(p["current_inputs_ref"])
    audit = e.json(current["source_audit_ref"])
    require(
        current["passed"]
        is current["complete"]
        is audit["passed"]
        is audit["complete"]
        is audit["eligible_for_model_validation"]
        is True
        and audit["native_stable_cases"] == 6
        and current["current_inputs"] == p["current_inputs"],
        "Terminal058 original synchronized six native controls required",
    )
    qualification = e.json(p["prerequisite_refs"]["bridge_qualification"])
    require(
        qualification["passed"] is qualification["complete"] is qualification["bridge_qualified"] is True,
        "Terminal058 original bridge not qualified",
    )
    # Independently use the frozen original matrix implementation, with only
    # the declared candidate identifier changed. Do not call the new builder.
    expected = baseline_entries(current["current_inputs"], p["request_template"], p["zero_cases"], qualification)
    for entry in expected:
        if entry["variant"] == "probe052":
            entry["variant"] = "terminal058"
            entry["name"] = entry["name"].replace("_probe052_s", "_terminal058_s")
            entry["request"]["name"] = entry["name"]
    require(expected == p["sampling_entries"], "Terminal058 independent complete99 matrix differs")
    head, zero = object(), object()
    worker_tree = ast.parse((REPO / "scripts/sample_wam_terminal_state.py").read_text())
    dispatch = next(
        n.value
        for n in ast.walk(worker_tree)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "readout" for t in n.targets)
    )
    expression = compile(ast.Expression(dispatch), "aggregate-actual-dispatch", "eval")
    selected = [
        eval(expression, {"entry": row, "select_readout": select_readout, "head": head, "zero": zero})
        for row in expected
    ]
    require(
        sum(v is head for v in selected) == sum(v is None for v in selected) == 48
        and sum(v is zero for v in selected) == 3,
        "Terminal058 actual dispatch differs",
    )
    for index in range(3, 99, 2):
        raw, candidate = expected[index : index + 2]
        a, b = copy.deepcopy(raw["request"]), copy.deepcopy(candidate["request"])
        a.pop("name")
        b.pop("name")
        require(
            a == b and candidate["paired_raw_generation_index"] == raw["index"], "Terminal058 input/RNG pair differs"
        )
    zero_nodes = sum(
        len(historical_source_summaries(qualification["generations"][row["qualified_generation_index"]]))
        for row in expected[:3]
    )
    require(zero_nodes == 105, "Terminal058 all historical zero nodes required")
    seed_plan = e.json(p["training_seed_plan_ref"])
    require(
        p["noise_roster"]
        == dict(training_seeds=[row["sampling_seed"] for row in seed_plan["entries"]], evaluation_seeds=[42, 43]),
        "Terminal058 training/evaluation noise roster differs",
    )
    require(len(p["weight_files"]) == 163, "Terminal058 complete checkpoint roster required")
    for path, expected_stat in p["weight_files"].items():
        stat = Path(path).stat()
        require(
            stat.st_size == expected_stat["bytes"] and stat.st_mtime_ns == expected_stat["mtime_ns"],
            "Terminal058 checkpoint changed",
        )
    proofs = body_proofs()
    verifier_ref = reference(Path(__file__))
    require(
        p["source_sha256"].get(verifier_ref["path"]) == verifier_ref["sha256"],
        "Terminal058 metadata verifier not frozen",
    )
    e.finish()
    return dict(
        schema_version="s2a.terminal-state-metadata.v1",
        complete=True,
        passed=True,
        prepared_ref=prepared_ref,
        verifier_ref=verifier_ref,
        requests_verified=99,
        current_only=True,
        paired_requests_verified=48,
        zero_nodes_available=105,
        body_proofs=proofs,
        input_sha256=e.inputs,
        execution_authorized=False,
        fresh_authorized=False,
        validation_future_archive_reads=0,
        new_model_initializations=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            publish(args.prepared.resolve().parent / "metadata_verified.json", verify(reference(args.prepared)))
        ),
        flush=True,
    )
