"""055 current-only metadata, unchanged matrix and executable dispatch admission."""

import argparse
import ast
import json
from pathlib import Path

from audit_wam_synchronous_state_prefix import verify as verify_prefix

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.synchronous_state_continuation import BUDGET, ROOT, checkpoint_stats, dispatch_matrix, prefix_evidence
from s2a.synchronous_state_dispatch import select_readout
from s2a.synchronous_state_sampling import candidate_head, paired_entries


def worker_expected_text(original):
    text = original[original.index("def run(") : original.index("\ndef main():")]
    text = text.replace(
        'REPO / "src/s2a/synchronous_state_sampling.py",',
        'REPO / "src/s2a/synchronous_state_sampling.py",\n        REPO / "src/s2a/synchronous_state_continuation.py",',
    )
    text = text.replace("for entry in entries[:3]:", 'for entry in [e for e in entries if e["variant"] == "zero"]:')
    text = text.replace(
        'result["parameters_before"] = parameters_identity(model)',
        'result["parameters_before"] = parameters_identity(model)\n        require(result["parameters_before"]["count"] == prepared["prefix_parameter_identity"]["count"], "055 model parameter count differs from audited prefix")',
    )
    text = text.replace("BoundaryCounter(model, 3465)", "BoundaryCounter(model, 3325)")
    text = text.replace("raw_by_pair = {}", "raw_by_pair = prefix_raw_records(prepared)")
    text = text.replace(
        'dict(raw=None, zero=zero, w2=head)[entry["variant"]]',
        'select_readout(entry["variant"], zero=zero, probe=head)',
    )
    return (
        text.replace("len(counter.calls) == 3465", "len(counter.calls) == 3325")
        .replace("bridge.total_head_calls == 3465", "bridge.total_head_calls == 3325")
        .replace('result["actual_requests"] == 99', 'result["actual_requests"] == 95')
        .replace("state99/3465 budget is incomplete", "continuation95/3325 budget is incomplete")
    )


def numerical_expected_text(original):
    text = original[original.index("def verify_generations(") : original.index("\ndef verify(refs, output):")]
    text = text.replace('roster = prepared["sampling_entries"]', 'roster = prepared["sampling_entries"][4:]')
    text = text.replace(
        'require(len(actual["generations"]) == 99, "all99 state outputs required")',
        'require(len(actual["generations"]) == 95, "all95 continuation outputs required")',
    )
    text = text.replace(
        "generated, pairs, changed = [], {}, 0",
        'prefix, failed = prefix_evidence(prepared)\n    prefix_raw = failed["generations"][3]\n    generated, pairs, changed = [], {(prefix_raw["anchor_name"], prefix_raw["sampling_seed"]): prefix_raw["initial_ref"]}, 0',
    )
    text = text.replace(
        "raw_directory = root / f\"{entry['index'] - 1:03d}_{roster[entry['index'] - 1]['name']}\"",
        'raw_directory = Path(pairs[key]["path"]).parent',
    )
    return text.replace("len(generated) == 96", "len(generated) == 95").replace(
        '"full48 identical-noise/current-condition pairs"', '"48 pairs including the cross-process prefix raw"'
    )


def body_proofs():
    proof = {}
    for label, oldname, newname, fn, transform in [
        (
            "worker",
            "sample_wam_synchronous_state.py",
            "sample_wam_synchronous_state_continuation.py",
            "run",
            worker_expected_text,
        ),
        (
            "numerical",
            "verify_wam_synchronous_state.py",
            "verify_wam_synchronous_state_continuation.py",
            "verify_generations",
            numerical_expected_text,
        ),
    ]:
        old = (REPO / "scripts" / oldname).read_text()
        new = ast.parse((REPO / "scripts" / newname).read_text())
        actual = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == fn)
        expected = ast.parse(transform(old)).body[0]
        require(ast.dump(actual) == ast.dump(expected), f"055 {label} changed beyond explicit suffix/dispatch seams")
        proof[label] = {
            "passed": True,
            "old_ref": reference(REPO / "scripts" / oldname),
            "new_ref": reference(REPO / "scripts" / newname),
        }
    return proof


def exercise_worker_dispatch(path, entries):
    tree = ast.parse(Path(path).read_text())
    assignments = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute)
            and isinstance(t.value, ast.Name)
            and t.value.id == "bridge"
            and t.attr == "readout"
            for t in n.targets
        )
    ]
    require(len(assignments) == 1, "055 must have one explicit actual readout assignment")
    expression = compile(ast.Expression(assignments[0].value), str(path), "eval")
    zero, head = object(), object()
    actual = []
    for entry in entries:
        actual.append(
            eval(
                expression,
                {"__builtins__": {"dict": dict}, "select_readout": select_readout},
                {"entry": entry, "zero": zero, "head": head},
            )
        )
    expected = dispatch_matrix(entries, zero=zero, probe=head)
    require(
        all(a is b for a, b in zip(actual, expected, strict=True)),
        "055 actual worker dispatch is not exact-object selection",
    )
    return {
        "requests": len(entries),
        "probe": sum(x is head for x in actual),
        "raw": sum(x is None for x in actual),
        "zero": sum(x is zero for x in actual),
    }


def verify(pref):
    p = read(pref)
    require(
        p["schema_version"] == "s2a.synchronous-state-continuation-preparation.v1"
        and Path(p["root"]) == ROOT
        and p["budget"] == BUDGET
        and p["execution_authorized"] is False,
        "055 preparation scope differs",
    )
    for mapping in [p["source_sha256"], p["snapshot_sha256"], p["input_sha256"]]:
        for path, digest in mapping.items():
            checked({"path": path, "sha256": digest})
    checkpoint_stats(p)
    prefix, failed = prefix_evidence(p)
    rechecked = verify_prefix(prefix["prepared_ref"], prefix["registration_ref"], ROOT / "prefix_rechecked.json")
    require(read(rechecked) == prefix, "055 fresh CPU prefix reconstruction differs from retained audit")
    old = read(prefix["prepared_ref"])
    require(
        p["current_inputs_ref"] == old["current_inputs_ref"]
        and p["candidate_params_ref"] == old["candidate_params_ref"],
        "055 altered current samples or trained parameters",
    )
    candidate_head(p["candidate_fit_ref"], p["candidate_verification_ref"])
    entries = paired_entries(
        p["current_inputs"],
        p["request_template"],
        p["zero_cases"],
        read(p["prerequisite_refs"]["bridge_qualification"]),
    )
    require(
        entries == old["sampling_entries"] == p["sampling_entries"] and p["remaining_indices"] == list(range(4, 99)),
        "055 original full99 matrix changed",
    )
    proofs = body_proofs()
    worker = REPO / "scripts/sample_wam_synchronous_state_continuation.py"
    full = exercise_worker_dispatch(worker, entries)
    suffix = exercise_worker_dispatch(worker, entries[4:])
    require(
        full == {"requests": 99, "probe": 48, "raw": 48, "zero": 3}
        and suffix == {"requests": 95, "probe": 48, "raw": 47, "zero": 0},
        "055 dispatch budget differs",
    )
    raw_initial = read(failed["generations"][3]["initial_ref"])
    allowed_binary = raw_initial["arrays"]["path"]
    require(
        all(Path(path).suffix in (".json", ".png") or path == allowed_binary for path in p["input_sha256"]),
        "055 model input allowlist contains a future archive",
    )
    return {
        "schema_version": "s2a.synchronous-state-continuation-metadata.v1",
        "passed": True,
        "complete": True,
        "prepared_ref": pref,
        "verifier_ref": reference(Path(__file__)),
        "body_proofs": proofs,
        "full_dispatch": full,
        "suffix_dispatch": suffix,
        "remaining_requests": 95,
        "prefix_verified_ref": p["prefix_verified_ref"],
        "prefix_rechecked_ref": rechecked,
        "source_entries": len(p["source_sha256"]),
        "snapshots": len(p["snapshot_sha256"]),
        "execution_authorized": False,
        "new_model_calls": 0,
        "validation_future_archive_reads": 0,
        "fresh_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            publish(args.prepared.resolve().parent / "metadata_verified.json", verify(reference(args.prepared)))
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
