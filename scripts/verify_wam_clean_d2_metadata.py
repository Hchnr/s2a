"""Independent056 JSON/current-image admission; no model or future decoding."""

import argparse
import ast
import copy
import json
from pathlib import Path

from verify_wam_clean_training_capture import Evidence, current_request, require

from s2a.artifacts import REPO


def body_proof():
    def tail(path):
        tree = ast.parse(path.read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        index = next(
            i
            for i, n in enumerate(run.body)
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "bridge" for t in n.targets)
        )
        return ast.Module(body=run.body[index:], type_ignores=[])

    class Changes(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Name(id="admit", ctx=node.ctx), node) if node.id == "admit_execution" else node

        def visit_IfExp(self, node):
            if (
                ast.dump(node.test) == ast.dump(ast.parse('round_name == "D0"', mode="eval").body)
                and isinstance(node.body, ast.Constant)
                and node.body.value == "raw"
                and isinstance(node.orelse, ast.Constant)
                and node.orelse.value == "w1"
            ):
                return ast.copy_location(ast.Constant("probe052"), node)
            return self.generic_visit(node)

    original = tail(REPO / "scripts/collect_wam_clean_training.py")
    actual = tail(REPO / "scripts/collect_wam_clean_d2.py")
    require(
        ast.dump(Changes().visit(original)) == ast.dump(actual), "056 capture body differs beyond registered changes"
    )
    return {
        "passed": True,
        "scope": "Complete capture body after bridge initialization; only admission and initial policy label changed",
    }


def verify(prepared_ref, output):
    evidence = Evidence()
    p = evidence.json(prepared_ref)
    root = REPO / "outputs/clean_on_policy_aggregation_20261024"
    require(
        p["schema_version"] == "s2a.clean-d2-execution-preparation.v1" and Path(p["root"]) == root,
        "056 preparation identity",
    )
    require(
        p["budget"]
        == dict(
            model_initializations=1,
            requests=300,
            denoise_calls=10500,
            head_calls=10500,
            future_label_reads=0,
            fits=0,
            physics_steps=0,
            fresh_cases=0,
        )
        and p["gpu_uuid"] == "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f",
        "056 fixed budget/UUID",
    )
    for mapping in (p["source_sha256"], p["snapshot_sha256"], p["input_sha256"]):
        require(bool(mapping), "056 empty closure")
        for path, digest in mapping.items():
            evidence.path(dict(path=path, sha256=digest))
    m = evidence.json(p["current_manifest_ref"])
    require(
        m["schema_version"] == "s2a.clean-d2-current-manifest.v1"
        and m["execution_authorized"] is False
        and m["budget"] == p["budget"],
        "056 current-only proposal required",
    )
    require(len(m["anchors"]) == 150 and len(m["entries"]) == 300, "056 full roster")
    require(all(set(a) == {"name", "task", "image_ref"} for a in m["anchors"]), "056 privileged anchor field")
    anchors = {a["name"]: a for a in m["anchors"]}
    require(len(anchors) == 150, "056 duplicate anchor")
    d0 = evidence.json(m["d0_collection_ref"])
    original = evidence.json(d0["prepared_ref"])
    require(
        m["anchors"]
        == [{k: copy.deepcopy(a[k]) for k in ("name", "task", "image_ref")} for a in original["training_anchors"]],
        "056 anchor identity drift",
    )
    seed_plan = evidence.json(m["training_seed_plan_ref"])
    audit, owner = evidence.json(p["d0_capture_audit_ref"]), evidence.json(p["d0_owner_ref"])
    require(
        all(v["complete"] is v["passed"] is True for v in (d0, audit, owner))
        and audit["collection_ref"] == owner["producer_ref"] == m["d0_collection_ref"]
        and owner["verification_ref"] == p["d0_capture_audit_ref"],
        "056 original D0 qualification",
    )
    require(
        d0["round_name"] == audit["round_name"] == owner["stage"] == "D0"
        and audit["actual_denoise_calls"] == 10500
        and len(d0["requests"]) == len(seed_plan["entries"]) == 300,
        "056 original D0 denominator",
    )
    seeds, names = [], set()
    for index, (entry, saved, seed) in enumerate(zip(m["entries"], d0["requests"], seed_plan["entries"], strict=True)):
        require(
            set(entry)
            == {"index", "name", "anchor_name", "sampling_seed", "request", "d0_initial_ref", "d0_request_ref"},
            "056 entry fields",
        )
        require(
            saved["complete"] is True
            and all(entry[k] == saved[k] == seed[k] for k in ("index", "name", "anchor_name", "sampling_seed"))
            and entry["index"] == index
            and entry["anchor_name"] == m["anchors"][index // 2]["name"]
            and entry["name"] not in names,
            "056 original order/pair/seed mismatch",
        )
        require(
            entry["d0_initial_ref"] == saved["initial_ref"]
            and entry["d0_request_ref"] == saved["request_ref"]
            and evidence.json(saved["request_ref"]) == entry["request"],
            "056 original request refs",
        )
        current_request(entry["request"], entry, anchors[entry["anchor_name"]])
        evidence.path(anchors[entry["anchor_name"]]["image_ref"])
        initial = evidence.json(saved["initial_ref"])
        evidence.path(initial["arrays"])
        require(
            initial["sampling_seed"] == entry["sampling_seed"] and initial["inference_only_current_image"] is True,
            "056 initial input provenance",
        )
        seeds.append(entry["sampling_seed"])
        names.add(entry["name"])
    require(
        len(set(seeds)) == 300
        and all(type(s) is int and 0 <= s < 2**31 and s not in (42, 43) for s in seeds)
        and m["noise_roster"] == dict(training_seeds=seeds, evaluation_seeds=[42, 43]),
        "056 noise split",
    )
    fit, qr = evidence.json(m["candidate_fit_ref"]), evidence.json(m["candidate_verification_ref"])
    require(
        all(v["complete"] is v["passed"] is True for v in (fit, qr))
        and qr["producer_ref"] == m["candidate_fit_ref"]
        and fit["params_ref"] == qr["params_ref"] == m["candidate_params_ref"]
        and m["candidate_params_ref"]["sha256"] == "ca01b9d8a49f2f1585c917f492e52a0a89cdb0b447f6bd2beee050ed0d2fccc0",
        "056 fixed052 full qualification",
    )
    require(
        qr["rtol"] == 1e-9
        and qr["atol"] == 1e-11
        and fit["fit_metadata"]["rows"] == qr["qr_metadata"]["rows"] == 840000
        and all(v["passed"] for v in qr["comparisons"].values()),
        "056 fixed QR qualification",
    )
    evidence.json(m["candidate_params_ref"])
    require(len(p["weight_files"]) == 163, "056 complete checkpoint roster")
    for path, value in p["weight_files"].items():
        stat = Path(path).stat()
        require(stat.st_size == value["bytes"] and stat.st_mtime_ns == value["mtime_ns"], "056 checkpoint changed")
    proof = body_proof()
    verifier_ref = dict(
        path=str(Path(__file__).resolve()), sha256=p["source_sha256"].get(str(Path(__file__).resolve()))
    )
    evidence.path(verifier_ref)
    evidence.finish()
    result = dict(
        schema_version="s2a.clean-d2-execution-metadata.v1",
        complete=True,
        passed=True,
        execution_authorized=False,
        prepared_ref=prepared_ref,
        verifier_ref=verifier_ref,
        requests_verified=300,
        current_only=True,
        body_proof=proof,
        input_sha256=evidence.inputs,
        future_archives_decoded=0,
        model_initializations=0,
    )
    require(Path(output).resolve().is_relative_to(root), "056 metadata output escaped root")
    with Path(output).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(dict(path=str(args.prepared.resolve()), sha256=args.prepared_sha256), args.output)
