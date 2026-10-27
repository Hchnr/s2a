"""051 full-case input hash preflight using only two generated metadata functions."""

import ast
import copy
import json
import sys
from pathlib import Path

from s2a.clean_training_state_handoff import GPU
from s2a.durable_stage_supervisor import REPO, publish, read, reference, require
from s2a.stability_position_simulation import execution_inputs

ROOT = REPO / "outputs/clean_training_development_dependency_repair_20261024"


def main():
    require(sys.version_info[:2] == (3, 11), "input preflight requires actual Python3.11 runtime")
    generation_ref = reference(ROOT / "source_generation_py311.json")
    generation = read(generation_ref)
    for item in generation["generated"]:
        require(reference(item["generated_ref"]["path"]) == item["generated_ref"], "generated source changed")
    source_ref = generation["generated"][0]["generated_ref"]
    tree = ast.parse(Path(source_ref["path"]).read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("make_case_config", "case_order")]
    require(len(selected) == 2, "two metadata functions required")
    namespace = dict(copy=copy, json=json, Path=Path, REPO=REPO, require=require, GPU=GPU)
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id in ("CHECKPOINT", "FIXTURE_SHA", "GPU", "PORT")
        ):
            namespace[statement.targets[0].id] = ast.literal_eval(statement.value)
    exec(compile(ast.Module(body=selected, type_ignores=[]), source_ref["path"], "exec"), namespace)
    old = read(reference(REPO / "outputs/clean_training_development_continuation_20261024/execution/admitted.json"))
    prepared = read(old["prepared_ref"])
    records = []
    for case in namespace["case_order"]():
        config = namespace["make_case_config"](prepared, case)
        ref = publish(ROOT / "preflight_py311/configs" / f"{case['case_id']}.json", config)
        hashes = execution_inputs(config, ref["path"])
        require("development_dependency_repair_20261024/cases/" in config["output_dir"], "old output path reused")
        records.append(dict(case=case, config_ref=ref, input_sha256=hashes))
    require(len(records) == 12 and sum(r["case"]["expected_requests"] for r in records) == 900, "full matrix differs")
    print(
        json.dumps(
            publish(
                ROOT / "preflight_py311/all_case_inputs.json",
                dict(
                    complete=True,
                    passed=True,
                    source_generation_ref=generation_ref,
                    owner_source_ref=source_ref,
                    records=records,
                    case_count=12,
                    expected_requests=900,
                    actual_model_calls=0,
                    actual_physics_steps=0,
                    array_decodes=0,
                    scope="actual generated case_order/make_case_config AST functions plus original execution_inputs; no whole-owner admission or simulator environment",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
