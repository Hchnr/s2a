"""Generate finite051 source copies without authorizing any execution."""

import json
import sys

from s2a.clean_training_development_handoff import ROOT as FAILED_ROOT
from s2a.clean_training_state_handoff import BUNDLE
from s2a.development_source_relocation import FILES, transform
from s2a.durable_stage_supervisor import REPO, check_sources, publish, read, reference, require

ROOT = REPO / "outputs/clean_training_development_dependency_repair_20261024"


def main():
    require(sys.version_info[:2] == (3, 11), "source generation requires actual Python3.11 runtime")
    registration = read(reference(FAILED_ROOT / "execution/admitted.json"))
    require(len(registration["source_sha256"]) == 562, "049 source basis differs")
    check_sources(registration["source_sha256"])
    check_sources(registration["snapshot_sha256"])
    relocation_ref = reference(ROOT / "path_relocation.json")
    relocation = read(relocation_ref)
    require(relocation["complete"] is relocation["passed"] is True, "tool contract not qualified")
    template = relocation["new_template_ref"]
    require(read(template)["tool_contract"] == relocation["safe_copy_ref"]["path"], "safe template differs")
    destination = ROOT / "source_py311"
    require(not destination.exists(), "finite source copies already exist")
    destination.mkdir()
    generated = []
    for name in FILES:
        original = reference(BUNDLE / name)
        require(registration["source_sha256"].get(original["path"]) == original["sha256"], "original source differs")
        code, changes = transform(
            (BUNDLE / name).read_text(),
            filename=name,
            old_bundle=str(BUNDLE.relative_to(REPO)),
            new_bundle=str(destination.relative_to(REPO)),
            template=template["path"],
        )
        path = destination / name
        with path.open("x") as stream:
            stream.write(code)
        generated.append(dict(original_ref=original, generated_ref=reference(path), changes=changes))
    print(
        json.dumps(
            publish(
                ROOT / "source_generation_py311.json",
                dict(
                    complete=True,
                    passed=True,
                    execution_authorized=False,
                    generated=generated,
                    relocation_ref=relocation_ref,
                    generator_ref=reference(REPO / "src/s2a/development_source_relocation.py"),
                    runtime_bridge_implemented=False,
                    new_model_calls=0,
                    new_physics_steps=0,
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
