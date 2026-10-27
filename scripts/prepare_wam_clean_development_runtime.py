"""Build051 role-aware source copies with the actual Python3.11 parser/printer."""

import json
import sys

from s2a.clean_training_state_handoff import BUNDLE
from s2a.development_repair_admission import BUNDLE as DESTINATION
from s2a.development_repair_admission import OLD_REF, ROOT
from s2a.development_runtime_sources import overlay_auditor, runtime_bridge_source
from s2a.development_source_relocation import FILES, transform
from s2a.durable_stage_supervisor import REPO, check_sources, publish, read, reference, require


def main():
    require(sys.version_info[:2] == (3, 11), "runtime generation requires actual Python3.11")
    old = read(OLD_REF)
    check_sources(old["source_sha256"])
    check_sources(old["snapshot_sha256"])
    relocation = read(reference(ROOT / "path_relocation.json"))
    template = relocation["new_template_ref"]
    read(template)
    require(not DESTINATION.exists(), "runtime copies exist; no overwrite")
    DESTINATION.mkdir()
    generated = []
    for name in FILES:
        original = reference(BUNDLE / name)
        require(old["source_sha256"].get(original["path"]) == original["sha256"], "original049 entrypoint changed")
        code, changes = transform(
            (BUNDLE / name).read_text(),
            filename=name,
            old_bundle=str(BUNDLE.relative_to(REPO)),
            new_bundle=str(DESTINATION.relative_to(REPO)),
            template=template["path"],
        )
        if name == FILES[-1]:
            code = overlay_auditor(code)
            changes.append(dict(kind="auditor_source_overlay", admission_only=True))
        path = DESTINATION / name
        compile(code, str(path), "exec")
        with path.open("x") as stream:
            stream.write(code)
        generated.append(dict(original_ref=original, generated_ref=reference(path), changes=changes))
    original = reference(BUNDLE / "continuation_runtime.py")
    require(old["source_sha256"].get(original["path"]) == original["sha256"], "original runtime changed")
    code = runtime_bridge_source(original)
    path = DESTINATION / "continuation_runtime.py"
    compile(code, str(path), "exec")
    with path.open("x") as stream:
        stream.write(code)
    generated.append(
        dict(original_ref=original, generated_ref=reference(path), changes=[dict(kind="runtime_import_bridge")])
    )
    print(
        json.dumps(
            publish(
                ROOT / "source_generation_runtime_py311.json",
                dict(
                    complete=True,
                    passed=True,
                    execution_authorized=False,
                    generated=generated,
                    python=sys.version,
                    actual_model_calls=0,
                    actual_physics_steps=0,
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
