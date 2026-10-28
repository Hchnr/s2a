"""Copy historically fingerprinted assets and compile once without a response."""

import argparse
import json
import os
import platform
from pathlib import Path

from execute_tracking_v5_prefix_trust_check_v2 import verify_native_qa

from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.prefix_trust_asset_repair import (
    HISTORICAL_MANIFEST_REF,
    ORIGINAL_XML_REF,
    REPAIR_SOURCES,
    SNAPSHOT_ROOT,
    repair_prerequisites,
)
from s2a.prefix_trust_control_evidence import CPU_ENVIRONMENT, REPO, finish, read, reference, remember, require
from s2a.preview_result_publication import publish_result
from s2a.terminal_development_execution import CONTROL_REF
from s2a.xml_asset_snapshot import prepare_asset_snapshot, verify_asset_snapshot


def main(qa_ref, dry=False):
    strict_native_environment()
    require(all(os.environ.get(k) == v for k, v in CPU_ENVIRONMENT.items()), "Fixed native CPU compilation environment")
    require(not SNAPSHOT_ROOT.exists(), "One exclusive snapshot creation; preserve failures without retry")
    qa = verify_native_qa(qa_ref)
    refs = repair_prerequisites(require_snapshot=False)
    remember(qa_ref, refs)
    remember(dict(path=str(Path(qa_ref["path"]).parent / "pytest.log"), sha256=qa["log_sha256"]), refs)
    finish(refs)
    if dry:
        print(json.dumps(dict(dry_run=True, passed=True, source_count=len(refs), new_responses=0)), flush=True)
        return
    manifest_ref = prepare_asset_snapshot(ORIGINAL_XML_REF, HISTORICAL_MANIFEST_REF, SNAPSHOT_ROOT)
    snapshot = verify_asset_snapshot(manifest_ref)
    result = dict(
        complete=False,
        passed=False,
        manifest_ref=manifest_ref,
        native_component_qa_ref=qa_ref,
        source_refs=[reference(REPO / path) for path in REPAIR_SOURCES],
        environment={key: os.environ.get(key) for key in CPU_ENVIRONMENT},
        response_evaluations=0,
        query_count=0,
        numerical_inner_steps=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        guard = read(CONTROL_REF)["cv2_guard_registration"]
        result["environment_admission"] = restore_cv2_environment(
            guard["exact_probe_ref"], guard["runtime_source_refs"]
        )
        os.environ.pop("MUJOCO_GL", None)
        os.environ.pop("PYOPENGL_PLATFORM", None)
        import mujoco
        import numpy as np

        model = mujoco.MjModel.from_xml_path(snapshot["migrated_xml_ref"]["path"])
        result["runtime"] = dict(python=platform.python_version(), mujoco=mujoco.__version__, numpy=np.__version__)
        require(
            result["runtime"]["mujoco"] == "2.3.7" and result["runtime"]["numpy"] == "1.26.4", "Fixed native compiler"
        )
        require(
            model.opt.timestep == 0.002 and model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER,
            "Original full Euler2ms model",
        )
        result["model_dimensions"] = {
            key: int(getattr(model, key))
            for key in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nmesh", "ntex", "nsite", "ntendon")
        }
        result["timestep"] = float(model.opt.timestep)
        verify_asset_snapshot(manifest_ref)
        finish(refs)
        strict_native_environment()
        result.update(complete=True, passed=True, source_refs_unchanged=list(refs.values()))
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish_result(SNAPSHOT_ROOT / "compile.json", result)
    print(
        json.dumps(
            dict(
                complete=True,
                passed=True,
                manifest_ref=manifest_ref,
                compile_ref=reference(SNAPSHOT_ROOT / "compile.json"),
                new_responses=0,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-ref", type=Path, required=True)
    parser.add_argument("--qa-sha256", required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(dict(path=str(args.qa_ref.resolve()), sha256=args.qa_sha256), args.dry)
