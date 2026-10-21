"""Explicit, probe-bound OpenCV import admission for CPU native collection."""

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

from s2a.artifacts import REPO


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_snapshot():
    return dict(
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
        LD_LIBRARY_PATH=os.environ.get("LD_LIBRARY_PATH"),
        cwd=str(Path.cwd().resolve()),
    )


def strict_native_environment(snapshot=None, *, expected_cwd=REPO):
    """Check the current process; never repair an unexpected mutation."""
    actual = environment_snapshot() if snapshot is None else snapshot
    require(
        actual == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", cwd=str(Path(expected_cwd).resolve()))
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        f"strict CPU native environment changed: {actual}",
    )
    return actual


class EnvironmentAdmissionError(ValueError):
    """Preserve actual import observations when admission fails."""

    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def restore_cv2_environment(probe_ref, runtime_source_refs, *, expected_cwd=REPO):
    """Import once, admit only the registered exact mutation, restore local LD.

    No simulator is imported. An unknown mutation is preserved and rejected;
    this function never resets CUDA, cwd or an unregistered LD value.
    """
    receipt = dict(
        schema_version="s2a.native-training-environment-readiness.v1",
        complete=False,
        passed=False,
        pid=os.getpid(),
        ppid=os.getppid(),
        argv=list(sys.argv),
        cv2_probe_ref=probe_ref,
        cv2_runtime_source_refs=runtime_source_refs,
        before=environment_snapshot(),
        bytecode_before=os.environ.get("PYTHONDONTWRITEBYTECODE"),
        restore_performed=False,
        new_environment_initializations=0,
        new_physics_steps=0,
        new_model_calls=0,
        new_hdf_reads=0,
    )
    try:
        strict_native_environment(receipt["before"], expected_cwd=expected_cwd)
        require(digest(probe_ref["path"]) == probe_ref["sha256"], "registered OpenCV probe SHA changed")
        probe = json.loads(Path(probe_ref["path"]).read_text())
        require(
            probe["schema_version"] == "s2a.native-training-import-environment-probe.v1"
            and probe["before"] == receipt["before"]
            and probe["after_import_cv2"]["cwd"] == receipt["before"]["cwd"]
            and probe["after_import_cv2"]["CUDA_VISIBLE_DEVICES"] == ""
            and isinstance(probe["after_import_cv2"]["LD_LIBRARY_PATH"], str)
            and bool(probe["after_import_cv2"]["LD_LIBRARY_PATH"])
            and probe["reproduces_empty_ld_guard_rejection"] is True
            and probe["new_environment_initializations"] == probe["new_physics_steps"] == probe["new_model_calls"] == 0
            and probe["new_hdf_reads"] == 0,
            "registered OpenCV import probe contract changed",
        )
        loader = Path(probe["cv2_loader_ref"]["path"]).resolve()
        expected_sources = {str(loader), str(loader.parent / "config.py"), str(loader.parent / "config-3.py")}
        require(
            len(runtime_source_refs) == 3
            and {str(Path(ref["path"]).resolve()) for ref in runtime_source_refs} == expected_sources
            and probe["cv2_loader_ref"] in runtime_source_refs,
            "registered OpenCV loader/config source roster changed",
        )
        before_sources = {}
        for ref in runtime_source_refs:
            require(digest(ref["path"]) == ref["sha256"], f"OpenCV runtime source changed: {ref['path']}")
            before_sources[ref["path"]] = ref["sha256"]
        receipt["source_before_sha256"] = before_sources
        require("cv2" not in sys.modules, "OpenCV must not be imported before explicit environment admission")
        cv2 = importlib.import_module("cv2")
        receipt["after_import_cv2"] = environment_snapshot()
        receipt["bytecode_after_import"] = os.environ.get("PYTHONDONTWRITEBYTECODE")
        receipt["actual_cv2_loader"] = str(Path(cv2.__file__).resolve())
        receipt["actual_cv2_loader_ref"] = dict(
            path=receipt["actual_cv2_loader"], sha256=digest(receipt["actual_cv2_loader"])
        )
        receipt["actual_cv2_version"] = cv2.__version__
        require(
            receipt["after_import_cv2"] == probe["after_import_cv2"]
            and receipt["bytecode_after_import"] == "1"
            and receipt["actual_cv2_loader"] == str(loader)
            and receipt["actual_cv2_loader_ref"] == probe["cv2_loader_ref"]
            and receipt["actual_cv2_version"] == probe["cv2_version"],
            "actual OpenCV import environment/loader differs from registered probe",
        )
        after_sources = {ref["path"]: digest(ref["path"]) for ref in runtime_source_refs}
        receipt["source_after_sha256"] = after_sources
        require(after_sources == before_sources, "OpenCV loader/config changed during import")
        os.environ["LD_LIBRARY_PATH"] = ""
        receipt["restore_performed"] = True
        receipt["after_restore"] = environment_snapshot()
        receipt["bytecode_after_restore"] = os.environ.get("PYTHONDONTWRITEBYTECODE")
        strict_native_environment(receipt["after_restore"], expected_cwd=expected_cwd)
        receipt.update(complete=True, passed=True)
        return receipt
    except Exception as error:
        receipt["failure_environment"] = environment_snapshot()
        receipt["error"] = dict(type=type(error).__name__, message=str(error))
        raise EnvironmentAdmissionError(str(error), receipt) from error
