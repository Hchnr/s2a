"""Explicit ordinary-Python metadata import isolation, without source rewriting.

These utilities neither qualify a fitted head nor authorize a model launch.
The caller must independently admit its protocol and bind all source refs.
"""

import copy
import importlib
import importlib.util
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from s2a.durable_stage_supervisor import REPO, checked, reference, require

_LOCK = threading.Lock()
ENV_PATH = "S2A_STATE_HANDOFF_PATH"
ENV_SHA = "S2A_STATE_HANDOFF_SHA256"


def _directory(path):
    path = Path(path)
    require(
        path.is_absolute() and path.resolve() == path and path.is_dir() and path.is_relative_to(REPO),
        "bridge directory must be canonical inside S2A",
    )
    return path


def child_environment(base, *, bridge_directory, bundle_directory, registration_ref):
    """Preserve the real command and allow explicit metadata module precedence."""
    bridge, bundle = _directory(bridge_directory), _directory(bundle_directory)
    path = Path(registration_ref["path"])
    digest = registration_ref["sha256"]
    require(
        path.is_absolute()
        and path.resolve() == path
        and path.is_relative_to(REPO / "outputs")
        and isinstance(digest, str)
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest),
        "bridge needs canonical registration path/SHA",
    )
    result = dict(base)
    for key, value in ((ENV_PATH, str(path)), (ENV_SHA, digest)):
        require(key not in result or result[key] == value, "conflicting inherited bridge registration")
        result[key] = value
    paths = [str(bridge), str(bundle), *(p for p in result.get("PYTHONPATH", "").split(os.pathsep) if p)]
    result["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    result["PYTHONSAFEPATH"] = "1"
    return result


def load_frozen_delegate(ref, *, name):
    """Load an actual old file under a private name; preserve its true __file__."""
    path = checked(ref)
    require(
        path.resolve() == path and path.is_relative_to(REPO) and path.suffix == ".py",
        "delegate must be an actual S2A Python source",
    )
    require(name.startswith("_s2a") and name not in sys.modules, "private delegate module name is occupied")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        require(
            Path(module.__file__).resolve() == path and reference(path) == ref, "delegate source changed during import"
        )
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module


@contextmanager
def metadata_import_scope(*, bridge_directory, bundle_directory, module_refs, registration_ref):
    """CPU-only temporary aliases; reject cached sources and restore all names."""
    require(_LOCK.acquire(blocking=False), "metadata import scope already active")
    saved_path = list(sys.path)
    saved_env = {key: os.environ.get(key) for key in (ENV_PATH, ENV_SHA, "PYTHONPATH", "PYTHONSAFEPATH")}
    created = []
    receipt = dict(imported_sources={}, source_guard_completed=False, restored=False)
    try:
        module_refs = copy.deepcopy(module_refs)
        registration_ref = copy.deepcopy(registration_ref)
        require(module_refs and "run_wam_clean_training" in module_refs, "STATE bridge alias is missing")
        require(all(name not in sys.modules for name in module_refs), "a reserved metadata module is already cached")
        for ref in module_refs.values():
            checked(ref)
        overlay = child_environment(
            os.environ,
            bridge_directory=bridge_directory,
            bundle_directory=bundle_directory,
            registration_ref=registration_ref,
        )
        for key in saved_env:
            os.environ[key] = overlay[key]
        sys.path[:] = list(dict.fromkeys([str(bridge_directory), str(bundle_directory), *saved_path]))
        before = set(sys.modules)
        bridge = None
        try:
            bridge = importlib.import_module("run_wam_clean_training")
            require(
                reference(Path(bridge.__file__).resolve()) == module_refs["run_wam_clean_training"],
                "resolved bridge source differs",
            )
            yield bridge, receipt
        finally:
            created = [name for name in set(sys.modules) - before if name in module_refs or name.startswith("_s2a")]
            if bridge is not None:
                require(sys.modules.get("run_wam_clean_training") is bridge, "bridge module was replaced during scope")
            for name, ref in module_refs.items():
                if name in sys.modules:
                    require(
                        reference(Path(sys.modules[name].__file__).resolve()) == ref,
                        "metadata module origin changed during scope",
                    )
                    receipt["imported_sources"][name] = ref
            for ref in module_refs.values():
                require(reference(Path(ref["path"])) == ref, "metadata source changed during scope")
            receipt["source_guard_completed"] = True
    finally:
        for name in created:
            sys.modules.pop(name, None)
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        sys.path[:] = saved_path
        receipt["restored"] = True
        _LOCK.release()
