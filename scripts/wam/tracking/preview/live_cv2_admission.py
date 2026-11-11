"""Probe and explicitly restore the model Python's known OpenCV environment change."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro

PROBE = """
import hashlib, json, os
from pathlib import Path
before = dict(os.environ)
cwd_before = str(Path.cwd().resolve())
import cv2
after = dict(os.environ)
loader = Path(cv2.__file__).resolve()
files = [loader, loader.parent / 'config.py', loader.parent / 'config-3.py']
sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
changes = {key: [before.get(key), after.get(key)] for key in sorted(set(before) | set(after))
           if before.get(key) != after.get(key)}
print(json.dumps(dict(version=cv2.__version__, cwd_before=cwd_before, cwd_after=str(Path.cwd().resolve()),
                      cuda_before=before.get('CUDA_VISIBLE_DEVICES'), cuda_after=after.get('CUDA_VISIBLE_DEVICES'),
                      ld_before=before.get('LD_LIBRARY_PATH'), ld_after=after.get('LD_LIBRARY_PATH'),
                      sources=sources, changes=changes)), flush=True)
"""


def run_cpu_probe(model_python, directory):
    """Import OpenCV in a separately waited CPU process using repository-local caches."""
    directory = Path(directory)
    prefix.require(
        directory.is_dir() and directory.is_relative_to(prefix.REPO / "outputs"), "Probe output must be local"
    )
    temporary, cache = directory / "tmp", directory / "cache"
    temporary.mkdir()
    cache.mkdir()
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH="src:.",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        LP_NUM_THREADS="1",
        TMPDIR=str(temporary),
        XDG_CACHE_HOME=str(cache),
    )
    process = subprocess.Popen(
        [str(model_python), "-c", PROBE],
        cwd=prefix.REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=60)
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.communicate()
        raise
    prefix.require(process.returncode == 0, f"CPU OpenCV probe failed: {stderr[-500:]}")
    value = json.loads(stdout.splitlines()[-1])
    prefix.require(
        value["cwd_before"] == value["cwd_after"] == str(prefix.REPO)
        and value["cuda_before"] == value["cuda_after"] == ""
        and value["ld_before"] == ""
        and type(value["ld_after"]) is str
        and bool(value["ld_after"])
        and len(value["sources"]) == 3
        and all(repro.sha256(path) == digest for path, digest in value["sources"].items()),
        "The actual CPU OpenCV import and its three source identities must be known",
    )
    value.update(actual_wait=True, reaped=True, returncode=process.returncode, gpu_work=0)
    return value


def admit_model_cv2(probe, *, gpu_uuid, importer=importlib.import_module):
    """Restore LD only after the actual model import matches the CPU probe exactly."""
    prefix.require(
        probe["actual_wait"] is probe["reaped"] is True
        and probe["returncode"] == 0
        and probe["gpu_work"] == 0
        and os.environ.get("CUDA_VISIBLE_DEVICES") == gpu_uuid
        and os.environ.get("LD_LIBRARY_PATH") == ""
        and str(Path.cwd().resolve()) == str(prefix.REPO)
        and "cv2" not in sys.modules,
        "A fresh GPU process and completed CPU OpenCV probe are required",
    )
    before = dict(os.environ)
    module = importer("cv2")
    after = dict(os.environ)
    changes = {
        key: [before.get(key), after.get(key)]
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    }
    loader = Path(module.__file__).resolve()
    paths = (loader, loader.parent / "config.py", loader.parent / "config-3.py")
    sources = {str(path): repro.sha256(path) for path in paths}
    prefix.require(
        module.__version__ == probe["version"]
        and sources == probe["sources"]
        and changes == probe["changes"]
        and after.get("CUDA_VISIBLE_DEVICES") == gpu_uuid
        and after.get("LD_LIBRARY_PATH") == probe["ld_after"]
        and str(Path.cwd().resolve()) == str(prefix.REPO),
        "OpenCV mutated the model environment differently from its audited CPU probe",
    )
    os.environ["LD_LIBRARY_PATH"] = ""
    prefix.require(os.environ["LD_LIBRARY_PATH"] == "", "Failed to restore the model process LD environment")
    return {
        "complete": True,
        "version": module.__version__,
        "sources": sources,
        "changes": changes,
        "ld_restored": True,
        "gpu_uuid": gpu_uuid,
    }
