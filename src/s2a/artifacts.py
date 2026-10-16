"""Strict JSON and reproducibility fingerprints; artifacts stay in this repo."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)

    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=convert) + "\n")


def code_fingerprint():
    paths = sorted(
        p
        for group in ["src", "scripts", "configs"]
        for p in (REPO / group).rglob("*")
        if p.is_file() and p.suffix in {".py", ".json"}
    )
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)),
        "files_sha256": {str(p.relative_to(REPO)): sha256(p) for p in paths},
        "python": platform.python_version(),
        "dependencies": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "pyarrow"]},
    }


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    out = Path(config["output_dir"])
    out = (REPO / out).resolve()
    if not out.is_relative_to(REPO / "outputs"):
        raise ValueError("output_dir must be inside repository outputs/")
    out.mkdir(parents=True, exist_ok=True)
    return config, out
