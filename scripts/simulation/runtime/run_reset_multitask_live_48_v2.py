"""Reuse the pinned case supervisor with the separately registered recovery inputs."""

import hashlib
from pathlib import Path

from scripts.simulation.runtime import reset_multitask_live_admission_v2 as admission
from scripts.simulation.runtime import reset_multitask_live_workers_v2 as workers
from scripts.simulation.runtime import run_reset_multitask_live_48 as previous

_OWNER_SHA256 = "5ed6a4582e697ae7239d074110b6815011ff0c91f69b9bd293277e077bb65265"
_OLD_WORKER_SHA256 = "e88e17a1c03ecdbe61026fd6a45779bddb449ed7a54ac023d751eed1bd15ccfb"
_OLD_ADMISSION_SHA256 = "a10f7f82833b25fa89252396134ce93582c3b7e3423aa1456230a3c3a91faebb"


def bind_recovery():
    """Bind only this process to the new inputs and corrected worker implementation."""
    root = admission.ROOT
    files = (
        (Path(previous.__file__), _OWNER_SHA256),
        (root / "scripts/simulation/runtime/reset_multitask_live_workers.py", _OLD_WORKER_SHA256),
        (root / "scripts/simulation/runtime/reset_multitask_live_admission.py", _OLD_ADMISSION_SHA256),
    )
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest for path, digest in files):
        raise ValueError("The previously registered supervisor and sources must remain unchanged")
    previous.admission = admission
    previous.workers = workers
    previous.ROOT, previous.OUTPUT, previous.QA, previous.DRY = (
        admission.ROOT,
        admission.OUTPUT,
        admission.QA,
        admission.DRY,
    )
    previous.MODULE = "scripts.simulation.runtime.run_reset_multitask_live_48_v2"
    return previous


def main():
    bind_recovery().main()


if __name__ == "__main__":
    main()
