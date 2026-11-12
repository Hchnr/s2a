"""Audit recovered pad snapshots and preserve the failed query budget."""

import hashlib
from pathlib import Path

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import audit_bowl_pad_alignment as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
SOURCE = {
    "path": str(ROOT / "outputs/bowl_pad_alignment_recovery_20261112/result.json"),
    "sha256": "3c4578e8cc7c3cadb850e014f554bf89fa9a3d2355378e3da0139338fc177080",
}
FAILURE = {
    "path": str(ROOT / "outputs/bowl_pad_alignment_snapshot_failure_20261112/result.json"),
    "sha256": "43d178f76fb90eed3f003952b7b6ec6078bd8a4fa9fdc9bafe2d6ec5ba198a10",
}
OUTPUT = ROOT / "outputs/bowl_pad_alignment_recovery_audit_20261112/result.json"
_OLD_SHA = "65df1ff1e07ba71fc907f5ac9d6a45ad46f4345b9d53da1e75fd4316cc3e1c78"


def recovered_publication(path, result):
    """Attach the preserved failure before publishing the prior vertex audit."""
    prefix.require(
        path == OUTPUT and result["source_ref"] == SOURCE and result["audited_points"] == 97,
        "The vertex audit must target the registered recovered result",
    )
    result.update(prior_failure_ref=FAILURE, cumulative_static_models=5, cumulative_static_forwards=398)
    return publish_result(path, result)


def main():
    prefix.environment()
    prefix.require(
        hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest() == _OLD_SHA,
        "The original independent vertex audit must remain unchanged",
    )
    failure, source = prefix.read(FAILURE), prefix.read(SOURCE)
    registration = prefix.read(source["registration_ref"])
    prefix.require(
        failure["complete"] is failure["passed"] is True
        and failure["source_diagnostic_qualified"] is False
        and failure["cumulative_static_models"] == 4
        and failure["cumulative_static_forwards"] == 301
        and registration["inputs"]["refs"]["snapshot_failure_audit"] == FAILURE,
        "The prior failure and its full cost must be bound to the new source",
    )
    previous.SOURCE, previous.OUTPUT, previous.publish_result = SOURCE, OUTPUT, recovered_publication
    previous.main()


if __name__ == "__main__":
    main()
