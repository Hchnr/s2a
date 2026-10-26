"""Owned final source/qualification audit for the fixed preview control registration."""

import argparse
import json
from pathlib import Path

from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import check_sources, cpu_environment, owned_stage, publish, sha

ROOT = REPO / "outputs/coupled_preview_qualification_20261026"
CHAIN = REPO / "outputs/coupled_pose_preview_continuous_chain_20261026/verified.json"
LOG = REPO / "outputs/coupled_preview_control_log_bound_20261026/independent_verified.json"


def verify():
    e = Evidence()
    e.read(__file__)
    chain, log = e.json(CHAIN), e.json(LOG)
    assert chain["complete"] and len(chain["results"]) == 18
    assert sum(c["periods"] for c in chain["results"]) == sum(c["source_commits"] for c in chain["results"]) == 4907
    assert max(c["maximum_kkt"] for c in chain["results"]) < 2e-6
    assert log["complete"] and log["independent_preview_control_log"]
    assert log["rows"] == log["json_roundtrip_rows"] == log["source_commit_cycles"] == 164
    assert len(log["corruptions_rejected"]) == 22 and log["max_stationarity_residual"] < 2e-6
    for proof in (chain, log):
        for path, digest in proof["source_sha256"].items():
            e.check(path, digest)
    e.recheck()
    publish(
        ROOT / "verified.json",
        dict(
            complete=True,
            source_sha256=e.hashes,
            record_cases=18,
            record_periods=4907,
            full_log_periods=164,
            corruptions_rejected=22,
            physical_control_qualified=False,
            new_physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "check"))
    args = parser.parse_args()
    if args.mode == "check":
        verify()
        return
    assert not ROOT.exists()
    sources = {str(Path(__file__).resolve()): sha(__file__), str(CHAIN): sha(CHAIN), str(LOG): sha(LOG)}
    for path in (CHAIN, LOG):
        for declared, digest in json.loads(path.read_text())["source_sha256"].items():
            normalized = str((REPO / declared).resolve())
            assert normalized not in sources or sources[normalized] == digest
            sources[normalized] = digest
    check_sources(sources)
    ROOT.mkdir()
    publish(ROOT / "source_sha256.json", sources)
    env = cpu_environment()
    env["PYTHONPATH"] = str(REPO / "src") + ":" + str(REPO / "scripts")
    stage = owned_stage(
        [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "check"],
        environment=env,
        directory=ROOT / "qualification",
        timeout_s=1800,
        source_sha256=sources,
    )
    check_sources(sources)
    proof = ROOT / "verified.json"
    assert json.loads(proof.read_text())["complete"]
    publish(
        ROOT / "execution_result.json",
        dict(
            complete=True,
            actual_owned_wait_recorded=True,
            stage_ref=stage,
            proof_ref=dict(path=str(proof), sha256=sha(proof)),
            physics_steps=0,
            automatic_retry=False,
        ),
    )
    print("owned preview qualification completed", flush=True)


if __name__ == "__main__":
    main()
