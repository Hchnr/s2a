"""CPU014 regression of old004/006 and already verified010 seed42 saved logs.

This entry accepts only the old registered development evidence. It has no
fresh/candidate/model/simulator execution option and writes new audit artifacts.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import audit_wam_earlier_position as old006
import audit_wam_position as old004
from run_wam_short_replan import cpu_environment, write_new
from wam_position_acceptance_checker import source_proof, verify_case, verify_query

from s2a.artifacts import REPO, sha256

ROOT = REPO / "outputs/wam_stability_20261021"
ORACLE_AUDIT = ROOT / "position_full_complete_20261021.json"
ORACLE_SHA = "fe714dad35dad3d6d3ab933be00e166c94c0c54cff6611730c439e782759299e"
EARLIER_AUDIT = ROOT / "earlier_position_final_audit.json"
EARLIER_SHA = "46a07b31bc0e7bcee9b8c740a2c853ea718fa4867354794bdd85724fd6d0cbdc"
METADATA_ONLY = {
    "reused_from",
    "profile",
    "interval",
    "model_rotation_alpha",
    "expected_queries",
    "shared_service_provenance",
}


def historical_first_query(config, record, variant, case, chunk, evidence, checkpoint, *, sampling_seed):
    """The existing006 callback authenticates only real historical seed42."""
    assert sampling_seed == 42
    return old006.verify_first_query006(config, record, variant, case, chunk, evidence, checkpoint)


def source_audit(path, signature, evidence):
    evidence.check(path, signature)
    value = evidence.json(path)
    for filename, digest in value["source_before_after_hashes"].items():
        evidence.check(filename, digest)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-replan-audit", required=True, type=Path)
    parser.add_argument("--short-replan-audit-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cpu_environment()
    assert not args.output.exists() and args.output.resolve().is_relative_to(REPO / "outputs")
    assert args.short_replan_audit.resolve().parent == ROOT / "short_replan_audit"
    evidence = old004.Evidence()
    evidence.read(__file__)
    proof = source_proof(evidence)
    pointers = (old004.verify_case, old004.verify_query, old006.verify_case)
    oracle = source_audit(ORACLE_AUDIT, ORACLE_SHA, evidence)
    assert oracle["whole_registered_protocol_complete"] and oracle["registered_oracle_cases"] == 9
    earlier = source_audit(EARLIER_AUDIT, EARLIER_SHA, evidence)
    assert earlier["complete"] and earlier["all12_cases_independently_verified"]
    short = source_audit(args.short_replan_audit, args.short_replan_audit_sha256, evidence)
    assert short["schema_version"] == "s2a.short-replan-readonly-audit.v1"
    registered = []
    registered += [("004", case) for case in oracle["cases"] if case["source"] == "oracle"]
    registered += [("006", case) for case in earlier["cases"]]
    registered += [("010", case) for case in short["cases"] if case["status"] == "verified"]
    assert sum(label == "004" for label, _ in registered) == 9
    assert sum(label == "006" for label, _ in registered) == 12
    assert any(label == "010" and c["raw_checkpoint_evidence"] and c["outcome"]["stable"] for label, c in registered)
    assert any(not c["outcome"]["stable"] for _, c in registered)
    verified = []
    for label, expected in registered:
        assert expected["status"] == "verified"
        case = Path(expected["case"])
        manifest = evidence.json(case / "manifest.json")
        config = manifest["experiment"]
        assert config["fixture"] == "outputs/wam_checkpoint_acceptance_20261019v2/simulation/fixtures.json"
        assert config["sampling_seeds"] == [42]
        assert manifest["record"]["name"] in {"task0_init0", "task1_init0", "task2_init0"}
        assert expected["source"] in {"wam", "oracle"}
        if expected["source"] == "wam":
            assert expected["checkpoint"] == "8400_masked"
        actual = verify_case(
            config,
            manifest["record"],
            manifest["variant"],
            case,
            evidence,
            source=expected["source"],
            checkpoint=expected["checkpoint"],
            full=True,
            sampling_seed=42,
            first_query_validator=historical_first_query if expected["source"] == "wam" else None,
        )
        assert set(expected) - set(actual) <= METADATA_ONLY
        assert actual == {key: value for key, value in expected.items() if key in actual}, (label, str(case))
        verified.append(
            dict(
                stage=label,
                case=str(case),
                source=actual["source"],
                sampling_seed=42,
                all_case_fields_exact=True,
                old_metadata_only_fields=sorted(set(expected) - set(actual)),
                result=actual,
            )
        )
        print(json.dumps(dict(stage=label, case=str(case), all_case_fields_exact=True)), flush=True)
    # These are negative API guards against the real42 config, never forged43 logs.
    example = next(case for label, case in registered if label == "010" and case["source"] == "wam")
    folder = Path(example["case"])
    manifest = evidence.json(folder / "manifest.json")
    config = manifest["experiment"]
    rejected = []
    for seed in (43, True):
        try:
            verify_case(
                config,
                manifest["record"],
                manifest["variant"],
                folder,
                evidence,
                source="wam",
                checkpoint="8400_masked",
                full=True,
                sampling_seed=seed,
                first_query_validator=historical_first_query,
            )
        except AssertionError:
            rejected.append(dict(seed=seed, guard="unregistered_or_noninteger_sampling_seed", rejected=True))
        else:
            raise AssertionError("unregistered seed incorrectly admitted")
    try:
        verify_query(
            config, folder, 0, evidence.json(folder / "chunks.json")[0], evidence, "8400_masked", sampling_seed=43
        )
    except AssertionError:
        rejected.append(dict(seed=43, guard="query_unregistered_sampling_seed", rejected=True))
    else:
        raise AssertionError("unregistered query seed incorrectly admitted")
    assert pointers == (old004.verify_case, old004.verify_query, old006.verify_case)
    evidence.recheck()
    result = dict(
        schema_version="s2a.seed-aware-checker-regression.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,
        all_case_fields_exact=True,
        source_proof=proof,
        cases=verified,
        old_log_cases_verified=len(verified),
        old_log_endpoints_verified=600 * len(verified),
        counts={label: sum(c["stage"] == label for c in verified) for label in ("004", "006", "010")},
        negative_seed_guards=rejected,
        original_function_objects_unchanged=True,
        new_model_requests=0,
        new_physics_steps=0,
        fresh_performance_opened=False,
        seed43_physics_verified=False,
        seed43_wam_request_response_verified=False,
        input_sha256=evidence.hashes,
        limitations=[
            "Only real existing seed42 logs are replayed; seed43 is not fabricated or inferred from42.",
            "Oracle qualification is separate from WAM capability and is excluded from any WAM success denominator.",
            "No new task execution, model query, candidate selection or fresh acceptance is authorized by this checker regression.",
            "Future first-query validation must be explicitly frozen against its own current inputs/fixture/initial physics; no historical raw identity is invented.",
        ],
    )
    write_new(args.output, result)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                sha256=sha256(args.output),
                complete=True,
                cases=len(verified),
                new_model_requests=0,
                new_physics_steps=0,
            )
        )
    )


if __name__ == "__main__":
    main()
