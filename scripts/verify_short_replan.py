"""Independently replay007 oracle execution, preserving all004/005 checkers."""

import argparse

from audit_wam_position import Evidence, verify_case
from qualify_short_replan import (
    CONFIGS,
    PRIOR_CONFIG,
    RECORDS,
    ROOT,
    VARIANTS,
    protocol,
    read,
    require_cpu,
    write_new,
)

from s2a.artifacts import REPO, sha256


def qualified(case):
    return (
        case["status"] == "verified"
        and case["steps"] == 600
        and case["outcome"]["stable"]
        and case["tracking"]["admitted"]["position_rmse_m"] <= 0.01
        and case["tracking"]["admitted"]["so3_rmse_rad"] <= 0.05
        and case["independent_scalar_gripper_replay"] == "passed"
        and case["independent_governor_replay"] in ("passed", "not_applicable")
    )


def verify_group(config, root, evidence):
    records = {r["name"]: r for r in evidence.json(config["fixture"])["records"]}
    cases = []
    for variant in config["variants"]:
        if variant["name"] not in VARIANTS:
            continue
        for name in RECORDS:
            folder = root / "simulation" / f"oracle_{variant['name']}" / f"{name}_s42"
            assert evidence.json(folder / "manifest.json")["environment"]["MUJOCO_GL"] == "osmesa"
            assert not any(folder.glob("request_*.json")) and not any(folder.glob("response_*.json"))
            case = verify_case(
                config,
                records[name],
                variant,
                folder,
                evidence,
                source="oracle",
                checkpoint=None,
                full=True,
            )
            case["qualified"] = qualified(case)
            case["max_observation_interval_steps"] = config["max_observation_interval_steps"]
            diagnostics = case["governor_diagnostics"]
            diagnostics["bridge_fraction"] = diagnostics["bridging_endpoint_count"] / 600
            details = diagnostics["chunk_consumption"]
            diagnostics["queries_with_no_model_progress"] = sum(c["consumed_model_s"] <= 1e-12 for c in details)
            diagnostics["query_intervals_steps"] = [c["physical_steps"] for c in details]
            cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--prior-only", action="store_true")
    args = parser.parse_args()
    require_cpu()
    evidence = Evidence()
    manifest_path = ROOT / "qualification_launch_manifest.json"
    launch = evidence.json(manifest_path)
    assert launch["registered_cases"] == 12 and launch["registered_physics_steps"] == 7200
    for signatures in (launch["input_sha256"], launch["review_source_sha256"]):
        for path, digest in signatures.items():
            evidence.check(path, digest)
    prior_config = evidence.json(PRIOR_CONFIG)
    prior_root = REPO / prior_config["output_dir"]
    prior_gate = evidence.json(prior_root / "qualification_independent_verification.json")
    assert prior_gate["qualified"] and prior_gate["verified_cases"] == 9
    prior = verify_group(prior_config, prior_root, evidence)
    assert len(prior) == 6 and all(c["qualified"] for c in prior)
    if args.prior_only:
        output = ROOT / "prior80_independent_verification.json"
        evidence.recheck()
        write_new(
            output,
            dict(
                seed=42,
                complete=True,
                qualified=True,
                verified_cases=6,
                new_physics_steps=0,
                new_model_requests=0,
                cases=prior,
                input_sha256=evidence.hashes,
            ),
        )
    else:
        execution = evidence.json(ROOT / "qualification_execution_launch.json")
        evidence.check(manifest_path, execution["launch_manifest_sha256"])
        prior_path = ROOT / "prior80_independent_verification.json"
        evidence.check(prior_path, execution["prior80_verification_sha256"])
        cases = []
        for path in CONFIGS:
            config, root = protocol(path)
            assert evidence.json(path) == config
            assert config["qualification"]["expected_physics_steps"] == 3600
            cases.extend(verify_group(config, root, evidence))
        assert len(cases) == 12 and sum(c["steps"] for c in cases) == 7200
        comparison = []
        old = {(c["variant"], c["record"]): c for c in prior}
        for case in cases:
            baseline = old[(case["variant"], case["record"])]
            comparison.append(
                dict(
                    interval=case["max_observation_interval_steps"],
                    variant=case["variant"],
                    record=case["record"],
                    stable=case["outcome"]["stable"],
                    tcp_rmse_m=case["tracking"]["admitted"]["position_rmse_m"],
                    so3_rmse_rad=case["tracking"]["admitted"]["so3_rmse_rad"],
                    queries=case["chunks"],
                    bridge_fraction=case["governor_diagnostics"]["bridge_fraction"],
                    model_consumed_s=case["governor_diagnostics"]["consumed_model_s"],
                    generated_reference_fraction=case["governor_diagnostics"]["generated_reference_fraction"],
                    queries_with_no_model_progress=case["governor_diagnostics"]["queries_with_no_model_progress"],
                    reference80=dict(
                        stable=baseline["outcome"]["stable"],
                        tcp_rmse_m=baseline["tracking"]["admitted"]["position_rmse_m"],
                        so3_rmse_rad=baseline["tracking"]["admitted"]["so3_rmse_rad"],
                        queries=baseline["chunks"],
                        bridge_fraction=baseline["governor_diagnostics"]["bridge_fraction"],
                        model_consumed_s=baseline["governor_diagnostics"]["consumed_model_s"],
                    ),
                )
            )
        evidence.recheck()
        output = ROOT / "qualification_independent_verification.json"
        write_new(
            output,
            dict(
                seed=42,
                complete=True,
                qualified=all(c["qualified"] for c in cases),
                verified_cases=12,
                verified_physics_steps=7200,
                reference80_replayed_cases=6,
                model_or_gpu_started=False,
                oracle_is_wam_success_evidence=False,
                cases=cases,
                reference80_cases=prior,
                comparison=comparison,
                input_sha256=evidence.hashes,
                limitations=[
                    "Privileged fixed-world oracle endpoints indexed by current physical step, not WAM task capability.",
                    "No model inference, new initial states, parameter tuning, failed-case restart or threshold relaxation.",
                    "Controller reconstruction uses shared unchanged OSC inverse plus independent scalar gripper/governor replay.",
                ],
            ),
        )
    print(dict(output=str(output), sha256=sha256(output), qualified=read(output)["qualified"]), flush=True)


if __name__ == "__main__":
    main()
