"""New aggregate identity must preserve the old metric rules and full QR gate."""

import ast
import copy

import pytest

from s2a.aggregate_state_metrics import diagnosis as aggregate_diagnosis
from s2a.artifacts import REPO
from s2a.clean_aggregation_candidate import candidate_head
from s2a.clean_training_aggregation import RIDGE, ROWS
from s2a.synchronous_state_metrics import diagnosis as previous_diagnosis


def test_metric_arithmetic_and_gate_thresholds_are_unchanged():
    class Rename(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(node.value.replace("probe052", "aggregate056")), node)
            return node

    def functions(name):
        return {
            n.name: n for n in ast.parse((REPO / "src/s2a" / name).read_text()).body if isinstance(n, ast.FunctionDef)
        }

    old = functions("synchronous_state_metrics.py")
    new = functions("aggregate_state_metrics.py")
    for name in ("pool", "state_gate", "diagnosis"):
        assert ast.dump(Rename().visit(old[name])) == ast.dump(new[name])


def test_all_groups_and_failure_checks_match_previous_scorer():
    rows = []
    for task in range(3):
        for demo in range(2):
            for anchor in range(4):
                for seed in (42, 43):
                    for variant in ("raw", "probe052"):
                        factor = 1.0 if variant == "raw" else (1.5 if (task, demo, seed) == (2, 1, 43) else 0.8)
                        errors = {
                            k: [factor * v] * 40 for k, v in (("position", 0.03), ("rotation", 0.1), ("opening", -0.08))
                        }
                        rows.append(
                            dict(
                                name=f"t{task}d{demo}a{anchor}_{variant}_s{seed}",
                                episode=f"t{task}d{demo}",
                                task_index=task,
                                sampling_seed=seed,
                                variant=variant,
                                errors=dict(
                                    wam=errors,
                                    hold={k: [v * 2 for v in a] for k, a in errors.items()},
                                    constant_velocity={k: [v * 3 for v in a] for k, a in errors.items()},
                                ),
                            )
                        )
    old_groups, old_gate = previous_diagnosis(rows)
    new_rows = copy.deepcopy(rows)
    for row in new_rows:
        if row["variant"] == "probe052":
            row["variant"] = "aggregate056"
            row["name"] = row["name"].replace("probe052", "aggregate056")
    new_groups, new_gate = aggregate_diagnosis(new_rows)

    def renamed(value):
        if isinstance(value, dict):
            return {
                (k.replace("probe052", "aggregate056") if isinstance(k, str) else k): renamed(v)
                for k, v in value.items()
            }
        return value

    assert new_groups == renamed(old_groups)
    assert new_gate == old_gate and new_gate["passed"] is False


def evidence():
    docs = {}
    process = dict(launch="launch", exit="exit", release="release")
    docs["launch"] = dict(child=dict(pid=11, start_ticks=22))
    docs["exit"] = dict(returncode=0, actual_wait=True, automatic_retry=False)
    docs["release"] = dict(
        pid=11,
        start_ticks=22,
        reaped=True,
        returncode=0,
        signal_sent=False,
        original_error=None,
        only_owned_child_signalled=True,
    )
    docs["owner"] = dict(
        schema_version="s2a.clean-aggregation-owner.v1",
        complete=True,
        passed=True,
        candidate_numerically_qualified=True,
        owned_commands=[process, process],
        fresh_authorized=False,
        validation_authorized=False,
        automatic_retry=False,
        producer_ref="fit",
        verification_ref="qr",
        params_ref="params",
        registration_ref="registration",
    )
    streams = dict(features="features-hash", target_clean_residual="target-hash", weights="weights-hash")
    docs["fit"] = dict(
        schema_version="s2a.clean-aggregation-fit.v1",
        complete=True,
        passed=True,
        registration_ref="registration",
        params_ref="params",
        producer_fits=1,
        collection_refs=["D0", "D1", "D2"],
        d2_owner_ref="d2-owner",
        fit_metadata=dict(rows=ROWS, replay_sha256=streams),
    )
    docs["qr"] = dict(
        schema_version="s2a.clean-aggregation-qr.v1",
        complete=True,
        passed=True,
        producer_ref="fit",
        params_ref="params",
        independent_reconstructions=1,
        collection_refs=["D0", "D1", "D2"],
        d2_owner_ref="d2-owner",
        rtol=1e-9,
        atol=1e-11,
        comparisons={k: dict(passed=True) for k in ("coefficients", "center", "scale")},
        qr_metadata=dict(
            rows=ROWS, replay_sha256=dict(features="features-hash", targets="target-hash", weights="weights-hash")
        ),
    )
    docs["params"] = dict(
        schema_version="s2a.clean-residual-readout.v1",
        coefficients=[[0.0] * 10 for _ in range(1291)],
        center=[0.0] * 1290,
        scale=1.0,
        ridge=RIDGE,
    )
    return docs


def test_aggregate_candidate_has_its_own_numeric_identity():
    docs = evidence()
    head, ref = candidate_head("owner", read_json=docs.__getitem__)
    assert ref == "params" and len(head.center) == 1290


@pytest.mark.parametrize("fault", ["old-schema", "rows", "stream", "tolerance", "unreaped", "params", "fresh"])
def test_incomplete_or_different_qualification_cannot_be_used_as_new_candidate(fault):
    docs = evidence()
    if fault == "old-schema":
        docs["fit"]["schema_version"] = "s2a.clean-regularization-probe.v1"
    elif fault == "rows":
        docs["qr"]["qr_metadata"]["rows"] = 840000
    elif fault == "stream":
        docs["qr"]["qr_metadata"]["replay_sha256"]["targets"] = "different"
    elif fault == "tolerance":
        docs["qr"]["rtol"] = 1e-6
    elif fault == "unreaped":
        docs["release"]["reaped"] = False
    elif fault == "params":
        docs["qr"]["params_ref"] = "other-params"
    else:
        docs["owner"]["fresh_authorized"] = True
    with pytest.raises(ValueError):
        candidate_head("owner", read_json=docs.__getitem__)
