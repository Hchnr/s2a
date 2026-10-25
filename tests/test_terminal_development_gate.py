import copy

import pytest

import s2a.terminal_development_gate as gate


def evidence(monkeypatch):
    def ref(name):
        return dict(path=str(gate.ROOT / name), sha256="a" * 64)

    docs = {}

    def put(name, value):
        r = ref(name)
        docs[r["path"]] = value
        return r

    commands = []
    for index, role in enumerate(("model", "independent", "scoring")):
        identity = dict(pid=11 + index, start_ticks=22 + index)
        commands.append(
            dict(
                launch=put(f"orchestration/{role}/launch.json", dict(child=identity)),
                exit=put(
                    f"orchestration/{role}/exit.json", dict(returncode=0, actual_wait=True, automatic_retry=False)
                ),
                release=put(
                    f"orchestration/{role}/ownership/release.json",
                    dict(
                        **identity,
                        reaped=True,
                        returncode=0,
                        signal_sent=False,
                        original_error=None,
                        only_owned_child_signalled=True,
                    ),
                ),
            )
        )
    gpu = put(
        "gpu.json", dict(released=True, uuid=gate.GPU, memory_mib=0, compute_apps=[], device_minor=6, visible_gpu=2)
    )
    proof, params = ref("proof.json"), ref("params.json")
    prepared = put("prepared.json", dict(root=str(gate.ROOT), candidate_proof_ref=proof, candidate_params_ref=params))
    numerical = put(
        "numerical.json",
        dict(
            schema_version="s2a.terminal-identity-state-audit.v1",
            passed=True,
            complete=True,
            numerical_contract_passed=True,
            lifecycle_passed=True,
            actual_requests=99,
            actual_nodes=3465,
            zero_nodes_verified=105,
            paired_initial_conditions_verified=48,
            fresh_cases=0,
            validation_future_archive_reads=0,
            worker_launch_ref=commands[0]["launch"],
            worker_exit_ref=commands[0]["exit"],
            worker_release_ref=commands[0]["release"],
            gpu_after_ref=gpu,
            prepared_ref=prepared,
            candidate_proof_ref=proof,
            candidate_params_ref=params,
        ),
    )
    score = put(
        "score.json",
        dict(
            schema_version="s2a.terminal-identity-state-score.v1",
            passed=True,
            complete=True,
            state_gate_passed=True,
            eligible_for_development=True,
            numerical_ref=numerical,
            prepared_ref=prepared,
            candidate_proof_ref=proof,
            candidate_params_ref=params,
            state_gate=dict(
                passed=True,
                checks={k: True for k in gate.CHECKS},
                component_checks={k: True for k in gate.COMPONENTS},
                episode_passes={str(i): True for i in range(6)},
            ),
        ),
    )
    owner = put(
        "orchestration/result.json",
        dict(
            schema_version="s2a.terminal-identity-state-owner.v1",
            passed=True,
            complete=True,
            state_gate_passed=True,
            eligible_for_development=True,
            automatic_retry=False,
            fresh_authorized=False,
            owned_commands=commands,
            gpu_after_ref=gpu,
            numerical_ref=numerical,
            state_score_ref=score,
            prepared_ref=prepared,
        ),
    )
    monkeypatch.setattr(gate, "candidate_head", lambda *a, **k: (object(), params))
    return owner, docs


def test_complete_gate_only_allows_preparation(monkeypatch):
    owner, docs = evidence(monkeypatch)
    _, admission = gate.admit_state(owner, read_json=lambda r: docs[r["path"]])
    assert admission["development_preconditions_passed"]
    assert admission["execution_authorized"] is admission["fresh_authorized"] is False


@pytest.mark.parametrize(
    "fault", ["gate-false", "missing-check", "component", "seed", "exit", "reap", "duplicated-role", "gpu", "params"]
)
def test_incomplete_or_legacy_exception_rejected(monkeypatch, fault):
    owner, docs = evidence(monkeypatch)
    o = docs[owner["path"]]
    score = docs[o["state_score_ref"]["path"]]
    if fault == "gate-false":
        score["state_gate_passed"] = False
    elif fault == "missing-check":
        score["state_gate"]["checks"].pop("all_components_protected")
    elif fault == "component":
        score["state_gate"]["component_checks"]["h8_opening_mae"] = False
    elif fault == "seed":
        score["state_gate"]["checks"]["both_seeds_protected"] = False
    elif fault == "exit":
        docs[o["owned_commands"][0]["exit"]["path"]]["returncode"] = 1
    elif fault == "reap":
        docs[o["owned_commands"][1]["release"]["path"]]["reaped"] = False
    elif fault == "duplicated-role":
        o["owned_commands"][1] = copy.deepcopy(o["owned_commands"][0])
    elif fault == "gpu":
        docs[o["gpu_after_ref"]["path"]]["memory_mib"] = 1
    else:
        score["candidate_params_ref"] = dict(path="different", sha256="b" * 64)
    with pytest.raises(ValueError):
        gate.admit_state(owner, read_json=lambda r: docs[r["path"]])
