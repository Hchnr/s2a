"""Exercise actual aggregate dispatch and prove the saved sampler body unchanged."""

import ast
import builtins
import sys

import pytest

from s2a.artifacts import REPO
from s2a.terminal_state_execution import GPU, select_readout


def tail(path):
    run = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "run")
    start = next(
        i
        for i, n in enumerate(run.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "bridge" for t in n.targets)
    )
    return ast.Module(body=run.body[start:], type_ignores=[])


def test_sampler_body_changes_only_variant_name_and_dispatch():
    old_dispatch = ast.parse('dict(raw=None, zero=zero, w2=head)[entry["variant"]]', mode="eval").body
    new_dispatch = ast.parse('select_readout(entry["variant"], zero=zero, candidate=head)', mode="eval").body

    class Expected(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(node.value.replace("probe052", "terminal058")), node)
            return node

        def visit_Subscript(self, node):
            if ast.dump(node) == ast.dump(old_dispatch):
                return ast.copy_location(new_dispatch, node)
            return self.generic_visit(node)

    original = tail(REPO / "scripts/sample_wam_synchronous_state.py")
    actual = tail(REPO / "scripts/sample_wam_terminal_state.py")
    assert ast.dump(Expected().visit(original)) == ast.dump(actual)


def test_actual_dispatch_expression_covers_all99_rows_without_old_key_error():
    tree = ast.parse((REPO / "scripts/sample_wam_terminal_state.py").read_text())
    assignment = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "readout" for t in n.targets)
    )
    expression = compile(ast.Expression(assignment.value), "aggregate-dispatch", "eval")
    head, zero = object(), object()
    variants = ["zero"] * 3 + [v for _ in range(48) for v in ("raw", "terminal058")]
    chosen = [
        eval(expression, {"select_readout": select_readout, "entry": {"variant": variant}, "head": head, "zero": zero})
        for variant in variants
    ]
    assert sum(v is head for v in chosen) == sum(v is None for v in chosen) == 48
    assert sum(v is zero for v in chosen) == 3
    with pytest.raises(ValueError, match="unregistered"):
        select_readout("probe052", zero=zero, candidate=head)


def test_actual_dry_cli_requires_new_owner_identity_and_never_imports_model(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import sample_wam_terminal_state as worker

    calls = []
    monkeypatch.setattr(worker, "admit", lambda *a, **k: ({}, {}, {}))
    monkeypatch.setattr(worker, "admit_sampling", lambda *a: calls.append(a))
    monkeypatch.setattr(worker.np, "load", lambda *a, **k: pytest.fail("dry decoded arrays"))
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("cosmos_framework"):
            pytest.fail("dry imported model")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    arguments = ["sample_wam_terminal_state.py"]
    for name in ("prepared", "execution-registration", "candidate-proof"):
        arguments.extend(
            ["--" + name, str(REPO / "outputs/test" / (name + ".json")), "--" + name + "-sha256", "1" * 64]
        )
    arguments.extend(["--gpu-uuid", GPU, "--seed", "42", "--dry"])
    monkeypatch.setattr(sys, "argv", arguments)
    worker.main()
    assert len(calls) == 1 and calls[0][-1]["path"].endswith("candidate-proof.json")
    assert '"model_initializations": 0' in capsys.readouterr().out


def test_scorer_rejects_old_qualified_receipt_before_any_future_access(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import score_wam_terminal_state as scorer

    pref = {"path": "prepared", "sha256": "1" * 64}
    receipt = dict(
        schema_version="s2a.synchronous-state-continuation-audit.v1",
        passed=True,
        complete=True,
        numerical_contract_passed=True,
        lifecycle_passed=True,
        prepared_ref=pref,
    )
    calls = []

    def read(ref):
        calls.append(ref)
        return receipt

    monkeypatch.setattr(scorer, "read", read)
    monkeypatch.setattr(scorer.np, "load", lambda *a, **k: pytest.fail("old receipt opened future labels"))
    with pytest.raises(ValueError, match="terminal058 numerical"):
        scorer.score(pref, "old-receipt", tmp_path / "score.json")
    assert calls == ["old-receipt"] and not (tmp_path / "score.json").exists()


def test_new_owner_command_parses_with_actual_worker_cli(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import sample_wam_terminal_state as worker
    from run_wam_terminal_state import worker_command

    def ref(name):
        return dict(path=str(REPO / "outputs/test" / (name + ".json")), sha256="1" * 64)

    pref, regref, candidate = ref("prepared"), ref("registration"), ref("candidate-proof")
    command = worker_command(
        dict(model_config=dict(model_python="python"), candidate_proof_ref=candidate), pref, regref
    )
    received = []
    monkeypatch.setattr(worker, "admit", lambda *a, **k: ({}, {}, {}))
    monkeypatch.setattr(worker, "admit_sampling", lambda *a: received.append(a))
    monkeypatch.setattr(sys, "argv", command[1:] + ["--dry"])
    worker.main()
    assert received == [({}, pref, regref, candidate)]
    assert '"requests": 0' in capsys.readouterr().out
