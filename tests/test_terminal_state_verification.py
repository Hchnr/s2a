"""Complete state math remains unchanged; lifecycle binds the new actual worker."""

import ast
import copy

import pytest

from s2a.artifacts import REPO
from s2a.clean_training_runtime import GPU


def test_complete_generation_audit_preserves_all_existing_numeric_checks():
    def function(filename):
        return next(
            n
            for n in ast.parse((REPO / "scripts" / filename).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "verify_generations"
        )

    class Rename(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(
                    ast.Constant(node.value.replace("probe052", "terminal058").replace("054 ", "Terminal058 ")), node
                )
            return node

    old = function("verify_wam_synchronous_state.py")
    actual = function("verify_wam_terminal_state.py")
    assert ast.dump(Rename().visit(old)) == ast.dump(actual)


def fixture(tmp_path):
    pref, regref, owner = [dict(path=str(tmp_path / (s + ".json")), sha256=s * 64) for s in ("a", "b", "c")]
    prepared = dict(root=str(tmp_path), model_config=dict(model_python="python"))
    actual = dict(candidate_proof_ref=owner)
    refs = {
        name: dict(path=str(tmp_path / (name + ".json")))
        for name in ("worker_launch", "worker_exit", "worker_release", "gpu_after")
    }
    command = ["python", "scripts/sample_wam_terminal_state.py"]
    for name, ref in (("prepared", pref), ("execution-registration", regref), ("candidate-proof", owner)):
        command.extend(["--" + name, ref["path"], "--" + name + "-sha256", ref["sha256"]])
    command.extend(["--gpu-uuid", GPU, "--seed", "42"])
    docs = dict(
        worker_launch=dict(
            child=dict(pid=11, start_ticks=22),
            selected_environment=dict(CUDA_VISIBLE_DEVICES=GPU, LD_LIBRARY_PATH=""),
            command=command,
        ),
        worker_exit=dict(returncode=0, actual_wait=True, automatic_retry=False),
        worker_release=dict(
            pid=11,
            start_ticks=22,
            reaped=True,
            returncode=0,
            signal_sent=False,
            original_error=None,
            only_owned_child_signalled=True,
        ),
        gpu_after=dict(released=True, uuid=GPU, memory_mib=0, compute_apps=[], device_minor=6, visible_gpu=2),
    )
    return pref, regref, prepared, actual, refs, docs


def verify_fixture(values):
    import verify_wam_terminal_state as verifier

    pref, regref, prepared, actual, refs, docs = values

    class Evidence:
        def json(self, ref):
            return docs[next(k for k, v in refs.items() if v == ref)]

    verifier.lifecycle(Evidence(), refs, pref, regref, actual, prepared)


def test_new_worker_normal_lifecycle_contract(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    verify_fixture(fixture(tmp_path))


@pytest.mark.parametrize(
    "fault", ["wait", "reap", "signal", "birth", "old-worker", "old-candidate", "cuda", "gpu-busy", "escaped-receipt"]
)
def test_wrong_or_partial_lifecycle_is_never_full_qualification(tmp_path, monkeypatch, fault):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    values = list(fixture(tmp_path))
    refs, docs = values[-2:]
    if fault == "wait":
        docs["worker_exit"]["actual_wait"] = False
    elif fault == "reap":
        docs["worker_release"]["reaped"] = False
    elif fault == "signal":
        docs["worker_release"]["signal_sent"] = True
    elif fault == "birth":
        docs["worker_release"]["start_ticks"] += 1
    elif fault == "old-worker":
        docs["worker_launch"]["command"][1] = "scripts/sample_wam_synchronous_state.py"
    elif fault == "old-candidate":
        command = docs["worker_launch"]["command"]
        command[command.index("--candidate-proof")] = "--candidate-fit"
    elif fault == "cuda":
        docs["worker_launch"]["selected_environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    elif fault == "gpu-busy":
        docs["gpu_after"]["memory_mib"] = 1
    else:
        refs["worker_exit"] = copy.deepcopy(refs["worker_exit"])
        refs["worker_exit"]["path"] = str(tmp_path.parent / "escaped.json")
    with pytest.raises(ValueError):
        verify_fixture(values)
