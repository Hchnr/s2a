"""Register the reset seed-43 trial with its actual simulation image root."""

from pathlib import Path

from scripts.simulation.runtime import registered_adaptive_trial as registered
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_live_second_admission as previous
from scripts.simulation.runtime import reset_seed43_live_admission as admission

MODULE = "scripts.simulation.runtime.run_reset_seed43_live_second_window"
QA_TARGETS = (
    "tests/research/native_runtime/test_reset_seed43_live_admission.py",
    "tests/research/native_runtime/test_reset_seed43_trial.py",
    "tests/research/native_runtime/test_reset_seed43_simulation_worker.py",
    "tests/research/native_runtime/test_reset_seed43_feedback_audit.py",
    "tests/research/native_runtime/test_reset_seed43_costs.py",
    "tests/research/native_runtime/test_reset_seed43_supervision.py",
    "tests/research/native_runtime/test_reset_seed43_execution.py",
    "tests/research/native_runtime/test_reset_live_two_window.py",
    "tests/research/native_runtime/test_reset_live_two_window_replay_v2.py",
    "tests/research/native_runtime/test_registered_adaptive_trial.py",
    "tests/research/wam/test_reset_seed43_live_request.py",
    "tests/research/wam/test_reset_seed43_model_worker.py",
)


def source_refs():
    """Include historical runtime, corrected checker, and new causal contracts."""
    names = (
        admission.CONFIG,
        admission.PLAN,
        Path(__file__).resolve(),
        prefix.REPO / "scripts/simulation/runtime/reset_seed43_live_admission.py",
        prefix.REPO / "scripts/simulation/runtime/reset_seed43_simulation_worker.py",
        prefix.REPO / "scripts/simulation/runtime/reset_seed43_feedback_audit.py",
        prefix.REPO / "scripts/simulation/runtime/reset_seed43_costs.py",
        prefix.REPO / "scripts/simulation/runtime/reset_seed43_supervision.py",
        prefix.REPO / "scripts/simulation/runtime/reset_seed43_execution.py",
        prefix.REPO / "scripts/simulation/runtime/run_reset_seed43_live_second_window.py",
        prefix.REPO / "scripts/simulation/runtime/registered_adaptive_execution.py",
        prefix.REPO / "src/s2a/research/native_runtime/physical_cost_bounds.py",
        prefix.REPO / "scripts/simulation/runtime/audit_reset_multitask_live_recovery.py",
        prefix.REPO / "scripts/simulation/runtime/registered_adaptive_trial.py",
        prefix.REPO / "scripts/simulation/runtime/registered_model_bundle.py",
        prefix.REPO / "scripts/simulation/runtime/reset_live_two_window_replay_v2.py",
        prefix.REPO / "scripts/wam/tracking/preview/adaptive_multiseed_contract.py",
        prefix.REPO / "scripts/wam/tracking/preview/reset_seed43_live_request.py",
        prefix.REPO / "scripts/wam/tracking/preview/reset_seed43_model_worker.py",
        *(prefix.REPO / path for path in QA_TARGETS),
    )
    return previous.source_refs() + [prefix.reference(path) for path in names]


class RegisteredResetTrial(registered.RegisteredAdaptiveTrial):
    """Reuse the admitted preflight with the reset runtime's later-created directory."""

    def model_bundle(self, inputs):
        return dict(
            input_receipt=self.input_receipt(inputs),
            prepared=inputs["prepared"],
            bridge=inputs["bridge"],
            template=inputs["template"],
            weight_hashes=inputs["weight_hashes"],
            model_registration=self.model_registration(inputs, self.output / "producer/simulation/runtime"),
        )


def make_trial():
    """Create the fixed local trial; this alone starts no model or simulation."""
    return RegisteredResetTrial(
        output=admission.OUTPUT,
        qa=admission.QA,
        dry=admission.DRY,
        module=MODULE,
        gpu_uuid=admission.GPU,
        required_qa_targets=QA_TARGETS,
        load_inputs=admission.load_inputs,
        input_receipt=admission.input_receipt,
        source_refs=source_refs,
        model_registration=admission.model_registration,
    )
