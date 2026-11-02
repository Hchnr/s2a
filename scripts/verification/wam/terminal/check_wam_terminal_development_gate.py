"""Check actual060 ->061 preparation gate, with JSON-only evidence reads."""

import os
from pathlib import Path

from s2a.research.wam.tracking.runtime.terminal_development_gate import ROOT, admit_state
from s2a.research.wam.training.clean.clean_training_runtime import publish, reference
from scripts.verification.wam.training.verify_wam_clean_training_capture import Evidence, require


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    e = Evidence()
    _, admission = admit_state(reference(ROOT / "orchestration/result.json"), read_json=e.json)
    e.finish()
    root = Path("outputs/terminal_development_preparation_20261025")
    root.mkdir(exist_ok=True)
    publish(
        root / "state_precondition.json",
        dict(
            passed=True,
            complete=True,
            admission=admission,
            input_sha256=e.inputs,
            source_refs=[reference(Path(__file__)), reference(Path("src/s2a/terminal_development_gate.py"))],
            new_model_calls=0,
            new_fits=0,
            physics_steps=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
