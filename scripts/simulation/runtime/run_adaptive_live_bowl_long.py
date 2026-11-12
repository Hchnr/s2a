"""Run the registered longer adaptive development trial and its checker."""

import argparse

from scripts.simulation.runtime import adaptive_live_bowl_long_admission as admission
from scripts.simulation.runtime import adaptive_live_bowl_long_owner as owner
from scripts.simulation.runtime import registered_adaptive_execution as execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", choices=("producer", "checker", "model"))
    args = parser.parse_args()
    trial = owner.make_trial()
    if args.dry_run:
        print(trial.dry_run(), flush=True)
    elif args.preflight:
        print(trial.preflight(), flush=True)
    elif args.phase:
        execution.execute(trial, args.phase)
    elif args.worker == "model":
        from scripts.wam.tracking.preview.adaptive_model_worker import run_model_worker

        run_model_worker(output=admission.OUTPUT, gpu_uuid=trial.gpu_uuid, loader=trial.model_worker_inputs)
    else:
        from scripts.simulation.runtime.managed_adaptive_simulation import run_adaptive_simulation_worker

        run_adaptive_simulation_worker(
            args.worker,
            output=admission.OUTPUT,
            load_inputs=admission.load_inputs,
            input_receipt=admission.input_receipt,
            source_refs=owner.source_refs,
            model_registration=admission.model_registration,
        )


if __name__ == "__main__":
    main()
