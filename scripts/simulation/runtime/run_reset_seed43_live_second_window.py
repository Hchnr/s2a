"""Run the registered seed-43 reset and current-image feedback trial."""

import argparse

from scripts.simulation.runtime.reset_seed43_trial import make_trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", choices=("producer", "checker", "model"))
    args = parser.parse_args()
    trial = make_trial()
    if args.dry_run:
        print(trial.dry_run(), flush=True)
    elif args.preflight:
        print(trial.preflight(), flush=True)
    elif args.phase:
        from scripts.simulation.runtime.reset_seed43_execution import execute

        execute(trial, args.phase)
    elif args.worker == "model":
        from scripts.wam.tracking.preview.reset_seed43_model_worker import run_model_worker

        run_model_worker(output=trial.output, gpu_uuid=trial.gpu_uuid, loader=trial.model_worker_inputs)
    else:
        from scripts.simulation.runtime.reset_seed43_simulation_worker import run_worker

        run_worker(trial, args.worker)


if __name__ == "__main__":
    main()
