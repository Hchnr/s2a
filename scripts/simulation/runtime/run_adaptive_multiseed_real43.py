"""Run one pre-registered real-model replay on a saved development image."""

import argparse

from scripts.simulation.runtime import adaptive_multiseed_real43_admission as admission
from scripts.simulation.runtime import adaptive_multiseed_real43_owner as owner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", choices=("replay", "model"))
    args = parser.parse_args()
    trial = owner.make_trial()
    if args.dry_run:
        print(trial.dry_run(), flush=True)
    elif args.preflight:
        print(trial.preflight(), flush=True)
    elif args.execute:
        owner.execute()
    elif args.worker == "replay":
        from scripts.simulation.runtime.adaptive_multiseed_real43_replay import replay_worker

        replay_worker(output=admission.OUTPUT, loader=owner.replay_worker_inputs)
    else:
        from scripts.wam.tracking.preview.adaptive_multiseed_model_worker import run_model_worker

        run_model_worker(output=admission.OUTPUT, gpu_uuid=admission.GPU, loader=trial.model_worker_inputs)


if __name__ == "__main__":
    main()
