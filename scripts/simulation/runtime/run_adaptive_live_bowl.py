"""Run the registered adaptive bowl preflight, producer or independent checker."""

import argparse

from scripts.simulation.runtime import adaptive_live_bowl_admission as admission
from scripts.simulation.runtime import adaptive_live_bowl_supervisor as owner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--phase", choices=("producer", "checker"))
    group.add_argument("--worker", choices=("producer", "checker", "model"))
    args = parser.parse_args()
    if args.dry_run:
        owner.dry_run()
    elif args.preflight:
        owner.preflight()
    elif args.phase:
        owner.execute(args.phase)
    elif args.worker == "model":
        from scripts.wam.tracking.preview.adaptive_model_worker import run_model_worker

        run_model_worker(output=admission.OUTPUT, gpu_uuid=owner.engine.GPU, loader=owner.model_worker_inputs)
    else:
        from scripts.simulation.runtime.adaptive_live_bowl_worker import simulation_worker

        simulation_worker(args.worker, source_refs=owner.source_refs)


if __name__ == "__main__":
    main()
