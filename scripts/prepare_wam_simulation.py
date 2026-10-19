"""Freeze official demonstration initial states without using model outcomes."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.libero_simulation import configure_simulator, sha256_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, out = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    dest = out / "simulation"
    dest.mkdir(exist_ok=False)
    template = json.loads(Path(c["policy_template"]).read_text())
    benchmark, _, _ = configure_simulator(template, dest)
    suite = benchmark.get_benchmark_dict()[template["task_suite"]]()
    tasks = {suite.get_task(i).language: i for i in range(suite.n_tasks)}
    rng = np.random.default_rng(args.seed)
    records = []
    for task_index, task in enumerate(c["tasks"]):
        filename = task.replace(" ", "_") + "_demo.hdf5"
        source = out / "assets" / filename
        if task_index == 0:
            source = Path("/share/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim/data") / filename
        with h5py.File(source) as data:
            old_tuning = {"demo_27", "demo_7", "demo_9", "demo_15", "demo_0"} if task_index == 0 else set()
            demos = sorted(set(data["data"]) - old_tuning, key=lambda x: int(x.split("_")[-1]))
            selected = rng.choice(demos, size=c["online_initial_states_per_task"], replace=False)
            for initial_index, demo in enumerate(selected):
                episode = data["data"][demo]
                initial = np.asarray(episode["states"][0], dtype=float)
                native_path = dest / f"task{task_index}_init{initial_index}_expert.npy"
                np.save(native_path, episode["actions"][:])
                records.append(
                    dict(
                        name=f"task{task_index}_init{initial_index}",
                        task_index=task_index,
                        task=task,
                        task_id=tasks[task],
                        initial_index=initial_index,
                        demo=str(demo),
                        source_hdf5=str(source),
                        source_hdf5_sha256=sha256(source),
                        episode=str(demo),
                        model_xml=episode.attrs["model_file"],
                        initial_state=initial,
                        initial_state_sha256=sha256_bytes(initial.tobytes()),
                        expert_path=str(native_path),
                        expert_sha256=sha256(native_path),
                        pretraining_consumption="unknown; initial states excluded from prior controller tuning",
                    )
                )
    write_json(dest / "fixtures.json", dict(code=code_fingerprint(), config=c, records=records))
    print(json.dumps([dict(name=r["name"], task_id=r["task_id"], demo=r["demo"]) for r in records]))


if __name__ == "__main__":
    main()
