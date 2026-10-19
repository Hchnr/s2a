"""Read checkpoint cursors and historical source without constructing a loader."""

import argparse
import ast
import hashlib
import io
import json
import pickle
import subprocess
from collections import Counter
from pathlib import Path

import yaml

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f"cursor contains non-primitive value: {module}.{name}")


def function_hash(source, qualified_name):
    node = ast.parse(source)
    for name in qualified_name.split("."):
        node = next(
            item for item in node.body if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name == name
        )
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    out = root / "training_history"
    out.mkdir(exist_ok=False)
    current = Path(c["checkpoints"][-1]["run"])
    run_config = yaml.safe_load((current / "config.yaml").read_text())
    metadata_path = current / "wandb/latest-run/files/wandb-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    training_commit = metadata["git"]["commit"]
    model_file = "cosmos_framework/model/generator/omni_mot_model.py"
    data_file = "cosmos_framework/data/generator/action/unified_action128.py"
    flow_file = "cosmos_framework/model/generator/algorithm/loss/flow_matching.py"
    targets = {
        model_file: [
            "OmniMoTModel." + x
            for x in (
                "_prepare_training_data",
                "_add_noise_to_input",
                "_compute_losses",
                "_get_velocity",
                "_tokenize_captions",
                "get_data_and_condition",
            )
        ],
        data_file: ["render_caption", "UnifiedAction128Dataset.get_item", "_copy_resized_camera"],
        flow_file: ["compute_flow_matching_loss"],
    }
    parity = []
    for path, functions in targets.items():
        historical = subprocess.check_output(
            ["git", "-C", c["cosmos_repo"], "show", f"{training_commit}:{path}"], text=True
        )
        present = (Path(c["cosmos_repo"]) / path).read_text()
        for name in functions:
            old, new = function_hash(historical, name), function_hash(present, name)
            parity.append(
                dict(file=path, function=name, historical_sha256=old, current_sha256=new, ast_identical=old == new)
            )
    checkpoints = []
    for ck in c["checkpoints"]:
        checkpoint = Path(ck["run"]) / "checkpoints" / ck["iteration"]
        rows = []
        for path in sorted((checkpoint / "dataloader").glob("rank_*.pkl")):
            data = PrimitiveUnpickler(io.BytesIO(path.read_bytes())).load()
            rows.append(
                dict(
                    rank=int(path.stem.split("_")[1]),
                    global_id=data.get("global_id"),
                    cursor={k: v for k, v in data.items() if "libero" in k},
                    sha256=sha256(path),
                    modified_ns=path.stat().st_mtime_ns,
                )
            )
        newest = max((r["global_id"] for r in rows), default=None)
        active = [r for r in rows if r["global_id"] == newest]
        result = dict(
            checkpoint=ck,
            completion_sha256=sha256(checkpoint / ".save_complete"),
            model_metadata_sha256=sha256(checkpoint / "model/.metadata"),
            cursor_global_id_counts=dict(Counter(str(r["global_id"]) for r in rows)),
            max_cursor_global_id=newest,
            max_cursor_ranks=[r["rank"] for r in active],
            cursors=rows,
        )
        checkpoints.append(result)
    write_json(
        out / "result.json",
        dict(
            code=code_fingerprint(),
            training_commit=training_commit,
            training_commit_evidence=str(metadata_path),
            metadata_sha256=sha256(metadata_path),
            current_cosmos_commit=subprocess.check_output(
                ["git", "-C", c["cosmos_repo"], "rev-parse", "HEAD"], text=True
            ).strip(),
            training_validation_enabled=run_config["trainer"]["run_validation"],
            historical_function_parity=parity,
            checkpoints=checkpoints,
            interpretation=[
                "Cursor epochs show substantial LIBERO traversal, not an episode-level consumed-sample ledger.",
                "Ranks with older global_id are retained files; do not count them as active ranks or infer mixed model weights.",
                "Study-heldout episodes are not certified unseen by pretraining; do not claim generalization to unseen training data.",
                "Historical commit is from the latest recorded W&B launch, not a per-tensor proof of every weight's complete history.",
            ],
        ),
    )
    print(
        json.dumps(
            dict(functions=len(parity), identical=sum(r["ast_identical"] for r in parity), checkpoints=len(checkpoints))
        )
    )


if __name__ == "__main__":
    main()
