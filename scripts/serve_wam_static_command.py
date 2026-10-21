"""022 unchanged WAM preparation with a persistent actual-caption ledger."""

import argparse
import json
import os
import types
from pathlib import Path

from run_wam_short_replan import verify_hashes, write_new

from s2a.artifacts import REPO, sha256
from s2a.wam_runtime import install_active_slot_mask, load_service


def align_empty_plan(batch, *, expanded):
    """Retain only the baseline's empty schema placeholder, never a plan."""
    if len(batch["ai_caption"]) != 1:
        raise ValueError("022 requires exactly one caption")
    caption = json.loads(batch["ai_caption"][0])
    if "current_subtask" in caption or caption.get("task_plan", []) != []:
        raise ValueError("022 forbids demonstration plans or current subtasks")
    if not expanded:
        if caption.get("task_plan") != []:
            raise ValueError("022 baseline lacks the registered empty plan placeholder")
        return False
    changed = "task_plan" not in caption
    caption["task_plan"] = []
    batch["ai_caption"] = [json.dumps(caption, ensure_ascii=False)]
    return changed


def install_ledger(service, protocol, directory):
    """Bind the actual prepare→predict handoff without changing either value."""
    original_prepare, original_predict = service.prepare, service.predict
    ordered = protocol["requests"]
    pending, count = {}, 0
    poisoned = False

    def prepare(self, request, output_dir):
        nonlocal poisoned
        if poisoned or pending or count >= len(ordered):
            raise ValueError("022 poisoned, overlapping or excess request; no retry")
        expected = ordered[count]
        if count >= 48:
            gate = json.loads((directory.parent / "reproduction_admission.json").read_text())
            if not gate["passed"] or gate["verified_queries"] != 48:
                raise ValueError("full48 exact reproduction admission is required")
        incoming = request.model_dump(mode="json", exclude_unset=True)
        if incoming != expected["request"]:
            poisoned = True
            raise ValueError("022 request differs from its fixed ordered input")
        prepared, batch = original_prepare(request, output_dir)
        if prepared.action_dim_mask != [True] * 10 + [False] * 118:
            poisoned = True
            raise ValueError("022 requires the ten active LIBERO slots")
        aligned = align_empty_plan(batch, expanded=expected["mode"] == "expanded")
        receipt = dict(
            name=request.name,
            seed=prepared.seed,
            caption=list(batch["ai_caption"]),
            request=incoming,
            image_sha256=sha256(incoming["vision_path"]),
            active_slot_mask=True,
            actual_action_dim_mask=prepared.action_dim_mask,
            stage="actual_prepare_return_before_predict",
            empty_plan_schema_alignment=aligned,
        )
        write_new(directory / f"{request.name}.prepare.json", receipt)
        pending.update(name=request.name, prepared_id=id(prepared), batch_id=id(batch), receipt=receipt)
        return prepared, batch

    def predict(self, prepared, batch):
        nonlocal count, poisoned
        if (
            poisoned
            or pending.get("prepared_id") != id(prepared)
            or pending.get("batch_id") != id(batch)
            or pending["receipt"]["caption"] != batch["ai_caption"]
        ):
            poisoned = True
            raise ValueError("022 actual prediction handoff changed")
        target = directory / f"{pending['name']}.predict.json"
        write_new(target, dict(pending["receipt"], stage="actual_predict_entry_same_prepared_and_batch"))
        try:
            result = original_predict(prepared, batch)
        except BaseException as exc:
            poisoned = True
            write_new(directory / f"{pending['name']}.error.json", dict(error=repr(exc), automatic_retry=False))
            raise
        count += 1
        pending.clear()
        return result

    service.prepare = types.MethodType(prepare, service)
    service.predict = types.MethodType(predict, service)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    args = parser.parse_args()
    assert Path.cwd().resolve() == REPO
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "7" and os.environ.get("LD_LIBRARY_PATH") == ""
    assert sha256(args.prepared) == args.prepared_sha256
    protocol = json.loads(args.prepared.read_text())
    verify_hashes(protocol["input_sha256"])
    config = json.loads(Path(protocol["configs"]["reproduction"]["path"]).read_text())
    output = Path(protocol["root"]) / "service"
    output.mkdir(exist_ok=False)
    service = load_service(config, "8400_masked", output)
    install_active_slot_mask(service.model, [True] * 10 + [False] * 118)
    ledger = Path(protocol["root"]) / "text_input_receipts"
    ledger.mkdir(exist_ok=False)
    install_ledger(service, protocol, ledger)
    service.info.update(
        port=10047,
        acceptance_intervention=dict(active_slot_mask=True, exact_published_caption=False),
        static_command_protocol=dict(path=str(args.prepared), sha256=args.prepared_sha256),
        actual_caption_ledger=str(ledger),
    )
    import uvicorn
    from cosmos_framework.inference.action_serving import create_app

    verify_hashes(protocol["input_sha256"])
    uvicorn.run(create_app(service), host="127.0.0.1", port=10047, workers=1)


if __name__ == "__main__":
    main()
