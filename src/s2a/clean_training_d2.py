"""056 current-only training roster checks; no model or future label decoding."""

from collections import Counter

from s2a.clean_training_runtime import require, validate_current_request
from s2a.wam_clean_capture import NoiseRoster

SCHEMA = "s2a.clean-d2-current-manifest.v1"
PARAMETER_SHA = "ca01b9d8a49f2f1585c917f492e52a0a89cdb0b447f6bd2beee050ed0d2fccc0"
BUDGET = {
    "model_initializations": 1,
    "requests": 300,
    "denoise_calls": 10500,
    "head_calls": 10500,
    "future_label_reads": 0,
    "fits": 0,
    "physics_steps": 0,
    "fresh_cases": 0,
}


def validate_roster(value, *, validate_request=validate_current_request):
    """Reject extra anchor/entry fields and any roster or noise-purpose drift.

    The default request checker hashes current PNG files and enforces the
    established template/conditioning schema. Tests may replace only that I/O.
    Original D0 request equality is checked separately by the preparer/verifier.
    """
    require(value["schema_version"] == SCHEMA, "056 current manifest schema differs")
    anchors, entries = value["anchors"], value["entries"]
    require(len(anchors) == 150 and len(entries) == 300, "056 full150/300 roster required")
    require(all(set(a) == {"name", "task", "image_ref"} for a in anchors), "056 anchors must be current-only")
    lookup = {a["name"]: a for a in anchors}
    require(len(lookup) == 150, "056 duplicate training anchor")
    seeds = [e["sampling_seed"] for e in entries]
    require(all(type(s) is int and 0 <= s < 2**31 for s in seeds), "056 actual31-bit seeds required")
    roster = NoiseRoster(tuple(seeds))
    require(value["noise_roster"] == {"training_seeds": seeds, "evaluation_seeds": [42, 43]}, "056 seed order differs")
    names = set()
    counts = Counter()
    for index, entry in enumerate(entries):
        require(
            set(entry)
            == {"index", "name", "anchor_name", "sampling_seed", "request", "d0_initial_ref", "d0_request_ref"},
            "056 entry fields differ",
        )
        require(entry["index"] == index and entry["name"] not in names, "056 request identity/order differs")
        require(entry["anchor_name"] == anchors[index // 2]["name"], "056 original two-noise anchor order required")
        anchor = lookup[entry["anchor_name"]]
        roster.admit(entry["sampling_seed"], "training")
        validate_request(
            entry["request"],
            name=entry["name"],
            seed=entry["sampling_seed"],
            image_ref=anchor["image_ref"],
            task=anchor["task"],
            template=value["request_templates"][anchor["task"]],
        )
        names.add(entry["name"])
        counts[entry["anchor_name"]] += 1
    require(set(counts) == set(lookup) and set(counts.values()) == {2}, "056 training coverage differs")
    require(value["candidate_params_ref"]["sha256"] == PARAMETER_SHA, "056 collection policy differs")
    require(value["budget"] == BUDGET and value["execution_authorized"] is False, "056 proposal cannot launch a model")
    return {
        "anchors": len(anchors),
        "requests": len(entries),
        "distinct_training_seeds": len(set(seeds)),
        "future_labels_read": 0,
    }
