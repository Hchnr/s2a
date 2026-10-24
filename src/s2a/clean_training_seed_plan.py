"""Explicit040 seed overlay for the immutable038 63-bit training identities.

This module performs no I/O, model work, RNG operations or label decoding. A
caller must register and verify the returned plan before authorizing execution.
The original prepared object and every original request identity remain intact.
"""

import copy
import hashlib
import json

DOMAIN = "s2a.clean-training-request-seed31.v1"
ALGORITHM = (
    "payload=json.dumps([DOMAIN,anchor_name,root_seed,original_sampling_seed],"
    "ensure_ascii=True,separators=(',',':')).encode(); "
    "sampling_seed=int.from_bytes(SHA256(payload).digest()[:4],'big') & ((1<<31)-1); "
    "DOMAIN='s2a.clean-training-request-seed31.v1'; reject collisions or42/43; no retry"
)
SCHEMA = "s2a.clean-training-seed-plan.v1"


def _require(value, message):
    if not value:
        raise ValueError(message)


def _reference(value):
    _require(
        isinstance(value, dict)
        and set(value) == {"path", "sha256"}
        and isinstance(value["path"], str)
        and value["path"].startswith("/")
        and isinstance(value["sha256"], str)
        and len(value["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in value["sha256"]),
        "seed plan requires an exact absolute path/SHA reference",
    )


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def build_seed_plan(prepared_ref, prepared):
    """Derive all300 actual31-bit seeds once; fail without retry on conflicts."""
    _reference(prepared_ref)
    _require("training_seed_plan_ref" not in prepared, "seed overlay requires the original unmodified preparation")
    anchors, requests = prepared["training_anchors"], prepared["training_requests"]
    _require(isinstance(anchors, list) and len(anchors) == 150, "seed plan requires all150 ordered anchors")
    names = [a["name"] for a in anchors]
    _require(all(isinstance(n, str) and n for n in names) and len(set(names)) == 150, "unique150 anchor identities")
    _require(isinstance(requests, list) and len(requests) == 300, "seed plan requires all300 original requests")
    old_seeds, request_names, entries, actual_seeds = [], set(), [], set()
    for index, entry in enumerate(requests):
        old_seed = entry["sampling_seed"]
        _require(
            type(entry["index"]) is int
            and entry["index"] == index
            and entry["anchor_name"] == names[index // 2]
            and type(entry["root_seed"]) is int
            and entry["root_seed"] == (101, 102)[index % 2],
            "fixed300 request index/anchor/root101102 order differs",
        )
        _require(
            isinstance(entry["name"], str) and entry["name"] and entry["name"] not in request_names,
            "unique300 request names required",
        )
        _require(
            type(old_seed) is int
            and 0 <= old_seed < 2**63
            and old_seed not in (42, 43)
            and type(entry["request"]["seed"]) is int
            and entry["request"]["seed"] == old_seed,
            "original63-bit request seed mismatch or evaluation overlap",
        )
        request_names.add(entry["name"])
        old_seeds.append(old_seed)
        payload = json.dumps(
            [DOMAIN, entry["anchor_name"], entry["root_seed"], old_seed], ensure_ascii=True, separators=(",", ":")
        ).encode()
        digest = hashlib.sha256(payload).digest()
        actual = int.from_bytes(digest[:4], "big") & ((1 << 31) - 1)
        _require(
            actual not in actual_seeds and actual not in (42, 43),
            "derived31-bit seed collision/evaluation overlap; no retry",
        )
        actual_seeds.add(actual)
        entries.append(
            dict(
                index=index,
                name=entry["name"],
                anchor_name=entry["anchor_name"],
                root_seed=entry["root_seed"],
                original_sampling_seed=old_seed,
                sampling_seed=actual,
                payload_sha256=digest.hex(),
            )
        )
    roster = prepared["noise_roster"]
    _require(
        len(set(old_seeds)) == 300
        and isinstance(roster["training_seeds"], list)
        and all(type(seed) is int for seed in roster["training_seeds"])
        and roster["training_seeds"] == old_seeds
        and roster["evaluation_seeds"] == [42, 43]
        and all(type(seed) is int for seed in roster["evaluation_seeds"]),
        "original noise roster/order differs",
    )
    return dict(
        schema_version=SCHEMA,
        complete=True,
        passed=True,
        prepared_ref=copy.deepcopy(prepared_ref),
        algorithm=ALGORITHM,
        entries=entries,
        evaluation_seeds=[42, 43],
        actual_seed_bits=31,
        original_seed_bits=63,
        model_calls=0,
        fits=0,
        physics_steps=0,
    )


def apply_seed_plan(prepared, prepared_ref, plan, plan_ref):
    """Return a deep copy with only the five explicitly registered seed changes."""
    _reference(plan_ref)
    expected = build_seed_plan(prepared_ref, prepared)
    _require(
        plan == expected and _canonical(plan) == _canonical(expected),
        "seed plan differs from the exact original300-request derivation",
    )
    effective = copy.deepcopy(prepared)
    seeds = []
    for entry, mapping in zip(effective["training_requests"], plan["entries"], strict=True):
        entry["sampling_seed"] = entry["request"]["seed"] = mapping["sampling_seed"]
        seeds.append(mapping["sampling_seed"])
    effective["noise_roster"]["training_seeds"] = seeds
    effective["noise_seed_algorithm"] = ALGORITHM
    effective["training_seed_plan_ref"] = copy.deepcopy(plan_ref)
    return effective
