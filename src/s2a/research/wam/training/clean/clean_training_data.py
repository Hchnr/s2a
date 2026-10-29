"""Pure038 identities, row weights and post-capture labels; no file or model I/O.

Callers own evidence admission and phase ordering. In particular, constructing
identities never grants permission to open future labels. Only after the whole
D0/D1 capture and exit audit may a CPU caller supply the measured arrays below.
Historical evidence SHAs and new execution-source SHAs remain separate records.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from s2a.geometry import finite, inverse, matrix_to_rot6d, validate_transform

DOMAIN = "franka_panda__parallel_gripper/eef20/observed_local_delta/libero_closed_pad_observed_endpoint_eef10_v1"
ROOT_SEEDS = (101, 102)
VALIDATION_DEMO_KEYS = frozenset(
    ((0, "demo_16"), (0, "demo_32"), (1, "demo_4"), (1, "demo_27"), (2, "demo_14"), (2, "demo_24"))
)
ROW_ORDER = "round_D0_then_D1/task_index/numeric_demo/anchor_frame/root_101_then_102/step_0_to_34/endpoint_0_to_39"
SEED_ALGORITHM = (
    "UTF8(json.dumps({task:int,demo:canonical_demo_N,anchor_frame:int,root_seed:int},"
    "sort_keys=True,separators=(',',':'),ensure_ascii=True)); "
    "int.from_bytes(SHA256(payload).digest()[:8],'big') & ((1<<63)-1); "
    "reject42/43/collisions_without_retry"
)
ACTIVE_MASK = np.array([True] * 10 + [False] * 118)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _demo_key(task_index, demo):
    _require(type(task_index) is int and task_index in (0, 1, 2), "task_index must be0/1/2")
    _require(isinstance(demo, str) and re.fullmatch(r"demo_(0|[1-9][0-9]*)", demo), "canonical demo_N required")
    return task_index, demo


@dataclass(frozen=True)
class SourceIdentity:
    task_index: int
    demo: str
    initial_state_sha256: str

    def __post_init__(self):
        _demo_key(self.task_index, self.demo)
        _require(
            isinstance(self.initial_state_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", self.initial_state_sha256),
            "actual initial state SHA256 required",
        )

    @property
    def demo_key(self):
        return self.task_index, self.demo


@dataclass(frozen=True)
class AnchorIdentity:
    source: SourceIdentity
    anchor_frame: int

    def __post_init__(self):
        _require(isinstance(self.source, SourceIdentity), "anchor needs a SourceIdentity")
        _require(type(self.anchor_frame) is int and self.anchor_frame >= 0, "anchor_frame must be nonnegative integer")

    def as_dict(self):
        return dict(
            task_index=self.source.task_index,
            demo=self.source.demo,
            initial_state_sha256=self.source.initial_state_sha256,
            anchor_frame=self.anchor_frame,
        )


def _anchor_key(anchor):
    return anchor.source.task_index, int(anchor.source.demo[5:]), anchor.anchor_frame


def _unique_sources(sources, name):
    sources = tuple(sources)
    _require(all(isinstance(source, SourceIdentity) for source in sources), f"{name} needs explicit source identities")
    _require(len({s.demo_key for s in sources}) == len(sources), f"{name} duplicate task/demo")
    _require(len({s.initial_state_sha256 for s in sources}) == len(sources), f"{name} duplicate initial state SHA")
    return sources


def build_training_roster(
    records,
    *,
    validation_sources,
    fresh_sources,
    validation_demo_keys=(),
    excluded_sources=(),
    forbidden_demo_keys=(),
    forbidden_initial_state_sha256=(),
):
    """Build all150 identities from034 case metadata, without opening trajectories.

    ``records`` uses task_index/demo/initial_state_sha256/native_steps/anchor_frames.
    Unknown validation initial hashes are represented by absent identities,
    never fabricated hashes. All six validation demo keys must still be excluded.
    The four native anchors and measured N+40 verification anchor are mandatory.
    """
    validation = _unique_sources(validation_sources, "validation")
    fresh = _unique_sources(fresh_sources, "fresh")
    excluded = _unique_sources(excluded_sources, "excluded")
    validation_keys = {_demo_key(*key) for key in validation_demo_keys}
    _require(validation_keys == VALIDATION_DEMO_KEYS, "all six fixed validation_demo_keys must be explicit")
    _require(all(s.demo_key in validation_keys for s in validation), "validation source is outside fixed six demos")
    _require(
        len(fresh) == 12 and all(sum(s.task_index == t for s in fresh) == 4 for t in range(3)),
        "all12 actual fresh identities, four per task, must be explicitly excluded",
    )
    _require(
        not ({s.demo_key for s in validation} & {s.demo_key for s in fresh})
        and not ({s.initial_state_sha256 for s in validation} & {s.initial_state_sha256 for s in fresh}),
        "validation/fresh identity overlap",
    )
    forbidden = {_demo_key(*key) for key in forbidden_demo_keys}
    forbidden.update(validation_keys)
    held = validation + fresh + excluded
    forbidden.update(s.demo_key for s in held)
    forbidden_hashes = {s.initial_state_sha256 for s in held}
    for digest in forbidden_initial_state_sha256:
        _require(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
            "forbidden initial state SHA256 must be real explicit metadata",
        )
        forbidden_hashes.add(digest)
    records = tuple(records)
    _require(len(records) == 30, "all30 training demos are required")
    sources, anchors = [], []
    for record in records:
        source = SourceIdentity(record["task_index"], record["demo"], record["initial_state_sha256"])
        _require(
            source.demo_key not in forbidden and source.initial_state_sha256 not in forbidden_hashes,
            "training demo/initial identity leaks into an excluded split",
        )
        n = record["native_steps"]
        _require(type(n) is int and 43 <= n <= 600, "native_steps must be43..600")
        expected = [j * (n - 40) // 3 for j in range(4)] + [n + 40]
        _require(record["anchor_frames"] == expected, "034 requires all five anchors including N+40")
        sources.append(source)
        anchors.extend(AnchorIdentity(source, frame) for frame in expected)
    _unique_sources(sources, "training")
    _require(all(sum(s.task_index == t for s in sources) == 10 for t in range(3)), "training needs10 demos per task")
    return tuple(sorted(anchors, key=_anchor_key))


def _noise_json(anchor, root_seed):
    _require(
        isinstance(anchor, AnchorIdentity) and type(root_seed) is int and root_seed in ROOT_SEEDS,
        "noise requires an admitted anchor and root101/102",
    )
    return json.dumps(
        dict(
            task=anchor.source.task_index,
            demo=anchor.source.demo,
            anchor_frame=anchor.anchor_frame,
            root_seed=root_seed,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def derive_noise_seed(anchor, root_seed):
    """Derive once from the documented identity; never retry a forbidden result."""
    payload = _noise_json(anchor, root_seed).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class NoiseRequest:
    anchor: AnchorIdentity
    root_seed: int
    sampling_seed: int
    canonical_json: str
    identity_sha256: str

    def as_dict(self):
        return dict(
            **self.anchor.as_dict(),
            root_seed=self.root_seed,
            sampling_seed=self.sampling_seed,
            canonical_json=self.canonical_json,
            identity_sha256=self.identity_sha256,
        )


def noise_roster(anchors):
    """Enumerate the complete300 actual seeds and reject aliases/leakage."""
    anchors = tuple(anchors)
    _require(len(anchors) == 150 and all(isinstance(a, AnchorIdentity) for a in anchors), "all150 anchors required")
    _require(len(set(anchors)) == 150, "duplicate anchor identity")
    sources = _unique_sources(set(a.source for a in anchors), "noise training")
    _require(
        len(sources) == 30 and all(sum(s.task_index == t for s in sources) == 10 for t in range(3)),
        "noise roster requires30 demos,10 per task",
    )
    _require(
        all(sum(a.source == s for a in anchors) == 5 for s in sources), "noise roster requires five anchors per demo"
    )
    result, seen = [], set()
    for anchor in sorted(anchors, key=_anchor_key):
        for root in ROOT_SEEDS:
            canonical = _noise_json(anchor, root)
            seed = derive_noise_seed(anchor, root)
            _require(
                type(seed) is int and 0 <= seed < 2**63 and seed not in (42, 43) and seed not in seen,
                "derived noise seed collides or leaks evaluation42/43; no retry",
            )
            seen.add(seed)
            result.append(NoiseRequest(anchor, root, seed, canonical, hashlib.sha256(canonical.encode()).hexdigest()))
    return tuple(result)


def _rounds(rounds):
    rounds = tuple(rounds)
    _require(rounds in (("D0",), ("D0", "D1")), "fit must use complete D0 or the D0/D1 union")
    return rounds


def row_count(*, rounds=("D0",)):
    return len(_rounds(rounds)) * 420000


def row_weights(start, stop, *, rounds=("D0",)):
    """Return global weights for one contiguous block, never block-normalized."""
    total = row_count(rounds=rounds)
    _require(type(start) is type(stop) is int and 0 <= start < stop <= total, "invalid complete-stream row range")
    return np.full(stop - start, 1.0 / total, dtype=np.float64)


def row_identity(index, requests, *, rounds=("D0",)):
    """Decode a stream offset to the frozen round/request/step/endpoint identity."""
    rounds = _rounds(rounds)
    _require(type(index) is int and 0 <= index < 420000 * len(rounds), "row index outside complete training stream")
    requests = _checked_requests(tuple(requests))
    round_index, within = divmod(index, 420000)
    request_index, within = divmod(within, 1400)
    step, endpoint = divmod(within, 40)
    return dict(
        round=rounds[round_index], **requests[request_index].as_dict(), step_index=step, endpoint_index=endpoint
    )


@lru_cache(maxsize=4)
def _checked_requests(requests):
    _require(
        len(requests) == 300
        and all(isinstance(r, NoiseRequest) for r in requests)
        and requests == noise_roster([r.anchor for r in requests[::2]]),
        "row requests differ from the complete ordered noise roster",
    )
    return requests


def _mask(value, shape, name):
    value = np.asarray(value)
    _require(value.dtype == np.bool_ and value.shape == shape, f"{name} must be an explicit bool array{shape}")
    return value


def post_capture_labels(
    source_poses,
    openings,
    sim_times,
    *,
    source_from_tcp,
    actual_source_site_poses,
    actual_eef_body_poses,
    state_mask,
    action_valid,
    action_mask,
):
    """Rebuild one measured H40 window, including terminal motion, after capture.

    No endpoint is synthesized: the hold label is the actual measured drift.
    Source position is grip SITE; source rotation is EEF BODY, not controller SITE.
    """
    source = validate_transform(source_poses)
    site, body = validate_transform(actual_source_site_poses), validate_transform(actual_eef_body_poses)
    h = validate_transform(source_from_tcp)
    _require(
        source.shape == site.shape == body.shape == (41, 4, 4) and h.shape == (4, 4), "need41 actual endpoint poses"
    )
    _require(
        np.allclose(source[:, :3, 3], site[:, :3, 3], atol=1e-10, rtol=0)
        and np.allclose(source[:, :3, :3], body[:, :3, :3], atol=1e-10, rtol=0),
        "source must use actual grip-site position and EEF BODY rotation",
    )
    opening, times = finite(openings, (41,)), finite(sim_times, (41,))
    _require(np.all((opening >= 0) & (opening <= 1)), "measured opening must be in[0,1]")
    _require(np.allclose(np.diff(times), 0.05, atol=1e-9, rtol=0), "actual endpoint interval must be0.05seconds")
    _require(
        np.array_equal(_mask(state_mask, (41, 128), "state_mask"), np.broadcast_to(ACTIVE_MASK, (41, 128))),
        "all41 measured states need active10 masks",
    )
    _require(_mask(action_valid, (40,), "action_valid").all(), "all40 outgoing measured labels must be valid")
    _require(
        np.array_equal(_mask(action_mask, (40, 128), "action_mask"), np.broadcast_to(ACTIVE_MASK, (40, 128))),
        "all40 labels require active10 only, with no invalid terminal padding",
    )
    tcp = validate_transform(source @ h)
    delta = inverse(tcp[:-1]) @ tcp[1:]
    action = np.zeros((40, 128), dtype=np.float64)
    action[:, :3] = delta[:, :3, 3]
    action[:, 3:9] = matrix_to_rot6d(delta[:, :3, :3])
    action[:, 9] = opening[1:]
    return dict(
        physical_action128=action,
        physical_active10=action[:, :10].copy(),
        tcp_pose=tcp,
        delta_pose=delta,
        sim_times=times.copy(),
        action_valid=np.ones(40, dtype=bool),
    )


def normalize_active10(physical_action128, *, transform, domain, normalization_id, projection_id):
    """Apply normalization31 with separate FP32 multiply/add, then masked clipping.

    Caller supplies the frozen actual transform and its domain identity. The
    returned SHA binds canonical JSON values, not an upstream file declaration.
    Clip counts are per active slot and include every row; no rows are removed.
    """
    _require(
        domain == DOMAIN
        and type(normalization_id) is int
        and normalization_id == 31
        and type(projection_id) is int
        and projection_id == 0,
        "full LIBERO domain/normalization31/projection0 required",
    )
    physical = finite(physical_action128, (40, 128))
    _require(not physical[:, 10:].any(), "physical inactive118 slots must be zero")
    _require(
        set(transform) == {"normalization_mask", "scale", "bias", "clip_min", "clip_max"}, "transform fields differ"
    )
    mask = _mask(transform["normalization_mask"], (128,), "normalization_mask")
    _require(not mask[10:].any(), "normalization31 cannot normalize inactive118")
    scale, bias = finite(transform["scale"], (128,)), finite(transform["bias"], (128,))
    low, high = transform["clip_min"], transform["clip_max"]
    _require(
        np.ndim(low) == np.ndim(high) == 0 and np.isfinite([low, high]).all() and low < high,
        "finite increasing clip bounds required",
    )
    values = physical.astype(np.float32)
    multiplied = np.multiply(values[:, mask], scale.astype(np.float32)[mask], dtype=np.float32)
    transformed = np.add(multiplied, bias.astype(np.float32)[mask], dtype=np.float32)
    _require(np.isfinite(transformed).all(), "FP32 normalized labels overflowed")
    clipped = np.zeros((40, 128), dtype=bool)
    clipped[:, mask] = (transformed < np.float32(low)) | (transformed > np.float32(high))
    values[:, mask] = np.clip(transformed, np.float32(low), np.float32(high))
    _require(np.isfinite(values).all(), "FP32 normalized labels contain nonfinite active values")
    identity = dict(
        normalization_mask=mask.tolist(),
        scale=scale.tolist(),
        bias=bias.tolist(),
        clip_min=float(low),
        clip_max=float(high),
    )
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return dict(
        normalized_active10=values[:, :10].copy(),
        clipped_count=int(clipped.sum()),
        clipped_per_active_slot=clipped[:, :10].sum(axis=0),
        transform_sha256=digest,
        normalization_id=31,
        projection_id=0,
        domain=DOMAIN,
    )


def clean_targets(normalized_x0, sample_fp32, raw_velocity, actual_sigma, *, velocity_semantics):
    """Use raw C0 for both D0 and W1-generated D1; never subtract W1 twice.

    Input action arrays are explicitly active10. Origin cannot be inferred from
    numbers alone: caller must bind raw_velocity to the saved raw capture field.
    """
    _require(velocity_semantics == "raw", "target C0 requires original raw velocity, never deployed velocity")
    x0, sample = np.asarray(normalized_x0), np.asarray(sample_fp32)
    _require(
        x0.dtype == sample.dtype == np.float32 and x0.shape == sample.shape == (40, 10),
        "normalized x0 and actual sampler sample must be FP32 active10",
    )
    x0, sample, raw = finite(x0, (40, 10)), finite(sample, (40, 10)), finite(raw_velocity, (40, 10))
    _require(
        type(actual_sigma) is float
        and np.isfinite(actual_sigma)
        and actual_sigma > 0
        and float(np.float32(actual_sigma)) == actual_sigma,
        "sigma must be actual positive observed FP32 scalar",
    )
    c0 = sample - actual_sigma * raw
    _require(np.isfinite(c0).all() and np.isfinite(x0 - c0).all(), "raw C0 or clean target overflowed")
    return dict(raw_c0_active10=c0, target_clean_residual=x0 - c0)
