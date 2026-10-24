"""Label-free, bounded-storage ledgers for actual clean sampler snapshots.

All source arrays are checked and hashed online. Only hidden/action data and
small timing/condition arrays are retained. Omitted vision values cannot be
reconstructed offline; their hashes and online checks are witnesses, not a
saved full-video trace. The caller owns initial-noise/condition provenance and
qualification of the bridge which produced each :class:`CleanCapturePoint`.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from s2a.wam_clean_capture import CleanCapturePoint

POINT_SCHEMA = "s2a.clean-compact-point.v1"
REQUEST_SCHEMA = "s2a.clean-compact-request.v1"
LEDGER_SCHEMA = "s2a.clean-compact-ledger.v1"
_MASK = np.arange(128) < 10
_SEMANTICS = "ideal FP64 sample_FP32 - actual_sigma * raw_velocity; not scheduler x0"
_SOURCE_KEYS = frozenset(
    (
        "packed_vision",
        "packed_action",
        "time_vision",
        "time_action",
        "condition_vision",
        "condition_action",
        "sample_fp32_flat",
        "hidden",
        "projection_ids",
        "head_base_full128",
        "sample_full128_fp32",
        "raw_velocity_full128",
        "c0_full128",
        "raw_velocity_flat",
        "deployed_velocity_flat",
        "deployed_velocity_full128",
        "clean_residual_active10",
        "velocity_delta_active10",
    )
)
_META_KEYS = frozenset(
    (
        "request_name",
        "noise_seed",
        "purpose",
        "step_index",
        "sigma_actual",
        "network_timestep",
        "action_offset",
        "action_dim_mask",
        "feature_terms",
        "original_dtypes",
        "clean_semantics",
        "returned_original_velocity_object",
    )
)
_ACTIVE = {
    "sample_active10_fp32": "sample_full128_fp32",
    "raw_velocity_active10": "raw_velocity_full128",
    "deployed_velocity_active10": "deployed_velocity_full128",
    "c0_active10": "c0_full128",
}
_SMALL = (
    "hidden",
    "projection_ids",
    "clean_residual_active10",
    "velocity_delta_active10",
    "time_vision",
    "time_action",
    "condition_vision",
    "condition_action",
)
_STORAGE_DTYPES = {
    "torch.bfloat16": np.dtype("float32"),
    "torch.float32": np.dtype("float32"),
    "torch.float64": np.dtype("float64"),
    "torch.int64": np.dtype("int64"),
    "torch.bool": np.dtype("bool"),
    "numpy.float64": np.dtype("float64"),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def _digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ref(path):
    return dict(path=str(Path(path).resolve()), sha256=_sha(path))


def _read(ref):
    _require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "requires an exact path/SHA reference")
    path = Path(ref["path"])
    _require(_sha(path) == ref["sha256"], "compact artifact SHA differs")
    return path


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_canonical(value) + b"\n")
    return _ref(path)


def _hex(value, *, nullable=False):
    return (nullable and value is None) or (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _same(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def _summary(array):
    return dict(
        shape=list(array.shape),
        storage_dtype=array.dtype.str,
        snapshot_bytes=array.nbytes,
        snapshot_sha256=_digest_bytes(memoryview(np.ascontiguousarray(array)).cast("B")),
    )


def _round(value, dtype):
    """NumPy gold for a direct FP64 target-dtype rounding, including BF16 ties."""
    _require(np.isfinite(value).all(), "ideal deployed velocity is nonfinite")
    if dtype == "torch.bfloat16":
        _, exponent = np.frexp(value)
        spacing = np.ldexp(np.ones_like(value), np.maximum(exponent - 8, -133))
        value = np.rint(value / spacing) * spacing
    else:
        _require(dtype == "torch.float32", "only qualified BF16/FP32 velocity is supported")
    with np.errstate(over="ignore"):
        result = value.astype(np.float32)
    _require(np.isfinite(result).all(), "deployed velocity overflowed its target dtype")
    return result


@dataclass(frozen=True)
class CompactRequest:
    """Only online identities; there is deliberately no label or future field."""

    request_name: str
    noise_seed: int
    purpose: str
    round_name: str
    anchor_id: str
    readout_sha256: str | None
    current_image_sha256: str

    def __post_init__(self):
        for name in ("request_name", "round_name", "anchor_id"):
            value = getattr(self, name)
            _require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty")
        _require(type(self.noise_seed) is int and 0 <= self.noise_seed < 2**63, "invalid noise seed")
        _require(self.purpose in ("training", "evaluation"), "purpose must be training or evaluation")
        _require(
            (self.noise_seed in (42, 43)) == (self.purpose == "evaluation"),
            "training noise must exclude evaluation42/43",
        )
        _require(_hex(self.readout_sha256, nullable=True), "readout must have an explicit SHA or be raw None")
        _require(_hex(self.current_image_sha256), "current image must have an explicit SHA")


def _grid(values):
    values = tuple(tuple(row) for row in values)
    _require(len(values) == 35 and all(len(row) == 2 for row in values), "requires the registered35 sigma/time pairs")
    result = []
    for sigma, time in values:
        _require(
            type(sigma) in (int, float)
            and np.isfinite(sigma)
            and 0 < sigma <= 1
            and float(np.float32(sigma)) == sigma
            and type(time) is int
            and time > 0,
            "grid requires actual positive FP32 sigma and integer network time",
        )
        result.append((float(sigma), time))
    _require(all(a[0] > b[0] and a[1] > b[1] for a, b in zip(result, result[1:])), "grid must descend strictly")
    return tuple(result)


def _metadata(meta, request, grid, offset):
    _require(set(meta) == _META_KEYS, "unexpected point metadata; future labels are not accepted")
    _require(
        all(meta[key] == getattr(request, key) for key in ("request_name", "noise_seed", "purpose")),
        "point request/source identity differs",
    )
    step = meta["step_index"]
    _require(type(step) is int and 0 <= step < 35, "invalid actual step index")
    _require(
        type(meta["sigma_actual"]) in (float, int)
        and type(meta["network_timestep"]) is int
        and (meta["sigma_actual"], meta["network_timestep"]) == grid[step],
        "actual sigma/time differs from the registered grid",
    )
    mask = np.asarray(meta["action_dim_mask"])
    _require(mask.dtype == np.bool_ and np.array_equal(mask, _MASK), "requires the full active10 mask")
    _require(type(meta["action_offset"]) is int and meta["action_offset"] == offset, "vision/action offset differs")
    _require(
        tuple(meta["feature_terms"]) == () and meta["clean_semantics"] == _SEMANTICS, "clean feature contract differs"
    )
    _require(type(meta["returned_original_velocity_object"]) is bool, "missing original-return witness")
    dtypes = dict(meta["original_dtypes"])
    _require(
        len(dtypes) == len(meta["original_dtypes"]) and set(dtypes) == _SOURCE_KEYS, "original dtype roster differs"
    )
    return dtypes


def _numeric(arrays, dtypes):
    _require(set(arrays) == _SOURCE_KEYS, "unexpected source array; future labels are not accepted")
    for name, array in arrays.items():
        _require(isinstance(array, np.ndarray) and array.size > 0, f"{name} must be a nonempty array")
        _require(array.dtype == _STORAGE_DTYPES.get(dtypes[name]), f"{name} storage/original dtype differs")
        _require(np.isfinite(array).all(), f"{name} is nonfinite")
        if dtypes[name] == "torch.bfloat16":
            _require(not np.any(array.view(np.uint32) & 0xFFFF), f"{name} is not a lossless BF16 snapshot")


def _actions(compact, meta, raw_dtype):
    """Check fully saved action math; usable independently after source omission."""
    for name, dtype in (
        ("sample_active10_fp32", np.float32),
        ("raw_velocity_active10", np.float32),
        ("deployed_velocity_active10", np.float32),
        ("c0_active10", np.float64),
        ("clean_residual_active10", np.float64),
        ("velocity_delta_active10", np.float64),
    ):
        value = compact[name]
        _require(
            value.shape == (40, 10) and value.dtype == dtype and np.isfinite(value).all(), f"{name} layout differs"
        )
    sample, raw, deployed, c0, residual, delta = (
        compact[key]
        for key in (
            "sample_active10_fp32",
            "raw_velocity_active10",
            "deployed_velocity_active10",
            "c0_active10",
            "clean_residual_active10",
            "velocity_delta_active10",
        )
    )
    sigma = meta["sigma_actual"]
    _require(
        _same(c0, sample.astype(np.float64) - sigma * raw.astype(np.float64)), "C0 does not use the actual FP32 sample"
    )
    expected_delta = -residual / sigma
    _require(
        np.array_equal(delta, expected_delta)
        if meta["returned_original_velocity_object"]
        else _same(delta, expected_delta),
        "clean-to-velocity sign/sigma differs",
    )
    if meta["returned_original_velocity_object"]:
        _require(not np.any(residual) and _same(raw, deployed), "raw/zero returned-object witness differs")
    else:
        gold = _round(raw.astype(np.float64) + delta, raw_dtype)
        _require(_same(gold, deployed), "deployed velocity does not match direct target-dtype rounding")


def _online(point, request, grid, vision_shape):
    _require(isinstance(point, CleanCapturePoint), "sink requires an actual CleanCapturePoint snapshot")
    arrays, meta = point.arrays, point.metadata
    offset = int(np.prod(vision_shape))
    dtypes = _metadata(meta, request, grid, offset)
    _numeric(arrays, dtypes)
    _require(arrays["packed_vision"].shape == vision_shape, "packed vision shape differs from the source contract")
    _require(arrays["hidden"].shape == (40, 1280), "hidden must be40x1280")
    ids = arrays["projection_ids"]
    _require(ids.shape == (40,) and ids.dtype == np.int64 and not np.any(ids), "LIBERO projection0 source differs")
    for name in ("packed_action", "head_base_full128", *tuple(_ACTIVE.values())):
        _require(arrays[name].shape == (40, 128), f"{name} must be40x128")
    for name in ("sample_fp32_flat", "raw_velocity_flat", "deployed_velocity_flat"):
        _require(arrays[name].shape == (offset + 5120,), f"{name} flat source layout differs")
    _require(dtypes["sample_fp32_flat"] == dtypes["sample_full128_fp32"] == "torch.float32", "sampler must be FP32")
    raw_dtype = dtypes["raw_velocity_flat"]
    _require(raw_dtype in ("torch.bfloat16", "torch.float32"), "unqualified raw velocity dtype")
    _require(
        all(
            dtypes[name] == raw_dtype
            for name in ("raw_velocity_full128", "deployed_velocity_flat", "deployed_velocity_full128")
        ),
        "raw/deployed original dtypes differ",
    )
    for flat, full in (
        ("sample_fp32_flat", "sample_full128_fp32"),
        ("raw_velocity_flat", "raw_velocity_full128"),
        ("deployed_velocity_flat", "deployed_velocity_full128"),
    ):
        _require(
            _same(arrays[flat][offset:].reshape(40, 128), arrays[full]), "action slice is not bound to its flat source"
        )
    _require(
        _same(arrays["raw_velocity_flat"][:offset], arrays["deployed_velocity_flat"][:offset]),
        "adaptation changed vision velocity bits",
    )
    for name in ("packed_action", *tuple(_ACTIVE.values())):
        _require(not np.any(arrays[name][:, 10:]), f"{name} inactive118 is not zero")
    _require(
        _same(arrays["raw_velocity_full128"][:, 10:], arrays["deployed_velocity_full128"][:, 10:]),
        "adaptation changed inactive velocity bits",
    )
    packed_dtype = dtypes["packed_action"]
    _require(
        packed_dtype in ("torch.bfloat16", "torch.float32")
        and _same(_round(arrays["sample_full128_fp32"].astype(np.float64), packed_dtype), arrays["packed_action"]),
        "packed action is not the declared rounding of the actual FP32 sample",
    )
    compact = {name: arrays[source][:, :10].copy() for name, source in _ACTIVE.items()}
    compact.update({name: arrays[name].copy() for name in _SMALL})
    compact["action_dim_mask"] = _MASK.copy()
    _actions(compact, meta, raw_dtype)
    if request.readout_sha256 is None:
        _require(
            meta["returned_original_velocity_object"] and not np.any(compact["clean_residual_active10"]),
            "raw request was adapted",
        )
    summaries = {}
    for name, value in arrays.items():
        summaries[name] = dict(_summary(value), original_dtype=dtypes[name])
    return compact, summaries


class CompactCaptureSink:
    """One new directory per request, one small NPZ/JSON per ordered node.

    ``expected_grid`` is the35 actual FP32-sigma/integer-network-time pairs from
    independent runtime qualification, never a grid inferred from integer time.
    Any rejection poisons this sink; no retry or final complete ledger follows.
    Source hashes describe NumPy snapshot bytes (BF16 is losslessly FP32), not
    unobserved Torch storage. ``finish`` does not claim task/model qualification.
    """

    def __init__(self, directory, request, *, expected_vision_shape, expected_grid):
        _require(isinstance(request, CompactRequest), "requires a CompactRequest")
        shape = tuple(expected_vision_shape)
        _require(len(shape) == 5 and all(type(n) is int and n > 0 for n in shape), "requires a frozen5D vision shape")
        self.request, self.grid, self.shape = request, _grid(expected_grid), shape
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        header = dict(
            schema_version=REQUEST_SCHEMA,
            request=asdict(request),
            expected_grid=self.grid,
            expected_vision_shape=shape,
            expected_nodes=35,
            source_hash_scope="C-order NumPy snapshot bytes",
            omitted_vision_reconstructible=False,
            labels_accepted=False,
        )
        self.request_ref = _write(self.directory / "request.json", header)
        self._stream = hashlib.sha256(_canonical(header))
        self._points, self._closed, self._failed = [], False, False

    def __call__(self, point):
        _require(not self._closed and not self._failed, "compact sink is closed or failed")
        try:
            _require(point.metadata["step_index"] == len(self._points), "duplicate or out-of-order step")
            compact, summaries = _online(point, self.request, self.grid, self.shape)
            step = len(self._points)
            path = self.directory / f"step{step:02d}.npz"
            with path.open("xb") as stream:
                np.savez(stream, **compact)
            metadata = dict(point.metadata)
            payload = dict(
                metadata=metadata,
                source_arrays=summaries,
                compact_arrays={name: _summary(value) for name, value in compact.items()},
            )
            next_stream = self._stream.copy()
            next_stream.update(_canonical(payload))
            record = dict(
                schema_version=POINT_SCHEMA,
                request_ref=self.request_ref,
                **payload,
                arrays_ref=_ref(path),
                source_stream_sha256=next_stream.hexdigest(),
                online_checks=dict(
                    vision_velocity_bits_unchanged=True,
                    inactive_velocity_bits_unchanged=True,
                    flat_action_binding=True,
                    active_clean_and_deployment_math=True,
                ),
                omitted_vision_reconstructible=False,
            )
            reference = _write(self.directory / f"step{step:02d}.json", record)
            self._points.append(reference)
            self._stream = next_stream
            return reference
        except Exception:
            self._failed = True
            raise

    def finish(self):
        _require(not self._closed and not self._failed, "compact sink is closed or failed")
        if len(self._points) != 35:
            self._failed = True
            raise ValueError("cannot complete a partial35-node request")
        result = _write(
            self.directory / "ledger.json",
            dict(
                schema_version=LEDGER_SCHEMA,
                complete=True,
                request_ref=self.request_ref,
                points=self._points,
                nodes=35,
                source_stream_sha256=self._stream.hexdigest(),
                omitted_vision_reconstructible=False,
                qualification_claim=False,
                future_labels_read=0,
            ),
        )
        self._closed = True
        return result


def load_compact_point(reference):
    """Return (metadata, read-only compact arrays), rechecking saved action math.

    Full vision invariance remains an online witness. Its omitted values are
    not recreated, rechecked numerically, or silently reported as saved data.
    """
    path = _read(reference)
    record = json.loads(path.read_text())
    _require(
        record["schema_version"] == POINT_SCHEMA and record["omitted_vision_reconstructible"] is False,
        "wrong compact point schema",
    )
    header_path = _read(record["request_ref"])
    _require(header_path.resolve() == path.parent / "request.json", "point request source path differs")
    header = json.loads(header_path.read_text())
    _require(header["schema_version"] == REQUEST_SCHEMA, "wrong compact request schema")
    request = CompactRequest(**header["request"])
    shape, grid = tuple(header["expected_vision_shape"]), _grid(header["expected_grid"])
    meta = record["metadata"]
    dtypes = _metadata(meta, request, grid, int(np.prod(shape)))
    array_path = _read(record["arrays_ref"])
    _require(array_path.resolve() == path.with_suffix(".npz"), "point action source path differs")
    with np.load(array_path, allow_pickle=False) as archive:
        _require(
            set(archive.files) == set(_ACTIVE) | set(_SMALL) | {"action_dim_mask"},
            "compact arrays include unregistered/future values",
        )
        arrays = {name: archive[name].copy() for name in archive.files}
    _require(set(record["compact_arrays"]) == set(arrays), "compact array summary roster differs")
    for name, value in arrays.items():
        _require(_summary(value) == record["compact_arrays"][name], "saved compact array shape/dtype/SHA differs")
        _require(np.isfinite(value).all(), "saved compact array is nonfinite")
        if name in _SMALL:
            summary = record["source_arrays"][name]
            _require(
                _summary(value) == {k: summary[k] for k in _summary(value)},
                "saved array differs from its online source",
            )
    _require(
        arrays["action_dim_mask"].dtype == np.bool_ and np.array_equal(arrays["action_dim_mask"], _MASK),
        "saved action mask differs",
    )
    _require(
        arrays["hidden"].shape == (40, 1280) and np.isfinite(arrays["hidden"]).all(), "saved hidden shape/value differs"
    )
    _require(
        arrays["projection_ids"].shape == (40,)
        and arrays["projection_ids"].dtype == np.int64
        and not np.any(arrays["projection_ids"]),
        "saved projection0 differs",
    )
    _actions(arrays, meta, dtypes["raw_velocity_flat"])
    for array in arrays.values():
        array.setflags(write=False)
    return meta, arrays


def iter_compact_points(ledger_reference):
    """Verify all35 source-summary chain links, then replay bounded NPZ blocks."""
    path = _read(ledger_reference)
    ledger = json.loads(path.read_text())
    _require(
        ledger["schema_version"] == LEDGER_SCHEMA
        and ledger["complete"] is True
        and ledger["nodes"] == len(ledger["points"]) == 35,
        "incomplete compact ledger",
    )
    header_path = _read(ledger["request_ref"])
    _require(header_path.resolve() == path.parent / "request.json", "ledger request source path differs")
    header = json.loads(header_path.read_text())
    digest = hashlib.sha256(_canonical(header))
    for index, reference in enumerate(ledger["points"]):
        point_path = _read(reference)
        _require(point_path.resolve() == path.parent / f"step{index:02d}.json", "ledger step path/order differs")
        record = json.loads(point_path.read_text())
        _require(
            record["request_ref"] == ledger["request_ref"] and record["metadata"]["step_index"] == index,
            "ledger step identity differs",
        )
        digest.update(
            _canonical(
                dict(
                    metadata=record["metadata"],
                    source_arrays=record["source_arrays"],
                    compact_arrays=record["compact_arrays"],
                )
            )
        )
        _require(digest.hexdigest() == record["source_stream_sha256"], "source-summary stream differs")
    _require(digest.hexdigest() == ledger["source_stream_sha256"], "final source-summary stream differs")
    for reference in ledger["points"]:
        yield load_compact_point(reference)
