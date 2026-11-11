"""CPU-side identity and sequencing checks for a causal WAM stream.

The caller must obtain the snapshots from the actual simulator and separately
verify physics, safety, model provenance and worker lifecycle. This module
does not dispatch actions or establish any of those facts by itself.
"""

import copy
import hashlib
import math
import os
import re
import stat

from s2a.infrastructure.repository_paths import repo_path, repository_root

SNAPSHOT_KEYS = {"step", "sim_time_s", "state_sha256"}
REQUEST_KEYS = {
    "version",
    "session",
    "request_id",
    "step",
    "sim_time_s",
    "state_sha256",
    "image",
    "task",
    "seed",
    "model_sha256",
    "denoise_steps",
}
RESPONSE_KEYS = {
    "version",
    "session",
    "request_id",
    "step",
    "sim_time_s",
    "state_sha256",
    "image_sha256",
    "seed",
    "model_sha256",
    "denoise_calls",
    "action_dim_mask",
    "action",
    "normalized_action",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _sha(value):
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _snapshot(value):
    if (
        type(value) is not dict
        or set(value) != SNAPSHOT_KEYS
        or type(value["step"]) is not int
        or value["step"] < 0
        or not _number(value["sim_time_s"])
        or value["sim_time_s"] < 0
        or not _sha(value["state_sha256"])
    ):
        raise ValueError("A current measured step, finite simulation time and state digest are required")
    return dict(value)


def _image_bytes(image, image_root, *, max_image_bytes):
    """Read one regular, registered image without following any path symlink."""
    if type(image) is not dict or set(image) != {"path", "sha256"} or not _sha(image["sha256"]):
        raise ValueError("One image path and SHA-256 are required")
    path = image["path"]
    if (
        type(path) is not str
        or not path
        or path.startswith("/")
        or "\\" in path
        or any(part in ("", ".", "..") for part in path.split("/"))
    ):
        raise ValueError("Image paths must be relative to the registered capture directory")
    if type(max_image_bytes) is not int or not 1 <= max_image_bytes <= 16 * 1024 * 1024:
        raise ValueError("A bounded image size is required")
    root = repository_root()
    capture = repo_path(image_root, root=root)
    if not capture.is_relative_to(root / "outputs") or not capture.is_dir():
        raise ValueError("The capture directory must be under repository outputs")
    common = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    directories = []
    try:
        directories.append(os.open(capture, common | os.O_DIRECTORY))
        parts = path.split("/")
        for part in parts[:-1]:
            directories.append(os.open(part, common | os.O_DIRECTORY, dir_fd=directories[-1]))
        descriptor = os.open(parts[-1], common, dir_fd=directories[-1])
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= max_image_bytes:
                raise ValueError("Current image must be a nonempty bounded regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                data = stream.read(max_image_bytes + 1)
        finally:
            os.close(descriptor)
    finally:
        for directory in reversed(directories):
            os.close(directory)
    if not 0 < len(data) <= max_image_bytes or hashlib.sha256(data).hexdigest() != image["sha256"]:
        raise ValueError("Current image bytes do not match their registered digest")
    return data


def _validate_request(request):
    if (
        type(request) is not dict
        or set(request) != REQUEST_KEYS
        or type(request["version"]) is not int
        or request["version"] != 1
        or type(request["session"]) is not str
        or not 0 < len(request["session"]) <= 128
        or type(request["request_id"]) is not int
        or request["request_id"] < 0
        or type(request["seed"]) is not int
        or request["seed"] < 0
        or not _sha(request["model_sha256"])
        or type(request["denoise_steps"]) is not int
        or not 1 <= request["denoise_steps"] <= 1000
    ):
        raise ValueError("A complete version-one request is required")
    _snapshot({key: request[key] for key in SNAPSHOT_KEYS})
    if type(request["task"]) is not str or not 0 < len(request["task"]) <= 4096:
        raise ValueError("A bounded public task is required")


def model_inputs(request, *, image_root, max_image_bytes=2 * 1024 * 1024):
    """Expose only verified current image bytes and the public task to the model.

    Sampling parameters are fixed by the surrounding registered model service;
    transport metadata, state hashes and future information are not model input.
    """
    _validate_request(request)
    return {
        "image_bytes": _image_bytes(request["image"], image_root, max_image_bytes=max_image_bytes),
        "task": request["task"],
    }


class CausalModelAdmission:
    """Bound the service-side sequence independently of the simulator ledger.

    The service remains responsible for measuring its actual denoise calls and
    for constructing a response from the matching admitted request.
    """

    def __init__(self, *, session, initial, image_root, task, seed, model_sha256, max_requests, denoise_steps=35):
        # Reuse registration checks without sharing mutable session state.
        registration = CausalStreamLedger(
            session=session,
            initial=initial,
            image_root=image_root,
            task=task,
            seed=seed,
            model_sha256=model_sha256,
            max_requests=max_requests,
            denoise_steps=denoise_steps,
        )
        self.session, self.initial, self.image_root = registration.session, registration.current, image_root
        self.task, self.seed, self.model_sha256 = task, seed, model_sha256
        self.max_requests, self.denoise_steps = max_requests, denoise_steps
        self.admitted_requests = 0
        self.phase = "ready"

    def admit(self, request):
        try:
            if self.phase != "ready" or self.admitted_requests >= self.max_requests:
                raise ValueError("One service request at a time within budget is required")
            _validate_request(request)
            index = self.admitted_requests
            if (
                request["session"] != self.session
                or request["request_id"] != index
                or request["step"] != self.initial["step"] + 8 * index
                or not math.isclose(
                    request["sim_time_s"], self.initial["sim_time_s"] + 0.4 * index, rel_tol=0, abs_tol=1e-9
                )
                or request["task"] != self.task
                or request["seed"] != self.seed
                or request["model_sha256"] != self.model_sha256
                or request["denoise_steps"] != self.denoise_steps
            ):
                raise ValueError("Service request differs from the registered sequence and sampling")
            inputs = model_inputs(request, image_root=self.image_root)
            self.admitted_requests += 1
            self.phase = "pending"
            return inputs
        except Exception:
            self.phase = "failed"
            raise

    def complete(self, *, request_id, denoise_calls):
        try:
            if (
                self.phase != "pending"
                or type(request_id) is not int
                or request_id != self.admitted_requests - 1
                or type(denoise_calls) is not int
                or denoise_calls != self.denoise_steps
            ):
                raise ValueError("Actual completion must match the pending request and denoise budget")
            self.phase = "done" if self.admitted_requests == self.max_requests else "ready"
        except Exception:
            self.phase = "failed"
            raise


def _action_matrix(value):
    return (
        type(value) is list
        and len(value) == 40
        and all(
            type(row) is list
            and len(row) == 128
            and all(_number(item) for item in row)
            and all(item == 0 for item in row[10:])
            for row in value
        )
    )


class CausalStreamLedger:
    """Sequence one current observation, one response, and eight completions.

    An invalid call poisons this ledger. The snapshots and 25-sample completion
    receipts are assertions from the owning runtime, not independently audited
    physical evidence. All callers must use a fresh ledger for each evaluation.
    """

    def __init__(self, *, session, initial, image_root, task, seed, model_sha256, max_requests, denoise_steps=35):
        if (
            type(session) is not str
            or not 0 < len(session) <= 128
            or type(task) is not str
            or not 0 < len(task) <= 4096
            or type(seed) is not int
            or seed < 0
            or not _sha(model_sha256)
            or type(max_requests) is not int
            or not 1 <= max_requests <= 75
            or type(denoise_steps) is not int
            or not 1 <= denoise_steps <= 1000
        ):
            raise ValueError("A bounded, explicitly registered session and sampling budget are required")
        self.session = session
        self._current = _snapshot(initial)
        self.image_root = image_root
        self.task = task
        self.seed = seed
        self.model_sha256 = model_sha256
        self.max_requests = max_requests
        self.denoise_steps = denoise_steps
        self.issued_requests = 0
        self.accepted_requests = 0
        self.completed_periods = 0
        self.segment_periods = 0
        self.phase = "ready"
        self._pending = None

    @property
    def current(self):
        """Return a copy so callers cannot silently alter the expected state."""
        return dict(self._current)

    def _fail(self, error):
        self.phase = "failed"
        raise error

    def issue(self, *, before_render, after_render, image):
        try:
            if self.phase != "ready" or self.issued_requests >= self.max_requests:
                raise ValueError("A fresh request requires a completed prior segment and remaining budget")
            if _snapshot(before_render) != self._current or _snapshot(after_render) != self._current:
                raise ValueError("Rendering must preserve the current measured state and simulation time")
            _image_bytes(image, self.image_root, max_image_bytes=2 * 1024 * 1024)
            request = {
                "version": 1,
                "session": self.session,
                "request_id": self.issued_requests,
                **self._current,
                "image": dict(image),
                "task": self.task,
                "seed": self.seed,
                "model_sha256": self.model_sha256,
                "denoise_steps": self.denoise_steps,
            }
            self._pending = copy.deepcopy(request)
            self.issued_requests += 1
            self.phase = "awaiting"
            return copy.deepcopy(request)
        except Exception as error:
            self._fail(error)

    def accept(self, response, *, after_wait):
        try:
            if self.phase != "awaiting" or self._pending is None:
                raise ValueError("Exactly one pending request is required")
            if _snapshot(after_wait) != self._current:
                raise ValueError("Simulation state and time changed while waiting for the model")
            if type(response) is not dict or set(response) != RESPONSE_KEYS:
                raise ValueError("A complete, unextended version-one response is required")
            request = self._pending
            for key in (
                "version",
                "session",
                "request_id",
                "step",
                "sim_time_s",
                "state_sha256",
                "seed",
                "model_sha256",
            ):
                if type(response[key]) is not type(request[key]) or response[key] != request[key]:
                    raise ValueError("Response identity differs from the current request")
            if (
                response["image_sha256"] != request["image"]["sha256"]
                or type(response["denoise_calls"]) is not int
                or response["denoise_calls"] != self.denoise_steps
                or response["action_dim_mask"] != [True] * 10 + [False] * 118
                or any(type(item) is not bool for item in response["action_dim_mask"])
                or not _action_matrix(response["action"])
                or not _action_matrix(response["normalized_action"])
            ):
                raise ValueError("Response image, sampling or finite Action128 contract differs")
            self._pending = None
            self.accepted_requests += 1
            self.segment_periods = 0
            self.phase = "executing"
            return copy.deepcopy(response["action"])
        except Exception as error:
            self._fail(error)

    def complete_period(self, *, before, after, inner_samples):
        try:
            if self.phase != "executing" or self.segment_periods >= 8:
                raise ValueError("A current accepted segment is required")
            before, after = _snapshot(before), _snapshot(after)
            if (
                before != self._current
                or after["step"] != before["step"] + 1
                or not math.isclose(after["sim_time_s"], before["sim_time_s"] + 0.05, rel_tol=0, abs_tol=1e-9)
                or type(inner_samples) is not int
                or inner_samples != 25
            ):
                raise ValueError("A consecutive completed 20 Hz period with 25 inner samples is required")
            self._current = after
            self.completed_periods += 1
            self.segment_periods += 1
            if self.segment_periods == 8:
                self.phase = "done" if self.issued_requests == self.max_requests else "ready"
        except Exception as error:
            self._fail(error)
