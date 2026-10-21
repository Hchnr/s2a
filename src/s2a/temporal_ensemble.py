"""Fixed020 causal equal-weight world references for identical physical endpoints.

No observation, task, object, contact or future measured state is accepted. A
query at physical step q supplies fixed world endpoints q+1 through q+H. The
same contributors and weights govern translation, SO(3), and aperture.
"""

from dataclasses import asdict, dataclass

import numpy as np

from s2a.geometry import validate_transform


@dataclass(frozen=True)
class TemporalEqualConfig:
    horizon_steps: int = 40
    query_interval_steps: int = 8
    identity_atol: float = 1e-12
    minimum_singular_value: float = 1e-6
    method: str = "uniform_chordal_world_same_endpoint"

    def __post_init__(self):
        fixed = dict(
            horizon_steps=40,
            query_interval_steps=8,
            identity_atol=1e-12,
            minimum_singular_value=1e-6,
            method="uniform_chordal_world_same_endpoint",
        )
        if asdict(self) != fixed or any(type(v) is not int for v in (self.horizon_steps, self.query_interval_steps)):
            raise ValueError("020 requires the exact preregistered temporal equal parameters")
        if any(isinstance(v, (bool, np.bool_)) for v in (self.identity_atol, self.minimum_singular_value)):
            raise ValueError("020 numerical thresholds cannot be booleans")


def _step(value):
    if type(value) is not int or value < 0:
        raise ValueError("physical steps must be nonnegative Python integers")
    return value


class CausalTemporalEqualEnsemble:
    """Fail closed on clock/coverage ambiguity; reset between independent cases.

    Pose validation follows the existing geometry.validate_transform contract.
    Inputs and returned diagnostics are copies; an old prediction is never
    recomputed from a newer measured anchor.
    """

    def __init__(self, *, config=None):
        self.config = TemporalEqualConfig() if config is None else config
        if not isinstance(self.config, TemporalEqualConfig):
            raise ValueError("requires an explicit TemporalEqualConfig")
        self.reset()

    def reset(self):
        self._next_step = 0
        self._next_query = 0
        self._references = []

    @property
    def next_step(self):
        return self._next_step

    @property
    def active_query_steps(self):
        return tuple(item[0] for item in self._references)

    def push(self, query_step, world_targets, openings):
        """Register the just-completed query before sampling its physical step."""
        q = _step(query_step)
        if q != self._next_step or q != self._next_query:
            raise ValueError("future, duplicate, skipped or out-of-order query")
        poses = validate_transform(world_targets).copy()
        aperture = np.asarray(openings, dtype=float).copy()
        if poses.ndim != 3 or not 1 <= len(poses) <= self.config.horizon_steps:
            raise ValueError("reference must contain between one and40 world poses")
        if (
            aperture.shape != (len(poses),)
            or not np.isfinite(aperture).all()
            or np.any((aperture < 0) | (aperture > 1))
        ):
            raise ValueError("one finite decoded opening in[0,1] is required per world endpoint")
        active = [item for item in self._references if item[0] + len(item[1]) > q]
        active.append((q, poses, aperture))
        if len(active) > self.config.horizon_steps // self.config.query_interval_steps:
            raise ValueError("more than the registered five active predictions")
        self._references = active
        self._next_query += self.config.query_interval_steps

    def sample(self, physical_step):
        """Return (world_TCP, opening, provenance) for endpoint physical_step+1."""
        k = _step(physical_step)
        if k != self._next_step:
            raise ValueError("sampling must advance exactly one physical step at a time")
        if self._next_query <= k:
            raise ValueError("the scheduled query is missing before sampling")
        endpoint = k + 1
        active = [item for item in self._references if item[0] < endpoint <= item[0] + len(item[1])]
        if not active:
            raise ValueError("no nonexpired prediction covers this physical endpoint")
        count = len(active)
        poses = np.stack([item[1][endpoint - item[0] - 1] for item in active])
        openings = np.asarray([item[2][endpoint - item[0] - 1] for item in active])
        latest, latest_opening = poses[-1], float(openings[-1])
        identical = bool(
            np.all(np.abs(poses - latest) <= self.config.identity_atol)
            and np.all(np.abs(openings - latest_opening) <= self.config.identity_atol)
        )
        determinant, minimum = None, None
        if identical:
            target, opening = latest.copy(), latest_opening
        else:
            mean_rotation = np.mean(poses[:, :3, :3], axis=0)
            u, singular, vt = np.linalg.svd(mean_rotation)
            determinant, minimum = float(np.linalg.det(mean_rotation)), float(singular[-1])
            if not np.isfinite(singular).all() or determinant <= 0 or minimum < self.config.minimum_singular_value:
                raise ValueError("ambiguous chordal rotation mean; no automatic fallback")
            target = np.eye(4)
            target[:3, 3] = np.mean(poses[:, :3, 3], axis=0)
            target[:3, :3] = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
            opening = float(np.mean(openings))
            validate_transform(target)
        provenance = dict(
            physical_step=k,
            absolute_endpoint=endpoint,
            contributor_count=count,
            equal_weight=1 / count,
            newest_query_step=active[-1][0],
            identity_fast_path=identical,
            mean_rotation_determinant=determinant,
            minimum_singular_value=minimum,
            contributors=[
                dict(
                    query_step=q,
                    endpoint_index=endpoint - q - 1,
                    absolute_endpoint=endpoint,
                    weight=1 / count,
                    target_tcp=poses[i].copy(),
                    opening=float(openings[i]),
                )
                for i, (q, _, _) in enumerate(active)
            ],
        )
        self._next_step += 1
        self._references = [item for item in self._references if item[0] + len(item[1]) > self._next_step]
        return target, opening, provenance
