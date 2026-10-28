"""Isolated accepted-point SQP with an exact complete initial-record gate.

The response/audit/cap and ranking contract is inherited from the frozen
prefix optimization reserve. QP algebra never calls the response model.
A finite numerical witness remains distinct from physical control success.
"""

import ast
from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType

import numpy as np

from s2a.accepted_point_sqp_core import damped_bfgs, solve_epigraph_qp
from s2a.geometry import finite, pose_error, validate_transform
from s2a.joint_preview_constraint_audit import audit_joint_preview_constraints

PREFIX_OPTIMIZATION_MARGIN = 1e-4
RADII = (0.01, 0.005, 0.0025)
BACKTRACKING_ALPHAS = tuple(2.0**-index for index in range(9))


class InitialRecordAdmissionError(ValueError):
    """A completed, charged initial response failed its mandatory record gate."""

    def __init__(self, message, failure):
        super().__init__(message)
        self.failure = failure


InitialRecordMismatch = InitialRecordAdmissionError


def _canonical(value):
    if isinstance(value, Mapping):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"Unsupported initial-record value: {type(value).__name__}")


def _readonly_snapshot(value):
    """Own every container and place numeric arrays over immutable byte buffers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _readonly_snapshot(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not initial-record numeric data")
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, (tuple, list)):
        return tuple(_readonly_snapshot(item) for item in value)
    if isinstance(value, np.generic) or value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"Unsupported initial-record value: {type(value).__name__}")


def _numeric_categories(value):
    if type(value) in (bool, int, float):
        return type(value).__name__
    if isinstance(value, list):
        result = [_numeric_categories(item) for item in value]
        return result if all(item is not None for item in result) else None
    return None


def _numeric_list(value):
    if not isinstance(value, list):
        return None
    categories = _numeric_categories(value)
    if categories is None:
        return None
    try:
        array, kinds = np.asarray(value, dtype=object), np.asarray(categories)
    except ValueError:
        return None
    return (array, kinds) if array.shape == kinds.shape else None


def canonical_initial_comparison(expected, observed):
    """Compare complete canonical records, retaining strict scalar categories.

    Array paths record every differing index; expected/observed full records
    provide the values. Canonical JSON does not preserve original NumPy dtype
    or hidden dimensions of empty arrays. Commands also require float64 bytes.
    """
    expected, observed = _canonical(expected), _canonical(observed)
    arrays, scalar_differences, structure_differences = {}, [], []

    def structure(path, kind, before, after):
        structure_differences.append(dict(path=path, kind=kind, expected=before, observed=after))

    def visit(before, after, path):
        if isinstance(before, dict) and isinstance(after, dict):
            if before.keys() != after.keys():
                structure(path, "mapping_keys", sorted(before), sorted(after))
            for key in sorted(before.keys() & after.keys()):
                visit(before[key], after[key], f"{path}.{key}" if path else key)
            return
        if isinstance(before, list) and isinstance(after, list):
            a, b = _numeric_list(before), _numeric_list(after)
            if a is not None and b is not None:
                av, ak = a
                bv, bk = b
                compatible = av.shape == bv.shape
                if compatible:
                    category_mismatch = ak != bk
                    values = list(zip(av.flat, bv.flat, strict=True))
                    finite_pair = all(
                        (type(a) is not float or np.isfinite(a)) and (type(b) is not float or np.isfinite(b))
                        for a, b in values
                    )
                    value_mismatch = np.array(
                        [
                            a != b
                            or (type(a) is float and not np.isfinite(a))
                            or (type(b) is float and not np.isfinite(b))
                            for a, b in values
                        ],
                        dtype=bool,
                    ).reshape(av.shape)
                    mismatch = category_mismatch | value_mismatch
                    maximum = max((float(abs(a - b)) for a, b in values), default=0.0) if finite_pair else None
                else:
                    structure(path, "sequence_length", list(av.shape), list(bv.shape))
                    category_mismatch, mismatch, maximum = None, None, None
                arrays[path] = dict(
                    shape=list(av.shape),
                    observed_shape=list(bv.shape),
                    categories_equal=bool(compatible and not np.any(category_mismatch)),
                    exact=bool(compatible and not np.any(mismatch)),
                    max_abs_difference=maximum,
                    mismatched_indices=np.argwhere(mismatch).tolist() if compatible else [],
                    category_mismatch_indices=np.argwhere(category_mismatch).tolist() if compatible else [],
                )
                return
            if len(before) != len(after):
                structure(path, "sequence_length", len(before), len(after))
            for index, (a, b) in enumerate(zip(before, after, strict=False)):
                visit(a, b, f"{path}[{index}]")
            return
        if isinstance(before, (dict, list)) or isinstance(after, (dict, list)):
            structure(path, "container_type", type(before).__name__, type(after).__name__)
            return
        same_category = type(before) is type(after)
        same_value = same_category and before == after
        if type(before) is float:
            same_value = same_value and np.isfinite(before)
        if type(after) is float:
            same_value = same_value and np.isfinite(after)
        if not same_value:
            scalar_differences.append(dict(path=path, expected=before, observed=after, category_equal=same_category))

    visit(expected, observed, "")
    try:
        a, b = np.asarray(expected["commands"], dtype=np.float64), np.asarray(observed["commands"], dtype=np.float64)
        commands_equal = a.shape == b.shape and a.tobytes() == b.tobytes()
    except (KeyError, TypeError, ValueError):
        commands_equal = False
    categories_equal = all(item["categories_equal"] for item in arrays.values()) and all(
        item["category_equal"] for item in scalar_differences
    )
    exact = (
        not structure_differences
        and not scalar_differences
        and commands_equal
        and all(item["exact"] for item in arrays.values())
    )
    return dict(
        comparison_scope="Canonical JSON arrays and scalars with strict scalar categories and float64 command bytes",
        exact=bool(exact),
        structure_equal=not structure_differences,
        scalar_categories_equal=bool(categories_equal),
        scalars_equal=not scalar_differences,
        commands_bytes_equal=bool(commands_equal),
        arrays=arrays,
        scalar_differences=scalar_differences,
        structure_differences=structure_differences,
        max_abs_difference=max((item["max_abs_difference"] or 0.0 for item in arrays.values()), default=0.0),
    )


def make_initial_record_hook(expected_entry, *, expected_ref):
    """Bind a source-located expected full entry before any response is called."""
    expected = _readonly_snapshot(expected_entry)
    reference = _canonical(expected_ref)
    if not isinstance(reference, dict) or any(
        not isinstance(reference.get(key), str) or not reference[key] for key in ("path", "sha256", "locator")
    ):
        raise ValueError("The initial record requires an explicit path, sha256 and locator")

    def hook(observed):
        comparison = canonical_initial_comparison(expected, observed)
        return dict(
            passed=comparison["exact"],
            exact=comparison["exact"],
            expected_ref=deepcopy(reference),
            comparison=comparison,
        )

    hook.expected_ref = deepcopy(reference)
    return hook


def _run_initial_record_hook(hook, initial, ledger):
    guard = dict(
        checked=False,
        passed=False,
        exact=False,
        hook_calls=0,
        request_index=0,
        response_index=0,
        request_count=len(ledger),
        response_count=1,
        stages_before=0,
        iterations_before=0,
        qp_iterations_before=0,
        expected_ref=None,
        comparison=None,
    )
    try:
        guard["expected_ref"] = _canonical(getattr(hook, "expected_ref", None))
        snapshot = _readonly_snapshot(initial)
        guard.update(checked=True, hook_calls=1)
        result = _canonical(hook(snapshot))
        if not isinstance(result, dict):
            raise TypeError("Initial record hook must return its complete guard report")
        guard.update(expected_ref=result.get("expected_ref"), comparison=result.get("comparison"))
        comparison = guard["comparison"]
        if (
            result.get("passed") is not True
            or result.get("exact") is not True
            or not isinstance(comparison, dict)
            or comparison.get("exact") is not True
            or not isinstance(guard["expected_ref"], dict)
        ):
            raise ValueError("The complete initial response did not exactly match its bound source record")
        guard.update(passed=True, exact=True)
        return guard
    except Exception as error:
        failure = dict(
            answer=None,
            observed_entry=deepcopy(initial),
            expected_ref=deepcopy(guard["expected_ref"]),
            comparison=deepcopy(guard["comparison"]),
            original_exception=dict(type=type(error).__name__, message=str(error)),
            initial_record_guard=deepcopy(guard),
            request_ledger=deepcopy(ledger),
            evaluated_candidates=[deepcopy(initial)],
            response_evaluations=1,
            total_iterations=0,
            total_qp_iterations=0,
            total_stages=0,
            phases=[],
            status="initial_record_admission_failed",
        )
        raise InitialRecordAdmissionError("Complete initial-record admission failed", failure) from error


def verify_restored_algorithm_ast(baseline_source, restored_source):
    """Invert only the registered solver deltas, never execute source text.

    This checks the entire algorithm function, its three mathematical constants,
    and original imports. It does not equate the new snapshot/comparison helpers
    to baseline code or replace response-level validation of the initial gate.
    """
    baseline, restored = ast.parse(baseline_source), ast.parse(restored_source)

    def dump(node):
        return ast.dump(node, include_attributes=False)

    def require(condition, message):
        if not condition:
            raise ValueError(f"Registered SQP AST difference violated: {message}")

    def function(module, name):
        matches = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name]
        require(len(matches) == 1, f"one {name} definition")
        return deepcopy(matches[0])

    old = function(baseline, "solve_accepted_point_sqp_response")
    new = function(restored, "solve_restored_anchor_sqp_response")
    new.name = old.name
    hook_arguments = [index for index, arg in enumerate(new.args.kwonlyargs) if arg.arg == "initial_record_hook"]
    require(len(hook_arguments) == 1, "one explicit initial-record hook argument")
    hook_index = hook_arguments[0]
    require(new.args.kw_defaults[hook_index] is None, "the hook is mandatory")
    require(
        dump(new.args.kwonlyargs[hook_index]) == dump(ast.arg(arg="initial_record_hook")), "unannotated hook argument"
    )
    del new.args.kwonlyargs[hook_index]
    del new.args.kw_defaults[hook_index]
    callable_guard = ast.parse(
        'if not callable(initial_record_hook):\n    raise TypeError("An explicit complete initial-record hook is required")'
    ).body[0]
    require(dump(new.body[1]) == dump(callable_guard), "callable admission immediately after the docstring")
    del new.body[1]

    def statement_index(node, statement):
        expected = dump(ast.parse(statement).body[0])
        matches = [index for index, item in enumerate(node.body) if dump(item) == expected]
        require(len(matches) == 1, f"one exact statement: {statement}")
        return matches[0]

    start_index = statement_index(new, "start = warm.ravel().copy()")
    new.body[start_index] = ast.parse("start = np.clip(warm.ravel(), lo, hi)").body[0]
    initial_index = statement_index(new, 'initial = evaluate(start, "warm")')
    hook_statement = "initial_record_guard = _run_initial_record_hook(initial_record_hook, initial, ledger)"
    require(statement_index(new, hook_statement) == initial_index + 1, "complete initial gate precedes all warm logic")
    del new.body[initial_index + 1]
    require(isinstance(new.body[-1], ast.Return) and isinstance(new.body[-1].value, ast.Call), "original final return")
    returned = new.body[-1].value
    hooks = [keyword for keyword in returned.keywords if keyword.arg == "initial_record_guard"]
    require(
        len(hooks) == 1 and dump(hooks[0].value) == dump(ast.Name(id="initial_record_guard", ctx=ast.Load())),
        "one guard field",
    )
    returned.keywords.remove(hooks[0])
    methods = [keyword for keyword in returned.keywords if keyword.arg == "method"]
    require(
        len(methods) == 1 and dump(methods[0].value) == dump(ast.Constant("restored_anchor_lagrangian_sqp")),
        "registered method name",
    )
    methods[0].value = ast.Constant("accepted_point_lagrangian_sqp")
    require(dump(new) == dump(old), "all remaining algorithm function AST is identical")

    constant_names = {"PREFIX_OPTIMIZATION_MARGIN", "RADII", "BACKTRACKING_ALPHAS"}

    def constants(module):
        return [
            dump(node)
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in constant_names for target in node.targets)
        ]

    require(
        len(constants(baseline)) == 3 and constants(restored) == constants(baseline), "original mathematical constants"
    )
    original_imports = [dump(node) for node in baseline.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    new_imports = [dump(node) for node in restored.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    require(all(new_imports.count(item) == 1 for item in original_imports), "original QP/BFGS and audit dependencies")
    # A second top-level binding could silently replace a checked constant/import/function.
    protected = constant_names | {
        "solve_restored_anchor_sqp_response",
        "np",
        "damped_bfgs",
        "solve_epigraph_qp",
        "finite",
        "pose_error",
        "validate_transform",
        "audit_joint_preview_constraints",
    }
    bindings = []
    for node in restored.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            bindings.append(node.name)
        elif isinstance(node, ast.Assign):
            bindings.extend(
                child.id for target in node.targets for child in ast.walk(target) if isinstance(child, ast.Name)
            )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bindings.extend(alias.asname or alias.name.split(".")[0] for alias in node.names)
        else:
            require(
                not any(
                    isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store) and child.id in protected
                    for child in ast.walk(node)
                ),
                "no hidden top-level algorithm rebinding",
            )
    require(all(bindings.count(name) == 1 for name in protected), "unique original algorithm bindings")
    return dict(
        passed=True,
        checked_function="solve_restored_anchor_sqp_response",
        original_function="solve_accepted_point_sqp_response",
        allowed_changes=[
            "function_and_method_names",
            "required_initial_record_hook_and_callable_admission",
            "copy_warm_start_without_clipping",
            "complete_initial_record_hook_before_warm_logic",
            "initial_record_guard_return_field",
        ],
        mathematical_constants_unchanged=True,
        original_imports_unchanged=True,
        helper_equivalence_claimed=False,
    )


def solve_restored_anchor_sqp_response(
    predict,
    previous,
    warm,
    interface,
    lower,
    upper,
    desired,
    site_from_tcp,
    *,
    past_inner_error_rad,
    terminal_normals,
    initial_record_hook,
    max_evaluations=4000,
    max_iterations=60,
    finite_difference_step=1e-4,
    max_stages=60,
):
    """Retain Lagrangian curvature only between actual accepted ray points.

    Each outer attempt solves one explicit algebraic QP. Numerical trajectory
    admission remains separate and requires the original full audit/caps.
    """
    if not callable(initial_record_hook):
        raise TypeError("An explicit complete initial-record hook is required")
    previous, warm = finite(previous, (6,)), finite(warm)
    if warm.ndim != 2 or warm.shape[1] != 6 or not 1 <= len(warm) <= 16:
        raise ValueError("One to sixteen explicit six-axis warm commands required")
    h, width = len(warm), warm.size
    lower, upper = finite(lower), finite(upper)
    desired, tool = validate_transform(desired), validate_transform(site_from_tcp)
    profile, normals = finite(past_inner_error_rad), finite(terminal_normals)
    if (
        lower.ndim != 1
        or not len(lower)
        or upper.shape != lower.shape
        or np.any(lower >= upper)
        or profile.shape != (25, len(lower))
        or np.any(profile < 0)
        or np.any(np.diff(profile, axis=0) < 0)
        or normals.ndim != 2
        or normals.shape[1] != len(lower)
        or desired.shape != (h, 4, 4)
        or tool.shape != (4, 4)
        or type(max_evaluations) is not int
        or max_evaluations < width + 1
        or type(max_iterations) is not int
        or max_iterations < 2
        or type(max_stages) is not int
        or max_stages < 1
        or not np.isfinite(finite_difference_step)
        or finite_difference_step <= 0
        or finite_difference_step >= RADII[-1]
    ):
        raise ValueError(
            "Valid trajectory/profile/limits, stage-compatible difference step and explicit budgets required"
        )
    lo = np.tile(finite(interface.input_min, (6,)), h)
    hi = np.tile(finite(interface.input_max, (6,)), h)
    lo[:6], hi[:6] = np.maximum(lo[:6], previous - 0.25), np.minimum(hi[:6], previous + 0.25)
    difference = np.eye(width) - np.eye(width, k=-6)
    center = np.r_[previous, np.zeros(width - 6)]
    if (
        np.any(hi - lo <= finite_difference_step)
        or np.any(warm.ravel() < lo - 1e-12)
        or np.any(warm.ravel() > hi + 1e-12)
        or np.any(np.abs(difference @ warm.ravel() - center) > 0.25 + 1e-12)
    ):
        raise ValueError("Box/rate feasible warm trajectory and nondegenerate command bounds required")
    start = warm.ravel().copy()
    safety_buffer = 1e-8
    cache, cache_indices, phases, ledger, accepted_anchors = {}, {}, [], [], []
    active_stage = None
    hessian, pending_pair = np.eye(width), None
    safety_names = ("inner_lower", "inner_upper", "braking_lower", "braking_upper", "terminal")

    class BudgetExhausted(Exception):
        pass

    class WitnessFound(Exception):
        pass

    def evaluate(x, role, *, axis=None, nominal_delta=None, alpha_index=None, alpha=None):
        x = finite(x, (width,))
        key = x.tobytes()
        request = dict(
            request_index=len(ledger),
            stage_index=active_stage["stage_index"] if active_stage is not None else None,
            role=role,
            command_key=key.hex(),
            response_index=None,
            new_response_index=None,
            outcome="pending",
            response_evaluations_before=len(cache),
            response_evaluations_after=len(cache),
            axis=axis,
            nominal_delta=nominal_delta,
            alpha_index=alpha_index,
            alpha=alpha,
        )
        ledger.append(request)
        bound_lo = lo if active_stage is None else active_stage["stage_lower_command"].ravel()
        bound_hi = hi if active_stage is None else active_stage["stage_upper_command"].ravel()
        if np.any(x < bound_lo - 1e-12) or np.any(x > bound_hi + 1e-12):
            request["outcome"] = "bounds_rejected"
            raise ValueError("Solver attempted a response outside its current command bounds")
        if key in cache:
            request.update(outcome="cache_hit", response_index=cache_indices[key])
        else:
            if len(cache) >= max_evaluations:
                request["outcome"] = "budget_rejected"
                raise BudgetExhausted
            commands = x.reshape(h, 6).copy()
            rollout = predict(commands.copy())
            audit = audit_joint_preview_constraints(
                rollout,
                commands,
                previous,
                interface,
                lower,
                upper,
                past_inner_error_rad=profile,
                terminal_normals=normals,
                braking_time_s=0.1,
            )
            poses = validate_transform(rollout["pose"])
            if poses.shape != (h, 4, 4):
                raise ValueError("Every predicted endpoint requires its own FK pose")
            errors = pose_error(desired, poses @ tool)
            normalized = finite(
                np.stack(
                    (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2),
                    axis=1,
                )
            )
            failures = np.flatnonzero(~np.all(normalized <= 1.0, axis=1))
            prefix = int(failures[0]) if len(failures) else h
            safety = np.concatenate([audit["groups"][name]["slack"].ravel() for name in safety_names])
            record = dict(
                commands=commands,
                rollout=rollout,
                audit=audit,
                tcp_errors=errors,
                normalized_squared_errors=normalized,
                safety_slack=safety,
                prefix_length=prefix,
                maximum_normalized_squared_error=float(np.max(normalized)),
                maximum_pose_cap_violation=float(max(0.0, np.max(np.sqrt(normalized)) - 1.0)),
                pose_loss=float(np.sum(normalized)),
                feasible=bool(audit["feasible"] and prefix == h),
            )
            index = len(cache)
            cache_indices[key], cache[key] = index, record
            request.update(
                outcome="new_response",
                response_index=index,
                new_response_index=index,
                response_evaluations_after=len(cache),
            )
        if active_stage is not None:
            active_stage["touched_response_indices"].append(request["response_index"])
        return cache[key]

    def witness_rank(record):
        return record["maximum_normalized_squared_error"], record["pose_loss"]

    def prefix_rank(record):
        prefix = record["prefix_length"]
        next_error = float(np.max(record["normalized_squared_errors"][prefix])) if prefix < h else 0.0
        return -prefix, next_error, record["pose_loss"]

    def select(records):
        witnesses = [record for record in records if record["feasible"]]
        return min(witnesses, key=witness_rank) if witnesses else min(records, key=prefix_rank)

    def global_witness():
        witnesses = [entry for entry in cache.values() if entry["feasible"]]
        return min(witnesses, key=witness_rank) if witnesses else None

    def index_of(record):
        return cache_indices[record["commands"].ravel().tobytes()]

    def anchor_record(record, before, stage=None):
        cursor = (
            stage["target_cursor"] if stage is not None else (None if record["feasible"] else record["prefix_length"])
        )
        before_max = float(np.max(before["normalized_squared_errors"][cursor])) if cursor is not None else None
        after_max = float(np.max(record["normalized_squared_errors"][cursor])) if cursor is not None else None
        return dict(
            is_initial=stage is None,
            stage_index=stage["stage_index"] if stage is not None else None,
            response_index=index_of(record),
            previous_response_index=index_of(before) if stage is not None else None,
            commands=record["commands"].copy(),
            prefix_length=record["prefix_length"],
            joint_feasible=record["audit"]["feasible"],
            safety_min=float(np.min(record["safety_slack"])),
            feasible=record["feasible"],
            audit=record["audit"],
            tcp_errors=record["tcp_errors"],
            normalized_squared_errors=record["normalized_squared_errors"],
            stage_target_cursor=cursor,
            stage_target_max_before=before_max,
            stage_target_max_after=after_max,
            target_improvement=before_max - after_max if cursor is not None else None,
            radius=stage["radius"] if stage is not None else None,
            acceptance_reason=stage["acceptance_reason"]
            if stage is not None
            else ("initial_witness" if record["feasible"] else "warm"),
        )

    def run_phase(base, radius_index):
        nonlocal active_stage, hessian, pending_pair
        cursor, radius = base["prefix_length"], RADII[radius_index]
        x = base["commands"].ravel()
        stage_lo, stage_hi = np.maximum(lo, x - radius), np.minimum(hi, x + radius)
        if np.any(stage_hi - stage_lo <= finite_difference_step):
            raise ValueError("Every stage requires a nondegenerate finite-difference interval")
        record = dict(
            phase="accepted_point_sqp",
            stage_index=len(phases),
            prefix_before=cursor,
            target_cursor=cursor,
            cursor=cursor,
            radius_index=radius_index,
            radius=radius,
            anchor_before_index=index_of(base),
            stage_lower_command=stage_lo.reshape(h, 6).copy(),
            stage_upper_command=stage_hi.reshape(h, 6).copy(),
            initial_commands=base["commands"].copy(),
            initial_epigraph=max(0.0, float(np.max(base["normalized_squared_errors"][cursor])) - 1.0) + 1e-6,
            initial_safety_min=float(np.min(base["safety_slack"])),
            evaluations_before=len(cache),
            response_budget_remaining_before=max_evaluations - len(cache),
            requests_before=len(ledger),
            iterations=1,
            qp_iterations=0,
            touched_response_indices=[],
            ray_attempts=[],
            hessian_before=hessian.copy(),
            hessian_after_update=hessian.copy(),
            pending_pair_source_stage=None if pending_pair is None else pending_pair["stage_index"],
            curvature_update=dict(updated=False, reason="no_pending_pair"),
            curvature_reset=None,
            qp=None,
            qp_direction=None,
            linearization=dict(complete=False, base_response_index=None, probe_response_indices=[], nominal_deltas=[]),
            status="started",
            ray_termination="no_direction",
        )
        phases.append(record)
        active_stage = record
        nonlinear_matrix, objective_gradient = None, None
        try:
            evaluate(x, "stage_anchor")
            reference = evaluate(x, "gradient_base")
            record["linearization"]["base_response_index"] = index_of(reference)
            ds, dp = [], []
            for axis in range(width):
                delta = (
                    finite_difference_step
                    if x[axis] + finite_difference_step <= stage_hi[axis]
                    else -finite_difference_step
                )
                other = x.copy()
                other[axis] += delta
                changed = evaluate(other, "gradient_probe", axis=axis, nominal_delta=delta)
                record["linearization"]["probe_response_indices"].append(index_of(changed))
                record["linearization"]["nominal_deltas"].append(delta)
                ds.append((changed["safety_slack"] - reference["safety_slack"]) / delta)
                dp.append(
                    (changed["normalized_squared_errors"].ravel() - reference["normalized_squared_errors"].ravel())
                    / delta
                )
            safety_jacobian, pose_jacobian = np.asarray(ds).T, np.asarray(dp).T
            objective_gradient = 1e-4 * np.sum(pose_jacobian, axis=0)
            nonlinear_matrix = np.vstack(
                (
                    np.c_[safety_jacobian, np.zeros(len(safety_jacobian))],
                    np.c_[-pose_jacobian[: 2 * cursor], np.zeros(2 * cursor)],
                    np.c_[-pose_jacobian[2 * cursor : 2 * cursor + 2], np.ones(2)],
                )
            )
            nonlinear_offset = np.r_[
                base["safety_slack"] - safety_buffer,
                1 - PREFIX_OPTIMIZATION_MARGIN - base["normalized_squared_errors"][:cursor].ravel(),
                1 - base["normalized_squared_errors"][cursor],
            ]
            record["linearization"].update(
                complete=True,
                objective_gradient=objective_gradient.copy(),
                nonlinear_row_count=len(nonlinear_offset),
            )
            if global_witness() is not None:
                raise WitnessFound
            if pending_pair is not None:
                if pending_pair["accepted_response_index"] != index_of(base) or pending_pair["cursor"] != cursor:
                    raise ValueError("Pending curvature must terminate at this exact accepted point and cursor")
                new_lagrangian = objective_gradient - nonlinear_matrix[:, :width].T @ pending_pair["multipliers"]
                displacement = x - pending_pair["base_commands"]
                change = new_lagrangian - pending_pair["lagrangian_gradient"]
                update = damped_bfgs(hessian, displacement, change)
                update.update(
                    source_stage=pending_pair["stage_index"],
                    source_response_index=pending_pair["base_response_index"],
                    accepted_response_index=index_of(base),
                    displacement=displacement.copy(),
                    gradient_change=change.copy(),
                    previous_lagrangian_gradient=pending_pair["lagrangian_gradient"].copy(),
                    current_lagrangian_gradient=new_lagrangian.copy(),
                    step_inf_over_fd=float(np.max(np.abs(displacement)) / finite_difference_step),
                )
                hessian = update["matrix"].copy()
                record["curvature_update"] = update
                pending_pair = None
            record["hessian_after_update"] = hessian.copy()
            condition = float(np.linalg.cond(hessian))
            record["hessian_condition_number"] = condition if np.isfinite(condition) else None
            rate_matrix = np.c_[difference, np.zeros(width)]
            command_matrix = np.c_[np.eye(width), np.zeros(width)]
            t_row = np.r_[np.zeros(width), 1.0][None]
            A = np.vstack((nonlinear_matrix, rate_matrix, -rate_matrix, command_matrix, -command_matrix, t_row))
            b = np.r_[
                nonlinear_offset,
                difference @ x - center + 0.25,
                0.25 - difference @ x + center,
                x - stage_lo,
                stage_hi - x,
                0.0,
            ]
            qp = solve_epigraph_qp(
                hessian,
                objective_gradient,
                A,
                b,
                record["initial_epigraph"],
                max_iterations=100,
                certificate_tolerance=1e-6,
            )
            record["qp"], record["qp_iterations"] = qp, qp["solver_nit"]
            if not 0 <= record["qp_iterations"] <= 100:
                raise ValueError("QP backend exceeded the explicit algebraic iteration budget")
            if not qp["qualified"]:
                record["status"] = "qp_certificate_failed"
            else:
                trial = x + qp["point"][:width]
                if np.any(trial < stage_lo - 1e-12) or np.any(trial > stage_hi + 1e-12):
                    record["status"] = "qp_command_bounds_failed"
                elif not np.any(trial != x):
                    record["status"] = "qp_zero_direction"
                else:
                    record["status"] = "qp_direction_qualified"
                    record["qp_direction"] = dict(
                        commands=trial.reshape(h, 6).copy(),
                        displacement=(trial - x).reshape(h, 6).copy(),
                        epigraph=float(qp["point"][width]),
                    )
        except WitnessFound:
            record["status"] = "witness_found"
        except BudgetExhausted:
            record["status"] = "response_budget"
            record["curvature_update"] = dict(updated=False, reason="incomplete_stencil")
        chosen = global_witness()
        direction = record["qp_direction"]
        record["generation_requests_after"] = len(ledger)
        record["ray_termination"] = (
            "witness"
            if chosen is not None
            else "response_budget"
            if record["status"] == "response_budget"
            else "no_direction"
        )
        base_target = float(np.max(base["normalized_squared_errors"][cursor]))
        if chosen is None and direction is not None:
            record["ray_termination"] = "alphas_exhausted"
            for alpha_index, alpha in enumerate(BACKTRACKING_ALPHAS):
                commands = (
                    direction["commands"].copy()
                    if alpha_index == 0
                    else base["commands"] + alpha * direction["displacement"]
                )
                request_index = len(ledger)
                attempt = dict(
                    alpha_index=alpha_index,
                    alpha=alpha,
                    request_index=request_index,
                    response_index=None,
                    new_response_index=None,
                    outcome=None,
                    commands=commands.copy(),
                    entry=None,
                    eligible=False,
                    selected=False,
                    rejection_reasons=[],
                )
                record["ray_attempts"].append(attempt)
                try:
                    candidate = evaluate(commands.ravel(), "backtracking", alpha_index=alpha_index, alpha=alpha)
                except BudgetExhausted:
                    attempt.update(outcome="budget_rejected", rejection_reasons=["response_budget"])
                    record["ray_termination"] = "response_budget"
                    chosen = global_witness()
                    if chosen is not None:
                        record["ray_termination"] = "witness"
                    break
                request = ledger[request_index]
                attempt.update(
                    response_index=request["response_index"],
                    new_response_index=request["new_response_index"],
                    outcome=request["outcome"],
                    entry=candidate,
                )
                reasons = []
                if not candidate["feasible"]:
                    if not candidate["audit"]["feasible"]:
                        reasons.append("joint_or_command")
                    if candidate["prefix_length"] < cursor:
                        reasons.append("prefix_lost")
                    if float(np.min(candidate["safety_slack"])) < safety_buffer:
                        reasons.append("safety_buffer")
                    change = base_target - float(np.max(candidate["normalized_squared_errors"][cursor]))
                    if not (candidate["prefix_length"] > cursor or change > 1e-12):
                        reasons.append("no_target_descent")
                attempt.update(eligible=not reasons, rejection_reasons=reasons)
                chosen = global_witness()
                if chosen is not None:
                    attempt["selected"] = index_of(chosen) == index_of(candidate)
                    record["ray_termination"] = "witness"
                    break
                if not reasons:
                    chosen = candidate
                    attempt["selected"] = True
                    record["ray_termination"] = "accepted"
                    break
        record["touched_response_indices"] = sorted(set(record["touched_response_indices"]))
        next_base = chosen if chosen is not None else base
        reason = (
            None
            if chosen is None
            else "witness"
            if chosen["feasible"]
            else "prefix_growth"
            if chosen["prefix_length"] > cursor
            else "target_descent"
        )
        if chosen is not None:
            if chosen["prefix_length"] != cursor:
                hessian, pending_pair = np.eye(width), None
                record["curvature_reset"] = "cursor_changed"
            elif not chosen["feasible"]:
                if record["qp"] is None or not record["qp"]["qualified"]:
                    raise ValueError("Ordinary accepted point requires a certified QP ray")
                multiplier = record["qp"]["multipliers"][: len(nonlinear_matrix)]
                pending_pair = dict(
                    stage_index=record["stage_index"],
                    cursor=cursor,
                    base_response_index=index_of(base),
                    accepted_response_index=index_of(chosen),
                    base_commands=x.copy(),
                    multipliers=multiplier.copy(),
                    lagrangian_gradient=objective_gradient - nonlinear_matrix[:, :width].T @ multiplier,
                )
        record.update(
            evaluations_after=len(cache),
            response_budget_remaining_after=max_evaluations - len(cache),
            requests_after=len(ledger),
            prefix_after=next_base["prefix_length"],
            selected_commands=next_base["commands"].copy(),
            anchor_after_index=index_of(next_base),
            accepted=chosen is not None,
            acceptance_reason=reason,
            radius_after=RADII[0]
            if chosen is not None
            else RADII[radius_index + 1]
            if radius_index + 1 < len(RADII)
            else None,
            hessian_after_stage=hessian.copy(),
            pending_pair_after_stage=None if pending_pair is None else pending_pair["stage_index"],
        )
        if chosen is not None:
            accepted_anchors.append(anchor_record(chosen, base, record))
        active_stage = None
        return next_base, chosen is not None

    initial = evaluate(start, "warm")
    initial_record_guard = _run_initial_record_hook(initial_record_hook, initial, ledger)
    if not initial["feasible"] and (
        not initial["audit"]["feasible"] or np.min(initial["safety_slack"]) < safety_buffer
    ):
        raise ValueError("Incomplete warm trajectory requires the full joint/command audit and1e-8 safety buffer")
    accepted_anchors.append(anchor_record(initial, initial))
    current, radius_index = initial, 0
    termination_reason = "witness" if initial["feasible"] else None
    while not current["feasible"]:
        used = sum(phase["iterations"] for phase in phases)
        if len(cache) >= max_evaluations:
            termination_reason = "response_budget"
            break
        if used >= max_iterations:
            termination_reason = "iteration_budget"
            break
        if len(phases) >= max_stages:
            termination_reason = "stage_budget"
            break
        if radius_index >= len(RADII):
            termination_reason = "trust_region_stalled"
            break
        current, accepted = run_phase(current, radius_index)
        radius_index = 0 if accepted else radius_index + 1
    safe = [record for record in cache.values() if record["audit"]["feasible"]]
    selected = select(safe)
    witnesses = [record for record in safe if record["feasible"]]

    def safety_rank(record):
        worst = min(group["minimum"] for group in record["audit"]["groups"].values() if group["minimum"] is not None)
        return max(0.0, -worst), record["pose_loss"]

    return dict(
        prefix_optimization_margin=PREFIX_OPTIMIZATION_MARGIN,
        previous_command=previous.copy(),
        method="restored_anchor_lagrangian_sqp",
        initial_record_guard=initial_record_guard,
        qp_iteration_limit=100,
        qp_certificate_tolerance=1e-6,
        total_qp_iterations=sum(phase["qp_iterations"] for phase in phases),
        final_hessian=hessian.copy(),
        pending_pair_source_stage=None if pending_pair is None else pending_pair["stage_index"],
        witness_found=bool(witnesses),
        selected=selected,
        closest_safety_attempt=min(cache.values(), key=safety_rank),
        initial=initial,
        initial_prefix_length=initial["prefix_length"],
        selected_prefix_length=selected["prefix_length"],
        supplied_warm=warm.copy(),
        used_start=start.reshape(h, 6),
        start_roundoff_clip=start.reshape(h, 6) - warm,
        lower_command=lo.reshape(h, 6),
        upper_command=hi.reshape(h, 6),
        horizon=h,
        pose_caps_horizon=h,
        all_pose_caps_enforced=True,
        pose_cap_enforcement_scope="All H caps are required for a witness; stages hard-constrain only the attained prefix",
        terminal_normals=normals.copy(),
        phases=phases,
        response_evaluations=len(cache),
        total_iterations=sum(phase["iterations"] for phase in phases),
        max_evaluations=max_evaluations,
        max_iterations=max_iterations,
        finite_difference_step=finite_difference_step,
        optimization_safety_buffer=safety_buffer,
        past_inner_error_rad=profile.copy(),
        evaluated_candidates=[
            dict(
                commands=record["commands"],
                safety_min=float(np.min(record["safety_slack"])),
                safety_slack=record["safety_slack"].copy(),
                joint_feasible=record["audit"]["feasible"],
                tcp_errors=record["tcp_errors"],
                prefix_length=record["prefix_length"],
                feasible=record["feasible"],
            )
            for record in cache.values()
        ],
        termination_reason="witness" if witnesses else termination_reason,
        status="witness" if witnesses else "finite_search_without_witness",
        request_ledger=ledger,
        accepted_anchors=accepted_anchors,
        max_stages=max_stages,
        total_stages=len(phases),
        backtracking_alphas=list(BACKTRACKING_ALPHAS),
        total_backtracking_attempts=sum(len(phase["ray_attempts"]) for phase in phases),
        total_backtracking_new_responses=sum(
            attempt["outcome"] == "new_response" for phase in phases for attempt in phase["ray_attempts"]
        ),
        final_selected_is_last_anchor=index_of(selected) == accepted_anchors[-1]["response_index"],
    )
