"""Validate the isolated stage budget, then use the frozen full search audit.

The response cap and every numerical gate remain fixed to the original global
envelope contract. Raising the stage cap raises the possible algebraic QP cost;
it neither grants extra responses nor changes any control qualification.
"""

from directional_derivative_audit import numerical_compare, require
from global_envelope_search_audit import audit_search as audit_global_search

_FIXED_CONFIG = dict(
    seed=42,
    max_stages=24,
    max_responses=122,
    alphas=[1.0, 0.5, 0.25, 0.125, 0.0625],
    radius_initial=0.005,
    radius_min=0.00015625,
    radius_max=0.02,
    global_displacement_limit=0.1,
    reserve_target=0.00001,
    reserve_entry_fraction=0.9,
    terminal_time_scale=0.1,
    pose_ceiling_fraction=0.01,
    safety_buffer=0.00000001,
    accept_ratio=0.1,
    shrink_ratio=0.25,
    expand_ratio=0.75,
    minimum_predicted_reduction=0.0001,
    minimum_actual_reduction=0.0001,
    pose_reduction_fraction=0.1,
    secant_min_squared_step=0.000000000001,
    secant_relative_limit=2.0,
    secant_row_norm_floor=0.000001,
)


def validate_config(config):
    """Independently bind the complete contract before inspecting any ledger."""
    require(isinstance(config, dict) and config.keys() == _FIXED_CONFIG.keys(), "Complete registered configuration")
    require(
        type(config["max_stages"]) is int and 1 <= config["max_stages"] <= 60, "Stage budget must be an integer1..60"
    )
    require(
        type(config["max_responses"]) is int and 2 <= config["max_responses"] <= 122,
        "Response budget must be an integer2..122 including initial and repeat",
    )
    for key, expected in _FIXED_CONFIG.items():
        if key in ("max_stages", "max_responses"):
            continue
        actual = config[key]
        require(type(actual) is type(expected), "Original configuration scalar category: " + key)
        if key == "alphas":
            require(
                len(actual) == len(expected) and all(type(item) is float for item in actual),
                "Original five explicit floating-point alphas",
            )
        require(actual == expected, "Original fixed configuration value: " + key)
    return config


def audit_search(
    answer,
    initial_Js,
    initial_Jq,
    historical_initial,
    previous,
    input_min,
    input_max,
    config,
    compare=numerical_compare,
):
    """Return the unmodified frozen audit result under a bounded stage cap."""
    validate_config(config)
    return audit_global_search(
        answer,
        initial_Js,
        initial_Jq,
        historical_initial,
        previous,
        input_min,
        input_max,
        config,
        compare=compare,
    )
