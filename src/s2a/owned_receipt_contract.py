"""Explicit legacy and durable wait/reap contracts; never synthesize receipt keys."""

from s2a.durable_stage_supervisor import require

LEGACY_RELEASE_KEYS = frozenset(
    ("only_owned_child_signalled", "original_error", "pid", "reaped", "returncode", "signal_sent", "start_ticks")
)


def normal_receipts(e, launch_ref, exit_ref, release_ref, *, format):
    require(format in ("legacy", "durable"), "explicit known receipt format required")
    launch, exited, released = (e.json(ref) for ref in (launch_ref, exit_ref, release_ref))
    require(
        exited.get("actual_wait") is True and exited.get("returncode") == 0 and exited.get("automatic_retry") is False,
        "normal actual wait0 missing",
    )
    if format == "legacy":
        require(
            exited == dict(actual_wait=True, returncode=0, automatic_retry=False)
            and set(released) == LEGACY_RELEASE_KEYS,
            "legacy exit/release schema differs",
        )
    else:
        require(
            released.get("actual_wait") is True and released.get("kill_sent") is False,
            "durable release lacks actual wait/no-kill",
        )
    require(
        released.get("reaped") is True
        and released.get("returncode") == 0
        and released.get("signal_sent") is False
        and released.get("original_error") is None
        and released.get("only_owned_child_signalled") is True
        and all(launch["child"][k] == released[k] for k in ("pid", "start_ticks")),
        "normal owned reap/no-signal/birth missing",
    )
    return launch


def d1_postcheck_failure(e, owned):
    """Only046's known after-wait metadata KeyError; not a normal-owner waiver."""
    require(
        owned.get("complete") is owned.get("passed") is False
        and owned.get("error") == dict(type="KeyError", message="'actual_wait'")
        and owned.get("timed_out") is False
        and owned.get("automatic_retry") is False,
        "not the exact046 post-wait field error",
    )
    launch, exited, released = (e.json(owned[k]) for k in ("launch_ref", "exit_ref", "release_ref"))
    require(
        exited.get("actual_wait") is True and exited.get("returncode") == 0 and exited.get("automatic_retry") is False,
        "046 wrapper did not actually exit0",
    )
    require(
        released.get("actual_wait") is released.get("reaped") is True
        and released.get("returncode") == 0
        and released.get("signal_sent") is released.get("kill_sent") is False
        and released.get("original_error") == "KeyError('actual_wait')"
        and released.get("only_owned_child_signalled") is True
        and all(launch["child"][k] == released[k] for k in ("pid", "start_ticks")),
        "046 wrapper lifecycle differs from the exact post-check failure",
    )
    return launch
