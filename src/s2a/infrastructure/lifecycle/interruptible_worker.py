"""Turn an owned worker's SIGTERM into a chance to publish partial evidence."""

import signal
import threading
from contextlib import contextmanager


class WorkerTermination(InterruptedError):
    """An owner requested termination of this worker."""


@contextmanager
def interruptible_sigterm():
    """Raise on SIGTERM and restore the previous handler when the scope exits.

    The worker must install this before experiment work and publish its own
    partial receipt in a finally block. SIGKILL, a second signal or a blocked
    native call can still prevent publication; a missing receipt stays unknown.
    """
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("SIGTERM handling must be installed on the main thread")

    def terminate(signum, _frame):
        raise WorkerTermination(f"Owned worker received signal {signum}")

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
