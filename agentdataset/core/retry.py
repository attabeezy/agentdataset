"""Small, dependency-free retry + circuit-breaker helpers."""

import logging
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retryable: Callable[[Exception], bool] | None = None,
) -> T:
    """Call ``fn`` up to ``attempts`` times with exponential backoff + jitter.

    ``retryable`` decides whether an exception is worth retrying; defaults to
    retrying on any exception (callers can narrow this to timeouts/429/5xx).
    """
    if retryable is None:
        retryable = lambda e: True

    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - intentional catch-all for retry
            last_exc = e
            if attempt >= attempts or not retryable(e):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = delay * (0.5 + random.random())  # jitter in [0.5x, 1.5x)
            logger.warning(
                "Attempt %d/%d failed (%s); retrying in %.2fs",
                attempt,
                attempts,
                type(e).__name__,
                delay,
            )
            time.sleep(delay)
    # Unreachable, but keeps type checkers happy.
    assert last_exc is not None
    raise last_exc


class CircuitBreaker:
    """Minimal circuit breaker: opens after ``failure_threshold`` consecutive
    failures and stays open for ``reset_timeout`` seconds."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if time.monotonic() - self._opened_at >= self.reset_timeout:
            # Half-open: allow one probe.
            self._opened_at = None
            self._failures = 0
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            logger.warning("Circuit breaker opened after %d failures", self._failures)
