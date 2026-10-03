"""Measure and report the maximum observed drift of a wall clock.

This module treats the clock supplied by the caller as the reference of truth
and the wall clock as the potentially-drifting signal. See DriftMonitor for
the contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional


# Type aliases for readability. A clock function takes no arguments and returns
# a monotonic time in seconds, exactly like time.monotonic and time.perf_counter.
Clock = Callable[[], float]


@dataclass(frozen=True)
class DriftSample:
    """A single paired observation of the reference clock and wall clock.

    Fields are intentionally raw floats. Comparisons in tests use the public
    DriftStats, not the samples directly, so equality of floats here is never
    asserted. The field order is (reference, wall) so callers can unpack the
    pair naturally.
    """

    reference_time: float
    wall_time: float


@dataclass(frozen=True)
class DriftStats:
    """Summary of all drift observed so far.

    Absolute drifts are reported because callers usually want the magnitude of
    the deviation, not its sign. Both the max_absolute_drift (magnitude) and the
    DriftSample that produced it are reported so the caller can reconstruct
    timing context if needed.
    """

    sample_count: int
    max_absolute_drift: float
    worst_sample: Optional[DriftSample]

    @classmethod
    def empty(cls) -> "DriftStats":
        """An empty stats object, before any samples have been taken."""
        return cls(sample_count=0, max_absolute_drift=float("-inf"), worst_sample=None)


class DriftMonitor:
    """Accumulates paired samples and reports the maximum observed drift.

    Interpretation chosen and documented in the README:
      - The clock passed as ``reference_clock`` is treated as the trusted source
        (e.g. time.perf_counter or an injected fake).
      - The clock passed as ``wall_clock`` is the one whose drift we care about
        (e.g. time.time, or an injected fake).
      - Drift at a sample is the difference between how far the wall clock
        advanced since the first sample and how far the reference advanced in
        the same interval: ``wall_delta - reference_delta``. A negative value
        means the wall clock ran slow relative to the reference.
      - ``max_absolute_drift`` is the largest magnitude of that signed drift.

    Drift is measured relative to the FIRST recorded sample, not a moving
    baseline. That keeps the statistic meaningful as samples accumulate: a
    drift that grows over time is captured, whereas a windowed baseline would
    hide a slow persistent divergence once the window moved past it.
    """

    def __init__(
        self,
        reference_clock: Clock = time.perf_counter,
        wall_clock: Clock = time.time,
    ) -> None:
        """Construct a monitor with the given clock functions.

        Defaulting to the stdlib clocks makes the object usable in production
        code, but tests are required to inject fakes so they remain
        deterministic. Both defaults are read at construction time; swapping
        ``time.perf_counter`` out from under this module after construction has
        no effect, which is the intended behaviour.
        """
        self._reference_clock = reference_clock
        self._wall_clock = wall_clock

        # We keep the first sample's raw values rather than normalising to zero.
        # Subtracting floats is cheap and avoids inventing a fake "zero" epoch
        # that could confuse a reader of the raw samples.
        self._first_sample: Optional[DriftSample] = None
        self._max_absolute_drift: float = float("-inf")
        self._worst_sample: Optional[DriftSample] = None
        self._sample_count: int = 0

    def sample(self) -> DriftSample:
        """Take one paired sample from both clocks and return it.

        The reference clock is read first, then the wall clock, so any
        scheduling jitter between the two reads biases toward the wall clock
        reading slightly later. That is acceptable because we report drift in
        magnitude and because the call ordering is stable across samples, so
        the bias cancels when comparing samples.
        """
        reference_time = self._reference_clock()
        wall_time = self._wall_clock()
        observed = DriftSample(reference_time, wall_time)
        self._observe(observed)
        return observed

    def _observe(self, observed: DriftSample) -> None:
        if self._first_sample is None:
            self._first_sample = observed
        reference_delta = observed.reference_time - self._first_sample.reference_time
        wall_delta = observed.wall_time - self._first_sample.wall_time
        # Float subtraction can produce exact zeros for identical inputs, but
        # we never assert == on floats in tests; stats report magnitude only.
        drift = wall_delta - reference_delta
        absolute = -drift if drift < 0 else drift
        if absolute >= self._max_absolute_drift:
            self._max_absolute_drift = absolute
            self._worst_sample = observed
        self._sample_count += 1

    def stats(self) -> DriftStats:
        """Return the current cumulative drift statistics.

        The returned object is a frozen snapshot; callers may hold it safely
        across later calls to ``sample``.
        """
        if self._sample_count == 0:
            return DriftStats.empty()
        return DriftStats(
            sample_count=self._sample_count,
            max_absolute_drift=self._max_absolute_drift,
            worst_sample=self._worst_sample,
        )
