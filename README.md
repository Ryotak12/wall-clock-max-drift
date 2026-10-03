# Wall Clock Max Drift

Measures and reports the maximum observed drift of a wall clock relative to a
trusted reference clock, given paired samples from each.

```python
import time

from wall_clock_max_drift import DriftMonitor, DriftSample, DriftStats

monitor = DriftMonitor(reference_clock=time.perf_counter, wall_clock=time.time)
monitor.sample()
# ... do work, then:
monitor.sample()
stats: DriftStats = monitor.stats()
print(stats.sample_count, stats.max_absolute_drift)
```

## Why this exists

Code that reads a wall clock (e.g. `time.time`) and later compares two
readings wants to know how far that clock could have wandered in the
meantime. `time.monotonic` and `time.perf_counter` are the usual trusted
sources, but they are not the clock whose drift you are worried about — the
wall clock is. This library accumulates paired samples from a reference clock
and a wall clock and reports the largest magnitude of drift seen so far,
relative to the first sample taken.

The trade-off: drift is measured against a fixed first-sample baseline, not a
sliding window. That means a slow, persistent divergence shows up as growing
drift the longer the monitor runs, which is exactly what you want for
"how bad could it have been" accounting. It also means you should not reuse a
single monitor across logically unrelated time spans; take a new one per span.

## The awkward edge

`max_absolute_drift` is reported in magnitude only. The signed worst sample is
not exposed because there is no single correct sign for "worst" — a caller
who needs the sign can subtract the first sample's values from
`stats.worst_sample` themselves. If you need per-sample signed drift, compute
it from the `DriftSample` objects that `sample()` returns; this library does
not retain the full sample history.

## Running the tests

```
PYTHONPATH=src python -m unittest discover -s tests
```
