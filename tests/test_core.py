import unittest

from wall_clock_max_drift import DriftMonitor, DriftSample, DriftStats


class FakeClock:
    """A controllable clock for deterministic tests.

    Returns the current value of ``self.now`` every time it is called, so
    tests advance time explicitly by mutating ``self.now``. This keeps tests
    free of sleeps and real wall-clock dependence.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class TestDriftStatsEmpty(unittest.TestCase):
    def test_empty_stats_before_any_samples(self):
        monitor = DriftMonitor(
            reference_clock=FakeClock(0.0), wall_clock=FakeClock(0.0)
        )
        stats = monitor.stats()
        self.assertIsInstance(stats, DriftStats)
        self.assertEqual(stats.sample_count, 0)
        self.assertEqual(stats.max_absolute_drift, float("-inf"))
        self.assertIsNone(stats.worst_sample)

    def test_empty_is_a_new_instance_each_call(self):
        a = DriftStats.empty()
        b = DriftStats.empty()
        self.assertIsNot(a, b)
        self.assertEqual(a, b)


class TestNoDrift(unittest.TestCase):
    def test_drift_is_zero_when_clocks_advance_together(self):
        ref = FakeClock(0.0)
        wall = FakeClock(0.0)
        monitor = DriftMonitor(reference_clock=ref, wall_clock=wall)

        monitor.sample()
        ref.now = 1.0
        wall.now = 1.0
        monitor.sample()
        ref.now = 2.0
        wall.now = 2.0
        monitor.sample()

        stats = monitor.stats()
        self.assertEqual(stats.sample_count, 3)
        # Zero is an exact result of subtracting equal floats, but assert with
        # a tolerance so the test encodes intent rather than float mechanics.
        self.assertLess(stats.max_absolute_drift, 1e-12)
        self.assertIsNotNone(stats.worst_sample)

    def test_sample_returns_the_observed_pair(self):
        ref = FakeClock(10.0)
        wall = FakeClock(1000.0)
        monitor = DriftMonitor(reference_clock=ref, wall_clock=wall)

        observed = monitor.sample()
        self.assertIsInstance(observed, DriftSample)
        self.assertEqual(observed.reference_time, 10.0)
        self.assertEqual(observed.wall_time, 1000.0)


class TestPositiveAndNegativeDrift(unittest.TestCase):
    def test_wall_clock_running_fast_produces_positive_drift_magnitude(self):
        ref = FakeClock(0.0)
        wall = FakeClock(0.0)
        monitor = DriftMonitor(reference_clock=ref, wall_clock=wall)

        monitor.sample()
        ref.now = 5.0
        wall.now = 6.0
        monitor.sample()

        stats = monitor.stats()
        self.assertEqual(stats.sample_count, 2)
        self.assertEqual(stats.max_absolute_drift, 1.0)
        self.assertEqual(stats.worst_sample.reference_time, 5.0)
        self.assertEqual(stats.worst_sample.wall_time, 6.0)

    def test_wall_clock_running_slow_produces_positive_drift_magnitude(self):
        ref = FakeClock(0.0)
        wall = FakeClock(0.0)
        monitor = DriftMonitor(reference_clock=ref, wall_clock=wall)

        monitor.sample()
        ref.now = 5.0
        wall.now = 4.0
        monitor.sample()

        stats = monitor.stats()
        self.assertEqual(stats.sample_count, 2)
        self.assertEqual(stats.max_absolute_drift, 1.0)
        self.assertEqual(stats.worst_sample.reference_time, 5.0)
        self.assertEqual(stats.worst_sample.wall_time, 4.0)

    def test_largest_magnitude_wins_when_signs_differ(self):
        ref = FakeClock(0.0)
        wall = FakeClock(0.0)
        monitor = DriftMonitor(reference_clock=ref, wall_clock=wall)

        monitor.sample()
        ref.now = 1.0
        wall.now = 0.5  # drift = -0.5, magnitude 0.5
        monitor.sample()
        ref.now = 2.0
        wall.now = 4.0  # drift = +2.0, magnitude 2.0 (new max)
        monitor.sample()
        ref.now = 3.0
        wall.now = 2.9  # drift = -0.1, magnitude 0.1 (not new max)
        monitor.sample()

        stats = monitor.stats()
        self.assertEqual(stats.sample_count, 4)
        self.assertEqual(stats.max_absolute_drift, 2.0)
        self.assertEqual(stats.worst_sample.reference_time, 2.0)
        self.assertEqual(stats.worst_sample.wall_time, 4.0)

    def test_ties_update_worst_sample_to_most_recent(self):
        ref = FakeClock(0.0)
        wall = FakeClock(0.0)
        monitor = DriftMonitor(reference_clock=ref, wall_clock=wall)

        monitor.sample()
        ref.now = 1.0
        wall.now = 2.0  # magnitude 1.0
        monitor.sample()
        ref.now = 2.0
        wall.now = 3.0  # magnitude 1.0 (tie)
        monitor.sample()

        stats = monitor.stats()
        self.assertEqual(stats.max_absolute_drift, 1.0)
        self.assertEqual(stats.worst_sample.reference_time, 2.0)
        self.assertEqual(stats.worst_sample.wall_time, 3.0)


class TestBaselineIsFirstSample(unittest.TestCase):
    def test_drift_is_measured_from_first_sample_not_previous(self):
        ref = FakeClock(0.0)
        wall = FakeClock(0.0)
        monitor = DriftMonitor(reference_clock=ref, wall_clock=wall)

        monitor.sample()
        ref.now = 1.0
        wall.now = 1.0
        monitor.sample()
        # From a per-step baseline this step would show no drift, but from the
        # fixed first-sample baseline the wall clock is now ahead by 1.0.
        ref.now = 2.0
        wall.now = 3.0
        monitor.sample()

        stats = monitor.stats()
        self.assertEqual(stats.max_absolute_drift, 1.0)
        self.assertEqual(stats.worst_sample.reference_time, 2.0)
        self.assertEqual(stats.worst_sample.wall_time, 3.0)

    def test_non_zero_first_sample_offsets_do_not_affect_drift(self):
        ref = FakeClock(1000.0)
        wall = FakeClock(5000.0)
        monitor = DriftMonitor(reference_clock=ref, wall_clock=wall)

        monitor.sample()
        ref.now = 1005.0
        wall.now = 5007.0
        monitor.sample()

        stats = monitor.stats()
        self.assertEqual(stats.max_absolute_drift, 2.0)


class TestStatsSnapshot(unittest.TestCase):
    def test_returned_stats_do_not_mutate_with_later_samples(self):
        ref = FakeClock(0.0)
        wall = FakeClock(0.0)
        monitor = DriftMonitor(reference_clock=ref, wall_clock=wall)

        monitor.sample()
        ref.now = 1.0
        wall.now = 2.0
        monitor.sample()
        snapshot = monitor.stats()

        ref.now = 100.0
        wall.now = 300.0
        monitor.sample()
        later = monitor.stats()

        self.assertEqual(snapshot.sample_count, 2)
        self.assertEqual(snapshot.max_absolute_drift, 1.0)
        self.assertEqual(later.sample_count, 3)
        self.assertEqual(later.max_absolute_drift, 200.0)

    def test_driftsample_is_immutable(self):
        sample = DriftSample(1.0, 2.0)
        with self.assertRaises(Exception):
            sample.reference_time = 9.9  # type: ignore[misc]

    def test_driftstats_is_immutable(self):
        stats = DriftStats(sample_count=1, max_absolute_drift=0.0, worst_sample=None)
        with self.assertRaises(Exception):
            stats.sample_count = 99  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
