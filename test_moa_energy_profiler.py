"""
test_moa_energy_profiler.py -- tests the energy-integration
arithmetic in moa_energy_profiler.py against known, hand-computable
cases. Does NOT test the actual GPU-polling or subprocess-launching
code, which requires real hardware not available in this environment
(see moa_energy_profiler.py's module docstring for what remains
unvalidated).
"""

import unittest
from moa_shape_derivation.moa_energy_profiler import integrate_power_samples


class TestPowerIntegration(unittest.TestCase):
    """Trapezoidal power-to-energy integration, checked against
    cases with a known, exactly-computable correct answer."""

    def test_constant_power_matches_simple_multiplication(self):
        """100W held constant for 10 seconds = exactly 1000 Joules --
        the simplest possible case, with no approximation error."""
        timestamps = [0.0, 5.0, 10.0]
        watts = [100.0, 100.0, 100.0]
        energy = integrate_power_samples(timestamps, watts)
        self.assertAlmostEqual(energy, 1000.0, places=6)

    def test_linear_ramp_matches_triangle_area(self):
        """Power ramping linearly from 0W to 100W over 10 seconds is
        exactly a triangle: area = 0.5 * base * height = 500 J.
        Trapezoidal integration is EXACT (not approximate) for any
        linear segment, so this should match to high precision."""
        timestamps = [0.0, 10.0]
        watts = [0.0, 100.0]
        energy = integrate_power_samples(timestamps, watts)
        self.assertAlmostEqual(energy, 500.0, places=6)

    def test_single_sample_returns_zero(self):
        """Cannot integrate over zero elapsed time with only one
        sample -- should return 0, not raise or guess."""
        energy = integrate_power_samples([0.0], [150.0])
        self.assertEqual(energy, 0.0)

    def test_empty_samples_returns_zero(self):
        energy = integrate_power_samples([], [])
        self.assertEqual(energy, 0.0)

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            integrate_power_samples([0.0, 1.0], [100.0])

    def test_non_monotonic_timestamps_raises(self):
        """Refuses to silently produce a nonsensical negative-time
        interval rather than guess what was meant."""
        with self.assertRaises(ValueError):
            integrate_power_samples([0.0, 2.0, 1.0], [100.0, 100.0, 100.0])

    def test_realistic_variable_power_trace(self):
        """A more realistic case: power varying sample to sample
        (as real GPU draw does under load), checked against a value
        computed independently by hand rather than by the same
        integration code being tested."""
        # Segments: 0->1s at 50->150W (avg 100, energy 100),
        #           1->2s at 150->150W (avg 150, energy 150),
        #           2->4s at 150->50W (avg 100, energy 200).
        # Total: 100 + 150 + 200 = 450 J.
        timestamps = [0.0, 1.0, 2.0, 4.0]
        watts = [50.0, 150.0, 150.0, 50.0]
        energy = integrate_power_samples(timestamps, watts)
        self.assertAlmostEqual(energy, 450.0, places=6)


class TestEnergyMeasurementDataclass(unittest.TestCase):
    """The small derived-property logic on EnergyMeasurement itself."""

    def test_energy_per_iteration_divides_correctly(self):
        from moa_shape_derivation.moa_energy_profiler import EnergyMeasurement
        m = EnergyMeasurement(
            total_energy_joules=1000.0, average_power_watts=200.0,
            duration_seconds=5.0, num_power_samples=50, iterations_run=100,
        )
        self.assertAlmostEqual(m.energy_per_iteration_joules, 10.0, places=6)

    def test_energy_per_iteration_handles_zero_iterations(self):
        """Should report NaN, not raise ZeroDivisionError, if somehow
        no iterations completed -- a measurement failure should be
        visible in the result, not crash the caller."""
        from moa_shape_derivation.moa_energy_profiler import EnergyMeasurement
        import math
        m = EnergyMeasurement(
            total_energy_joules=0.0, average_power_watts=0.0,
            duration_seconds=5.0, num_power_samples=50, iterations_run=0,
        )
        self.assertTrue(math.isnan(m.energy_per_iteration_joules))


if __name__ == "__main__":
    unittest.main(verbosity=2)
