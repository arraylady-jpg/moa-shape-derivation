"""
test_moa_performance_predictor.py -- validates the roofline
predictor's arithmetic against hand-computable cases, and confirms
it exactly reproduces this project's own already-published
cross-machine prediction results (position paper, Section 6.3;
engineering TR, Section 11) -- not approximately, but to the exact
percentage points reported there, since this module's numbers ARE
those results, not a re-implementation of them.
"""

import unittest
from moa_shape_derivation.moa_performance_predictor import (
    PeakSpecs, attention_flops_and_bytes, roofline_time_seconds,
    calibrate_efficiency, predict_time_seconds, predict_held_out,
)


class TestAttentionFlopsAndBytes(unittest.TestCase):
    """Hand-computable cases for the FLOPs/bytes formulas themselves."""

    def test_matches_hand_calculation(self):
        # B=1, n=512, d=64, fp64 (8 bytes): flops = 4*1*512*512*64
        flops, bytes_moved = attention_flops_and_bytes(1, 512, 64, 8)
        self.assertEqual(flops, 4 * 1 * 512 * 512 * 64)
        self.assertEqual(bytes_moved, 4 * 1 * 512 * 64 * 8)

    def test_bytes_scale_linearly_with_n_not_quadratically(self):
        """The whole point of a fused kernel: bytes moved is O(n),
        not O(n^2) -- doubling n should exactly double bytes moved,
        not quadruple it."""
        _, bytes_512 = attention_flops_and_bytes(1, 512, 64, 8)
        _, bytes_1024 = attention_flops_and_bytes(1, 1024, 64, 8)
        self.assertEqual(bytes_1024, bytes_512 * 2)

    def test_flops_scale_quadratically_with_n(self):
        flops_512, _ = attention_flops_and_bytes(1, 512, 64, 8)
        flops_1024, _ = attention_flops_and_bytes(1, 1024, 64, 8)
        self.assertEqual(flops_1024, flops_512 * 4)


class TestRooflineTimeSeconds(unittest.TestCase):
    """Hand-computable roofline bound cases."""

    def test_compute_bound_when_arithmetic_intensity_high(self):
        # 1000 FLOPs, 1 byte moved, peak 1000 FLOPs/s, peak 1 GB/s
        # -> compute_time = 1000/1000 = 1s, memory_time = 1/1e9 ~ 0
        # -> compute-bound, result should be ~1s
        peak = PeakSpecs("test", peak_flops=1000.0, peak_bandwidth=1e9)
        t = roofline_time_seconds(flops=1000.0, bytes_moved=1.0, peak=peak)
        self.assertAlmostEqual(t, 1.0, places=6)

    def test_memory_bound_when_arithmetic_intensity_low(self):
        # 1 FLOP, 1000 bytes, peak 1e9 FLOPs/s, peak 1000 bytes/s
        # -> compute_time ~0, memory_time = 1000/1000 = 1s
        peak = PeakSpecs("test", peak_flops=1e9, peak_bandwidth=1000.0)
        t = roofline_time_seconds(flops=1.0, bytes_moved=1000.0, peak=peak)
        self.assertAlmostEqual(t, 1.0, places=6)


class TestCalibrateEfficiency(unittest.TestCase):
    def test_single_point_returns_itself(self):
        eff = calibrate_efficiency([(1.0, 2.0)])
        self.assertAlmostEqual(eff, 0.5, places=6)

    def test_averages_multiple_points(self):
        # ratios: 1/2=0.5, 1/4=0.25 -> average 0.375
        eff = calibrate_efficiency([(1.0, 2.0), (1.0, 4.0)])
        self.assertAlmostEqual(eff, 0.375, places=6)

    def test_empty_calibration_raises_rather_than_guesses(self):
        with self.assertRaises(ValueError):
            calibrate_efficiency([])


class TestPredictHeldOut(unittest.TestCase):
    """The real test: exact reproduction of this project's own
    already-published cross-machine prediction results. These are
    not approximate or independently-derived expected values -- they
    are the exact numbers reported in the position paper and TR,
    computed here by the same method to confirm this module IS that
    method, correctly captured in reusable code."""

    V100 = PeakSpecs("V100", 7.8e12, 900e9)
    A100 = PeakSpecs("A100", 9.7e12, 1555e9)
    H100 = PeakSpecs("H100", 34e12, 3350e9)

    # Real measured tiled-OpenACC forward-pass times (seconds),
    # B=1, d=64, fp64, n=512..8192 -- from this project's own
    # real-hardware sweeps.
    MEASURED_S = {
        "V100": [0.628e-3, 2.523e-3, 10.041e-3, 31.384e-3, 111.305e-3],
        "A100": [0.736e-3, 2.663e-3, 8.628e-3, 27.561e-3, 90.042e-3],
        "H100": [0.414e-3, 0.818e-3, 3.201e-3, 12.919e-3, 51.434e-3],
    }
    N_VALUES = [512, 1024, 2048, 4096, 8192]
    PEAKS = {"V100": V100, "A100": A100, "H100": H100}

    # Published error percentages, position paper Section 6.3 / TR
    # Table 16, reproduced here exactly (not approximately).
    PUBLISHED_ERROR_PCT = {
        "V100": [93.40, 36.11, 20.81, 35.78, 34.19],
        "A100": [1.80, 10.77, 8.84, 17.60, 32.85],
        "H100": [58.72, 19.68, 25.65, 41.82, 50.21],
    }

    def test_reproduces_published_errors_exactly(self):
        for held_out_name in ["V100", "A100", "H100"]:
            others = [n for n in self.PEAKS if n != held_out_name]
            for i, n in enumerate(self.N_VALUES):
                calibration = [
                    (self.PEAKS[o], self.MEASURED_S[o][i]) for o in others
                ]
                predicted = predict_held_out(
                    self.PEAKS[held_out_name], (1, n, 64, 8), calibration
                )
                real = self.MEASURED_S[held_out_name][i]
                error_pct = abs(predicted - real) / real * 100
                expected = self.PUBLISHED_ERROR_PCT[held_out_name][i]
                self.assertAlmostEqual(
                    error_pct, expected, places=1,
                    msg=f"{held_out_name} at n={n}: got {error_pct:.2f}%, "
                        f"published value is {expected}%"
                )

    def test_mean_error_matches_published_headline_number(self):
        """The paper's headline claim: mean error 32.5% across all
        15 held-out points. (Note: earlier drafts of the paper and
        TR stated 32.6% -- this test caught a real, small rounding
        discrepancy; both documents were corrected to 32.5% to match
        this precise, independently-computed value.)"""
        all_errors = []
        for held_out_name in ["V100", "A100", "H100"]:
            others = [n for n in self.PEAKS if n != held_out_name]
            for i, n in enumerate(self.N_VALUES):
                calibration = [
                    (self.PEAKS[o], self.MEASURED_S[o][i]) for o in others
                ]
                predicted = predict_held_out(
                    self.PEAKS[held_out_name], (1, n, 64, 8), calibration
                )
                real = self.MEASURED_S[held_out_name][i]
                all_errors.append(abs(predicted - real) / real * 100)
        mean_error = sum(all_errors) / len(all_errors)
        self.assertAlmostEqual(mean_error, 32.5, places=1)

    def test_target_own_measured_time_never_required_as_input(self):
        """A structural check, not just a numeric one: confirm
        predict_held_out's signature genuinely has no way to accept
        the target GPU's own measured time -- the held-out property
        is enforced by the function's design, not just by convention
        in how it happens to be called."""
        import inspect
        sig = inspect.signature(predict_held_out)
        param_names = list(sig.parameters.keys())
        self.assertEqual(param_names, ["target_peak", "workload_shape", "calibration_gpus"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
