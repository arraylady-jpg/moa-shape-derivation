"""
moa_performance_predictor.py -- predicts absolute kernel performance
on a target GPU using only its public peak specifications and an
efficiency ratio calibrated from OTHER GPUs' real measured data.

This is a genuinely different, harder prediction than
moa_derive_params.py's parameter derivation. Deriving a tile size or
worker count from measured shape and checking it against that SAME
GPU's own hardware is one kind of claim (and, in this project's
real-hardware results, an exactly-right one every time checked).
Predicting a wall-clock NUMBER for a machine whose own real timing
data was never used to produce that number is a categorically harder
claim, and this module's own real-hardware validation (see this
project's position paper, Section 6.3) reports a mean error of
32.6% -- real signal, correct order of magnitude and scaling trend,
but nowhere near the zero-error result parameter derivation achieves.
This module does not pretend otherwise.

METHOD: a standard roofline lower bound --
    T_roofline = max(FLOPs / peak_FLOPs, bytes_moved / peak_bandwidth)
-- is computed from the workload's own data shape and the target's
public peak specs. This bound is deliberately not itself the
prediction: real kernels never approach 100% of peak. An efficiency
ratio (roofline_time / measured_time) is computed from GPUs whose
real timing IS known, averaged, and applied to the target GPU's own
roofline bound to produce the actual prediction -- the target's own
real timing data is never used to predict itself.

HONEST LIMITATION: this two-source-error model does not separate
fixed kernel-launch overhead from the roofline's scaling term, which
is exactly why error is worst at small problem sizes (Section 6.3).
A more accurate model would fit overhead as a separate additive
term; this module implements the simpler version actually validated
in this project's published results, not a hypothetical improved one.
"""

from dataclasses import dataclass


@dataclass
class PeakSpecs:
    """A GPU's peak theoretical throughput, from its own public
    datasheet -- not measured, and not the same thing as the
    occupancy-relevant MachineShape fields in moa_shape_parser.py."""
    device_name: str
    peak_flops: float       # FLOPs per second (e.g. 7.8e12 for 7.8 TFLOPS)
    peak_bandwidth: float   # bytes per second (e.g. 900e9 for 900 GB/s)


def attention_flops_and_bytes(B: int, n: int, d: int, dtype_bytes: int):
    """FLOPs and bytes moved for one fused attention forward pass,
    batch B, sequence length n, head dimension d.

    FLOPs: the dominant terms are QK^T and softmax(.)V, each
    O(B*n^2*d) multiply-adds, counted as 2 FLOPs each -> 4*B*n^2*d.
    Bytes moved: a FUSED kernel (this project's own, and the one
    these formulas were validated against) never materializes the
    full n x n attention matrix -- it reads Q, K, V and writes O,
    each B*n*d elements -> O(n), not O(n^2), moved in total.
    """
    flops = 4 * B * n * n * d
    bytes_moved = 4 * B * n * d * dtype_bytes
    return flops, bytes_moved


def roofline_time_seconds(flops: float, bytes_moved: float, peak: PeakSpecs) -> float:
    """The roofline lower bound: the best any kernel with this many
    FLOPs and this many bytes to move could possibly do on this
    GPU's peak theoretical throughput. Not a prediction by itself --
    see calibrate_efficiency and predict_time below."""
    compute_time = flops / peak.peak_flops
    memory_time = bytes_moved / peak.peak_bandwidth
    return max(compute_time, memory_time)


def calibrate_efficiency(calibration_points):
    """Given real (roofline_time_seconds, measured_time_seconds)
    pairs from GPUs whose real timing IS known, return the average
    efficiency ratio (roofline / measured -- always <= 1, since the
    roofline is a lower bound and real kernels never beat it).

    calibration_points: iterable of (roofline_time, measured_time) tuples.
    """
    points = list(calibration_points)
    if not points:
        raise ValueError(
            "calibrate_efficiency requires at least one calibration "
            "point -- refusing to guess an efficiency ratio from nothing."
        )
    ratios = [roofline / measured for roofline, measured in points]
    return sum(ratios) / len(ratios)


def predict_time_seconds(target_roofline_time: float, calibrated_efficiency: float) -> float:
    """Apply an externally-calibrated efficiency ratio to a target
    GPU's own roofline bound to predict its absolute time -- the
    target's own real measured time is never an input to this
    function, by construction of how it's meant to be called."""
    if calibrated_efficiency <= 0:
        raise ValueError("efficiency ratio must be positive")
    return target_roofline_time / calibrated_efficiency


def predict_held_out(
    target_peak: PeakSpecs,
    workload_shape,  # (B, n, d, dtype_bytes)
    calibration_gpus,  # list of (PeakSpecs, measured_time_seconds) for the SAME workload_shape
) -> float:
    """Full pipeline: compute the target's roofline bound, calibrate
    efficiency from the other GPUs' real data at the same workload
    shape, and predict the target's time -- without ever using the
    target's own real measured time (this function doesn't even
    accept it as an argument, by design)."""
    B, n, d, dtype_bytes = workload_shape
    flops, bytes_moved = attention_flops_and_bytes(B, n, d, dtype_bytes)

    calibration_points = []
    for gpu_peak, measured_time in calibration_gpus:
        gpu_roofline = roofline_time_seconds(flops, bytes_moved, gpu_peak)
        calibration_points.append((gpu_roofline, measured_time))

    efficiency = calibrate_efficiency(calibration_points)
    target_roofline = roofline_time_seconds(flops, bytes_moved, target_peak)
    return predict_time_seconds(target_roofline, efficiency)
