"""
moa_energy_profiler.py -- measures real GPU energy consumption during
a kernel workload, as the first concrete step toward an
energy-efficient PREDICTIVE MoA system: deriving not just expected
runtime from data shape and machine shape (already demonstrated
elsewhere in this project), but expected ENERGY consumption too.

This module only MEASURES energy, from a real, running workload. It
does not yet PREDICT energy from shape alone -- that would require a
cost function fit to real measured data across many configurations,
the same way this project's existing time-based cost functions were
fit (R^2 > 0.999) against real timing measurements. No such energy
data exists yet; this module is what would collect it.

HONEST MEASUREMENT LIMITATION, stated directly rather than glossed
over: GPU power-query tools (nvidia-smi, rocm-smi) sample the
underlying driver's power sensor at approximately 1 Hz internally,
regardless of how frequently they are queried. This project's kernels
run in single-digit-to-low-hundreds of milliseconds -- far shorter
than one power-sensor sample period. A single kernel launch CANNOT be
meaningfully power-profiled this way. This module is therefore built
to measure energy over a SUSTAINED, REPEATED execution (the same
kernel run back-to-back for several seconds), not a single invocation,
and reports per-invocation energy as (total energy) / (iteration
count) -- an average over the sustained run, not a measurement of any
single kernel launch in isolation.

STATUS: the energy-integration arithmetic below is independently
tested against synthetic power-sample data (see
test_moa_energy_profiler.py) and does not require a GPU to verify.
The actual subprocess-launching and live power-polling code has NOT
been run against a real GPU as of this writing -- no GPU was
available in the environment this was written in. That validation is
the immediate next step once real hardware access allows it.
"""

import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class EnergyMeasurement:
    """Result of measuring energy over a sustained workload run."""
    total_energy_joules: float
    average_power_watts: float
    duration_seconds: float
    num_power_samples: int
    iterations_run: int

    @property
    def energy_per_iteration_joules(self) -> float:
        if self.iterations_run == 0:
            return float("nan")
        return self.total_energy_joules / self.iterations_run


def integrate_power_samples(timestamps: list, watts: list) -> float:
    """Trapezoidal integration of a power-vs-time curve into total
    energy in Joules (Watts * seconds = Joules). Pure arithmetic, no
    GPU or subprocess involved -- this is the part of this module
    that can be, and is, fully tested without real hardware.

    timestamps: seconds, monotonically increasing.
    watts: power draw at each corresponding timestamp.
    """
    if len(timestamps) != len(watts):
        raise ValueError("timestamps and watts must be the same length")
    if len(timestamps) < 2:
        return 0.0

    energy = 0.0
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt < 0:
            raise ValueError("timestamps must be monotonically increasing")
        avg_power = (watts[i] + watts[i - 1]) / 2.0
        energy += avg_power * dt
    return energy


def _poll_nvidia_power(samples: list, stop_event: threading.Event,
                        poll_interval_s: float = 0.1):
    """Background-thread target: repeatedly query nvidia-smi's power
    draw and append (timestamp, watts) to the shared samples list.
    Polling faster than the sensor's own ~1 Hz update rate does not
    yield more real information, but does not hurt correctness either
    -- it only means adjacent samples may repeat the same underlying
    sensor reading, which the trapezoidal integration handles
    correctly regardless (repeated equal values contribute their
    correct, unchanged share of the integral)."""
    start = time.monotonic()
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            watts = float(result.stdout.strip().split("\n")[0])
            samples.append((time.monotonic() - start, watts))
        except (subprocess.SubprocessError, ValueError, IndexError):
            pass  # a single failed sample is not fatal; keep polling
        time.sleep(poll_interval_s)


def _poll_amd_power(samples: list, stop_event: threading.Event,
                     poll_interval_s: float = 0.1):
    """Same as _poll_nvidia_power, for rocm-smi. NOT yet validated
    against real rocm-smi output -- built from its documented
    --showpower flag, same discipline as this project's rocminfo
    parser: based on documented format, not yet checked against a
    real capture."""
    start = time.monotonic()
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                ["rocm-smi", "--showpower", "--csv"],
                capture_output=True, text=True, timeout=5,
            )
            # rocm-smi --csv format: a header row, then one row per
            # GPU with a power-draw column. Parsed defensively; not
            # yet confirmed against a real capture.
            lines = [l for l in result.stdout.strip().split("\n") if l]
            if len(lines) >= 2:
                header = lines[0].split(",")
                row = lines[1].split(",")
                power_col = next(
                    i for i, h in enumerate(header)
                    if "power" in h.lower()
                )
                watts = float(row[power_col])
                samples.append((time.monotonic() - start, watts))
        except (subprocess.SubprocessError, ValueError, StopIteration, IndexError):
            pass
        time.sleep(poll_interval_s)


def measure_energy(command: list, min_duration_s: float = 5.0,
                    rocm: bool = False, poll_interval_s: float = 0.1) -> EnergyMeasurement:
    """Run `command` repeatedly (back-to-back) for at least
    min_duration_s seconds, sampling GPU power draw concurrently, and
    return the total energy consumed and the average per-iteration
    share of it.

    command: the kernel binary (and args) to run repeatedly, e.g.
        ["./moa_forward_openacc_tiled"].
    min_duration_s: run for at least this long -- must be long enough,
        relative to the power sensor's ~1 Hz update rate, to get a
        meaningful number of independent samples. 5 seconds is a
        reasonable default; shorter risks measuring mostly sensor
        noise rather than real average draw.

    NOT YET VALIDATED against a real GPU (see module docstring).
    """
    samples = []
    stop_event = threading.Event()
    poll_fn = _poll_amd_power if rocm else _poll_nvidia_power
    poll_thread = threading.Thread(
        target=poll_fn, args=(samples, stop_event, poll_interval_s)
    )
    poll_thread.start()

    start = time.monotonic()
    iterations = 0
    try:
        while (time.monotonic() - start) < min_duration_s:
            subprocess.run(command, capture_output=True, timeout=30)
            iterations += 1
    finally:
        stop_event.set()
        poll_thread.join(timeout=poll_interval_s * 5)

    duration = time.monotonic() - start
    if not samples:
        raise RuntimeError(
            "No power samples collected -- check that nvidia-smi/rocm-smi "
            "is on PATH and a GPU is actually present on this node."
        )

    timestamps = [s[0] for s in samples]
    watts = [s[1] for s in samples]
    total_energy = integrate_power_samples(timestamps, watts)
    avg_power = sum(watts) / len(watts)

    return EnergyMeasurement(
        total_energy_joules=total_energy,
        average_power_watts=avg_power,
        duration_seconds=duration,
        num_power_samples=len(samples),
        iterations_run=iterations,
    )
