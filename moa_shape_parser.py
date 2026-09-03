"""
moa_shape_parser.py -- Aim 1 prototype: parse nvaccelinfo output into a
structured, machine-readable shape descriptor, automatically.

This replaces the by-hand reading of nvaccelinfo output (done manually,
three times, once per GPU generation, earlier in this project) with a
script. It is a first, narrow instance of the automation this project's
NSF Specific Aims document identifies as Aim 1: formalizing machine
shape as a general, automatically-measurable input -- not yet the full
generalization (this parses one specific tool's output format), but a
real, working step rather than only a plan.
"""

import re
import subprocess
from dataclasses import dataclass, asdict


@dataclass
class MachineShape:
    """A structured descriptor of a GPU's shape, as measured directly
    from hardware via nvaccelinfo -- not looked up from vendor specs."""
    device_name: str
    sms: int
    warp_size: int
    max_threads_per_sm: int
    max_threads_per_block: int
    shared_mem_per_block_bytes: int
    registers_per_block: int
    l2_cache_bytes: int

    @property
    def max_warps_per_sm(self) -> int:
        return self.max_threads_per_sm // self.warp_size


# Field name -> attribute name, matching nvaccelinfo's real, observed
# output format (verified against real captured output from V100, A100,
# and H100 earlier in this project -- not a guessed format).
_FIELD_MAP = {
    "Device Name":                  ("device_name", str),
    "Number of Multiprocessors":    ("sms", int),
    "Warp Size":                    ("warp_size", int),
    "Max Threads Per SMP":          ("max_threads_per_sm", int),
    "Maximum Threads per Block":    ("max_threads_per_block", int),
    "Total Shared Memory per Block": ("shared_mem_per_block_bytes", int),
    "Registers per Block":          ("registers_per_block", int),
    "L2 Cache Size":                ("l2_cache_bytes", int),
}


def parse_nvaccelinfo(text: str) -> MachineShape:
    """Parse nvaccelinfo's real text output into a MachineShape.

    Raises ValueError if a required field is missing -- this is
    deliberate: silently defaulting a missing hardware constant would
    reintroduce exactly the "assumed, not measured" problem this
    project's whole machine-shape argument is built on rejecting.
    """
    values = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        if key not in _FIELD_MAP:
            continue
        attr, caster = _FIELD_MAP[key]
        raw = rest.strip()
        if caster is int:
            m = re.search(r"[\d]+", raw.replace(",", ""))
            if not m:
                continue
            values[attr] = int(m.group())
        else:
            values[attr] = raw

    missing = [attr for attr, _ in _FIELD_MAP.values() if attr not in values]
    if missing:
        raise ValueError(
            f"nvaccelinfo output missing required field(s): {missing}. "
            f"Refusing to guess -- re-run nvaccelinfo on a real compute "
            f"node (not a login node, which reports no accelerator)."
        )

    return MachineShape(**values)


def measure_local_gpu() -> MachineShape:
    """Run nvaccelinfo directly on this machine and parse its output.
    Requires an nvhpc module/toolchain already loaded, and a real GPU
    (will raise if run on a login node with no accelerator attached)."""
    try:
        result = subprocess.run(
            ["nvaccelinfo"], capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        raise RuntimeError(
            "nvaccelinfo not found on PATH -- load the nvhpc module first."
        )
    if "No accelerators found" in result.stdout:
        raise RuntimeError(
            "nvaccelinfo found no GPU -- this must be run on a real "
            "compute node (via srun/sbatch), not a login node."
        )
    return parse_nvaccelinfo(result.stdout)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            shape = parse_nvaccelinfo(f.read())
    else:
        shape = measure_local_gpu()
    for k, v in asdict(shape).items():
        print(f"{k:30s} {v}")
    print(f"{'max_warps_per_sm':30s} {shape.max_warps_per_sm}")
