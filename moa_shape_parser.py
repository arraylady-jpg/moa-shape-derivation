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
    from hardware -- not looked up from vendor specs. Vendor-neutral
    by design: NVIDIA's nvaccelinfo and AMD's rocminfo use different
    terminology for the same underlying concepts (a compute unit is
    what NVIDIA calls an SM; a wavefront is what NVIDIA calls a warp),
    and both map onto this same structure so downstream derivation
    code (moa_derive_params.py) never needs to know which vendor
    produced the measurement.

    registers_per_block is optional: nvaccelinfo reports it, but
    rocminfo (AMD) does not appear to report a directly analogous
    field in any output format checked so far -- and no current
    derivation in this project actually uses it, so it is not
    required."""
    device_name: str
    sms: int
    warp_size: int
    max_threads_per_sm: int
    max_threads_per_block: int
    shared_mem_per_block_bytes: int
    l2_cache_bytes: int
    registers_per_block: int = None

    @property
    def max_warps_per_sm(self) -> int:
        return self.max_threads_per_sm // self.warp_size


# Field name -> attribute name, matching nvaccelinfo's real, observed
# output format (verified against real captured output from V100, A100,
# and H100 earlier in this project -- not a guessed format).
_NVIDIA_FIELD_MAP = {
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
        if key not in _NVIDIA_FIELD_MAP:
            continue
        attr, caster = _NVIDIA_FIELD_MAP[key]
        raw = rest.strip()
        if caster is int:
            m = re.search(r"[\d]+", raw.replace(",", ""))
            if not m:
                continue
            values[attr] = int(m.group())
        else:
            values[attr] = raw

    missing = [attr for attr, _ in _NVIDIA_FIELD_MAP.values() if attr not in values]
    if missing:
        raise ValueError(
            f"nvaccelinfo output missing required field(s): {missing}. "
            f"Refusing to guess -- re-run nvaccelinfo on a real compute "
            f"node (not a login node, which reports no accelerator)."
        )

    return MachineShape(**values)


def parse_rocminfo(text: str, agent_name: str = None) -> MachineShape:
    """Parse rocminfo's real text output into a MachineShape.

    UNLIKE nvaccelinfo, rocminfo reports on every HSA "Agent" in the
    system -- CPUs and GPUs both -- in one combined output, not just
    the GPU. This function splits the output into per-agent blocks,
    keeps only agents with "Device Type: GPU", and parses the first
    one found (or the one matching agent_name, if given, for
    multi-GPU nodes).

    As with parse_nvaccelinfo, missing required fields raise rather
    than silently default -- the whole point of measuring machine
    shape is to not assume it.

    STATUS: this parser's field mapping is based on rocminfo's
    documented, real output format (verified against AMD's own
    published documentation and multiple real, publicly posted
    rocminfo runs), not a guessed format. It has NOT yet been run
    against real output from this project's own target hardware
    (Delta's MI100 partition) -- that validation is the immediate
    next step once cluster access is available, not yet done.
    """
    # Split into per-agent blocks. rocminfo delimits agents with a
    # "*******" rule followed by "Agent N" -- split on that marker.
    blocks = re.split(r"\*{3,}\s*\nAgent\s+\d+\s*\n\*{3,}", text)

    gpu_block = None
    for block in blocks:
        if "Device Type:" not in block:
            continue
        device_type_match = re.search(r"Device Type:\s*(\w+)", block)
        if not device_type_match or device_type_match.group(1) != "GPU":
            continue
        if agent_name is not None:
            name_match = re.search(r"\n\s*Name:\s*(.+)", block)
            if not name_match or agent_name not in name_match.group(1):
                continue
        gpu_block = block
        break

    if gpu_block is None:
        raise ValueError(
            "No GPU agent (Device Type: GPU) found in rocminfo output"
            + (f" matching name {agent_name!r}" if agent_name else "")
            + ". Refusing to guess -- confirm rocminfo actually ran on "
              "a node with a GPU attached, not a login node."
        )

    def find_int(pattern):
        m = re.search(pattern, gpu_block)
        if not m:
            return None
        return int(m.group(1).replace(",", ""))

    device_name = re.search(r"\n\s*Name:\s*(.+)", gpu_block)
    sms = find_int(r"Compute Unit:\s*(\d+)")
    wavefront = find_int(r"Wavefront Size:\s*(\d+)")
    max_workitem_per_cu = find_int(r"Max Work-item Per CU:\s*(\d+)")
    workgroup_max = find_int(r"Workgroup Max Size:\s*(\d+)")
    l2_kb = find_int(r"L2:\s*(\d+)\(0x[0-9a-fA-F]+\)\s*KB")
    # The GROUP-segment memory pool is rocminfo's equivalent of
    # NVIDIA's "shared memory per block" -- AMD's Local Data Share
    # (LDS), the fast, per-workgroup on-chip scratch memory.
    lds_kb = find_int(r"Segment:\s*GROUP[^\n]*\n\s*Size:\s*(\d+)\(0x[0-9a-fA-F]+\)\s*KB")

    values = {
        "device_name": device_name.group(1).strip() if device_name else None,
        "sms": sms,
        "warp_size": wavefront,
        "max_threads_per_sm": max_workitem_per_cu,
        "max_threads_per_block": workgroup_max,
        "shared_mem_per_block_bytes": lds_kb * 1024 if lds_kb is not None else None,
        "l2_cache_bytes": l2_kb * 1024 if l2_kb is not None else None,
    }

    missing = [k for k, v in values.items() if v is None]
    if missing:
        raise ValueError(
            f"rocminfo GPU agent block missing required field(s): "
            f"{missing}. Refusing to guess -- this may indicate a "
            f"rocminfo output format this parser has not yet been "
            f"validated against (see this function's STATUS note)."
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


def measure_local_amd_gpu(agent_name: str = None) -> MachineShape:
    """Run rocminfo directly on this machine and parse its output.
    Requires /opt/rocm/bin (or wherever ROCm is installed) on PATH,
    and a real AMD GPU attached to this node."""
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        raise RuntimeError(
            "rocminfo not found on PATH -- add ROCm's bin directory "
            "(e.g. /opt/rocm/bin) to PATH first."
        )
    return parse_rocminfo(result.stdout, agent_name=agent_name)


if __name__ == "__main__":
    import sys
    if "--rocm" in sys.argv:
        sys.argv.remove("--rocm")
        if len(sys.argv) > 1:
            with open(sys.argv[1]) as f:
                shape = parse_rocminfo(f.read())
        else:
            shape = measure_local_amd_gpu()
    elif len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            shape = parse_nvaccelinfo(f.read())
    else:
        shape = measure_local_gpu()
    for k, v in asdict(shape).items():
        print(f"{k:30s} {v}")
    print(f"{'max_warps_per_sm':30s} {shape.max_warps_per_sm}")
