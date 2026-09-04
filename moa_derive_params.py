"""
moa_derive_params.py -- Aim 2 prototype (narrow first instance): derive
OpenACC kernel parameters from a measured MachineShape and a target
computation's data shape, automatically.

This replaces the by-hand arithmetic performed repeatedly this
project (computing num_workers=16 from "64 max warps / 4 schedulers",
and solving for a 16x16 tile size by trying candidate sizes against a
48 KiB shared-memory budget) with a function. Every formula here is
exactly the one already verified by hand against real hardware
earlier in this project -- this file automates applying it, it does
not introduce a new derivation.

HONEST LIMITATION, stated directly rather than glossed over:
schedulers_per_sm (used below to derive num_workers) is NOT reported
by nvaccelinfo -- it was an architectural assumption (4, consistent
across Volta/Ampere/Hopper) in every hand derivation this project
performed, never independently confirmed by device introspection.
This function exposes it as an explicit, named parameter rather than
a silently hard-coded constant, so that assumption stays visible
rather than being buried in automation.
"""

from dataclasses import dataclass
from moa_shape_parser import MachineShape


@dataclass
class OpenACCParams:
    num_workers: int
    vector_length: int
    block_m: int
    block_n: int
    tile_footprint_bytes: int
    shared_mem_budget_bytes: int


_DTYPE_BYTES = {"fp64": 8, "fp32": 4, "fp16": 2, "bf16": 2}


def derive_num_workers(shape: MachineShape, schedulers_per_sm: int = 4) -> int:
    """num_workers = max resident warps/SM / warp schedulers/SM.

    schedulers_per_sm defaults to 4, matching every NVIDIA architecture
    measured in this project (Volta, Ampere, Hopper) -- but this is an
    architectural assumption, not something nvaccelinfo reports
    directly (see module docstring). Pass an explicit value for any
    architecture where this has not been independently confirmed.
    """
    return shape.max_warps_per_sm // schedulers_per_sm


def derive_vector_length(shape: MachineShape) -> int:
    """vector_length = the machine's own warp width, measured directly."""
    return shape.warp_size


def _tile_footprint_bytes(block_m: int, block_n: int, head_dim: int, dtype_bytes: int) -> int:
    """Simultaneous working-set size for one gang: Q tile, K tile, V
    tile, score tile, output accumulator -- the exact five arrays
    accounted for when this tile size was first derived by hand."""
    q = block_m * head_dim * dtype_bytes
    k = block_n * head_dim * dtype_bytes
    v = block_n * head_dim * dtype_bytes
    s = block_m * block_n * dtype_bytes
    acc = block_m * head_dim * dtype_bytes
    return q + k + v + s + acc


def derive_tile_size(shape: MachineShape, head_dim: int, dtype: str,
                      candidates=((32, 32), (32, 16), (16, 32), (16, 16), (8, 16), (16, 8), (8, 8))):
    """Search candidate (BLOCK_M, BLOCK_N) tile sizes, largest first,
    and return the first that fits within the machine's measured
    shared-memory-per-block budget -- the same search order used by
    hand when this tile size was first derived (largest first, until
    one fits with real margin)."""
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"Unknown dtype {dtype!r}; known: {list(_DTYPE_BYTES)}")
    dtype_bytes = _DTYPE_BYTES[dtype]
    budget = shape.shared_mem_per_block_bytes

    for block_m, block_n in candidates:
        footprint = _tile_footprint_bytes(block_m, block_n, head_dim, dtype_bytes)
        if footprint <= budget:
            return block_m, block_n, footprint
    raise ValueError(
        f"No candidate tile size fits within {budget} bytes for "
        f"head_dim={head_dim}, dtype={dtype}. Widen the candidates "
        f"list or reduce head_dim."
    )


def derive_openacc_params(shape: MachineShape, head_dim: int, dtype: str,
                           schedulers_per_sm: int = 4) -> OpenACCParams:
    """The full derivation this project performed by hand, three
    times (once per GPU generation), now a single function call."""
    num_workers = derive_num_workers(shape, schedulers_per_sm)
    vector_length = derive_vector_length(shape)
    block_m, block_n, footprint = derive_tile_size(shape, head_dim, dtype)
    return OpenACCParams(
        num_workers=num_workers,
        vector_length=vector_length,
        block_m=block_m,
        block_n=block_n,
        tile_footprint_bytes=footprint,
        shared_mem_budget_bytes=shape.shared_mem_per_block_bytes,
    )


if __name__ == "__main__":
    import sys
    from moa_shape_parser import parse_nvaccelinfo, parse_rocminfo

    if len(sys.argv) < 2:
        print("Usage: python3 moa_derive_params.py <shape_output.txt> "
              "[--rocm] [--dtype fp64] [--head-dim 64]")
        sys.exit(1)

    dtype = "fp64"
    head_dim = 64
    use_rocm = "--rocm" in sys.argv
    if use_rocm:
        sys.argv.remove("--rocm")
    if "--dtype" in sys.argv:
        dtype = sys.argv[sys.argv.index("--dtype") + 1]
    if "--head-dim" in sys.argv:
        head_dim = int(sys.argv[sys.argv.index("--head-dim") + 1])

    with open(sys.argv[1]) as f:
        text = f.read()
    shape = parse_rocminfo(text) if use_rocm else parse_nvaccelinfo(text)

    params = derive_openacc_params(shape, head_dim=head_dim, dtype=dtype)

    print(f"Device: {shape.device_name}")
    print(f"  Measured: {shape.sms} SMs, warp={shape.warp_size}, "
          f"max_warps/SM={shape.max_warps_per_sm}, "
          f"shared_mem/block={shape.shared_mem_per_block_bytes}B")
    print(f"Derived OpenACC parameters (dtype={dtype}, head_dim={head_dim}):")
    print(f"  num_workers({params.num_workers})")
    print(f"  vector_length({params.vector_length})")
    print(f"  BLOCK_M={params.block_m}, BLOCK_N={params.block_n} "
          f"({params.tile_footprint_bytes}B of {params.shared_mem_budget_bytes}B budget)")
