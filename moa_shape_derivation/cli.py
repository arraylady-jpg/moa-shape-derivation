"""
cli.py -- the single, unified command-line entry point for this
project: `moa-shape <subcommand>`.

Subcommands:
  measure     Measure the local GPU's shape (NVIDIA or AMD) and print it.
  derive      Derive OpenACC parameters from a measured shape file.
  generate    Generate a complete, compilable kernel file from a shape.
  contribute  Prepare a shape measurement for community contribution
              (see CONTRIBUTING.md) -- never submits anything
              automatically; only ever writes a local file and prints
              clear next-step instructions for the user to review and
              submit themselves.
"""

import argparse
import datetime
import json
import platform
import sys
from dataclasses import asdict

from .moa_shape_parser import (
    parse_nvaccelinfo, parse_rocminfo,
    measure_local_gpu, measure_local_amd_gpu,
)
from .moa_derive_params import derive_openacc_params
from .moa_generate_kernel import generate_kernel
from .moa_energy_profiler import measure_energy


def _measure_shape(args):
    if args.rocm:
        if args.input:
            with open(args.input) as f:
                return parse_rocminfo(f.read(), agent_name=args.agent)
        return measure_local_amd_gpu(agent_name=args.agent)
    else:
        if args.input:
            with open(args.input) as f:
                return parse_nvaccelinfo(f.read())
        return measure_local_gpu()


def cmd_measure(args):
    shape = _measure_shape(args)
    for k, v in asdict(shape).items():
        print(f"{k:30s} {v}")
    print(f"{'max_warps_per_sm':30s} {shape.max_warps_per_sm}")


def cmd_derive(args):
    shape = _measure_shape(args)
    params = derive_openacc_params(shape, head_dim=args.head_dim, dtype=args.dtype)
    print(f"Device: {shape.device_name}")
    print(f"  Measured: {shape.sms} SMs/CUs, warp/wavefront={shape.warp_size}, "
          f"max_warps/SM={shape.max_warps_per_sm}, "
          f"shared_mem/block={shape.shared_mem_per_block_bytes}B")
    print(f"Derived OpenACC parameters (dtype={args.dtype}, head_dim={args.head_dim}):")
    print(f"  num_workers({params.num_workers})")
    print(f"  vector_length({params.vector_length})")
    print(f"  BLOCK_M={params.block_m}, BLOCK_N={params.block_n} "
          f"({params.tile_footprint_bytes}B of {params.shared_mem_budget_bytes}B budget)")


def cmd_generate(args):
    shape = _measure_shape(args)
    source = generate_kernel(shape, head_dim=args.head_dim, dtype=args.dtype)
    with open(args.output, "w") as f:
        f.write(source)
    print(f"Generated {args.output} for {shape.device_name} "
          f"(dtype={args.dtype}, head_dim={args.head_dim})")


def cmd_contribute(args):
    """Prepare a shape measurement for community contribution.

    Deliberately conservative by design: this command NEVER makes a
    network call, NEVER submits anything automatically, and only
    collects architectural fields already present in MachineShape
    (compute unit / SM count, warp or wavefront width, cache and
    memory sizes, device name) -- nothing about the host system,
    user, institution, or network. It writes one local JSON file and
    prints exactly what to do next; the user reviews and submits it
    themselves.
    """
    shape = _measure_shape(args)
    contribution = {
        "schema_version": 1,
        "contributed_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "vendor": "AMD" if args.rocm else "NVIDIA",
        "measurement_tool": "rocminfo" if args.rocm else "nvaccelinfo",
        "python_platform_machine": platform.machine(),  # e.g. "x86_64" -- CPU
                                                          # architecture only,
                                                          # not a hostname or
                                                          # any host-identifying
                                                          # value
        "shape": asdict(shape),
    }
    out_path = args.output or f"contribution_{shape.device_name.replace(' ', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(contribution, f, indent=2)

    print(f"Wrote {out_path}\n")
    print("This file contains ONLY architectural GPU specifications")
    print("(the fields above) -- no hostname, username, IP address, or")
    print("any other host-identifying information.")
    print()
    print("Nothing has been submitted or transmitted anywhere. To")
    print("actually contribute this to the project's shared dataset,")
    print("please review the file above and then either:")
    print(f"  1. Open a pull request adding it to contributed_shapes/, or")
    print(f"  2. Open a GitHub issue and attach this file")
    print("at the project's repository. See CONTRIBUTING.md for the")
    print("current, exact submission process.")


def cmd_energy(args):
    """Measure real energy consumption of a repeated kernel run.

    NOT a prediction -- a measurement. See moa_energy_profiler.py's
    module docstring for why this measures a SUSTAINED, REPEATED run
    rather than a single kernel launch, and for what has and has not
    been validated against real hardware yet.
    """
    result = measure_energy(
        command=args.command, min_duration_s=args.duration,
        rocm=args.rocm, poll_interval_s=args.poll_interval,
    )
    print(f"Ran {result.iterations_run} iterations over "
          f"{result.duration_seconds:.2f}s ({result.num_power_samples} power samples)")
    print(f"Average power:        {result.average_power_watts:.2f} W")
    print(f"Total energy:         {result.total_energy_joules:.2f} J")
    print(f"Energy per iteration: {result.energy_per_iteration_joules:.4f} J "
          f"(average over the sustained run, not a single-launch measurement)")


def main():
    parser = argparse.ArgumentParser(prog="moa-shape",
                                      description="Derive GPU kernel parameters from measured hardware shape.")
    parser.add_argument("--rocm", action="store_true",
                         help="Measure/parse AMD (rocminfo) rather than NVIDIA (nvaccelinfo) output.")
    parser.add_argument("--agent", default=None,
                         help="For multi-GPU nodes with --rocm: substring to match the target GPU's Name field.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_measure = sub.add_parser("measure", help="Measure and print the local GPU's shape.")
    p_measure.add_argument("input", nargs="?", default=None,
                            help="Path to a saved nvaccelinfo/rocminfo output file. Omit to run the tool live.")
    p_measure.set_defaults(func=cmd_measure)

    for name, helptext, func in [
        ("derive", "Derive OpenACC parameters from a measured shape.", cmd_derive),
        ("generate", "Generate a complete kernel file from a measured shape.", cmd_generate),
        ("contribute", "Prepare a shape measurement for community contribution (never auto-submits).", cmd_contribute),
    ]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("input", nargs="?", default=None,
                        help="Path to a saved nvaccelinfo/rocminfo output file. Omit to run the tool live.")
        p.add_argument("--dtype", default="fp64", choices=["fp64", "fp32", "fp16", "bf16"])
        p.add_argument("--head-dim", type=int, default=64)
        if name == "generate":
            p.add_argument("--output", "-o", required=True, help="Output .c file path.")
        if name == "contribute":
            p.add_argument("--output", "-o", default=None, help="Output JSON path (default: auto-named).")
        p.set_defaults(func=func)

    p_energy = sub.add_parser("energy", help="Measure real GPU energy consumption of a repeated kernel run (NOT yet a prediction -- a measurement).")
    p_energy.add_argument("command", nargs=argparse.REMAINDER,
                           help="The kernel command to run repeatedly, e.g.: moa-shape energy -- ./my_kernel")
    p_energy.add_argument("--duration", type=float, default=5.0,
                           help="Minimum sustained run duration in seconds (default 5.0 -- see moa_energy_profiler.py for why this must not be too short).")
    p_energy.add_argument("--poll-interval", type=float, default=0.1,
                           help="Power-sampling poll interval in seconds (default 0.1; the underlying sensor itself updates at ~1 Hz regardless).")
    p_energy.set_defaults(func=cmd_energy)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
