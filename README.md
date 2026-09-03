# moa-shape-derivation

A prototype tool that derives GPU kernel code generation parameters
directly from a target machine's **measured** hardware shape, rather
than from tuning, search, or vendor documentation.

## The idea

Most GPU kernel tuning today works one of two ways: an expert
hand-tunes parameters for a specific chip, or an autotuner searches a
space of candidates on real hardware until something fast is found.
Both approaches treat the target machine's actual shape (how many
warps/wavefronts it can run at once, how much fast on-chip memory a
compute unit has, and so on) as something to be discovered indirectly,
through trial and error.

This project instead measures that shape directly (via
`nvaccelinfo`, NVIDIA's own device-introspection tool) and derives
kernel parameters from it in closed form — the same tile size,
worker count, and vector width every time for the same measured
shape, no search involved. It is a companion to the Mathematics of
Arrays (MoA) research program, which does the analogous thing for a
program's *data* shape; this tool applies the identical discipline to
the *machine's* shape.

Every formula here reproduces, exactly, values that were first
derived by hand and validated on real NVIDIA V100, A100, and H100
GPU hardware. This tool automates that derivation — it does not
introduce a new one.

## What's here

| File | What it does |
|---|---|
| `moa_shape_parser.py` | Parses `nvaccelinfo` output into a structured `MachineShape` |
| `moa_derive_params.py` | Derives OpenACC kernel parameters (`num_workers`, `vector_length`, tile size) from a `MachineShape` |
| `moa_generate_kernel.py` | Generates a complete, compilable OpenACC kernel file from those derived parameters |
| `moa_forward_openacc_template.c` | The kernel template — a real, hardware-validated attention kernel with its machine-derived values as placeholders |
| `test_moa_automation.py` | Automated tests, including a real compile-and-run check on generated output |
| `examples/` | Real, captured `nvaccelinfo` output from three GPU generations, and one example generated kernel |

## Quick start

No external dependencies — pure Python standard library.

```bash
# Derive parameters for a measured GPU shape
python3 moa_derive_params.py examples/v100_real.txt --dtype fp64 --head-dim 64

# Generate a complete, ready-to-compile kernel file
python3 moa_generate_kernel.py examples/h100_real.txt my_kernel.c --dtype fp64

# Run the test suite
python3 test_moa_automation.py
```

To measure a real GPU on a machine with the NVIDIA HPC SDK installed:

```bash
python3 moa_shape_parser.py          # runs nvaccelinfo directly and parses its output
```

## Status

This is an early-stage research prototype, not production tooling.
It currently:

- **Supports:** NVIDIA GPUs via `nvaccelinfo`, OpenACC kernel generation, one operator (Transformer attention forward pass).
- **Does not yet support:** AMD GPUs (`rocminfo` parsing is planned but not implemented), operators other than attention, or OpenMP/OpenMPI targets.
- **Known, stated assumption:** the derivation for `num_workers` assumes 4 warp schedulers per SM, an architectural constant consistent across every NVIDIA GPU generation tested (Volta, Ampere, Hopper) but not something `nvaccelinfo` reports directly — this is exposed as an explicit function parameter (`schedulers_per_sm`) rather than a hidden constant, precisely so it stays visible rather than silently assumed.

Cross-vendor validation (AMD, via `rocminfo`) is the immediate next
target.

## License

Apache License 2.0 — see `LICENSE`.
