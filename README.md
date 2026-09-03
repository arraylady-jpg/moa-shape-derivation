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
| `moa_shape_parser.py` | Parses `nvaccelinfo` (NVIDIA) or `rocminfo` (AMD) output into a structured, vendor-neutral `MachineShape` |
| `moa_derive_params.py` | Derives OpenACC kernel parameters (`num_workers`, `vector_length`, tile size) from a `MachineShape` |
| `moa_generate_kernel.py` | Generates a complete, compilable OpenACC kernel file from those derived parameters |
| `moa_forward_openacc_template.c` | The kernel template — a real, hardware-validated attention kernel with its machine-derived values as placeholders |
| `test_moa_automation.py` | Automated tests (16), including a real compile-and-run check on generated output |
| `examples/` | Real, captured `nvaccelinfo` output from three NVIDIA GPU generations; a `rocminfo` fixture built from AMD's documented format; one example generated kernel |

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

Or, on a machine with ROCm installed:

```bash
python3 moa_shape_parser.py --rocm   # runs rocminfo directly and parses its output
```

## Status

This is an early-stage research prototype, not production tooling.
It currently:

- **Supports:** NVIDIA GPUs via `nvaccelinfo`, and a first prototype
  of AMD GPU support via `rocminfo` (parser built from AMD's own
  documented output format, tested against a fixture built from that
  documentation and real, publicly posted examples — **not yet
  validated against real captured output from a real MI100 or other
  AMD accelerator**; that validation is the immediate next step now
  that AMD GPU cluster access is in progress).
- **A genuinely interesting early result:** applied to AMD's
  documented shape (wider 64-wide wavefronts, larger measured LDS
  capacity than NVIDIA's shared memory per block), the same
  derivation produces a *different* tile size (32×16, not NVIDIA's
  16×16) — real evidence the derivation responds to actual measured
  hardware rather than coincidentally always returning the same
  answer.
- **A flagged, unresolved question for AMD specifically:** the
  `num_workers`-equivalent derivation still uses the
  `schedulers_per_sm=4` assumption carried over from NVIDIA, never
  validated for AMD's architecture. Interestingly, `rocminfo` directly
  *reports* a field called `SIMDs per CU` (4, in every fixture
  checked so far) — a genuinely measured AMD analog, rather than an
  assumption, that this parser does not yet extract or use. Using it
  instead of the carried-over NVIDIA assumption is a concrete,
  well-scoped next step.
- **Does not yet support:** operators other than attention, or
  OpenMP/OpenMPI targets.

Real-hardware AMD validation (Delta's MI100 partition, via ACCESS) is
the immediate next target.

## License

Apache License 2.0 — see `LICENSE`.
