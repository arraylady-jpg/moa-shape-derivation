# moa-shape-derivation

A prototype tool that derives GPU kernel code generation parameters
directly from a target machine's **measured** hardware shape, rather
than from tuning, search, or vendor documentation.

## Install and try it in one line

```bash
pip install moa-shape-derivation      # once published; for now: pip install -e . from a clone
moa-shape derive                      # measures your local NVIDIA GPU and derives kernel parameters
moa-shape derive --rocm               # same, for a local AMD GPU
```

No external dependencies -- pure Python standard library, works
anywhere Python 3.8+ does.

## Help this project by contributing your GPU's shape

This project's core claim -- that kernel parameters can be derived
from measured shape rather than hand-tuned per chip -- has so far
only been checked against three NVIDIA GPU generations. **If you have
access to any other GPU**, especially anything non-NVIDIA, running
one command and opening a pull request is the most useful thing you
can do for this project:

```bash
moa-shape contribute            # NVIDIA
moa-shape contribute --rocm     # AMD
```

This writes one local JSON file containing only architectural
specifications (compute unit count, warp width, cache sizes) --
never a hostname, username, or anything else identifying you or your
institution -- and **never transmits anything automatically**. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for exactly what gets collected
and how to submit it.

## The idea

Most GPU kernel tuning today works one of two ways: an expert
hand-tunes parameters for a specific chip, or an autotuner searches a
space of candidates on real hardware until something fast is found.
Both treat the target machine's actual shape (how many warps or
wavefronts it can run at once, how much fast on-chip memory a compute
unit has) as something to discover indirectly, through trial and
error.

This project instead measures that shape directly (via `nvaccelinfo`
for NVIDIA, `rocminfo` for AMD) and derives kernel parameters from it
in closed form -- the same tile size, worker count, and vector width
every time for the same measured shape, no search involved. It is a
companion to the Mathematics of Arrays (MoA) research program, which
does the analogous thing for a program's *data* shape; this tool
applies the identical discipline to the *machine's* shape.

Every formula here reproduces, exactly, values first derived by hand
and validated on real NVIDIA V100, A100, and H100 GPU hardware. This
tool automates that derivation -- it does not introduce a new one.

## What's here

```
moa_shape_derivation/
    moa_shape_parser.py             Parses nvaccelinfo/rocminfo output into a MachineShape
    moa_derive_params.py            Derives OpenACC parameters from a MachineShape
    moa_generate_kernel.py          Generates a complete kernel file from those parameters
    moa_forward_openacc_template.c  The kernel template (hardware-validated, values parameterized)
    cli.py                          The moa-shape command (measure / derive / generate / contribute)
examples/                           Real captured NVIDIA output (3 GPUs) + a documented-format AMD fixture
contributed_shapes/                 Community-contributed measurements (see CONTRIBUTING.md)
test_moa_automation.py              16 automated tests, including a real compile-and-run check
```

## Command reference

```bash
moa-shape measure    [file] [--rocm]                       # print a shape, measured live or from a saved file
moa-shape derive     [file] [--rocm] [--dtype fp64] [--head-dim 64]
moa-shape generate   [file] [--rocm] -o kernel.c [--dtype fp64] [--head-dim 64]
moa-shape contribute [file] [--rocm] [-o out.json]          # see "Help this project" above
moa-shape energy -- <command...> [--duration 5.0]           # measure real GPU energy of a repeated kernel run
```

Omit `[file]` to measure the local GPU directly (requires
`nvaccelinfo` or `rocminfo`, respectively, on PATH). Pass a saved
output file instead to work from a capture made elsewhere.

`energy` measures, it does not yet predict: this project's existing
time-based cost functions are fit to real measurement (R² > 0.999);
no equivalent energy-based cost function exists yet, because no
energy dataset exists yet to fit one to. This command is the
measurement tool that would collect it. It also cannot meaningfully
profile a single fast kernel launch -- GPU power sensors update at
roughly 1 Hz internally, far coarser than this project's
millisecond-scale kernels -- so it runs the given command repeatedly
for a sustained duration and reports the average energy per
iteration, not a single-launch measurement.

## Status

This is an early-stage research prototype, not production tooling.

- **Supports:** NVIDIA GPUs via `nvaccelinfo`, and a first prototype
  of AMD GPU support via `rocminfo` (parser built from AMD's own
  documented output format -- **not yet validated against real
  captured output from real AMD hardware**; that validation is in
  progress).
- **A genuinely interesting early result:** applied to AMD's
  documented shape (wider 64-wide wavefronts, larger measured LDS
  capacity than NVIDIA's shared memory per block), the same
  derivation produces a *different* tile size (32×16, not NVIDIA's
  16×16) -- real evidence the derivation responds to actual measured
  hardware rather than coincidentally always returning the same
  answer.
- **A flagged, unresolved question for AMD specifically:** the
  `num_workers`-equivalent derivation still uses the
  `schedulers_per_sm=4` assumption carried over from NVIDIA, never
  validated for AMD's architecture. `rocminfo` directly reports a
  field called `SIMDs per CU` (a genuinely measured AMD analog,
  rather than an assumption) that this tool does not yet use --
  a concrete, well-scoped next step.
- **Does not yet support:** operators other than attention, or
  OpenMP/OpenMPI targets.
- **New, and explicitly a measurement tool, not yet a predictive
  one:** `moa-shape energy` measures real GPU energy consumption of a
  sustained, repeated kernel run. It does not predict energy from
  shape the way this project's existing cost functions predict
  runtime -- that would require fitting a cost function to real
  energy data across many configurations, which does not exist yet.
  The integration arithmetic is independently tested against
  hand-computable cases; the actual GPU-polling and subprocess code
  has **not yet been run against a real GPU**, no GPU being available
  in the environment this was written in.

## License

Apache License 2.0 -- see `LICENSE`.
