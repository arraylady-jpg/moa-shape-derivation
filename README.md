# moa-shape-derivation

A prototype tool that derives GPU kernel code generation parameters
directly from a target machine's **measured** hardware shape, rather
than from tuning, search, or vendor documentation.

## Install and try it in one line

```bash
pip install moa-shape-derivation      # once published; for now: pip install -e . from a clone
moa-shape derive                      # measures your local NVIDIA GPU and derives kernel parameters
moa-shape --rocm derive               # same, for a local AMD GPU
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
moa-shape --rocm contribute     # AMD
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
    moa_energy_profiler.py          Measures real GPU energy over a sustained kernel run
    moa_performance_predictor.py    Predicts a held-out GPU's absolute time from public specs + calibration
    cli.py                          The moa-shape command (measure / derive / generate / contribute / energy / predict)
examples/                           Real captured NVIDIA output (3 GPUs), a documented-format AMD fixture,
                                     and a predict config reproducing a published held-out result exactly
contributed_shapes/                 Community-contributed measurements (see CONTRIBUTING.md)
test_moa_automation.py              16 automated tests, including a real compile-and-run check
test_moa_energy_profiler.py         9 tests for the energy-integration arithmetic
test_moa_performance_predictor.py   11 tests, including exact reproduction of published cross-machine results
intel_tools/query_intel_shape.cpp   Minimal SYCL program: queries Intel GPU shape fields not exposed by clinfo
```

## Command reference

```bash
moa-shape [--rocm|--intel] measure    [file]                       # print a shape, measured live or from a saved file
moa-shape [--rocm|--intel] derive     [file] [--dtype fp64] [--head-dim 64]
moa-shape [--rocm|--intel] generate   [file] -o kernel.c [--dtype fp64] [--head-dim 64]
moa-shape [--rocm|--intel] contribute [file] [-o out.json]          # see "Help this project" above
moa-shape energy -- <command...> [--duration 5.0]                  # measure real GPU energy of a repeated kernel run
moa-shape predict config.json                                       # predict a held-out GPU's absolute time (see examples/)
```

**`--rocm`/`--intel`, when used, must come *before* the subcommand**
(`moa-shape --rocm derive`, not `moa-shape derive --rocm`) -- this is
how Python's `argparse` subcommand parsing works, and is easy to get
backwards (an earlier version of this README did, for every example,
until real use caught it).

**Intel GPUs work differently from NVIDIA/AMD**: there is no single
standard introspection tool (nothing playing the role `nvaccelinfo`
or `rocminfo` do). `--intel` always requires a captured output file
-- there is no live-measurement mode. First, compile and run this
project's own query tool on the target machine:
```bash
module load intel   # or your site's equivalent oneAPI environment
icpx -fsycl intel_tools/query_intel_shape.cpp -o query_intel_shape
./query_intel_shape > my_intel_shape.txt
```
then parse its captured output the same way as any other file:
```bash
moa-shape --intel derive my_intel_shape.txt
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

`predict` is a genuinely different, harder kind of claim than
`derive`. `derive` computes a parameter from a GPU's *own* measured
shape and checks it against that *same* GPU's own hardware -- a
claim this project has checked 12 independent times with zero error.
`predict` instead estimates a GPU's *absolute* wall-clock time using
only its public peak specs plus other GPUs' real measured timing --
the target's own real data is never an input, by construction (see
`moa_performance_predictor.py`'s docstring). This project's own
validation of this harder claim, across 15 held-out points, reports
32.5% mean error -- real signal, correct order of magnitude and
scaling trend, but nowhere near `derive`'s zero-error track record.
`examples/predict_h100_from_v100_a100.json` reproduces one of those
15 points exactly, using this project's own real, published data.

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
- **A third vendor, validated on real hardware, not predicted:**
  Intel Data Center GPU Max 1550, via this project's own
  `intel_tools/query_intel_shape.cpp` (there is no single standard
  Intel introspection tool the way `nvaccelinfo`/`rocminfo` are for
  NVIDIA/AMD), run for real on TACC Stampede3. Intel's larger local
  memory (128 KiB) produces the largest tile of any vendor tested
  (32×32) -- the tile size tracking measured capacity exactly as
  expected across a third, architecturally distinct vendor. A real
  wrinkle discovered only by actually running this: SYCL enumerates
  each physical GPU twice (once via OpenCL, once via Level-Zero), and
  only the Level-Zero view exposes the fields this project needs --
  the same *kind* of multi-agent-in-one-output issue `rocminfo`
  has (CPU and GPU agents together), a different specific cause.
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
- **`moa-shape predict` reproduces this project's own published
  cross-machine prediction results exactly** (test suite validates
  this to the exact percentage points reported in the position
  paper and TR, not approximately) -- but the underlying result
  itself is honestly mixed: 32.5% mean error across 15 held-out
  points, a categorically harder and less precise claim than
  `derive`'s zero-error parameter checks, with error worst at small
  problem sizes (unmodeled launch overhead) and for GPUs at either
  end of the tested generational range (extrapolation, not
  interpolation). See the module's docstring for the honest
  breakdown.

## License

Apache License 2.0 -- see `LICENSE`.
