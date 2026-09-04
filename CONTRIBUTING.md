# Contributing Hardware Shape Data

This project's core claim is that GPU kernel parameters can be
*derived* from measured hardware shape rather than tuned per chip.
That claim gets stronger with every additional, genuinely different
GPU it's been checked against — and the project's own access to
hardware is limited to whatever a handful of researchers can reach.
If you have access to a GPU this project hasn't tested yet
(especially anything non-NVIDIA, or an NVIDIA generation not yet in
`examples/`), running the tool and contributing what it measures is
the single most useful thing you can do for this project.

## What gets collected, precisely

Only architectural specifications already present in this project's
`MachineShape` structure: device name, compute unit / SM count,
warp or wavefront width, maximum resident threads per compute unit,
shared memory / LDS capacity per block, and L2 cache size. Nothing
else.

**Never collected, by design:** hostname, username, IP address,
institution name, cluster name, file paths, environment variables,
or anything else that identifies you, your organization, or your
network. The `contribute` command's own code
(`moa_shape_derivation/cli.py`) is short and readable — please check
it yourself rather than take this claim on faith.

## Nothing is ever submitted automatically

Running `moa-shape contribute` **only ever writes a local JSON file
on your own machine.** It never makes a network call, never opens a
connection, never submits anything anywhere. You decide, afterward,
whether to share that file, by manually completing one of the two
steps below.

## How to contribute

```bash
# On the machine with the GPU you want to contribute:
pip install moa-shape-derivation      # or: pip install -e . from a clone
moa-shape contribute                  # NVIDIA, via nvaccelinfo
moa-shape contribute --rocm           # AMD, via rocminfo
```

This writes a file like `contribution_Tesla_V100-SXM2-32GB.json` in
your current directory. Open it and check it yourself — it should
contain only the fields listed above. Then, either:

1. **Open a pull request** adding your file to `contributed_shapes/`
   in this repository, or
2. **Open a GitHub issue** and attach the file, if you'd rather not
   open a pull request yourself.

Either way, please also mention (in the PR description or issue) the
actual marketing name of the GPU if `rocminfo`'s `Name:` field
reported an architecture codename rather than a product name (e.g.
`gfx908` rather than "MI100") — this project doesn't have a complete
codename-to-product mapping and would rather ask than guess.

## Why this helps

Every additional real device shape checked against this project's
derivation is a genuine test of whether it generalizes, not just a
bigger demo. A contribution that *breaks* the derivation (produces
an obviously wrong or nonsensical parameter) is exactly as valuable
as one that confirms it -- please contribute it either way, and
consider opening an issue describing what you saw.
