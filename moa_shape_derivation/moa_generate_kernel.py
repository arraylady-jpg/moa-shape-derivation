"""
moa_generate_kernel.py -- Aim 2 prototype, second instance: given a
measured MachineShape, generate an actual, compilable OpenACC kernel
file with its parallelism structure and tile dimensions already
filled in -- not just computed as numbers (moa_derive_params.py),
but substituted directly into real source code.

This closes the loop this project's automation work has been building
toward: measure shape (moa_shape_parser.py) -> derive parameters
(moa_derive_params.py) -> generate code (this file) -> compile and
verify (test_moa_automation.py extends to cover this below). No step
in this pipeline is hand-edited.

The template (moa_forward_openacc_template.c) is not a new kernel --
it is the exact, already-validated tiled kernel from this project's
real-hardware work (verified correct on V100, A100, and H100), with
its four machine-derived values replaced by placeholder tokens rather
than hand-edited per target.
"""

from pathlib import Path
from .moa_shape_parser import MachineShape
from .moa_derive_params import derive_openacc_params

_TEMPLATE_PATH = Path(__file__).parent / "moa_forward_openacc_template.c"


def generate_kernel(shape: MachineShape, head_dim: int, dtype: str,
                     schedulers_per_sm: int = 4) -> str:
    """Return the complete, compilable kernel source text for this
    machine shape -- no placeholders remain in the output."""
    params = derive_openacc_params(shape, head_dim, dtype, schedulers_per_sm)
    template = _TEMPLATE_PATH.read_text()

    generated = (
        template
        .replace("{{BLOCK_M}}", str(params.block_m))
        .replace("{{BLOCK_N}}", str(params.block_n))
        .replace("{{NUM_WORKERS}}", str(params.num_workers))
        .replace("{{VECTOR_LENGTH}}", str(params.vector_length))
        .replace("{{DEVICE_NAME}}", shape.device_name)
    )

    remaining = [tok for tok in ("{{BLOCK_M}}", "{{BLOCK_N}}",
                                  "{{NUM_WORKERS}}", "{{VECTOR_LENGTH}}",
                                  "{{DEVICE_NAME}}") if tok in generated]
    if remaining:
        raise RuntimeError(
            f"Template substitution incomplete -- placeholder(s) "
            f"{remaining} still present. Refusing to emit a file with "
            f"unfilled placeholders rather than silently shipping one."
        )

    return generated


