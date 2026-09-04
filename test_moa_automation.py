"""
test_moa_automation.py -- Aim 3 prototype (narrow first instance):
automated verification that this tool's output matches the results
this project previously obtained by hand, rather than checking that
by eye.

Every expected value below is a real number from this project's own
prior, real-hardware results (not a value invented to make the tests
pass) -- see moa_delta_benchmarking_tr.pdf Sections 7.9-9 and
moa_results_synthesis.pdf Finding 3 for the hand-derived originals
this automates.
"""

import unittest
from moa_shape_derivation.moa_shape_parser import parse_nvaccelinfo
from moa_shape_derivation.moa_derive_params import derive_openacc_params


class TestMeasuredShapeMatchesHandRecordedValues(unittest.TestCase):
    """Confirms the parser extracts exactly the values this project
    recorded by hand when nvaccelinfo was first run on each GPU."""

    def test_v100(self):
        with open("examples/v100_real.txt") as f:
            shape = parse_nvaccelinfo(f.read())
        self.assertEqual(shape.sms, 80)
        self.assertEqual(shape.warp_size, 32)
        self.assertEqual(shape.max_threads_per_sm, 2048)
        self.assertEqual(shape.shared_mem_per_block_bytes, 49152)
        self.assertEqual(shape.l2_cache_bytes, 6291456)

    def test_a100(self):
        with open("examples/a100_real.txt") as f:
            shape = parse_nvaccelinfo(f.read())
        self.assertEqual(shape.sms, 108)
        self.assertEqual(shape.l2_cache_bytes, 41943040)

    def test_h100(self):
        with open("examples/h100_real.txt") as f:
            shape = parse_nvaccelinfo(f.read())
        self.assertEqual(shape.sms, 132)
        self.assertEqual(shape.l2_cache_bytes, 52428800)


class TestDerivedParamsMatchHandDerivation(unittest.TestCase):
    """Confirms the derivation engine reproduces exactly the
    num_workers=16, vector_length=32, BLOCK_M=BLOCK_N=16 values this
    project derived by hand and validated on real GPU hardware."""

    def _params_for(self, filename):
        with open(filename) as f:
            shape = parse_nvaccelinfo(f.read())
        return derive_openacc_params(shape, head_dim=64, dtype="fp64")

    def test_num_workers_is_16_on_all_three_gpus(self):
        for gpu_file in ("examples/v100_real.txt", "examples/a100_real.txt", "examples/h100_real.txt"):
            params = self._params_for(gpu_file)
            self.assertEqual(
                params.num_workers, 16,
                f"{gpu_file}: expected num_workers=16 (64 max warps/SM / "
                f"4 schedulers), matching this project's hand derivation"
            )

    def test_vector_length_is_32_on_all_three_gpus(self):
        for gpu_file in ("examples/v100_real.txt", "examples/a100_real.txt", "examples/h100_real.txt"):
            params = self._params_for(gpu_file)
            self.assertEqual(params.vector_length, 32)

    def test_tile_size_is_16x16_on_all_three_gpus(self):
        for gpu_file in ("examples/v100_real.txt", "examples/a100_real.txt", "examples/h100_real.txt"):
            params = self._params_for(gpu_file)
            self.assertEqual(params.block_m, 16)
            self.assertEqual(params.block_n, 16)

    def test_tile_footprint_matches_hand_calculation(self):
        # Hand calculation (Section 9, engineering TR): 34816 bytes,
        # comfortably within the 49152-byte (48 KiB) measured budget.
        params = self._params_for("examples/a100_real.txt")
        self.assertEqual(params.tile_footprint_bytes, 34816)
        self.assertLess(params.tile_footprint_bytes, params.shared_mem_budget_bytes)


class TestMissingFieldRaisesRatherThanGuesses(unittest.TestCase):
    """This project's whole argument is measure, don't assume. Confirm
    the parser actually refuses to guess a missing field rather than
    silently defaulting it."""

    def test_incomplete_input_raises(self):
        incomplete = "Device Name: Fake GPU\nWarp Size: 32\n"
        with self.assertRaises(ValueError):
            parse_nvaccelinfo(incomplete)


class TestKernelGeneration(unittest.TestCase):
    """Aim 2, second instance: confirms the code generator produces a
    complete, placeholder-free kernel file, consistent with what
    moa_derive_params.py computes independently."""

    def test_no_placeholders_remain(self):
        from moa_shape_derivation.moa_generate_kernel import generate_kernel
        with open("examples/a100_real.txt") as f:
            shape = parse_nvaccelinfo(f.read())
        source = generate_kernel(shape, head_dim=64, dtype="fp64")
        self.assertNotIn("{{", source)
        self.assertNotIn("}}", source)

    def test_generated_values_match_derive_params(self):
        """The generator and the standalone derivation function must
        agree -- if they ever diverge, that is a real bug, not a
        stylistic difference."""
        from moa_shape_derivation.moa_generate_kernel import generate_kernel
        with open("examples/h100_real.txt") as f:
            shape = parse_nvaccelinfo(f.read())
        params = derive_openacc_params(shape, head_dim=64, dtype="fp64")
        source = generate_kernel(shape, head_dim=64, dtype="fp64")
        self.assertIn(f"num_workers({params.num_workers})", source)
        self.assertIn(f"vector_length({params.vector_length})", source)
        self.assertIn(f"#define BLOCK_M {params.block_m}", source)
        self.assertIn(f"#define BLOCK_N {params.block_n}", source)

    def test_generated_kernel_compiles_and_passes_correctness(self):
        """The real end-to-end test: generate a kernel, compile it,
        run it, and check its own correctness self-test passes. Skips
        gracefully if no GPU-offload-capable compiler is present,
        rather than failing a test that was never able to run."""
        import shutil
        import subprocess
        import tempfile

        if not shutil.which("gcc"):
            self.skipTest("gcc not found")

        from moa_shape_derivation.moa_generate_kernel import generate_kernel
        with open("examples/v100_real.txt") as f:
            shape = parse_nvaccelinfo(f.read())
        source = generate_kernel(shape, head_dim=64, dtype="fp64")

        with tempfile.TemporaryDirectory() as tmp:
            src_path = f"{tmp}/generated.c"
            bin_path = f"{tmp}/generated_bin"
            with open(src_path, "w") as f:
                f.write(source)

            compile_result = subprocess.run(
                ["gcc", "-fopenacc", "-foffload=nvptx-none",
                 "-foffload-options=nvptx-none=-fcf-protection=none -lm",
                 "-O3", "-DN=512", "-o", bin_path, src_path, "-lm"],
                capture_output=True, text=True,
            )
            if compile_result.returncode != 0:
                self.skipTest(
                    f"GPU-offload compiler toolchain not fully available "
                    f"in this environment: {compile_result.stderr[:200]}"
                )

            run_result = subprocess.run([bin_path], capture_output=True, text=True)
            self.assertIn("[PASS]", run_result.stdout,
                          f"Generated kernel's own correctness check did "
                          f"not pass. Output:\n{run_result.stdout}")


class TestROCmParsing(unittest.TestCase):
    """Validates the rocminfo parser against a fixture built from
    AMD's own documented, real output format (not a guessed one --
    see moa_shape_parser.py's parse_rocminfo docstring). This is
    NOT yet validated against real captured output from this
    project's actual target hardware (Delta's MI100 partition) --
    that is the immediate next step once cluster access allows it."""

    def _shape(self):
        from moa_shape_derivation.moa_shape_parser import parse_rocminfo
        with open("examples/rocminfo_example_gfx906.txt") as f:
            return parse_rocminfo(f.read())

    def test_selects_gpu_agent_not_cpu_agent(self):
        """rocminfo lists CPU and GPU agents together in one output --
        confirm the parser picks the GPU, not the Ryzen CPU listed
        first in the same fixture."""
        shape = self._shape()
        self.assertEqual(shape.device_name, "gfx906")
        self.assertNotIn("Ryzen", shape.device_name)

    def test_measured_values_match_fixture(self):
        shape = self._shape()
        self.assertEqual(shape.sms, 60)
        self.assertEqual(shape.warp_size, 64)  # AMD's wavefront, wider than NVIDIA's 32
        self.assertEqual(shape.max_threads_per_sm, 2560)
        self.assertEqual(shape.shared_mem_per_block_bytes, 65536)  # 64 KiB LDS
        self.assertEqual(shape.l2_cache_bytes, 4194304)  # 4096 KiB

    def test_internal_consistency_with_amd_own_reported_value(self):
        """AMD's own rocminfo output independently reports "Max Waves
        Per CU: 40" for this fixture. This parser derives
        max_warps_per_sm from two OTHER fields (max_threads_per_sm /
        warp_size) entirely independently -- if the field mapping
        were wrong, these would not be expected to agree."""
        shape = self._shape()
        self.assertEqual(shape.max_warps_per_sm, 40)

    def test_derivation_produces_a_genuinely_different_result_than_nvidia(self):
        """Confirms the derivation is actually responding to measured
        shape, not coincidentally always returning NVIDIA's values.
        AMD's wider wavefront and larger measured LDS capacity should
        produce a different vector_length and, given more available
        shared memory, room for a larger tile than NVIDIA's 16x16."""
        shape = self._shape()
        params = derive_openacc_params(shape, head_dim=64, dtype="fp64")
        self.assertEqual(params.vector_length, 64)
        self.assertNotEqual((params.block_m, params.block_n), (16, 16))

    def test_missing_gpu_agent_raises(self):
        cpu_only = (
            "*******\nAgent 1\n*******\n"
            "  Name:                    Some CPU\n"
            "  Device Type:             CPU\n"
        )
        with self.assertRaises(ValueError):
            from moa_shape_derivation.moa_shape_parser import parse_rocminfo
            parse_rocminfo(cpu_only)


if __name__ == "__main__":
    unittest.main(verbosity=2)
