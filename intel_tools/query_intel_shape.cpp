// query_intel_shape.cpp -- queries the exact hardware occupancy
// properties nvaccelinfo/rocminfo give directly, but which clinfo's
// OpenCL view did not expose: EU count, hardware threads per EU,
// slices, and subslices-per-slice. sycl-ls --verbose already showed
// these as available Level-Zero extended device info aspects
// (ext_intel_gpu_eu_count, ext_intel_gpu_hw_threads_per_eu,
// ext_intel_gpu_slices, ext_intel_gpu_subslices_per_slice,
// ext_intel_gpu_eu_count_per_subslice) -- this program queries them
// directly rather than guessing from public spec sheets.
//
// Compile:   icpx -fsycl query_intel_shape.cpp -o query_intel_shape
// Run:       ./query_intel_shape

#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) {
        std::cerr << "No SYCL GPU devices found.\n";
        return 1;
    }

    for (size_t i = 0; i < gpus.size(); ++i) {
        auto &d = gpus[i];
        std::cout << "=== Device " << i << ": "
                  << d.get_info<sycl::info::device::name>() << " ===\n";

        std::cout << "  max_compute_units: "
                  << d.get_info<sycl::info::device::max_compute_units>() << "\n";
        std::cout << "  max_work_group_size: "
                  << d.get_info<sycl::info::device::max_work_group_size>() << "\n";
        std::cout << "  local_mem_size (bytes): "
                  << d.get_info<sycl::info::device::local_mem_size>() << "\n";
        std::cout << "  global_mem_cache_size (bytes): "
                  << d.get_info<sycl::info::device::global_mem_cache_size>() << "\n";

        // Intel Level-Zero extended device info -- the fields
        // sycl-ls --verbose listed as available Aspects on this
        // hardware, queried directly here rather than assumed.
        if (d.has(sycl::aspect::ext_intel_gpu_eu_count)) {
            std::cout << "  ext_intel_gpu_eu_count: "
                      << d.get_info<sycl::ext::intel::info::device::gpu_eu_count>() << "\n";
        }
        if (d.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu)) {
            std::cout << "  ext_intel_gpu_hw_threads_per_eu: "
                      << d.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>() << "\n";
        }
        if (d.has(sycl::aspect::ext_intel_gpu_slices)) {
            std::cout << "  ext_intel_gpu_slices: "
                      << d.get_info<sycl::ext::intel::info::device::gpu_slices>() << "\n";
        }
        if (d.has(sycl::aspect::ext_intel_gpu_subslices_per_slice)) {
            std::cout << "  ext_intel_gpu_subslices_per_slice: "
                      << d.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>() << "\n";
        }
        if (d.has(sycl::aspect::ext_intel_gpu_eu_count_per_subslice)) {
            std::cout << "  ext_intel_gpu_eu_count_per_subslice: "
                      << d.get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>() << "\n";
        }
        if (d.has(sycl::aspect::ext_intel_gpu_eu_simd_width)) {
            std::cout << "  ext_intel_gpu_eu_simd_width: "
                      << d.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>() << "\n";
        }
        std::cout << "\n";
    }
    return 0;
}
