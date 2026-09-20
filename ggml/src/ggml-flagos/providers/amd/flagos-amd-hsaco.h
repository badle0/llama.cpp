#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace flagos_amd {

struct hsaco_argument_metadata {
    uint64_t offset = 0;
    uint64_t size = 0;
    std::string value_kind;
    std::string address_space;
};

struct hsaco_kernel_metadata {
    std::string target;
    std::string name;
    std::string symbol;
    std::vector<hsaco_argument_metadata> arguments;
    uint64_t kernarg_segment_size = 0;
    uint64_t max_flat_workgroup_size = 0;
    uint64_t wavefront_size = 0;
    uint64_t group_segment_fixed_size = 0;
    uint64_t private_segment_fixed_size = 0;
};

// Parse the ELF64 AMDGPU code-object note without opening a HIP device.  The
// parser accepts the complete MessagePack scalar/container encoding but keeps
// strict size, nesting, and element limits before allocating any metadata.
// A FlagOS package uses one Triton kernel per HSACO, so multiple metadata notes
// or kernel records are rejected rather than leaving symbol selection
// ambiguous.
bool parse_hsaco_metadata(
        const std::vector<uint8_t> & image,
        hsaco_kernel_metadata & metadata,
        std::string & error);

} // namespace flagos_amd
