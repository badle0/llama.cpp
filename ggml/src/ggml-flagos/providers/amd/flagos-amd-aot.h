#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <array>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace flagos_amd {

inline std::string_view base_gcn_arch(std::string_view arch) {
    return arch.substr(0, arch.find(':'));
}

inline constexpr char tuning_profile_gfx1150_q4ffn_v1[] = "gfx1150-wave32-q4ffn-v1";
struct tuned_kernel_abi {
    std::string_view name;
    size_t argument_count;
    int block_size;
    int tile_m;
    int tile_n;
    int tile_k;
    int num_warps;
    int warp_size;
};

enum class kernel_argument_kind : uint8_t {
    by_value,
    global_buffer,
};

struct kernel_argument_abi {
    size_t offset = 0;
    size_t size = 0;
    kernel_argument_kind kind = kernel_argument_kind::by_value;
};

inline constexpr tuned_kernel_abi tuning_profile_gfx1150_q4ffn_v1_kernels[] = {
#define FLAGOS_AMD_TUNED_KERNEL(name, argument_count, block_size, tile_m, tile_n, tile_k, num_warps, warp_size) \
    { name, argument_count, block_size, tile_m, tile_n, tile_k, num_warps, warp_size },
#include "flagos-amd-tuning-profile.inc"
#undef FLAGOS_AMD_TUNED_KERNEL
};

// The metadata is the ABI contract between FlagTree's Triton compiler and the
// C++ launcher.  block_size is the logical elements per Triton program;
// threads is the physical HIP workgroup size.
struct kernel_metadata {
    std::string name;
    std::string symbol;
    std::string file;
    std::string sha256;
    size_t binary_size = 0;
    int shared_memory = 0;
    // Static LDS is declared by the HSACO itself; shared_memory is Triton's
    // per-launch dynamic LDS request from manifest.json.
    size_t static_shared_memory = 0;
    size_t private_segment_size = 0;
    size_t kernarg_segment_size = 0;
    int num_warps = 0;
    int warp_size = 0;
    int threads = 0;
    int block_size = 0;
    // When true, block_size is an exact logical-shape contract rather than
    // only a launch capacity. Providers must check it before dispatch.
    bool exact_block_size = false;
    // Optional multidimensional tile shape for kernels whose launch grid is
    // not described by one logical block size.  Zero means legacy ABI.
    int tile_m = 0;
    int tile_n = 0;
    int tile_k = 0;
    size_t global_scratch_size = 0;
    size_t global_scratch_align = 1;
    size_t profile_scratch_size = 0;
    size_t profile_scratch_align = 1;
    // Read from a new manifest or populated from a known tuned-profile
    // contract. Negative means a legacy package with no declared call ABI.
    int argument_count = -1;
    std::vector<kernel_argument_abi> argument_abi;
};

// Kernel argument vectors are tiny and are rebuilt for every AOT launch.  A
// fixed-capacity stack representation avoids a host heap allocation for each
// decode token while retaining the same packed-pointer ABI as std::vector.
class kernel_arguments {
public:
    static constexpr size_t capacity = 30;

    kernel_arguments() = default;

    template <typename... T>
    kernel_arguments(T *... values) {
        assign(values...);
    }

    template <typename... T>
    void assign(T *... values) {
        size_ = 0;
        valid_ = true;
        (push_back(values), ...);
    }

    template <typename T>
    void push_back(T * value) {
        static_assert(!std::is_void<T>::value, "pass the address of a host argument value");
        if (size_ < capacity) {
            values_[size_] = const_cast<void *>(static_cast<const void *>(value));
            sizes_[size_] = sizeof(T);
            kinds_[size_] = std::is_pointer<typename std::remove_cv<T>::type>::value
                ? kernel_argument_kind::global_buffer
                : kernel_argument_kind::by_value;
            ++size_;
        } else {
            valid_ = false;
        }
    }

    void * const * data() const { return values_.data(); }
    size_t size() const { return size_; }
    bool valid() const { return valid_; }
    bool matches(const std::vector<kernel_argument_abi> & abi) const {
        if (!valid_ || size_ != abi.size()) {
            return false;
        }
        for (size_t index = 0; index < size_; ++index) {
            if (sizes_[index] != abi[index].size || kinds_[index] != abi[index].kind) {
                return false;
            }
        }
        return true;
    }

private:
    std::array<void *, capacity> values_ {};
    std::array<size_t, capacity> sizes_ {};
    std::array<kernel_argument_kind, capacity> kinds_ {};
    size_t size_ = 0;
    bool valid_ = true;
};

class kernel_registry {
public:
    kernel_registry() = default;
    ~kernel_registry();

    kernel_registry(const kernel_registry &) = delete;
    kernel_registry & operator=(const kernel_registry &) = delete;

    // Load every per-kernel HSACO listed in manifest.json.  The operation is
    // transactional: a malformed entry or failed module load unloads all
    // modules and returns false.
    bool initialize(const std::filesystem::path & directory, int device,
                    const std::string & arch, bool load_modules = true);
    void reset();

    const kernel_metadata * find(std::string_view name) const;
    size_t size() const;
    const std::string & tuning_profile() const;

    bool launch(std::string_view name, hipStream_t stream,
                unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                const kernel_arguments & arguments);

    // When FLAGOS_PROFILE_KERNELS=1 is set, launch records device-side elapsed
    // time per HSACO symbol.  Profiling is deliberately opt-in and is skipped
    // during HIP graph capture so it cannot change capture eligibility.
    void log_profile() const;

private:
    struct loaded_kernel {
        kernel_metadata metadata;
        hipModule_t module = nullptr;
        hipFunction_t function = nullptr;
    };

    struct profile_entry {
        uint64_t launches = 0;
        double milliseconds = 0.0;
    };

    std::unordered_map<std::string, loaded_kernel> kernels_;
    std::unordered_map<std::string_view, loaded_kernel *> kernel_index_;
    std::string tuning_profile_;
    int device_ = -1;
    std::array<unsigned int, 3> max_grid_size_ {};
    unsigned int max_threads_per_block_ = 0;
    unsigned int device_warp_size_ = 0;
    size_t max_shared_memory_per_block_ = 0;
    std::mutex profile_launch_mutex_;
    mutable std::mutex profile_mutex_;
    std::unordered_map<std::string, profile_entry> profile_;
    hipEvent_t profile_start_ = nullptr;
    hipEvent_t profile_end_ = nullptr;
    mutable uint64_t last_logged_launches_ = 0;
    uint64_t profile_launches_ = 0;
    uint64_t profile_dump_launch_interval_ = 0;
    bool profile_capture_skip_logged_ = false;
    bool profile_error_logged_ = false;
    bool profile_enabled_ = false;
};

} // namespace flagos_amd
