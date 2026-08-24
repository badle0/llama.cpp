#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <array>
#include <filesystem>
#include <initializer_list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace flagos_amd {

// The metadata is the ABI contract between FlagTree's Triton compiler and the
// C++ launcher.  block_size is the logical elements per Triton program;
// threads is the physical HIP workgroup size.
struct kernel_metadata {
    std::string name;
    std::string symbol;
    std::string file;
    int shared_memory = 0;
    int num_warps = 0;
    int warp_size = 0;
    int threads = 0;
    int block_size = 0;
    // Optional multidimensional tile shape for kernels whose launch grid is
    // not described by one logical block size.  Zero means legacy ABI.
    int tile_m = 0;
    int tile_n = 0;
    int tile_k = 0;
    size_t profile_scratch_size = 0;
    size_t profile_scratch_align = 1;
};

// Kernel argument vectors are tiny and are rebuilt for every AOT launch.  A
// fixed-capacity stack representation avoids a host heap allocation for each
// decode token while retaining the same packed-pointer ABI as std::vector.
class kernel_arguments {
public:
    static constexpr size_t capacity = 24;

    kernel_arguments() = default;

    kernel_arguments(std::initializer_list<void *> values) {
        assign(values);
    }

    kernel_arguments & operator=(std::initializer_list<void *> values) {
        assign(values);
        return *this;
    }

    void push_back(void * value) {
        if (size_ < capacity) {
            values_[size_++] = value;
        } else {
            valid_ = false;
        }
    }

    void * const * data() const { return values_.data(); }
    size_t size() const { return size_; }
    bool valid() const { return valid_; }

private:
    void assign(std::initializer_list<void *> values) {
        size_ = 0;
        valid_ = true;
        for (void * value : values) {
            push_back(value);
        }
    }

    std::array<void *, capacity> values_ {};
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
                    const std::string & arch);
    void reset();

    const kernel_metadata * find(const std::string & name) const;
    size_t size() const;

    bool launch(const std::string & name, hipStream_t stream,
                unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                const std::vector<void *> & arguments);

    bool launch(const std::string & name, hipStream_t stream,
                unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                const kernel_arguments & arguments);

    bool launch(const std::string & name, hipStream_t stream,
                unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                void * const * arguments, size_t argument_count);

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
    mutable std::mutex profile_mutex_;
    std::unordered_map<std::string, profile_entry> profile_;
    mutable uint64_t last_logged_launches_ = 0;
    uint64_t profile_launches_ = 0;
    uint64_t profile_dump_launch_interval_ = 0;
    bool profile_capture_skip_logged_ = false;
    bool profile_error_logged_ = false;
    bool profile_enabled_ = false;
};

} // namespace flagos_amd
