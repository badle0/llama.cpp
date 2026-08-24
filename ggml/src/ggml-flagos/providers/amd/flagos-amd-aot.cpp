#include "flagos-amd-aot.h"

#include "../../../ggml-impl.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <limits>
#include <array>
#include <cstdlib>
#include <cstring>

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

static bool hip_check(hipError_t result, const char * expression) {
    if (result == hipSuccess) {
        return true;
    }
    GGML_LOG_ERROR("FlagOS AMD AOT: %s failed: %s\n", expression, hipGetErrorString(result));
    return false;
}

static bool is_safe_relative_path(const fs::path & path) {
    if (path.empty() || path.is_absolute()) {
        return false;
    }
    for (const auto & component : path) {
        if (component == "..") {
            return false;
        }
    }
    return true;
}

static bool read_binary(const fs::path & path, std::vector<uint8_t> & bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        GGML_LOG_ERROR("FlagOS AMD AOT: HSACO not found: %s\n", path.string().c_str());
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff length = input.tellg();
    if (length <= 0 || static_cast<uint64_t>(length) > std::numeric_limits<size_t>::max()) {
        GGML_LOG_ERROR("FlagOS AMD AOT: invalid HSACO size: %s\n", path.string().c_str());
        return false;
    }
    input.seekg(0, std::ios::beg);
    bytes.resize(static_cast<size_t>(length));
    input.read(reinterpret_cast<char *>(bytes.data()), length);
    if (!input) {
        GGML_LOG_ERROR("FlagOS AMD AOT: failed to read HSACO: %s\n", path.string().c_str());
        return false;
    }
    return true;
}

static bool parse_metadata(const json & item, const fs::path & directory,
                           const std::string & arch, flagos_amd::kernel_metadata & metadata) {
    if (!item.is_object() || !item.contains("name") || !item.contains("file")) {
        return false;
    }
    metadata.name = item.at("name").get<std::string>();
    metadata.symbol = item.value("symbol", metadata.name);
    metadata.file = item.at("file").get<std::string>();
    metadata.shared_memory = item.value("shared", 0);
    metadata.num_warps = item.value("num_warps", 0);
    metadata.warp_size = item.value("warp_size", 0);
    metadata.block_size = item.value("block_size", 0);
    metadata.tile_m = item.value("tile_m", 0);
    metadata.tile_n = item.value("tile_n", 0);
    metadata.tile_k = item.value("tile_k", 0);
    metadata.profile_scratch_size = item.value("profile_scratch_size", size_t(0));
    metadata.profile_scratch_align = item.value("profile_scratch_align", size_t(1));
    if (metadata.name.empty() || metadata.symbol.empty() ||
        !is_safe_relative_path(metadata.file) || metadata.shared_memory < 0 ||
        metadata.num_warps <= 0 || metadata.warp_size <= 0 ||
        metadata.block_size <= 0 || metadata.num_warps > 1024 ||
        metadata.warp_size > 1024 / metadata.num_warps ||
        metadata.profile_scratch_align == 0 || metadata.tile_m < 0 ||
        metadata.tile_n < 0 || metadata.tile_k < 0) {
        return false;
    }
    metadata.threads = metadata.num_warps * metadata.warp_size;
    if (!arch.empty() && item.contains("arch") && item.at("arch").get<std::string>() != arch) {
        return false;
    }
    return fs::is_regular_file(directory / fs::u8path(metadata.file));
}

} // namespace

namespace flagos_amd {

kernel_registry::~kernel_registry() {
    // Some applications retain backend objects until the process tears down
    // the provider registry.  Emit the final opt-in profile from the AOT
    // registry as well, so diagnostics do not depend on the upper layer's
    // backend-destruction policy.
    log_profile();
    reset();
}

void kernel_registry::reset() {
    for (auto & pair : kernels_) {
        if (pair.second.module != nullptr) {
            hip_check(hipModuleUnload(pair.second.module), "hipModuleUnload");
        }
    }
    kernels_.clear();
    std::lock_guard<std::mutex> lock(profile_mutex_);
    profile_.clear();
    last_logged_launches_ = 0;
    profile_launches_ = 0;
    profile_capture_skip_logged_ = false;
    profile_error_logged_ = false;
}

bool kernel_registry::initialize(const fs::path & directory, int device, const std::string & arch) {
    reset();
    const char * profile = std::getenv("FLAGOS_PROFILE_KERNELS");
    profile_enabled_ = profile != nullptr && std::strcmp(profile, "0") != 0;
    const char * dump_interval = std::getenv("FLAGOS_PROFILE_DUMP_LAUNCH_INTERVAL");
    if (dump_interval != nullptr && dump_interval[0] != '\0') {
        char * end = nullptr;
        const unsigned long long parsed = std::strtoull(dump_interval, &end, 10);
        if (end != dump_interval && *end == '\0') {
            profile_dump_launch_interval_ = static_cast<uint64_t>(parsed);
        } else {
            GGML_LOG_WARN("FlagOS AMD AOT: invalid FLAGOS_PROFILE_DUMP_LAUNCH_INTERVAL=%s; disabling launch profile dumps\n",
                dump_interval);
            profile_dump_launch_interval_ = 0;
        }
    } else {
        profile_dump_launch_interval_ = 0;
    }
    if (profile_enabled_) {
        GGML_LOG_INFO("FlagOS AMD AOT: kernel profiling enabled (launch_dump_interval=%llu)\n",
            static_cast<unsigned long long>(profile_dump_launch_interval_));
    }
    if (!fs::is_directory(directory)) {
        GGML_LOG_ERROR("FlagOS AMD AOT: kernel directory not found: %s\n", directory.string().c_str());
        return false;
    }
    std::ifstream input(directory / "manifest.json");
    if (!input) {
        GGML_LOG_ERROR("FlagOS AMD AOT: manifest not found: %s\n", (directory / "manifest.json").string().c_str());
        return false;
    }

    try {
        json manifest;
        input >> manifest;
        if (manifest.value("format", 0) < 2 || !manifest.contains("kernels") ||
            !manifest.at("kernels").is_array()) {
            GGML_LOG_ERROR("FlagOS AMD AOT: unsupported manifest: %s\n",
                (directory / "manifest.json").string().c_str());
            return false;
        }
        const std::string manifest_arch = manifest.value("arch", "");
        if (!manifest_arch.empty() && !arch.empty() && manifest_arch != arch) {
            GGML_LOG_ERROR("FlagOS AMD AOT: manifest arch %s does not match device arch %s\n",
                manifest_arch.c_str(), arch.c_str());
            return false;
        }

        if (!hip_check(hipSetDevice(device), "hipSetDevice")) {
            return false;
        }
        for (const auto & item : manifest.at("kernels")) {
            kernel_metadata metadata;
            if (!parse_metadata(item, directory, arch, metadata) ||
                kernels_.find(metadata.name) != kernels_.end()) {
                GGML_LOG_ERROR("FlagOS AMD AOT: invalid or duplicate kernel entry\n");
                reset();
                return false;
            }
            std::vector<uint8_t> image;
            if (!read_binary(directory / fs::u8path(metadata.file), image)) {
                reset();
                return false;
            }
            loaded_kernel loaded;
            loaded.metadata = std::move(metadata);
            if (!hip_check(hipModuleLoadData(&loaded.module, image.data()), "hipModuleLoadData") ||
                !hip_check(hipModuleGetFunction(&loaded.function, loaded.module,
                    loaded.metadata.symbol.c_str()), "hipModuleGetFunction")) {
                if (loaded.module != nullptr) {
                    hip_check(hipModuleUnload(loaded.module), "hipModuleUnload");
                }
                reset();
                return false;
            }
            kernels_.emplace(loaded.metadata.name, std::move(loaded));
        }
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("FlagOS AMD AOT: cannot parse manifest: %s\n", error.what());
        reset();
        return false;
    }
    return !kernels_.empty();
}

const kernel_metadata * kernel_registry::find(const std::string & name) const {
    const auto it = kernels_.find(name);
    return it == kernels_.end() ? nullptr : &it->second.metadata;
}

size_t kernel_registry::size() const {
    return kernels_.size();
}

bool kernel_registry::launch(const std::string & name, hipStream_t stream,
                             unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                             const std::vector<void *> & arguments) {
    return launch(name, stream, grid_x, grid_y, grid_z,
        arguments.data(), arguments.size());
}

bool kernel_registry::launch(const std::string & name, hipStream_t stream,
                             unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                             const kernel_arguments & arguments) {
    if (!arguments.valid()) {
        GGML_LOG_ERROR("FlagOS AMD AOT: argument vector overflow for kernel %s\n", name.c_str());
        return false;
    }
    return launch(name, stream, grid_x, grid_y, grid_z,
        arguments.data(), arguments.size());
}

bool kernel_registry::launch(const std::string & name, hipStream_t stream,
                             unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                             void * const * arguments, size_t argument_count) {
    const auto it = kernels_.find(name);
    if (it == kernels_.end()) {
        GGML_LOG_ERROR("FlagOS AMD AOT: kernel %s is not present in the loaded package\n",
            name.c_str());
        return false;
    }
    if (it->second.function == nullptr || grid_x == 0 || grid_y == 0 || grid_z == 0) {
        GGML_LOG_ERROR("FlagOS AMD AOT: invalid launch for %s (function=%p grid=%u,%u,%u)\n",
            name.c_str(), static_cast<void *>(it->second.function), grid_x, grid_y, grid_z);
        return false;
    }
    const auto & metadata = it->second.metadata;
    // Triton AMD's generated launcher appends global_scratch and
    // profile_scratch after the explicit kernel signature.  The current
    // registry has no scratch allocator, so pass null for both and reject
    // packages that require profile scratch until that allocator is added.
    if (metadata.profile_scratch_size != 0) {
        GGML_LOG_ERROR("FlagOS AMD AOT: kernel %s requires profile scratch (%zu bytes)\n",
            name.c_str(), metadata.profile_scratch_size);
        return false;
    }
    void * global_scratch = nullptr;
    void * profile_scratch = nullptr;
    // The generated AMD kernels currently have at most 19 explicit ABI
    // parameters.  Keep the two Triton scratch slots in a small stack buffer
    // rather than copying every argument vector into a heap-allocated vector
    // on every launch.  This matters for decode, where a Qwen token can issue
    // hundreds of AOT launches and the host-side allocator becomes visible in
    // the profile.
    constexpr size_t max_launch_arguments = 32;
    if (arguments == nullptr || argument_count > max_launch_arguments - 2) {
        GGML_LOG_ERROR("FlagOS AMD AOT: kernel %s has too many arguments (%zu)\n",
            name.c_str(), argument_count);
        return false;
    }
    std::array<void *, max_launch_arguments> launch_arguments {};
    std::memcpy(launch_arguments.data(), arguments, argument_count * sizeof(void *));
    launch_arguments[argument_count] = &global_scratch;
    launch_arguments[argument_count + 1] = &profile_scratch;
    const auto launch_raw = [&]() {
        return hip_check(hipModuleLaunchKernel(
            it->second.function, grid_x, grid_y, grid_z,
            static_cast<unsigned int>(metadata.threads), 1, 1,
            static_cast<unsigned int>(metadata.shared_memory), stream,
            launch_arguments.data(), nullptr), "hipModuleLaunchKernel");
    };
    if (!profile_enabled_) {
        return launch_raw();
    }

    const auto log_profile_error_once = [&](const char * stage, hipError_t result) {
        bool log_error = false;
        {
            std::lock_guard<std::mutex> lock(profile_mutex_);
            if (!profile_error_logged_) {
                profile_error_logged_ = true;
                log_error = true;
            }
        }
        if (log_error) {
            GGML_LOG_WARN("FlagOS AMD AOT: kernel profiling failed at %s: %s\n",
                stage, hipGetErrorString(result));
        }
    };

    // Events cannot be inserted into an active HIP graph without changing the
    // graph topology.  Keep profiling transparent to capture experiments.
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    const hipError_t capture_result = hipStreamIsCapturing(stream, &capture_status);
    if (capture_result != hipSuccess || capture_status != hipStreamCaptureStatusNone) {
        bool log_skip = false;
        {
            std::lock_guard<std::mutex> lock(profile_mutex_);
            if (!profile_capture_skip_logged_) {
                profile_capture_skip_logged_ = true;
                log_skip = true;
            }
        }
        if (log_skip) {
            GGML_LOG_WARN("FlagOS AMD AOT: kernel profiling skipped because hipStreamIsCapturing returned %s (status=%d)\n",
                hipGetErrorString(capture_result), static_cast<int>(capture_status));
        }
        return launch_raw();
    }
    hipEvent_t start = nullptr;
    hipEvent_t end = nullptr;
    const hipError_t create_start = hipEventCreate(&start);
    const hipError_t create_end = create_start == hipSuccess ? hipEventCreate(&end) : create_start;
    if (create_start != hipSuccess || create_end != hipSuccess) {
        log_profile_error_once("hipEventCreate", create_start != hipSuccess ? create_start : create_end);
        if (start != nullptr) {
            hip_check(hipEventDestroy(start), "hipEventDestroy (profile start)");
        }
        if (end != nullptr) {
            hip_check(hipEventDestroy(end), "hipEventDestroy (profile end)");
        }
        return launch_raw();
    }
    const hipError_t record_start = hipEventRecord(start, stream);
    if (record_start != hipSuccess) {
        log_profile_error_once("hipEventRecord(start)", record_start);
        hip_check(hipEventDestroy(start), "hipEventDestroy (profile start)");
        hip_check(hipEventDestroy(end), "hipEventDestroy (profile end)");
        return launch_raw();
    }
    if (!launch_raw()) {
        hip_check(hipEventDestroy(start), "hipEventDestroy (profile start)");
        hip_check(hipEventDestroy(end), "hipEventDestroy (profile end)");
        return false;
    }
    // The kernel has already been enqueued successfully.  If a profiling
    // event cannot be recorded/synchronized, preserve the launch result and
    // simply omit this sample rather than turning an optional diagnostic into
    // a backend failure.
    const hipError_t record_end = hipEventRecord(end, stream);
    const hipError_t synchronize_end = record_end == hipSuccess
        ? hipEventSynchronize(end) : record_end;
    if (record_end != hipSuccess) {
        log_profile_error_once("hipEventRecord(end)", record_end);
    } else if (synchronize_end != hipSuccess) {
        log_profile_error_once("hipEventSynchronize(end)", synchronize_end);
    }
    float elapsed_ms = 0.0f;
    const hipError_t elapsed_result = synchronize_end == hipSuccess
        ? hipEventElapsedTime(&elapsed_ms, start, end) : synchronize_end;
    if (synchronize_end == hipSuccess && elapsed_result != hipSuccess) {
        log_profile_error_once("hipEventElapsedTime", elapsed_result);
    }
    const bool timed = elapsed_result == hipSuccess;
    hip_check(hipEventDestroy(start), "hipEventDestroy (profile start)");
    hip_check(hipEventDestroy(end), "hipEventDestroy (profile end)");
    if (timed) {
        const std::string profile_key = name + " grid=" + std::to_string(grid_x) +
            "x" + std::to_string(grid_y) + "x" + std::to_string(grid_z);
        bool dump_profile = false;
        uint64_t launch_count = 0;
        {
            std::lock_guard<std::mutex> lock(profile_mutex_);
            auto & entry = profile_[profile_key];
            ++entry.launches;
            entry.milliseconds += elapsed_ms;
            launch_count = ++profile_launches_;
            dump_profile = profile_dump_launch_interval_ != 0 &&
                launch_count % profile_dump_launch_interval_ == 0;
        }
        if (dump_profile) {
            GGML_LOG_INFO("FlagOS AMD kernel profile snapshot: timed_launches=%llu\n",
                static_cast<unsigned long long>(launch_count));
            log_profile();
        }
    }
    return true;
}

void kernel_registry::log_profile() const {
    if (!profile_enabled_) {
        return;
    }
    std::vector<std::pair<std::string, profile_entry>> snapshot;
    {
        std::lock_guard<std::mutex> lock(profile_mutex_);
        uint64_t total_launches = 0;
        snapshot.reserve(profile_.size());
        for (const auto & pair : profile_) {
            total_launches += pair.second.launches;
            snapshot.push_back(pair);
        }
        // Backend destruction and provider teardown may both ask for the same
        // cumulative snapshot.  Suppress exact duplicates and empty dumps.
        if (total_launches == 0 || total_launches == last_logged_launches_) {
            return;
        }
        last_logged_launches_ = total_launches;
    }
    std::sort(snapshot.begin(), snapshot.end(), [](const auto & left, const auto & right) {
        if (left.second.milliseconds != right.second.milliseconds) {
            return left.second.milliseconds > right.second.milliseconds;
        }
        return left.first < right.first;
    });
    for (const auto & pair : snapshot) {
        const auto & entry = pair.second;
        const double average_us = entry.launches == 0
            ? 0.0 : entry.milliseconds * 1000.0 / static_cast<double>(entry.launches);
        GGML_LOG_INFO("FlagOS AMD kernel profile: name=%s launches=%llu time_ms=%.3f avg_us=%.3f\n",
            pair.first.c_str(), static_cast<unsigned long long>(entry.launches),
            entry.milliseconds, average_us);
    }
}

} // namespace flagos_amd
