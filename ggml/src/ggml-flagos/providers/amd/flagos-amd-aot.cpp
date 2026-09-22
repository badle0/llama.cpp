#include "flagos-amd-aot.h"
#include "flagos-amd-hsaco.h"

#include "../../../ggml-impl.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <unordered_set>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

static constexpr uintmax_t MAX_MANIFEST_BYTES = 4ULL * 1024 * 1024;
static constexpr uintmax_t MAX_HSACO_BYTES = 256ULL * 1024 * 1024;
static constexpr uint64_t MAX_PACKAGE_BYTES = 4ULL * 1024 * 1024 * 1024;
static constexpr size_t MAX_KERNELS = 4096;

static bool hip_check(hipError_t result, const char * expression) {
    if (result == hipSuccess) {
        return true;
    }
    GGML_LOG_ERROR("FlagOS AMD AOT: %s failed: %s\n", expression, hipGetErrorString(result));
    return false;
}

static bool is_safe_relative_path(const fs::path & path) {
    if (path.empty() || path.is_absolute() || path.has_parent_path()) {
        return false;
    }
    const std::string native = path.string();
    if (native == "." || native == ".." || native.find('\0') != std::string::npos ||
        native.find('/') != std::string::npos || native.find('\\') != std::string::npos) {
        return false;
    }
    for (const auto & component : path) {
        if (component == "..") {
            return false;
        }
    }
    return true;
}

static bool is_regular_package_file(const fs::path & path) {
    std::error_code error;
    const fs::file_status status = fs::symlink_status(path, error);
    return !error && fs::is_regular_file(status);
}

static bool read_regular_file(
        const fs::path & path, uintmax_t expected_size, uintmax_t maximum_size,
        const char * description, std::vector<uint8_t> & bytes) {
    bytes.clear();
#if !defined(_WIN32)
    int flags = O_RDONLY;
#if defined(O_CLOEXEC)
    flags |= O_CLOEXEC;
#endif
#if defined(O_NOFOLLOW)
    flags |= O_NOFOLLOW;
#endif
    const int descriptor = open(path.c_str(), flags);
    if (descriptor < 0) {
        GGML_LOG_ERROR("FlagOS AMD AOT: cannot open %s %s: %s\n",
            description, path.string().c_str(), std::strerror(errno));
        return false;
    }
    struct stat before {};
    if (fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode) || before.st_size <= 0 ||
        static_cast<uintmax_t>(before.st_size) > maximum_size ||
        static_cast<uintmax_t>(before.st_size) > std::numeric_limits<size_t>::max() ||
        (expected_size != 0 && static_cast<uintmax_t>(before.st_size) != expected_size)) {
        GGML_LOG_ERROR("FlagOS AMD AOT: invalid %s size: %s\n",
            description, path.string().c_str());
        close(descriptor);
        return false;
    }
    bytes.resize(static_cast<size_t>(before.st_size));
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = read(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            GGML_LOG_ERROR("FlagOS AMD AOT: failed to read %s: %s\n",
                description, path.string().c_str());
            close(descriptor);
            bytes.clear();
            return false;
        }
        offset += static_cast<size_t>(count);
    }
    struct stat after {};
    const bool stable = fstat(descriptor, &after) == 0 &&
        before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
        before.st_size == after.st_size &&
        before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
        before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
        before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
        before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
    close(descriptor);
    if (!stable) {
        GGML_LOG_ERROR("FlagOS AMD AOT: %s changed while being read: %s\n",
            description, path.string().c_str());
        bytes.clear();
        return false;
    }
    return true;
#else
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        GGML_LOG_ERROR("FlagOS AMD AOT: %s not found: %s\n",
            description, path.string().c_str());
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff length = input.tellg();
    if (length <= 0 || static_cast<uint64_t>(length) > std::numeric_limits<size_t>::max() ||
        static_cast<uint64_t>(length) > maximum_size ||
        (expected_size != 0 && static_cast<uint64_t>(length) != expected_size)) {
        GGML_LOG_ERROR("FlagOS AMD AOT: invalid %s size: %s\n",
            description, path.string().c_str());
        return false;
    }
    input.seekg(0, std::ios::beg);
    bytes.resize(static_cast<size_t>(length));
    input.read(reinterpret_cast<char *>(bytes.data()), length);
    if (!input) {
        GGML_LOG_ERROR("FlagOS AMD AOT: failed to read %s: %s\n",
            description, path.string().c_str());
        return false;
    }
    return true;
#endif
}

static bool read_binary(const fs::path & path, size_t expected_size, std::vector<uint8_t> & bytes) {
    return read_regular_file(path, expected_size, MAX_HSACO_BYTES, "HSACO", bytes);
}

struct sha256_context {
    uint32_t state[8];
    uint64_t bit_length;
    uint8_t buffer[64];
    size_t buffer_length;
};

static constexpr uint32_t SHA256_CONSTANTS[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static uint32_t rotate_right(uint32_t value, unsigned int shift) {
    return (value >> shift) | (value << (32 - shift));
}

static void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
    uint32_t words[64];
    for (int i = 0; i < 16; ++i) {
        words[i] = (uint32_t(block[i * 4]) << 24) |
                   (uint32_t(block[i * 4 + 1]) << 16) |
                   (uint32_t(block[i * 4 + 2]) << 8) |
                    uint32_t(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotate_right(words[i - 15], 7) ^ rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3);
        const uint32_t s1 = rotate_right(words[i - 2], 17) ^ rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const uint32_t choose = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + sum1 + choose + SHA256_CONSTANTS[i] + words[i];
        const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

static void sha256_initialize(sha256_context & context) {
    context.state[0] = 0x6a09e667;
    context.state[1] = 0xbb67ae85;
    context.state[2] = 0x3c6ef372;
    context.state[3] = 0xa54ff53a;
    context.state[4] = 0x510e527f;
    context.state[5] = 0x9b05688c;
    context.state[6] = 0x1f83d9ab;
    context.state[7] = 0x5be0cd19;
    context.bit_length = 0;
    context.buffer_length = 0;
}

static void sha256_update(sha256_context & context, const uint8_t * data, size_t length) {
    context.bit_length += uint64_t(length) * 8;
    if (context.buffer_length != 0) {
        const size_t copy_length = std::min(length, sizeof(context.buffer) - context.buffer_length);
        std::memcpy(context.buffer + context.buffer_length, data, copy_length);
        context.buffer_length += copy_length;
        data += copy_length;
        length -= copy_length;
        if (context.buffer_length == sizeof(context.buffer)) {
            sha256_compress(context.state, context.buffer);
            context.buffer_length = 0;
        }
    }
    while (length >= sizeof(context.buffer)) {
        sha256_compress(context.state, data);
        data += sizeof(context.buffer);
        length -= sizeof(context.buffer);
    }
    if (length != 0) {
        std::memcpy(context.buffer, data, length);
        context.buffer_length = length;
    }
}

static std::string sha256_hex(const std::vector<uint8_t> & bytes) {
    sha256_context context;
    sha256_initialize(context);
    sha256_update(context, bytes.data(), bytes.size());
    const uint64_t bit_length = context.bit_length;
    context.buffer[context.buffer_length++] = 0x80;
    if (context.buffer_length > 56) {
        while (context.buffer_length < sizeof(context.buffer)) {
            context.buffer[context.buffer_length++] = 0;
        }
        sha256_compress(context.state, context.buffer);
        context.buffer_length = 0;
    }
    while (context.buffer_length < 56) {
        context.buffer[context.buffer_length++] = 0;
    }
    for (int i = 7; i >= 0; --i) {
        context.buffer[context.buffer_length++] = uint8_t(bit_length >> (i * 8));
    }
    sha256_compress(context.state, context.buffer);

    static constexpr char HEX[] = "0123456789abcdef";
    std::string result(64, '0');
    for (int i = 0; i < 32; ++i) {
        const uint8_t value = uint8_t(context.state[i / 4] >> (24 - (i % 4) * 8));
        result[i * 2] = HEX[value >> 4];
        result[i * 2 + 1] = HEX[value & 0x0f];
    }
    return result;
}

static bool is_sha256(const std::string & value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
    });
}

static bool hsaco_target_matches_arch(const std::string & target, const std::string & arch) {
    static constexpr char prefix[] = "amdgcn-amd-amdhsa--";
    if (arch.empty() || target.compare(0, sizeof(prefix) - 1, prefix) != 0) {
        return false;
    }
    const std::string target_arch = target.substr(sizeof(prefix) - 1);
    const size_t feature = target_arch.find(':');
    return target_arch.substr(0, feature) == arch;
}

static bool validate_hsaco_contract(
        const std::vector<uint8_t> & image,
        const std::string & manifest_arch,
        flagos_amd::kernel_metadata & metadata) {
    flagos_amd::hsaco_kernel_metadata hsaco;
    std::string error;
    if (!flagos_amd::parse_hsaco_metadata(image, hsaco, error)) {
        GGML_LOG_ERROR("FlagOS AMD AOT: invalid HSACO metadata for %s: %s\n",
            metadata.file.c_str(), error.c_str());
        return false;
    }
    const std::string code_symbol = metadata.symbol + ".kd";
    if (!hsaco_target_matches_arch(hsaco.target, manifest_arch) ||
        hsaco.name != metadata.symbol || hsaco.symbol != code_symbol ||
        hsaco.wavefront_size != static_cast<uint64_t>(metadata.warp_size) ||
        hsaco.max_flat_workgroup_size != static_cast<uint64_t>(metadata.threads) ||
        hsaco.arguments.size() < 2 || hsaco.arguments.size() - 2 > 30) {
        GGML_LOG_ERROR(
            "FlagOS AMD AOT: HSACO/manifest contract mismatch for %s "
            "(target=%s name=%s symbol=%s args=%zu wave=%llu workgroup=%llu)\n",
            metadata.name.c_str(), hsaco.target.c_str(), hsaco.name.c_str(), hsaco.symbol.c_str(),
            hsaco.arguments.size(), static_cast<unsigned long long>(hsaco.wavefront_size),
            static_cast<unsigned long long>(hsaco.max_flat_workgroup_size));
        return false;
    }
    const int explicit_arguments = static_cast<int>(hsaco.arguments.size() - 2);
    if (metadata.argument_count >= 0 && metadata.argument_count != explicit_arguments) {
        GGML_LOG_ERROR("FlagOS AMD AOT: HSACO argument count mismatch for %s: manifest=%d HSACO=%d\n",
            metadata.name.c_str(), metadata.argument_count, explicit_arguments);
        return false;
    }
    if (hsaco.group_segment_fixed_size > std::numeric_limits<size_t>::max() ||
        hsaco.private_segment_fixed_size > std::numeric_limits<size_t>::max() ||
        hsaco.kernarg_segment_size > std::numeric_limits<size_t>::max()) {
        GGML_LOG_ERROR("FlagOS AMD AOT: HSACO resource metadata overflows host size_t for %s\n",
            metadata.name.c_str());
        return false;
    }
    metadata.argument_abi.clear();
    metadata.argument_abi.reserve(static_cast<size_t>(explicit_arguments));
    for (int index = 0; index < explicit_arguments; ++index) {
        const auto & argument = hsaco.arguments[static_cast<size_t>(index)];
        if (argument.offset > std::numeric_limits<size_t>::max() ||
            argument.size > std::numeric_limits<size_t>::max() ||
            (argument.value_kind != "by_value" && argument.value_kind != "global_buffer")) {
            GGML_LOG_ERROR("FlagOS AMD AOT: unsupported HSACO argument ABI for %s at index %d\n",
                metadata.name.c_str(), index);
            return false;
        }
        metadata.argument_abi.push_back({
            static_cast<size_t>(argument.offset),
            static_cast<size_t>(argument.size),
            argument.value_kind == "global_buffer"
                ? flagos_amd::kernel_argument_kind::global_buffer
                : flagos_amd::kernel_argument_kind::by_value,
        });
    }
    metadata.argument_count = explicit_arguments;
    metadata.static_shared_memory = static_cast<size_t>(hsaco.group_segment_fixed_size);
    metadata.private_segment_size = static_cast<size_t>(hsaco.private_segment_fixed_size);
    metadata.kernarg_segment_size = static_cast<size_t>(hsaco.kernarg_segment_size);
    return true;
}

template <typename T>
static bool read_integer(const json & item, const char * key, T fallback, T & value) {
    if (!item.contains(key)) {
        value = fallback;
        return true;
    }
    const json & field = item.at(key);
    if (!field.is_number_integer() && !field.is_number_unsigned()) {
        return false;
    }
    // nlohmann::json permits narrowing integer conversions (including signed
    // to unsigned) rather than reporting them as errors.  Validate in the
    // source representation first so values such as 2^32+1 cannot wrap to a
    // small, apparently valid block size or warp count.
    if (field.type() == json::value_t::number_integer) {
        const json::number_integer_t parsed = field.get<json::number_integer_t>();
        if constexpr (std::numeric_limits<T>::is_signed) {
            if (parsed < static_cast<json::number_integer_t>(std::numeric_limits<T>::min()) ||
                parsed > static_cast<json::number_integer_t>(std::numeric_limits<T>::max())) {
                return false;
            }
        } else {
            if (parsed < 0 ||
                static_cast<json::number_unsigned_t>(parsed) >
                    static_cast<json::number_unsigned_t>(std::numeric_limits<T>::max())) {
                return false;
            }
        }
        value = static_cast<T>(parsed);
    } else {
        const json::number_unsigned_t parsed = field.get<json::number_unsigned_t>();
        if (parsed > static_cast<json::number_unsigned_t>(std::numeric_limits<T>::max())) {
            return false;
        }
        value = static_cast<T>(parsed);
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
    metadata.sha256 = item.value("sha256", "");
    const bool declares_argument_count = item.contains("argument_count");
    if (item.contains("exact_block_size") && !item.at("exact_block_size").is_boolean()) {
        return false;
    }
    metadata.exact_block_size = item.value("exact_block_size", false);
    if (!read_integer(item, "size_bytes", size_t(0), metadata.binary_size) ||
        !read_integer(item, "shared", 0, metadata.shared_memory) ||
        !read_integer(item, "num_warps", 0, metadata.num_warps) ||
        !read_integer(item, "warp_size", 0, metadata.warp_size) ||
        !read_integer(item, "block_size", 0, metadata.block_size) ||
        !read_integer(item, "tile_m", 0, metadata.tile_m) ||
        !read_integer(item, "tile_n", 0, metadata.tile_n) ||
        !read_integer(item, "tile_k", 0, metadata.tile_k) ||
        !read_integer(item, "argument_count", -1, metadata.argument_count) ||
        !read_integer(item, "global_scratch_size", size_t(0), metadata.global_scratch_size) ||
        !read_integer(item, "global_scratch_align", size_t(1), metadata.global_scratch_align) ||
        !read_integer(item, "profile_scratch_size", size_t(0), metadata.profile_scratch_size) ||
        !read_integer(item, "profile_scratch_align", size_t(1), metadata.profile_scratch_align) ||
        metadata.name.empty() || metadata.symbol.empty() ||
        metadata.name.find('\0') != std::string::npos ||
        metadata.symbol.find('\0') != std::string::npos ||
        !is_safe_relative_path(metadata.file) || metadata.shared_memory < 0 ||
        metadata.num_warps <= 0 || metadata.warp_size <= 0 ||
        metadata.block_size <= 0 || metadata.num_warps > 1024 ||
        metadata.warp_size > 1024 / metadata.num_warps ||
        metadata.global_scratch_align == 0 || metadata.profile_scratch_align == 0 ||
        metadata.global_scratch_size != 0 || metadata.profile_scratch_size != 0 ||
        metadata.tile_m < 0 ||
        metadata.tile_n < 0 || metadata.tile_k < 0 ||
        metadata.argument_count < -1 || metadata.argument_count > 30 ||
        (declares_argument_count && metadata.argument_count < 0) ||
        metadata.binary_size > MAX_HSACO_BYTES ||
        (!metadata.sha256.empty() && !is_sha256(metadata.sha256))) {
        return false;
    }
    metadata.threads = metadata.num_warps * metadata.warp_size;
    if (!arch.empty() && item.contains("arch") && item.at("arch").get<std::string>() != arch) {
        return false;
    }
    return is_regular_package_file(directory / fs::u8path(metadata.file));
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
    if (device_ >= 0) {
        hip_check(hipSetDevice(device_), "hipSetDevice before registry reset");
    }
    {
        std::lock_guard<std::mutex> launch_lock(profile_launch_mutex_);
        if (profile_start_ != nullptr) {
            hip_check(hipEventDestroy(profile_start_), "hipEventDestroy (profile start)");
            profile_start_ = nullptr;
        }
        if (profile_end_ != nullptr) {
            hip_check(hipEventDestroy(profile_end_), "hipEventDestroy (profile end)");
            profile_end_ = nullptr;
        }
    }
    kernel_index_.clear();
    tuning_profile_.clear();
    for (auto & pair : kernels_) {
        if (pair.second.module != nullptr) {
            hip_check(hipModuleUnload(pair.second.module), "hipModuleUnload");
        }
    }
    kernels_.clear();
    device_ = -1;
    max_grid_size_ = {};
    max_threads_per_block_ = 0;
    device_warp_size_ = 0;
    max_shared_memory_per_block_ = 0;
    std::lock_guard<std::mutex> lock(profile_mutex_);
    profile_.clear();
    last_logged_launches_ = 0;
    profile_launches_ = 0;
    profile_dump_launch_interval_ = 0;
    profile_capture_skip_logged_ = false;
    profile_error_logged_ = false;
    profile_enabled_ = false;
}

bool kernel_registry::initialize(
        const fs::path & directory, int device, const std::string & arch, bool load_modules) {
    reset();
    const char * profile = std::getenv("FLAGOS_PROFILE_KERNELS");
    profile_enabled_ = load_modules && profile != nullptr && std::strcmp(profile, "0") != 0;
    const char * dump_interval = std::getenv("FLAGOS_PROFILE_DUMP_LAUNCH_INTERVAL");
    if (dump_interval != nullptr && dump_interval[0] != '\0') {
        char * end = nullptr;
        errno = 0;
        const unsigned long long parsed = std::strtoull(dump_interval, &end, 10);
        if (dump_interval[0] >= '0' && dump_interval[0] <= '9' &&
            end != dump_interval && *end == '\0' && errno != ERANGE &&
            parsed <= std::numeric_limits<uint64_t>::max()) {
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
        reset();
        return false;
    }
    const fs::path manifest_path = directory / "manifest.json";
    if (!is_regular_package_file(manifest_path)) {
        GGML_LOG_ERROR("FlagOS AMD AOT: manifest must be a package-local regular file: %s\n",
            manifest_path.string().c_str());
        reset();
        return false;
    }
    std::error_code manifest_error;
    const uintmax_t manifest_size = fs::file_size(manifest_path, manifest_error);
    if (manifest_error || manifest_size == 0 || manifest_size > MAX_MANIFEST_BYTES) {
        GGML_LOG_ERROR("FlagOS AMD AOT: invalid manifest size: %s\n", manifest_path.string().c_str());
        reset();
        return false;
    }
    std::vector<uint8_t> manifest_bytes;
    if (!read_regular_file(
            manifest_path, manifest_size, MAX_MANIFEST_BYTES, "manifest", manifest_bytes)) {
        reset();
        return false;
    }

    try {
        const json manifest = json::parse(manifest_bytes.begin(), manifest_bytes.end());
        int manifest_format = 0;
        if (!manifest.is_object() || !read_integer(manifest, "format", 0, manifest_format) ||
            manifest_format != 2 || !manifest.contains("arch") ||
            !manifest.at("arch").is_string() || !manifest.contains("kernels") ||
            !manifest.at("kernels").is_array() || manifest.at("kernels").empty() ||
            manifest.at("kernels").size() > MAX_KERNELS) {
            GGML_LOG_ERROR("FlagOS AMD AOT: unsupported manifest: %s\n",
                (directory / "manifest.json").string().c_str());
            reset();
            return false;
        }
        const std::string manifest_arch = manifest.at("arch").get<std::string>();
        if (manifest_arch.empty() || manifest_arch.find('\0') != std::string::npos) {
            GGML_LOG_ERROR("FlagOS AMD AOT: invalid manifest architecture: %s\n",
                manifest_path.string().c_str());
            reset();
            return false;
        }
        if (manifest.contains("package")) {
            const auto & package = manifest.at("package");
            if (!package.is_object() || package.value("abi", "") != "flagos-amd-aot-v2" ||
                package.value("backend", "") != "hip" || package.value("binary_format", "") != "hsaco") {
                GGML_LOG_ERROR("FlagOS AMD AOT: incompatible package metadata: %s\n",
                    (directory / "manifest.json").string().c_str());
                reset();
                return false;
            }
            const auto safe_optional_string = [&package](const char * field) {
                if (!package.contains(field)) {
                    return true;
                }
                return package.at(field).is_string() &&
                    package.at(field).get_ref<const std::string &>().find('\0') == std::string::npos;
            };
            bool valid_package_strings = true;
            for (const char * field : {
                    "tuning_profile", "provenance", "source_sha256", "generator", "python_version" }) {
                valid_package_strings = valid_package_strings && safe_optional_string(field);
            }
            json compiler = json::object();
            if (package.contains("compiler")) {
                compiler = package.at("compiler");
                if (!compiler.is_object()) {
                    valid_package_strings = false;
                } else {
                    for (const char * field : { "name", "version" }) {
                        if (compiler.contains(field) &&
                            (!compiler.at(field).is_string() ||
                             compiler.at(field).get_ref<const std::string &>().find('\0') != std::string::npos)) {
                            valid_package_strings = false;
                        }
                    }
                }
            }
            if (!valid_package_strings) {
                GGML_LOG_ERROR("FlagOS AMD AOT: invalid package string metadata: %s\n",
                    manifest_path.string().c_str());
                reset();
                return false;
            }
            tuning_profile_ = package.value("tuning_profile", "");
            if (!tuning_profile_.empty() &&
                tuning_profile_ != tuning_profile_gfx1150_q4ffn_v1 &&
                tuning_profile_ != tuning_profile_gfx1150_qwen35_q4km_v2 &&
                tuning_profile_ != tuning_profile_gfx1150_qwen35_q4km_v3) {
                GGML_LOG_ERROR("FlagOS AMD AOT: unsupported tuning profile: %s\n",
                    tuning_profile_.c_str());
                reset();
                return false;
            }
            GGML_LOG_INFO("FlagOS AMD AOT: package compiler=%s version=%s source=%s provenance=%s\n",
                compiler.value("name", "unknown").c_str(),
                compiler.value("version", "unknown").c_str(),
                package.value("source_sha256", "unknown").c_str(),
                package.value("provenance", "unknown").c_str());
        }
        if (!arch.empty() && manifest_arch != arch) {
            GGML_LOG_ERROR("FlagOS AMD AOT: manifest arch %s does not match device arch %s\n",
                manifest_arch.c_str(), arch.c_str());
            reset();
            return false;
        }
        if (!tuning_profile_.empty() && manifest_arch != "gfx1150") {
            GGML_LOG_ERROR("FlagOS AMD AOT: tuning profile %s requires gfx1150\n",
                tuning_profile_.c_str());
            reset();
            return false;
        }

        if (load_modules) {
            hipDeviceProp_t properties {};
            if (!hip_check(hipSetDevice(device), "hipSetDevice") ||
                !hip_check(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties") ||
                properties.maxGridSize[0] <= 0 || properties.maxGridSize[1] <= 0 ||
                properties.maxGridSize[2] <= 0 || properties.maxThreadsPerBlock <= 0 ||
                properties.warpSize <= 0) {
                reset();
                return false;
            }
            for (size_t dimension = 0; dimension < max_grid_size_.size(); ++dimension) {
                max_grid_size_[dimension] = static_cast<unsigned int>(properties.maxGridSize[dimension]);
            }
            max_threads_per_block_ = static_cast<unsigned int>(properties.maxThreadsPerBlock);
            device_warp_size_ = static_cast<unsigned int>(properties.warpSize);
            max_shared_memory_per_block_ = properties.sharedMemPerBlock;
        }
        device_ = load_modules ? device : -1;
        if (load_modules && profile_enabled_ &&
            (!hip_check(hipEventCreate(&profile_start_), "hipEventCreate (profile start)") ||
             !hip_check(hipEventCreate(&profile_end_), "hipEventCreate (profile end)"))) {
            if (profile_start_ != nullptr) {
                hip_check(hipEventDestroy(profile_start_), "hipEventDestroy (profile start)");
                profile_start_ = nullptr;
            }
            if (profile_end_ != nullptr) {
                hip_check(hipEventDestroy(profile_end_), "hipEventDestroy (profile end)");
                profile_end_ = nullptr;
            }
            profile_enabled_ = false;
            GGML_LOG_WARN("FlagOS AMD AOT: kernel profiling disabled because reusable events could not be created\n");
        }
        uint64_t package_binary_bytes = 0;
        std::unordered_set<std::string> artifact_files;
        artifact_files.reserve(manifest.at("kernels").size());
        for (const auto & item : manifest.at("kernels")) {
            kernel_metadata metadata;
            if (!parse_metadata(item, directory, manifest_arch, metadata) ||
                (!tuning_profile_.empty() && (metadata.sha256.empty() || metadata.binary_size == 0)) ||
                (load_modules &&
                 (static_cast<unsigned int>(metadata.warp_size) != device_warp_size_ ||
                  static_cast<unsigned int>(metadata.threads) > max_threads_per_block_)) ||
                kernels_.find(metadata.name) != kernels_.end() ||
                !artifact_files.insert(metadata.file).second) {
                GGML_LOG_ERROR("FlagOS AMD AOT: invalid or duplicate kernel entry\n");
                reset();
                return false;
            }
            std::vector<uint8_t> image;
            if (!read_binary(directory / fs::u8path(metadata.file), metadata.binary_size, image)) {
                reset();
                return false;
            }
            if (image.size() > MAX_PACKAGE_BYTES - package_binary_bytes) {
                GGML_LOG_ERROR("FlagOS AMD AOT: package binaries exceed %llu bytes\n",
                    static_cast<unsigned long long>(MAX_PACKAGE_BYTES));
                reset();
                return false;
            }
            package_binary_bytes += image.size();
            if (!metadata.sha256.empty() && metadata.sha256 != sha256_hex(image)) {
                GGML_LOG_ERROR("FlagOS AMD AOT: SHA-256 mismatch for %s\n", metadata.file.c_str());
                reset();
                return false;
            }
            if (!validate_hsaco_contract(image, manifest_arch, metadata) ||
                (load_modules &&
                 (metadata.static_shared_memory > max_shared_memory_per_block_ ||
                  static_cast<size_t>(metadata.shared_memory) >
                      max_shared_memory_per_block_ - metadata.static_shared_memory))) {
                GGML_LOG_ERROR("FlagOS AMD AOT: invalid HSACO launch resources for %s\n",
                    metadata.name.c_str());
                reset();
                return false;
            }
            loaded_kernel loaded;
            loaded.metadata = std::move(metadata);
            if (load_modules &&
                (!hip_check(hipModuleLoadData(&loaded.module, image.data()), "hipModuleLoadData") ||
                !hip_check(hipModuleGetFunction(&loaded.function, loaded.module,
                    loaded.metadata.symbol.c_str()), "hipModuleGetFunction"))) {
                if (loaded.module != nullptr) {
                    hip_check(hipModuleUnload(loaded.module), "hipModuleUnload");
                }
                reset();
                return false;
            }
            kernels_.emplace(loaded.metadata.name, std::move(loaded));
        }
        kernel_index_.reserve(kernels_.size());
        for (auto & pair : kernels_) {
            kernel_index_.emplace(pair.first, &pair.second);
        }
        const auto apply_known_abis = [this](const auto & contracts) {
            for (const tuned_kernel_abi & abi : contracts) {
                const auto kernel = kernels_.find(std::string(abi.name));
                if (kernel == kernels_.end()) {
                    continue;
                }
                if (kernel->second.metadata.argument_count >= 0 &&
                    kernel->second.metadata.argument_count != static_cast<int>(abi.argument_count)) {
                    GGML_LOG_ERROR("FlagOS AMD AOT: kernel ABI contract mismatch for %.*s\n",
                        static_cast<int>(abi.name.size()), abi.name.data());
                    return false;
                }
                kernel->second.metadata.argument_count = static_cast<int>(abi.argument_count);
            }
            return true;
        };
        if (!apply_known_abis(tuning_profile_gfx1150_q4ffn_v1_kernels) ||
            !apply_known_abis(tuning_profile_gfx1150_qwen35_q4km_v2_kernels) ||
            !apply_known_abis(tuning_profile_gfx1150_qwen35_q4km_v3_kernels)) {
            reset();
            return false;
        }
        const auto validate_profile = [this](const auto & contracts) {
            if (kernels_.size() != std::size(contracts)) {
                GGML_LOG_ERROR("FlagOS AMD AOT: tuning profile %s has %zu kernels, expected %zu\n",
                    tuning_profile_.c_str(), kernels_.size(),
                    std::size(contracts));
                return false;
            }
            for (const tuned_kernel_abi & abi : contracts) {
                const auto kernel = kernels_.find(std::string(abi.name));
                if (kernel == kernels_.end()) {
                    GGML_LOG_ERROR("FlagOS AMD AOT: tuning profile %s lacks kernel %.*s\n",
                        tuning_profile_.c_str(), static_cast<int>(abi.name.size()), abi.name.data());
                    return false;
                }
                const auto & metadata = kernel->second.metadata;
                if (metadata.exact_block_size != abi.exact_block_size ||
                    metadata.block_size != abi.block_size ||
                    metadata.tile_m != abi.tile_m || metadata.tile_n != abi.tile_n ||
                    metadata.tile_k != abi.tile_k || metadata.num_warps != abi.num_warps ||
                    metadata.warp_size != abi.warp_size) {
                    GGML_LOG_ERROR("FlagOS AMD AOT: tuning profile launch contract mismatch for %.*s\n",
                        static_cast<int>(abi.name.size()), abi.name.data());
                    return false;
                }
            }
            return true;
        };
        if ((tuning_profile_ == tuning_profile_gfx1150_q4ffn_v1 &&
             !validate_profile(tuning_profile_gfx1150_q4ffn_v1_kernels)) ||
            (tuning_profile_ == tuning_profile_gfx1150_qwen35_q4km_v2 &&
             !validate_profile(tuning_profile_gfx1150_qwen35_q4km_v2_kernels)) ||
            (tuning_profile_ == tuning_profile_gfx1150_qwen35_q4km_v3 &&
             !validate_profile(tuning_profile_gfx1150_qwen35_q4km_v3_kernels))) {
            reset();
            return false;
        }
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("FlagOS AMD AOT: cannot parse manifest: %s\n", error.what());
        reset();
        return false;
    }
    return !kernels_.empty();
}

const kernel_metadata * kernel_registry::find(std::string_view name) const {
    const auto it = kernel_index_.find(name);
    return it == kernel_index_.end() ? nullptr : &it->second->metadata;
}

size_t kernel_registry::size() const {
    return kernels_.size();
}

const std::string & kernel_registry::tuning_profile() const {
    return tuning_profile_;
}

bool kernel_registry::launch(std::string_view name, hipStream_t stream,
                             unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                             const kernel_arguments & arguments) {
    if (!arguments.valid()) {
        GGML_LOG_ERROR("FlagOS AMD AOT: argument vector overflow for kernel %.*s\n",
            static_cast<int>(name.size()), name.data());
        return false;
    }
    const auto it = kernel_index_.find(name);
    if (it == kernel_index_.end()) {
        GGML_LOG_ERROR("FlagOS AMD AOT: kernel %.*s is not present in the loaded package\n",
            static_cast<int>(name.size()), name.data());
        return false;
    }
    const loaded_kernel & kernel = *it->second;
    if (kernel.function == nullptr || grid_x == 0 || grid_y == 0 || grid_z == 0) {
        GGML_LOG_ERROR("FlagOS AMD AOT: invalid launch for %.*s (function=%p grid=%u,%u,%u)\n",
            static_cast<int>(name.size()), name.data(), static_cast<void *>(kernel.function), grid_x, grid_y, grid_z);
        return false;
    }
    const auto & metadata = kernel.metadata;
    if (grid_x > max_grid_size_[0] || grid_y > max_grid_size_[1] ||
        grid_z > max_grid_size_[2] ||
        static_cast<unsigned int>(metadata.threads) > max_threads_per_block_ ||
        metadata.static_shared_memory > max_shared_memory_per_block_ ||
        static_cast<size_t>(metadata.shared_memory) >
            max_shared_memory_per_block_ - metadata.static_shared_memory) {
        GGML_LOG_ERROR("FlagOS AMD AOT: launch geometry exceeds device limits for %.*s\n",
            static_cast<int>(name.size()), name.data());
        return false;
    }
    if (metadata.argument_count >= 0 &&
        arguments.size() != static_cast<size_t>(metadata.argument_count)) {
        GGML_LOG_ERROR("FlagOS AMD AOT: kernel %.*s ABI expects %d arguments, received %zu\n",
            static_cast<int>(name.size()), name.data(), metadata.argument_count, arguments.size());
        return false;
    }
    if (!arguments.matches(metadata.argument_abi)) {
        GGML_LOG_ERROR("FlagOS AMD AOT: kernel %.*s host argument types do not match the HSACO ABI\n",
            static_cast<int>(name.size()), name.data());
        return false;
    }
    // Triton AMD's generated launcher appends global_scratch and
    // profile_scratch after the explicit kernel signature.  initialize()
    // rejects nonzero requirements because this registry does not yet own a
    // scratch allocator; the null ABI slots remain mandatory for zero-scratch
    // kernels.
    void * global_scratch = nullptr;
    void * profile_scratch = nullptr;
    // The validated package ABI permits at most 30 explicit parameters. Keep
    // the two Triton scratch slots in a small stack buffer
    // rather than copying every argument vector into a heap-allocated vector
    // on every launch.  This matters for decode, where a Qwen token can issue
    // hundreds of AOT launches and the host-side allocator becomes visible in
    // the profile.
    constexpr size_t max_launch_arguments = 32;
    if (arguments.size() > max_launch_arguments - 2) {
        GGML_LOG_ERROR("FlagOS AMD AOT: kernel %.*s has too many arguments (%zu)\n",
            static_cast<int>(name.size()), name.data(), arguments.size());
        return false;
    }
    std::array<void *, max_launch_arguments> launch_arguments {};
    std::memcpy(launch_arguments.data(), arguments.data(), arguments.size() * sizeof(void *));
    launch_arguments[arguments.size()] = &global_scratch;
    launch_arguments[arguments.size() + 1] = &profile_scratch;
    const auto launch_raw = [&]() {
        return hip_check(hipModuleLaunchKernel(
            kernel.function, grid_x, grid_y, grid_z,
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
    std::lock_guard<std::mutex> launch_lock(profile_launch_mutex_);
    if (profile_start_ == nullptr || profile_end_ == nullptr) {
        return launch_raw();
    }
    const hipError_t record_start = hipEventRecord(profile_start_, stream);
    if (record_start != hipSuccess) {
        log_profile_error_once("hipEventRecord(start)", record_start);
        return launch_raw();
    }
    if (!launch_raw()) {
        return false;
    }
    // The kernel has already been enqueued successfully.  If a profiling
    // event cannot be recorded/synchronized, preserve the launch result and
    // simply omit this sample rather than turning an optional diagnostic into
    // a backend failure.
    const hipError_t record_end = hipEventRecord(profile_end_, stream);
    const hipError_t synchronize_end = record_end == hipSuccess
        ? hipEventSynchronize(profile_end_) : record_end;
    if (record_end != hipSuccess) {
        log_profile_error_once("hipEventRecord(end)", record_end);
    } else if (synchronize_end != hipSuccess) {
        log_profile_error_once("hipEventSynchronize(end)", synchronize_end);
    }
    float elapsed_ms = 0.0f;
    const hipError_t elapsed_result = synchronize_end == hipSuccess
        ? hipEventElapsedTime(&elapsed_ms, profile_start_, profile_end_) : synchronize_end;
    if (synchronize_end == hipSuccess && elapsed_result != hipSuccess) {
        log_profile_error_once("hipEventElapsedTime", elapsed_result);
    }
    const bool timed = elapsed_result == hipSuccess;
    if (timed) {
        const std::string profile_key = std::string(name) + " grid=" + std::to_string(grid_x) +
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
