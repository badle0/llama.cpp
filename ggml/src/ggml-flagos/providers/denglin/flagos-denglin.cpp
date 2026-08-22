#include "../../ggml-flagos.h"
#include "../../flagos-graph-plan.h"
#include "../../flagos-provider.h"
#include "flagos-denglin-api.h"

#include "../../../ggml-backend-impl.h"
#include "../../../ggml-impl.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef FLAGOS_KERNEL_DIR
#define FLAGOS_KERNEL_DIR ""
#endif

namespace fs = std::filesystem;

static bool flagos_denglin_is_backend(ggml_backend_t backend);

static bool flagos_cuda_check(cudaError_t result, const char * expression) {
    if (result == cudaSuccess) {
        return true;
    }

    GGML_LOG_ERROR("FlagOS: %s failed: %s\n", expression, cudaGetErrorString(result));
    return false;
}

static bool flagos_driver_check(CUresult result, const char * expression) {
    if (result == CUDA_SUCCESS) {
        return true;
    }

    const char * name = nullptr;
    const char * description = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    GGML_LOG_ERROR("FlagOS: %s failed: %s (%s)\n", expression,
        name ? name : "unknown", description ? description : "no description");
    return false;
}

static bool flagos_blas_check(cublasStatus_t result, const char * expression) {
    if (result == CUBLAS_STATUS_SUCCESS) {
        return true;
    }
    GGML_LOG_ERROR("FlagOS: %s failed with dlblas status %d\n", expression, static_cast<int>(result));
    return false;
}

static bool flagos_q4_dlblas_requested() {
    if (std::getenv("FLAGOS_Q4_DLBLAS") == nullptr) {
        return false;
    }

    // This recovered Denglin SDK aborts inside libdleol instead of returning a
    // status when its quantized GEMV JIT cannot produce a CU function. Refuse
    // to register the optional variant unless the known-required JIT contract
    // is explicit. A vendor release with packaged AOT kernels can replace this
    // preflight with a module/capability query.
    const char * use_dlcc = std::getenv("DLEOL_JIT_USE_DLCC");
    const char * compile_options = std::getenv("DLEOL_CU_COMPILE_OPTIONS");
    if (use_dlcc != nullptr && std::strcmp(use_dlcc, "1") == 0 &&
        compile_options != nullptr && compile_options[0] != '\0') {
        return true;
    }

    static std::atomic<bool> warned { false };
    if (!warned.exchange(true)) {
        GGML_LOG_WARN(
            "FlagOS: FLAGOS_Q4_DLBLAS ignored because the Denglin JIT preflight is incomplete "
            "(require DLEOL_JIT_USE_DLCC=1 and non-empty DLEOL_CU_COMPILE_OPTIONS); using direct AOT\n");
    }
    return false;
}

static cudaError_t flagos_cuda_graph_instantiate(cudaGraphExec_t * instance, cudaGraph_t graph) {
#if CUDART_VERSION >= 12000
    return cudaGraphInstantiate(instance, graph, 0);
#else
    return cudaGraphInstantiate(instance, graph, nullptr, nullptr, 0);
#endif
}

// Row-wise kernels are AOT-compiled with BLOCK spanning a full row; rows wider
// than this must fall back to the CPU. Keep in sync with ROW_BLOCK_SIZE in
// generate_flagos_kernels.py.
static constexpr int64_t FLAGOS_ROW_BLOCK = 1024;
// Row widths that are not a multiple of 4 give wrong reduction results on this
// backend: the masked tail lanes still contribute, so restrict to widths where
// the vectorised path is exact. Measured with test-backend-ops SUM_ROWS/CUMSUM.
static constexpr int64_t FLAGOS_ROW_WIDTH_MULTIPLE = 4;

// Compiled block width of the RMS norm kernels (RMS_NORM_BLOCK_SIZE in
// kernels/generate_flagos_kernels.py). The kernel reduces a row in one
// BLOCK-wide pass, so wider rows must fall back to the CPU.
static constexpr int64_t FLAGOS_RMS_NORM_MAX_COLS = 4096;

// Columns each batched quantized GEMM program computes (MUL_MAT_COLS_PER_BLOCK
// in kernels/generate_flagos_kernels.py). Must match the compiled constant.
static constexpr int FLAGOS_MUL_MAT_COLS_PER_BLOCK = 16;
static constexpr size_t FLAGOS_MAX_CUDA_GRAPHS = 16;

struct flagos_kernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    int block_size = 256;
    int threads = 128;
    int shared_memory = 0;
};

struct flagos_kernel_registry {
    flagos_kernel add_f32;
    flagos_kernel mul_f32;
    flagos_kernel scale_f32;
    flagos_kernel copy_f32;
    flagos_kernel copy_strided_f32;
    flagos_kernel swiglu_split_f32;
    flagos_kernel set_rows_f32_f16;
    flagos_kernel flash_attn_decode_f32_f16;
    flagos_kernel rope_neox_f32;
    flagos_kernel cast_f32_f16;
    flagos_kernel cast_f16_f32;
    flagos_kernel dequant_q4_k_f16;
    flagos_kernel dequant_q6_k_f16;
    flagos_kernel mul_mat_q4_k_f32;
    flagos_kernel mul_mat_q4_k_f32_batched;
    flagos_kernel mul_mat_q6_k_f32;
    flagos_kernel mul_mat_q6_k_f32_batched;
    flagos_kernel rms_norm_f32;
    flagos_kernel rms_norm_mul_f32;
    flagos_kernel get_rows_q4_k_f32;
    flagos_kernel get_rows_q6_k_f32;
    flagos_kernel get_rows_f32;
    flagos_kernel ssm_conv_f32;
    flagos_kernel sub_f32;
    flagos_kernel div_f32;
    flagos_kernel sigmoid_f32;
    flagos_kernel exp_f32;
    flagos_kernel softplus_f32;
    flagos_kernel fill_f32;
    flagos_kernel sum_rows_f32;
    flagos_kernel l2_norm_f32;
    flagos_kernel norm_f32;
    flagos_kernel cumsum_f32;
    flagos_kernel soft_max_f32;

    struct kernel_entry {
        flagos_kernel * kernel;
        const char * name;
        std::string file;
    };

    std::vector<kernel_entry> entries() {
        return {
            { &add_f32,                   "flagos_add_f32",                   {} },
            { &mul_f32,                   "flagos_mul_f32",                   {} },
            { &scale_f32,                 "flagos_scale_f32",                 {} },
            { &copy_f32,                  "flagos_copy_f32",                  {} },
            { &copy_strided_f32,          "flagos_copy_strided_f32",          {} },
            { &swiglu_split_f32,          "flagos_swiglu_split_f32",          {} },
            { &set_rows_f32_f16,          "flagos_set_rows_f32_f16",          {} },
            { &flash_attn_decode_f32_f16, "flagos_flash_attn_decode_f32_f16", {} },
            { &rope_neox_f32,             "flagos_rope_neox_f32",             {} },
            { &rms_norm_f32,              "flagos_rms_norm_f32",              {} },
            { &rms_norm_mul_f32,          "flagos_rms_norm_mul_f32",          {} },
            { &cast_f32_f16,              "flagos_cast_f32_f16",              {} },
            { &cast_f16_f32,              "flagos_cast_f16_f32",              {} },
            { &dequant_q4_k_f16,          "flagos_dequant_q4_k_f16",          {} },
            { &dequant_q6_k_f16,          "flagos_dequant_q6_k_f16",          {} },
            { &mul_mat_q4_k_f32,          "flagos_mul_mat_q4_k_f32",          {} },
            { &mul_mat_q4_k_f32_batched,  "flagos_mul_mat_q4_k_f32_batched",  {} },
            { &mul_mat_q6_k_f32,          "flagos_mul_mat_q6_k_f32",          {} },
            { &mul_mat_q6_k_f32_batched,  "flagos_mul_mat_q6_k_f32_batched",  {} },
            { &get_rows_q4_k_f32,         "flagos_get_rows_q4_k_f32",         {} },
            { &get_rows_q6_k_f32,         "flagos_get_rows_q6_k_f32",         {} },
            { &get_rows_f32,              "flagos_get_rows_f32",              {} },
            { &ssm_conv_f32,              "flagos_ssm_conv_f32",              {} },
            { &sub_f32,                   "flagos_sub_f32",                   {} },
            { &div_f32,                   "flagos_div_f32",                   {} },
            { &sigmoid_f32,               "flagos_sigmoid_f32",               {} },
            { &exp_f32,                   "flagos_exp_f32",                   {} },
            { &softplus_f32,              "flagos_softplus_f32",              {} },
            { &fill_f32,                  "flagos_fill_f32",                  {} },
            { &sum_rows_f32,              "flagos_sum_rows_f32",              {} },
            { &l2_norm_f32,               "flagos_l2_norm_f32",               {} },
            { &norm_f32,                  "flagos_norm_f32",                  {} },
            { &cumsum_f32,                "flagos_cumsum_f32",                {} },
            { &soft_max_f32,              "flagos_soft_max_f32",              {} },
        };
    }

    ~flagos_kernel_registry() {
        for (auto & entry : entries()) {
            unload(*entry.kernel);
        }
        if (merged_module != nullptr) {
            flagos_driver_check(cuModuleUnload(merged_module), "cuModuleUnload merged");
            merged_module = nullptr;
        }
    }

    static void unload(flagos_kernel & kernel) {
        if (kernel.module != nullptr) {
            flagos_driver_check(cuModuleUnload(kernel.module), "cuModuleUnload");
            kernel.module = nullptr;
            kernel.function = nullptr;
        }
    }

    static bool load(flagos_kernel & kernel, const fs::path & path, const char * name) {
        if (!fs::is_regular_file(path)) {
            GGML_LOG_ERROR("FlagOS: AOT kernel not found: %s\n", path.string().c_str());
            return false;
        }
        if (!flagos_driver_check(cuModuleLoad(&kernel.module, path.string().c_str()), "cuModuleLoad")) {
            return false;
        }
        if (!flagos_driver_check(cuModuleGetFunction(&kernel.function, kernel.module, name), "cuModuleGetFunction")) {
            unload(kernel);
            return false;
        }
        return true;
    }

    // Every kernel lives in one merged module (kernels/merge_flagos_module.py).
    // Owned here because the individual flagos_kernel entries only borrow the
    // CUmodule; unloading it invalidates all of their CUfunction handles.
    CUmodule merged_module = nullptr;

    bool load_manifest(
            const fs::path & path,
            std::vector<kernel_entry> & kernel_entries,
            std::string & merged_file) {
        try {
            std::ifstream input(path);
            if (!input) {
                GGML_LOG_ERROR("FlagOS: AOT manifest not found: %s\n", path.string().c_str());
                return false;
            }

            nlohmann::json manifest;
            input >> manifest;
            if (manifest.value("format", 0) < 2 || !manifest.contains("kernels")) {
                GGML_LOG_ERROR("FlagOS: unsupported AOT manifest format in %s\n", path.string().c_str());
                return false;
            }
            merged_file = manifest.value("module", "");

            const auto & metadata = manifest.at("kernels");
            for (auto & entry : kernel_entries) {
                const auto item = std::find_if(metadata.begin(), metadata.end(), [&entry](const nlohmann::json & value) {
                    return value.value("name", "") == entry.name;
                });
                if (item == metadata.end()) {
                    GGML_LOG_ERROR("FlagOS: AOT manifest is missing %s\n", entry.name);
                    return false;
                }

                entry.file = item->at("file").get<std::string>();
                const int num_warps = item->at("num_warps").get<int>();
                const int warp_size = item->at("warp_size").get<int>();
                entry.kernel->block_size = item->at("block_size").get<int>();
                entry.kernel->shared_memory = item->at("shared").get<int>();
                entry.kernel->threads = num_warps * warp_size;
                if (entry.file.empty() || entry.kernel->block_size <= 0 || entry.kernel->shared_memory < 0 ||
                    num_warps <= 0 || warp_size <= 0 || entry.kernel->threads > 1024) {
                    GGML_LOG_ERROR("FlagOS: invalid AOT metadata for %s\n", entry.name);
                    return false;
                }
            }
        } catch (const std::exception & error) {
            GGML_LOG_ERROR("FlagOS: cannot parse AOT manifest %s: %s\n", path.string().c_str(), error.what());
            return false;
        }
        return true;
    }

    bool initialize() {
        const char * env_dir = std::getenv("FLAGOS_KERNEL_DIR");
        const fs::path kernel_dir = env_dir && env_dir[0] != '\0' ? env_dir : FLAGOS_KERNEL_DIR;
        if (kernel_dir.empty()) {
            GGML_LOG_ERROR("FlagOS: FLAGOS_KERNEL_DIR is not configured\n");
            return false;
        }

        auto kernel_entries = entries();
        std::string merged_file;
        if (!load_manifest(kernel_dir / "manifest.json", kernel_entries, merged_file)) {
            return false;
        }

        const fs::path merged_path = kernel_dir / merged_file;
        if (!merged_file.empty() && fs::is_regular_file(merged_path)) {
            if (load_merged(merged_path, kernel_entries)) {
                return true;
            }
            GGML_LOG_WARN("FlagOS: merged module unusable, falling back to per-kernel modules\n");
        }

        for (auto & entry : kernel_entries) {
            if (!load(*entry.kernel, kernel_dir / entry.file, entry.name)) {
                return false;
            }
        }
        return true;
    }

    // Resolve every kernel out of one module. On failure the module is
    // released so the per-kernel fallback starts from a clean slate.
    bool load_merged(const fs::path & path, const std::vector<kernel_entry> & kernel_entries) {
        if (!flagos_driver_check(cuModuleLoad(&merged_module, path.string().c_str()),
                "cuModuleLoad merged")) {
            merged_module = nullptr;
            return false;
        }

        for (const auto & entry : kernel_entries) {
            if (!flagos_driver_check(cuModuleGetFunction(&entry.kernel->function, merged_module, entry.name),
                    "cuModuleGetFunction merged")) {
                GGML_LOG_ERROR("FlagOS: merged module is missing %s\n", entry.name);
                release_merged(kernel_entries);
                return false;
            }
            entry.kernel->module = nullptr;
        }

        return true;
    }

    void release_merged(const std::vector<kernel_entry> & kernel_entries) {
        for (const auto & entry : kernel_entries) {
            entry.kernel->function = nullptr;
            entry.kernel->module = nullptr;
        }
        if (merged_module != nullptr) {
            flagos_driver_check(cuModuleUnload(merged_module), "cuModuleUnload merged");
            merged_module = nullptr;
        }
    }

    // Multi-dimensional grid variant, for kernels that use program_id(1)/(2).
    // Diagnostic helper: kernel faults surface asynchronously, so without an explicit
    // sync the error is reported by whichever later launch or memcpy runs next.
    static bool check_launch_sync(
            const flagos_kernel & kernel,
            cudaStream_t stream,
            const char * variant,
            unsigned int gx, unsigned int gy, unsigned int gz) {
        if (!getenv("GGML_FLAGOS_SYNC_LAUNCH")) {
            return true;
        }
        const cudaError_t sync = cudaStreamSynchronize(stream);
        if (sync != cudaSuccess) {
            GGML_LOG_ERROR("FlagOS: kernel fn=%p via %s (grid=%u,%u,%u threads=%d shmem=%d) faulted: %s\n",
                (void *) kernel.function, variant, gx, gy, gz,
                kernel.threads, kernel.shared_memory, cudaGetErrorString(sync));
            return false;
        }
        return true;
    }

    static bool launch_3d(
            const flagos_kernel & kernel,
            cudaStream_t stream,
            unsigned int grid_x,
            unsigned int grid_y,
            unsigned int grid_z,
            void ** arguments) {
        const bool launched = flagos_driver_check(cuLaunchKernel(
            kernel.function,
            grid_x, grid_y, grid_z,
            kernel.threads, 1, 1,
            kernel.shared_memory,
            reinterpret_cast<CUstream>(stream),
            arguments,
            nullptr), "cuLaunchKernel");
        return launched && check_launch_sync(kernel, stream, "launch_3d", grid_x, grid_y, grid_z);
    }

    static bool launch(
            const flagos_kernel & kernel,
            cudaStream_t stream,
            unsigned int grid,
            void ** arguments) {
        const bool launched = flagos_driver_check(cuLaunchKernel(
            kernel.function,
            grid, 1, 1,
            kernel.threads, 1, 1,
            kernel.shared_memory,
            reinterpret_cast<CUstream>(stream),
            arguments,
            nullptr), "cuLaunchKernel");
        return launched && check_launch_sync(kernel, stream, "launch", grid, 1, 1);
    }

    static bool launch_2d(
            const flagos_kernel & kernel,
            cudaStream_t stream,
            unsigned int grid_x,
            unsigned int grid_y,
            void ** arguments) {
        const bool launched = flagos_driver_check(cuLaunchKernel(
            kernel.function,
            grid_x, grid_y, 1,
            kernel.threads, 1, 1,
            kernel.shared_memory,
            reinterpret_cast<CUstream>(stream),
            arguments,
            nullptr), "cuLaunchKernel");
        return launched && check_launch_sync(kernel, stream, "launch_2d", grid_x, grid_y, 1);
    }

    bool launch_add(
            ggml_tensor * dst,
            void * src0_data,
            void * src1_data,
            cudaStream_t stream) const {
        int n_elements = static_cast<int>(ggml_nelements(dst));
        void * dst_data = dst->data;
        void * arguments[] = { &src0_data, &src1_data, &dst_data, &n_elements };
        const unsigned int grid = (n_elements + add_f32.block_size - 1) / add_f32.block_size;
        return launch(add_f32, stream, grid, arguments);
    }

    bool launch_mul(
            const ggml_tensor * src1,
            ggml_tensor * dst,
            void * src0_data,
            void * src1_data,
            cudaStream_t stream) const {
        int n_elements = static_cast<int>(ggml_nelements(dst));
        int src1_elements = static_cast<int>(ggml_nelements(src1));
        void * dst_data = dst->data;
        void * arguments[] = { &src0_data, &src1_data, &dst_data, &n_elements, &src1_elements };
        const unsigned int grid = (n_elements + mul_f32.block_size - 1) / mul_f32.block_size;
        return launch(mul_f32, stream, grid, arguments);
    }

    bool launch_scale(
            ggml_tensor * dst,
            void * src0_data,
            cudaStream_t stream) const {
        // GGML_OP_SCALE computes dst = src0 * scale + bias, both packed in op_params.
        float scale = 1.0f;
        float bias  = 0.0f;
        memcpy(&scale, reinterpret_cast<const float *>(dst->op_params) + 0, sizeof(float));
        memcpy(&bias,  reinterpret_cast<const float *>(dst->op_params) + 1, sizeof(float));
        int n_elements = static_cast<int>(ggml_nelements(dst));
        void * dst_data = dst->data;
        void * arguments[] = { &src0_data, &dst_data, &scale, &bias, &n_elements };
        const unsigned int grid = (n_elements + scale_f32.block_size - 1) / scale_f32.block_size;
        return launch(scale_f32, stream, grid, arguments);
    }

    // GGML_OP_SUB / GGML_OP_DIV share the broadcast contract of launch_mul.
    bool launch_binary(
            const flagos_kernel & kernel,
            const ggml_tensor * src1,
            ggml_tensor * dst,
            void * src0_data,
            void * src1_data,
            cudaStream_t stream) const {
        int n_elements = static_cast<int>(ggml_nelements(dst));
        int src1_elements = static_cast<int>(ggml_nelements(src1));
        void * dst_data = dst->data;
        void * arguments[] = { &src0_data, &src1_data, &dst_data, &n_elements, &src1_elements };
        const unsigned int grid = (n_elements + kernel.block_size - 1) / kernel.block_size;
        return launch(kernel, stream, grid, arguments);
    }

    // Elementwise unary: sigmoid / exp / softplus.
    bool launch_unary(
            const flagos_kernel & kernel,
            ggml_tensor * dst,
            void * src0_data,
            cudaStream_t stream) const {
        int n_elements = static_cast<int>(ggml_nelements(dst));
        void * dst_data = dst->data;
        void * arguments[] = { &src0_data, &dst_data, &n_elements };
        const unsigned int grid = (n_elements + kernel.block_size - 1) / kernel.block_size;
        return launch(kernel, stream, grid, arguments);
    }

    // Row-wise reductions: one program per row of src0.
    bool launch_row_reduce(
            const flagos_kernel & kernel,
            ggml_tensor * dst,
            void * src0_data,
            cudaStream_t stream,
            bool has_eps) const {
        const ggml_tensor * src0 = dst->src[0];
        int n_cols = static_cast<int>(src0->ne[0]);
        const unsigned int grid = static_cast<unsigned int>(ggml_nrows(src0));
        void * dst_data = dst->data;
        float eps = 0.0f;
        if (has_eps) {
            memcpy(&eps, dst->op_params, sizeof(float));
        }
        if (has_eps) {
            void * arguments[] = { &src0_data, &dst_data, &n_cols, &eps };
            return launch(kernel, stream, grid, arguments);
        }
        void * arguments[] = { &src0_data, &dst_data, &n_cols };
        return launch(kernel, stream, grid, arguments);
    }

    bool launch_soft_max(
            ggml_tensor * dst,
            void * src0_data,
            void * src1_data,
            cudaStream_t stream) const {
        const ggml_tensor * src0 = dst->src[0];
        int n_cols = static_cast<int>(src0->ne[0]);
        const unsigned int grid = static_cast<unsigned int>(ggml_nrows(src0));
        float scale = 1.0f;
        memcpy(&scale, dst->op_params, sizeof(float));
        void * dst_data = dst->data;
        // HAS_MASK is a compile-time constant in the AOT kernel, so a null mask
        // still needs a valid pointer argument; src0 is safe and never read.
        GGML_ASSERT(src1_data != nullptr);
        void * mask_data = src1_data;
        void * arguments[] = { &src0_data, &mask_data, &dst_data, &n_cols, &scale };
        return launch(soft_max_f32, stream, grid, arguments);
    }

    // GGML_OP_CPY / GGML_OP_CONT on dense tensors: same-type copy, or F32->F16 cast.
    bool launch_copy(
            ggml_tensor * dst,
            void * src0_data,
            cudaStream_t stream) const {
        int n_elements = static_cast<int>(ggml_nelements(dst));
        void * dst_data = dst->data;
        if (dst->src[0]->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            return launch_cast_f32_f16(src0_data, dst_data, n_elements, stream);
        }
        const ggml_tensor * src0 = dst->src[0];
        if (std::getenv("FLAGOS_LOG_COPY") != nullptr) {
            GGML_LOG_ERROR("FlagOS copy: %s dst %s[%lld,%lld,%lld,%lld]nb[%zu,%zu,%zu,%zu] "
                "src %s[%lld,%lld,%lld,%lld]nb[%zu,%zu,%zu,%zu] dense=%d\n",
                dst->name,
                ggml_type_name(dst->type),
                (long long)dst->ne[0], (long long)dst->ne[1],
                (long long)dst->ne[2], (long long)dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                ggml_type_name(src0->type),
                (long long)src0->ne[0], (long long)src0->ne[1],
                (long long)src0->ne[2], (long long)src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                (int)(ggml_is_contiguous(dst) && ggml_is_contiguous(src0)));
        }
        if (ggml_is_contiguous(dst) && ggml_is_contiguous(src0)) {
            void * arguments[] = { &src0_data, &dst_data, &n_elements };
            const unsigned int grid = (n_elements + copy_f32.block_size - 1) / copy_f32.block_size;
            return launch(copy_f32, stream, grid, arguments);
        }
        // Strided path: convert ggml byte strides to element strides.
        // Pass full 4D shapes to match the 4D strides — decomposition must
        // cover all dimensions for the addressing to work correctly.
        int xe0 = static_cast<int>(src0->ne[0]);
        int xe1 = static_cast<int>(src0->ne[1]);
        int xe2 = static_cast<int>(src0->ne[2]);
        int xe3 = static_cast<int>(src0->ne[3]);
        int ye0 = static_cast<int>(dst->ne[0]);
        int ye1 = static_cast<int>(dst->ne[1]);
        int ye2 = static_cast<int>(dst->ne[2]);
        int ye3 = static_cast<int>(dst->ne[3]);
        int sx0 = static_cast<int>(src0->nb[0] / sizeof(float));
        int sx1 = static_cast<int>(src0->nb[1] / sizeof(float));
        int sx2 = static_cast<int>(src0->nb[2] / sizeof(float));
        int sx3 = static_cast<int>(src0->nb[3] / sizeof(float));
        int sy0 = static_cast<int>(dst->nb[0] / sizeof(float));
        int sy1 = static_cast<int>(dst->nb[1] / sizeof(float));
        int sy2 = static_cast<int>(dst->nb[2] / sizeof(float));
        int sy3 = static_cast<int>(dst->nb[3] / sizeof(float));

        void * arguments[] = { &src0_data, &dst_data,
                               &xe0, &xe1, &xe2, &xe3,
                               &ye0, &ye1, &ye2, &ye3,
                               &sx0, &sx1, &sx2, &sx3,
                               &sy0, &sy1, &sy2, &sy3,
                               &n_elements };
        if (n_elements == 0) {
            return true;   // nothing to copy; a zero grid would be INVALID_VALUE
        }
        const unsigned int grid =
            (n_elements + copy_strided_f32.block_size - 1) / copy_strided_f32.block_size;
        return launch(copy_strided_f32, stream, grid, arguments);
    }

    bool launch_swiglu_split(
            ggml_tensor * dst,
            void * gate_data,
            void * up_data,
            cudaStream_t stream) const {
        int n_elements = static_cast<int>(ggml_nelements(dst));
        void * dst_data = dst->data;
        void * arguments[] = { &gate_data, &up_data, &dst_data, &n_elements };
        const unsigned int grid = (n_elements + swiglu_split_f32.block_size - 1) / swiglu_split_f32.block_size;
        return launch(swiglu_split_f32, stream, grid, arguments);
    }

    bool launch_set_rows(ggml_tensor * dst, cudaStream_t stream) const {
        int n_cols = static_cast<int>(dst->src[0]->ne[0]);
        int n_rows = static_cast<int>(dst->src[0]->ne[1]);
        // Row capacity of the destination view; the kernel clamps stores to it so a
        // stale or out-of-range cache index cannot fault the device.
        int n_dst_rows = static_cast<int>(dst->ne[1]);
        void * src_data = dst->src[0]->data;
        void * index_data = dst->src[1]->data;
        void * dst_data = dst->data;
        void * arguments[] = { &src_data, &index_data, &dst_data, &n_cols, &n_rows, &n_dst_rows };
        const unsigned int grid = (n_cols * n_rows + set_rows_f32_f16.block_size - 1) /
            set_rows_f32_f16.block_size;
        return launch(set_rows_f32_f16, stream, grid, arguments);
    }

    // GET_ROWS over a Q4_K table: dequantize just the rows the indices name.
    // Grid is (n_tokens, blocks_per_row); each program emits 256 elements.
    bool launch_get_rows_q4_k(ggml_tensor * dst, cudaStream_t stream) const {
        const ggml_tensor * table = dst->src[0];
        const ggml_tensor * index = dst->src[1];

        int n_cols = static_cast<int>(dst->ne[0]);
        const int n_tokens = static_cast<int>(ggml_nelements(index));
        const int blocks_per_row = n_cols / 256;

        // The f16 scale pair sits at the head of each 144-byte block, so the
        // same base pointer is read both as bytes and as halves.
        void * weights_u8 = table->data;
        void * weights_f16 = table->data;
        void * index_data = index->data;
        void * dst_data = dst->data;
        void * arguments[] = { &weights_u8, &weights_f16, &index_data, &dst_data, &n_cols };
        return launch_3d(get_rows_q4_k_f32, stream,
            static_cast<unsigned int>(n_tokens),
            static_cast<unsigned int>(blocks_per_row), 1, arguments);
    }

    bool launch_get_rows_q6_k(ggml_tensor * dst, cudaStream_t stream) const {
        const ggml_tensor * table = dst->src[0];
        const ggml_tensor * index = dst->src[1];

        int n_cols = static_cast<int>(dst->ne[0]);
        const int n_tokens = static_cast<int>(ggml_nelements(index));
        const int blocks_per_row = n_cols / 256;

        void * weights_u8 = table->data;
        void * weights_f16 = table->data;
        void * index_data = index->data;
        void * dst_data = dst->data;
        void * arguments[] = { &weights_u8, &weights_f16, &index_data, &dst_data, &n_cols };
        return launch_3d(get_rows_q6_k_f32, stream,
            static_cast<unsigned int>(n_tokens),
            static_cast<unsigned int>(blocks_per_row), 1, arguments);
    }

    bool launch_get_rows_f32(ggml_tensor * dst, cudaStream_t stream) const {
        const ggml_tensor * table = dst->src[0];
        const ggml_tensor * index = dst->src[1];

        int n_cols = static_cast<int>(dst->ne[0]);
        const int n_tokens = static_cast<int>(ggml_nelements(index));
        const int blocks_per_row =
            (n_cols + get_rows_f32.block_size - 1) / get_rows_f32.block_size;

        void * table_data = table->data;
        void * index_data = index->data;
        void * dst_data = dst->data;
        void * arguments[] = { &table_data, &index_data, &dst_data, &n_cols };
        return launch_3d(get_rows_f32, stream,
            static_cast<unsigned int>(n_tokens),
            static_cast<unsigned int>(blocks_per_row), 1, arguments);
    }

    bool launch_get_rows(ggml_tensor * dst, cudaStream_t stream) const {
        switch (dst->src[0]->type) {
            case GGML_TYPE_Q4_K:
                return launch_get_rows_q4_k(dst, stream);
            case GGML_TYPE_Q6_K:
                return launch_get_rows_q6_k(dst, stream);
            case GGML_TYPE_F32:
                return launch_get_rows_f32(dst, stream);
            default:
                return false;
        }
    }

    // SSM_CONV: causal depthwise conv1d over the rolling conv state.
    // Grid is (n_tokens, n_seqs, channel tiles).
    bool launch_ssm_conv(ggml_tensor * dst, cudaStream_t stream) const {
        const ggml_tensor * state  = dst->src[0];
        const ggml_tensor * weight = dst->src[1];

        int d_conv   = static_cast<int>(weight->ne[0]);
        int d_inner  = static_cast<int>(dst->ne[0]);
        int n_tokens = static_cast<int>(dst->ne[1]);
        int n_seqs   = static_cast<int>(dst->ne[2]);

        // Element strides; nb[0] is unit stride on all three (checked in supports_op).
        int ss1 = static_cast<int>(state->nb[1] / sizeof(float));
        int ss2 = static_cast<int>(state->nb[2] / sizeof(float));
        int sc1 = static_cast<int>(weight->nb[1] / sizeof(float));
        int so0 = static_cast<int>(dst->nb[1] / sizeof(float));
        int so1 = static_cast<int>(dst->nb[2] / sizeof(float));

        void * state_data  = state->data;
        void * weight_data = weight->data;
        void * dst_data    = dst->data;
        void * arguments[] = {
            &state_data, &weight_data, &dst_data,
            &d_conv, &d_inner, &n_tokens, &n_seqs,
            &ss1, &ss2, &sc1, &so0, &so1,
        };
        const unsigned int tiles =
            (d_inner + ssm_conv_f32.block_size - 1) / ssm_conv_f32.block_size;
        return launch_3d(ssm_conv_f32, stream,
            static_cast<unsigned int>(n_tokens),
            static_cast<unsigned int>(n_seqs), tiles, arguments);
    }

    bool launch_flash_attn_decode(ggml_tensor * dst, cudaStream_t stream) const {
        const ggml_tensor * q = dst->src[0];
        const ggml_tensor * k = dst->src[1];
        const ggml_tensor * v = dst->src[2];
        const ggml_tensor * mask = dst->src[3];
        void * q_data = q->data;
        void * k_data = k->data;
        void * v_data = v->data;
        void * mask_data = mask->data;
        void * dst_data = dst->data;
        int key_length = static_cast<int>(k->ne[1]);
        int q_heads = static_cast<int>(q->ne[2]);
        int kv_heads = static_cast<int>(k->ne[2]);
        int q_per_kv = q_heads / kv_heads;
        int stride_q_token = static_cast<int>(q->nb[1] / sizeof(float));
        int stride_q_head = static_cast<int>(q->nb[2] / sizeof(float));
        int stride_k_token = static_cast<int>(k->nb[1] / sizeof(ggml_fp16_t));
        int stride_k_head = static_cast<int>(k->nb[2] / sizeof(ggml_fp16_t));
        int stride_v_token = static_cast<int>(v->nb[1] / sizeof(ggml_fp16_t));
        int stride_v_head = static_cast<int>(v->nb[2] / sizeof(ggml_fp16_t));
        int stride_output_token = static_cast<int>(dst->nb[2] / sizeof(float));
        int stride_output_head = static_cast<int>(dst->nb[1] / sizeof(float));
        float scale = 0.0f;
        std::memcpy(&scale, dst->op_params, sizeof(float));
        void * arguments[] = {
            &q_data, &k_data, &v_data, &mask_data, &dst_data,
            &key_length, &q_per_kv, &stride_q_token, &stride_q_head,
            &stride_k_token, &stride_k_head, &stride_v_token, &stride_v_head,
            &stride_output_token, &stride_output_head, &scale,
        };
        return launch(flash_attn_decode_f32_f16, stream, static_cast<unsigned int>(q_heads), arguments);
    }

    bool launch_cast_f32_f16(
            void * src_data,
            void * dst_data,
            int n_elements,
            cudaStream_t stream) const {
        void * arguments[] = { &src_data, &dst_data, &n_elements };
        const unsigned int grid = (n_elements + cast_f32_f16.block_size - 1) / cast_f32_f16.block_size;
        return launch(cast_f32_f16, stream, grid, arguments);
    }

    bool launch_cast_f16_f32(
            void * src_data,
            void * dst_data,
            int n_elements,
            cudaStream_t stream) const {
        void * arguments[] = { &src_data, &dst_data, &n_elements };
        const unsigned int grid = (n_elements + cast_f16_f32.block_size - 1) / cast_f16_f32.block_size;
        return launch(cast_f16_f32, stream, grid, arguments);
    }

    bool launch_dequant_qk_f16(
            ggml_type type,
            void * src_data,
            void * dst_data,
            unsigned int n_blocks,
            cudaStream_t stream) const {
        void * src_u8 = src_data;
        void * src_f16 = src_data;
        void * arguments[] = { &src_u8, &src_f16, &dst_data };
        const flagos_kernel & kernel = type == GGML_TYPE_Q4_K
            ? dequant_q4_k_f16
            : dequant_q6_k_f16;
        return launch(kernel, stream, n_blocks, arguments);
    }

    bool launch_rope_neox(
            const ggml_tensor * src,
            ggml_tensor * dst,
            void * src_data,
            void * positions_data,
            cudaStream_t stream) const {
        int n_elements = static_cast<int>(ggml_nelements(dst));
        int ne0 = static_cast<int>(src->ne[0]);
        int ne1 = static_cast<int>(src->ne[1]);
        int ne2 = static_cast<int>(src->ne[2]);
        const int32_t * params = static_cast<const int32_t *>(dst->op_params);
        int n_dims = params[1];
        float freq_base = 0.0f;
        float freq_scale = 0.0f;
        std::memcpy(&freq_base, params + 5, sizeof(float));
        std::memcpy(&freq_scale, params + 6, sizeof(float));
        void * dst_data = dst->data;
        void * arguments[] = {
            &src_data, &positions_data, &dst_data, &n_elements,
            &ne0, &ne1, &ne2, &n_dims, &freq_base, &freq_scale,
        };
        const unsigned int grid = (n_elements / 2 + rope_neox_f32.block_size - 1) / rope_neox_f32.block_size;
        return launch(rope_neox_f32, stream, grid, arguments);
    }

    bool launch_mul_mat(
            const ggml_tensor * src0,
            ggml_tensor * dst,
            void * weights_data,
            void * src1_data,
            cudaStream_t stream) const {
        int k = static_cast<int>(src0->ne[0]);
        int rows = static_cast<int>(src0->ne[1]);
        void * weights_u8 = weights_data;
        void * weights_f16 = weights_data;
        void * dst_data = dst->data;
        void * arguments[] = { &weights_u8, &weights_f16, &src1_data, &dst_data, &k, &rows };
        const flagos_kernel & kernel = src0->type == GGML_TYPE_Q4_K
            ? mul_mat_q4_k_f32
            : mul_mat_q6_k_f32;
        return launch(kernel, stream, static_cast<unsigned int>(rows), arguments);
    }

    bool launch_mul_mat_batched(
            const ggml_tensor * src0,
            ggml_tensor * dst,
            void * weights_data,
            void * src1_data,
            int columns,
            cudaStream_t stream) const {
        int k = static_cast<int>(src0->ne[0]);
        int rows = static_cast<int>(src0->ne[1]);
        void * weights_u8 = weights_data;
        void * weights_f16 = weights_data;
        void * dst_data = dst->data;
        void * arguments[] = { &weights_u8, &weights_f16, &src1_data, &dst_data, &k, &rows, &columns };
        const flagos_kernel & kernel = src0->type == GGML_TYPE_Q6_K
            ? mul_mat_q6_k_f32_batched
            : mul_mat_q4_k_f32_batched;
        const unsigned int grid_x = static_cast<unsigned int>(rows);
        // Each program handles FLAGOS_MUL_MAT_COLS_PER_BLOCK columns and masks
        // the tail, so the y grid covers column tiles rather than columns.
        const unsigned int grid_y = static_cast<unsigned int>(
            (columns + FLAGOS_MUL_MAT_COLS_PER_BLOCK - 1) / FLAGOS_MUL_MAT_COLS_PER_BLOCK);
        return launch_2d(kernel, stream, grid_x, grid_y, arguments);
    }

    bool launch_rms_norm(const ggml_tensor * src, ggml_tensor * dst, cudaStream_t stream) const {
        int n_cols = static_cast<int>(src->ne[0]);
        const unsigned int rows = static_cast<unsigned int>(ggml_nelements(src) / n_cols);
        float eps = 0.0f;
        std::memcpy(&eps, dst->op_params, sizeof(float));
        void * src_data = src->data;
        void * dst_data = dst->data;
        void * arguments[] = { &dst_data, &src_data, &n_cols, &eps };
        return launch(rms_norm_f32, stream, rows, arguments);
    }

    bool launch_rms_norm_mul(
            const ggml_tensor * src,
            ggml_tensor * norm_dst,
            ggml_tensor * mul_dst,
            void * weight_data,
            cudaStream_t stream) const {
        int n_cols = static_cast<int>(src->ne[0]);
        const unsigned int rows = static_cast<unsigned int>(ggml_nelements(src) / n_cols);
        float eps = 0.0f;
        std::memcpy(&eps, norm_dst->op_params, sizeof(float));
        void * norm_dst_data = norm_dst->data;
        void * mul_dst_data = mul_dst->data;
        void * src_data = src->data;
        void * arguments[] = { &norm_dst_data, &mul_dst_data, &src_data, &weight_data, &n_cols, &eps };
        return launch(rms_norm_mul_f32, stream, rows, arguments);
    }
};

struct flagos_cached_weight {
    void * device_ptr;
    size_t size;
};

struct flagos_dlblas_q4_weight {
    void * packed = nullptr;
    void * scales = nullptr;
    void * zeros = nullptr;
    int k = 0;
    int rows = 0;
    size_t bytes = 0;
};

struct ggml_backend_flagos_buffer_context {
    int device;
    void * device_ptr;
    uint64_t revision;
    std::unordered_map<const void *, flagos_dlblas_q4_weight> q4_dlblas_weights;
};

static std::atomic<uint64_t> flagos_buffer_revision { 1 };

static uint64_t flagos_next_buffer_revision() {
    return flagos_buffer_revision.fetch_add(1, std::memory_order_relaxed);
}

struct flagos_dequantized_weight {
    void * device_ptr;
    size_t n_elements;
    ggml_type source_type;
    uint64_t source_revision;
};

struct flagos_tensor_properties {
    const ggml_tensor * identity = nullptr;
    void * data = nullptr;
    int type = -1;
    int op = -1;
    std::array<int64_t, GGML_MAX_DIMS> ne {};
    std::array<size_t, GGML_MAX_DIMS> nb {};
    std::array<uint8_t, GGML_MAX_OP_PARAMS> op_params {};

    bool operator==(const flagos_tensor_properties & other) const {
        return identity == other.identity && data == other.data && type == other.type && op == other.op &&
            ne == other.ne && nb == other.nb && op_params == other.op_params;
    }
};

struct flagos_graph_node_properties {
    flagos_tensor_properties node;
    std::array<flagos_tensor_properties, GGML_MAX_SRC> src;
    std::array<const void *, GGML_MAX_SRC> resolved_src_data {};

    bool operator==(const flagos_graph_node_properties & other) const {
        return node == other.node && src == other.src && resolved_src_data == other.resolved_src_data;
    }
};

struct flagos_cuda_graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    bool warmup_complete = false;
    uint64_t last_used = 0;
    std::vector<flagos_graph_node_properties> node_properties;

    ~flagos_cuda_graph() {
        if (instance != nullptr) {
            flagos_cuda_check(cudaGraphExecDestroy(instance), "cudaGraphExecDestroy");
        }
        if (graph != nullptr) {
            flagos_cuda_check(cudaGraphDestroy(graph), "cudaGraphDestroy");
        }
    }
};

struct ggml_backend_flagos_context {
    int device = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    void * f16_activation = nullptr;
    size_t f16_activation_capacity = 0;
    void * f16_output = nullptr;
    size_t f16_output_capacity = 0;
    const void * last_cast_source = nullptr;
    int last_cast_elements = 0;
    std::unique_ptr<flagos_kernel_registry> kernels;
    bool log_kernel_stats = false;
    bool fusion_enabled = true;
    bool graph_capture_enabled = false;
    bool dequant_blas_enabled = false;
    bool q4_dlblas_enabled = false;
    uint64_t graph_calls = 0;
    uint64_t graph_captures = 0;
    uint64_t graph_replays = 0;
    uint64_t graph_cache_tick = 0;
    uint64_t graph_cache_evictions = 0;
    uint64_t add_calls = 0;
    uint64_t mul_calls = 0;
    uint64_t scale_calls = 0;
    uint64_t copy_calls = 0;
    uint64_t swiglu_calls = 0;
    uint64_t set_rows_calls = 0;
    uint64_t get_rows_calls = 0;
    uint64_t ssm_conv_calls = 0;
    uint64_t flash_attn_decode_calls = 0;
    uint64_t rope_calls = 0;
    uint64_t rms_norm_calls = 0;
    uint64_t rms_norm_mul_calls = 0;
    uint64_t mul_mat_q4_k_calls = 0;
    uint64_t mul_mat_q6_k_calls = 0;
    uint64_t dequant_blas_calls = 0;
    uint64_t dequant_cache_misses = 0;
    size_t dequant_cache_bytes = 0;
    uint64_t q4_dlblas_calls = 0;
    uint64_t q4_dlblas_cache_misses = 0;
    size_t q4_dlblas_cache_bytes = 0;
    uint64_t weight_cache_hits = 0;
    uint64_t weight_cache_misses = 0;
    size_t weight_cache_bytes = 0;
    std::unordered_map<const void *, flagos_cached_weight> weight_cache;
    std::unordered_map<const void *, flagos_dequantized_weight> dequantized_weights;
    std::unordered_map<const void *, flagos_dlblas_q4_weight> q4_dlblas_weights;
    std::unordered_set<const void *> q4_dlblas_seen;
    std::unordered_map<const ggml_tensor *, void *> weight_aliases;
    flagos_graph_plan_cache graph_plans;
    std::unordered_map<const void *, std::unique_ptr<flagos_cuda_graph>> cuda_graphs;
};

static void * flagos_resolve_data(const ggml_backend_flagos_context * context, const ggml_tensor * tensor) {
    const auto alias = context->weight_aliases.find(tensor);
    return alias == context->weight_aliases.end() ? tensor->data : alias->second;
}

// Host-side description of one GGUF Q4_K super-block. Keep local to the
// provider: the generic backend layer must not depend on a vendor packing.
struct flagos_q4_k_block {
    ggml_fp16_t d;
    ggml_fp16_t dmin;
    uint8_t scales[12];
    uint8_t quants[128];
};

static_assert(sizeof(flagos_q4_k_block) == 144, "unexpected Q4_K block layout");

static void flagos_q4_k_scale_min(
        const flagos_q4_k_block & block,
        int group,
        uint8_t & scale,
        uint8_t & minimum) {
    if (group < 4) {
        scale = block.scales[group] & 63;
        minimum = block.scales[group + 4] & 63;
    } else {
        scale = (block.scales[group + 4] & 15) | ((block.scales[group - 4] >> 6) << 4);
        minimum = (block.scales[group + 4] >> 4) | ((block.scales[group] >> 6) << 4);
    }
}

static uint8_t flagos_q4_k_quant(const flagos_q4_k_block & block, int index) {
    const int within_64 = index & 63;
    const uint8_t packed = block.quants[(index / 64) * 32 + (within_64 & 31)];
    return within_64 < 32 ? packed & 15 : packed >> 4;
}

static bool flagos_pack_q4_k_for_dlblas(
        const ggml_tensor * tensor,
        const void * host_data,
        std::vector<uint8_t> & packed,
        std::vector<ggml_fp16_t> & scales,
        std::vector<ggml_fp16_t> & zeros) {
    const int64_t k = tensor->ne[0];
    const int64_t rows = tensor->ne[1];
    if (tensor->type != GGML_TYPE_Q4_K || host_data == nullptr ||
        k <= 0 || k % 256 != 0 || rows <= 0 || rows % 2 != 0 ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1 || !ggml_is_contiguous(tensor)) {
        return false;
    }

    const int64_t blocks_per_row = k / 256;
    const int64_t groups_per_row = k / 32;
    const auto * blocks = static_cast<const flagos_q4_k_block *>(host_data);
    packed.resize(static_cast<size_t>(rows * k / 2));
    scales.resize(static_cast<size_t>(rows * groups_per_row));
    zeros.resize(static_cast<size_t>(rows * groups_per_row));
    std::vector<uint8_t> zero_scale(static_cast<size_t>(rows * groups_per_row), 0);

    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t block_index = 0; block_index < blocks_per_row; ++block_index) {
            const flagos_q4_k_block & block = blocks[row * blocks_per_row + block_index];
            const float d = ggml_fp16_to_fp32(block.d);
            const float dmin = ggml_fp16_to_fp32(block.dmin);
            for (int group = 0; group < 8; ++group) {
                uint8_t scale_code = 0;
                uint8_t minimum_code = 0;
                flagos_q4_k_scale_min(block, group, scale_code, minimum_code);
                const float original_scale = d * scale_code;
                const float offset = dmin * minimum_code;
                // DLBLAS stores group parameters as [K/group_size, rows],
                // matching the contiguous layout emitted by the vendor
                // quantizer after reshaping scales to [-1, output_rows].
                const int64_t global_group = block_index * 8 + group;
                const size_t parameter_index = static_cast<size_t>(global_group * rows + row);

                ggml_fp16_t scale_half = ggml_fp32_to_fp16(original_scale);
                float rounded_scale = ggml_fp16_to_fp32(scale_half);
                if (rounded_scale == 0.0f || !std::isfinite(rounded_scale)) {
                    // A zero-scale Q4_K group is constant. Re-encode it with
                    // q=0 so DLBLAS can still express the additive minimum.
                    const float replacement = offset == 0.0f ? 1.0f : std::fabs(offset);
                    scale_half = ggml_fp32_to_fp16(replacement);
                    rounded_scale = ggml_fp16_to_fp32(scale_half);
                    zero_scale[parameter_index] = 1;
                }
                if (rounded_scale == 0.0f || !std::isfinite(rounded_scale)) {
                    return false;
                }

                const float zero_point = offset / rounded_scale;
                if (!std::isfinite(zero_point)) {
                    return false;
                }
                scales[parameter_index] = scale_half;
                zeros[parameter_index] = ggml_fp32_to_fp16(zero_point);
            }
        }
    }

    // DLBLAS interprets A as column-major [rows, K] and packs adjacent rows
    // into low/high nibbles. GGUF instead packs two K positions per row, so a
    // one-time transpose/repack is required for each immutable model weight.
    const int64_t packed_rows = rows / 2;
    for (int64_t column = 0; column < k; ++column) {
        const int64_t block_index = column / 256;
        const int within_block = static_cast<int>(column % 256);
        const int group = within_block / 32;
        const int64_t global_group = block_index * 8 + group;
        for (int64_t row = 0; row < rows; row += 2) {
            const flagos_q4_k_block & low_block = blocks[row * blocks_per_row + block_index];
            const flagos_q4_k_block & high_block = blocks[(row + 1) * blocks_per_row + block_index];
            uint8_t low = zero_scale[static_cast<size_t>(global_group * rows + row)]
                ? 0 : flagos_q4_k_quant(low_block, within_block);
            uint8_t high = zero_scale[static_cast<size_t>(global_group * rows + row + 1)]
                ? 0 : flagos_q4_k_quant(high_block, within_block);
            packed[static_cast<size_t>(column * packed_rows + row / 2)] = low | (high << 4);
        }
    }
    return true;
}

static void flagos_free_q4_dlblas_weight(flagos_dlblas_q4_weight & cached) {
    if (cached.packed != nullptr) flagos_cuda_check(cudaFree(cached.packed), "cudaFree DLBLAS Q4 packed");
    if (cached.scales != nullptr) flagos_cuda_check(cudaFree(cached.scales), "cudaFree DLBLAS Q4 scales");
    if (cached.zeros != nullptr) flagos_cuda_check(cudaFree(cached.zeros), "cudaFree DLBLAS Q4 zeros");
    cached = {};
}

static bool flagos_create_q4_dlblas_weight(
        const ggml_tensor * host_tensor,
        const void * host_data,
        cudaStream_t stream,
        flagos_dlblas_q4_weight & cached) {
    std::vector<uint8_t> packed;
    std::vector<ggml_fp16_t> scales;
    std::vector<ggml_fp16_t> zeros;
    if (!flagos_pack_q4_k_for_dlblas(host_tensor, host_data, packed, scales, zeros)) {
        return false;
    }

    cached.k = static_cast<int>(host_tensor->ne[0]);
    cached.rows = static_cast<int>(host_tensor->ne[1]);
    cached.bytes = packed.size() + (scales.size() + zeros.size()) * sizeof(ggml_fp16_t);
    const size_t parameter_bytes = scales.size() * sizeof(ggml_fp16_t);
    const bool allocated =
        flagos_cuda_check(cudaMalloc(&cached.packed, packed.size()), "cudaMalloc DLBLAS Q4 packed") &&
        flagos_cuda_check(cudaMalloc(&cached.scales, parameter_bytes), "cudaMalloc DLBLAS Q4 scales") &&
        flagos_cuda_check(cudaMalloc(&cached.zeros, parameter_bytes), "cudaMalloc DLBLAS Q4 zeros");
    if (!allocated) {
        flagos_free_q4_dlblas_weight(cached);
        return false;
    }

    bool copied = false;
    if (stream != nullptr) {
        copied =
            flagos_cuda_check(cudaMemcpyAsync(cached.packed, packed.data(), packed.size(),
                cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync DLBLAS Q4 packed") &&
            flagos_cuda_check(cudaMemcpyAsync(cached.scales, scales.data(), parameter_bytes,
                cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync DLBLAS Q4 scales") &&
            flagos_cuda_check(cudaMemcpyAsync(cached.zeros, zeros.data(), parameter_bytes,
                cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync DLBLAS Q4 zeros") &&
            flagos_cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize DLBLAS Q4 cache");
    } else {
        copied =
            flagos_cuda_check(cudaMemcpy(cached.packed, packed.data(), packed.size(),
                cudaMemcpyHostToDevice), "cudaMemcpy DLBLAS Q4 packed") &&
            flagos_cuda_check(cudaMemcpy(cached.scales, scales.data(), parameter_bytes,
                cudaMemcpyHostToDevice), "cudaMemcpy DLBLAS Q4 scales") &&
            flagos_cuda_check(cudaMemcpy(cached.zeros, zeros.data(), parameter_bytes,
                cudaMemcpyHostToDevice), "cudaMemcpy DLBLAS Q4 zeros");
    }
    if (!copied) {
        flagos_free_q4_dlblas_weight(cached);
        return false;
    }
    return true;
}

static bool flagos_cache_q4_k_for_dlblas(
        ggml_backend_flagos_context * context,
        const ggml_tensor * host_tensor,
        void * raw_device_ptr) {
    if (!context->q4_dlblas_enabled || raw_device_ptr == nullptr ||
        context->q4_dlblas_weights.find(raw_device_ptr) != context->q4_dlblas_weights.end()) {
        return true;
    }

    flagos_dlblas_q4_weight cached;
    if (!flagos_create_q4_dlblas_weight(host_tensor, host_tensor->data, context->stream, cached)) {
        return false;
    }

    context->q4_dlblas_weights.emplace(raw_device_ptr, cached);
    ++context->q4_dlblas_cache_misses;
    context->q4_dlblas_cache_bytes += cached.bytes;
    return true;
}

static bool ggml_backend_buft_is_flagos(ggml_backend_buffer_type_t buft);

static const flagos_dlblas_q4_weight * flagos_find_q4_dlblas_weight(
        ggml_backend_flagos_context * context,
        const ggml_tensor * tensor,
        const void * weights_data) {
    const auto backend_cached = context->q4_dlblas_weights.find(weights_data);
    if (backend_cached != context->q4_dlblas_weights.end()) {
        return &backend_cached->second;
    }
    if (tensor->buffer == nullptr || !ggml_backend_buft_is_flagos(tensor->buffer->buft)) {
        return nullptr;
    }
    auto * buffer_context = static_cast<ggml_backend_flagos_buffer_context *>(tensor->buffer->context);
    const auto buffer_cached = buffer_context->q4_dlblas_weights.find(weights_data);
    if (buffer_cached == buffer_context->q4_dlblas_weights.end()) {
        return nullptr;
    }
    if (context->q4_dlblas_seen.insert(weights_data).second) {
        ++context->q4_dlblas_cache_misses;
        context->q4_dlblas_cache_bytes += buffer_cached->second.bytes;
    }
    return &buffer_cached->second;
}

static bool flagos_launch_mul_mat_q4_dlblas(
        ggml_backend_flagos_context * context,
        const ggml_tensor * src0,
        ggml_tensor * dst,
        const flagos_dlblas_q4_weight & cached,
        void * activation_data) {
    const int k = static_cast<int>(src0->ne[0]);
    const int rows = static_cast<int>(src0->ne[1]);
    const int columns = static_cast<int>(dst->ne[1]);
    const int activation_elements = k * columns;
    const int output_elements = rows * columns;
    if (cached.k != k || cached.rows != rows || columns != 1 ||
        static_cast<size_t>(activation_elements) > context->f16_activation_capacity ||
        static_cast<size_t>(output_elements) > context->f16_output_capacity) {
        return false;
    }

    if (context->last_cast_source != activation_data ||
        context->last_cast_elements != activation_elements) {
        if (!context->kernels->launch_cast_f32_f16(
                activation_data, context->f16_activation, activation_elements, context->stream)) {
            return false;
        }
        context->last_cast_source = activation_data;
        context->last_cast_elements = activation_elements;
    }

    flagos_dlblas_quant_parameters_v2 parameters {};
    parameters.a_group_size_m = 1;
    parameters.a_group_size_k = 32;
    parameters.a_zeropoints = cached.zeros;
    parameters.a_zeropoints_type = CUDA_R_16F;
    parameters.a_scales = cached.scales;
    parameters.a_scales_type = CUDA_R_16F;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (!flagos_blas_check(dlblasGemmExV2(
            context->blas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            rows,
            columns,
            k,
            &alpha,
            cached.packed,
            CUDA_R_4U,
            rows,
            context->f16_activation,
            CUDA_R_16F,
            k,
            &beta,
            context->f16_output,
            CUDA_R_16F,
            rows,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            &parameters), "dlblasGemmExV2 Q4_K") ||
        !context->kernels->launch_cast_f16_f32(
            context->f16_output, dst->data, output_elements, context->stream)) {
        return false;
    }
    ++context->q4_dlblas_calls;
    return true;
}

static bool flagos_launch_mul_mat_dequant_blas(
        ggml_backend_flagos_context * context,
        const ggml_tensor * src0,
        ggml_tensor * dst,
        void * weights_data,
        void * activation_data) {
    const size_t n_elements = static_cast<size_t>(src0->ne[0]) * src0->ne[1];
    const auto * source_buffer = weights_data == src0->data && src0->buffer != nullptr
        ? static_cast<const ggml_backend_flagos_buffer_context *>(src0->buffer->context)
        : nullptr;
    const uint64_t source_revision = source_buffer != nullptr ? source_buffer->revision : 0;
    auto dequantized = context->dequantized_weights.find(weights_data);
    if (dequantized != context->dequantized_weights.end() &&
        (dequantized->second.n_elements != n_elements ||
         dequantized->second.source_type != src0->type ||
         dequantized->second.source_revision != source_revision)) {
        flagos_cuda_check(cudaFree(dequantized->second.device_ptr), "cudaFree stale dequantized weight");
        context->dequant_cache_bytes -= dequantized->second.n_elements * sizeof(ggml_fp16_t);
        context->dequantized_weights.erase(dequantized);
        dequantized = context->dequantized_weights.end();
    }
    if (dequantized == context->dequantized_weights.end()) {
        void * device_ptr = nullptr;
        const size_t size = n_elements * sizeof(ggml_fp16_t);
        if (!flagos_cuda_check(cudaMalloc(&device_ptr, size), "cudaMalloc dequantized weight")) {
            return false;
        }
        const unsigned int n_blocks = static_cast<unsigned int>(n_elements / 256);
        if (!context->kernels->launch_dequant_qk_f16(
                src0->type, weights_data, device_ptr, n_blocks, context->stream)) {
            flagos_cuda_check(cudaFree(device_ptr), "cudaFree dequantized weight");
            return false;
        }
        dequantized = context->dequantized_weights.emplace(
            weights_data,
            flagos_dequantized_weight { device_ptr, n_elements, src0->type, source_revision }).first;
        ++context->dequant_cache_misses;
        context->dequant_cache_bytes += size;
    }

    const int k = static_cast<int>(src0->ne[0]);
    const int rows = static_cast<int>(src0->ne[1]);
    const int columns = static_cast<int>(dst->ne[1]);
    const int activation_elements = k * columns;
    if (static_cast<size_t>(activation_elements) > context->f16_activation_capacity) {
        return false;
    }
    if (context->last_cast_source != activation_data ||
        context->last_cast_elements != activation_elements) {
        if (!context->kernels->launch_cast_f32_f16(
                activation_data, context->f16_activation, activation_elements, context->stream)) {
            return false;
        }
        context->last_cast_source = activation_data;
        context->last_cast_elements = activation_elements;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    return flagos_blas_check(cublasGemmEx(
        context->blas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        rows,
        columns,
        k,
        &alpha,
        dequantized->second.device_ptr,
        CUDA_R_16F,
        k,
        context->f16_activation,
        CUDA_R_16F,
        k,
        &beta,
        dst->data,
        CUDA_R_32F,
        rows,
        CUBLAS_COMPUTE_32F,
        columns == 1 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT), "cublasGemmEx") &&
        (++context->dequant_blas_calls, true);
}

struct ggml_backend_flagos_buffer_type_context {
    int device;
    std::string name;
};

struct ggml_backend_flagos_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    ggml_backend_buffer_type buffer_type;
    ggml_backend_flagos_buffer_type_context buffer_type_context;
};

static const char * ggml_backend_flagos_buffer_type_name(ggml_backend_buffer_type_t buft) {
    auto * context = static_cast<ggml_backend_flagos_buffer_type_context *>(buft->context);
    return context->name.c_str();
}

static bool ggml_backend_buft_is_flagos(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_flagos_buffer_type_name;
}

static void ggml_backend_flagos_buffer_free(ggml_backend_buffer_t buffer) {
    auto * context = static_cast<ggml_backend_flagos_buffer_context *>(buffer->context);
    for (auto & item : context->q4_dlblas_weights) {
        flagos_free_q4_dlblas_weight(item.second);
    }
    if (context->device_ptr != nullptr) {
        flagos_cuda_check(cudaSetDevice(context->device), "cudaSetDevice");
        flagos_cuda_check(cudaFree(context->device_ptr), "cudaFree");
    }
    delete context;
}

static void * ggml_backend_flagos_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * context = static_cast<ggml_backend_flagos_buffer_context *>(buffer->context);
    return context->device_ptr;
}

static void ggml_backend_flagos_buffer_memset_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor,
        uint8_t value,
        size_t offset,
        size_t size) {
    auto * context = static_cast<ggml_backend_flagos_buffer_context *>(buffer->context);
    if (!flagos_cuda_check(cudaSetDevice(context->device), "cudaSetDevice") ||
        !flagos_cuda_check(cudaMemset(static_cast<char *>(tensor->data) + offset, value, size), "cudaMemset")) {
        GGML_ABORT("FlagOS: device memset failed");
    }
    auto cached = context->q4_dlblas_weights.find(tensor->data);
    if (cached != context->q4_dlblas_weights.end()) {
        flagos_free_q4_dlblas_weight(cached->second);
        context->q4_dlblas_weights.erase(cached);
    }
    context->revision = flagos_next_buffer_revision();
}

static void ggml_backend_flagos_buffer_set_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    auto * context = static_cast<ggml_backend_flagos_buffer_context *>(buffer->context);
    // The buffer interface has no stream, so these copies go to the default
    // stream. dlgpu does not give the default stream the implicit ordering
    // against other streams that NVIDIA does, so drain in-flight kernels
    // explicitly or the copy races them and the driver rejects it.
    if (!flagos_cuda_check(cudaSetDevice(context->device), "cudaSetDevice") ||
        !flagos_cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize before H2D") ||
        !flagos_cuda_check(cudaMemcpy(static_cast<char *>(tensor->data) + offset, data, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D")) {
        GGML_LOG_ERROR("FlagOS: H2D target %s %s[%lld,%lld,%lld,%lld] data=%p offset=%zu size=%zu base=%p span=%zu\n",
            tensor->name, ggml_type_name(tensor->type),
            static_cast<long long>(tensor->ne[0]), static_cast<long long>(tensor->ne[1]),
            static_cast<long long>(tensor->ne[2]), static_cast<long long>(tensor->ne[3]),
            tensor->data, offset, size, context->device_ptr, static_cast<size_t>(0));
        GGML_ABORT("FlagOS: host-to-device copy failed");
    }
    // Any write can invalidate the repacked representation. Rebuild only when
    // this call provides a complete immutable Q4_K tensor; partial writes keep
    // the ordinary GGUF representation and therefore use the direct AOT path.
    auto old = context->q4_dlblas_weights.find(tensor->data);
    if (old != context->q4_dlblas_weights.end()) {
        flagos_free_q4_dlblas_weight(old->second);
        context->q4_dlblas_weights.erase(old);
    }
    if (flagos_q4_dlblas_requested() &&
        tensor->type == GGML_TYPE_Q4_K && offset == 0 && size == ggml_nbytes(tensor)) {
        flagos_dlblas_q4_weight cached;
        if (flagos_create_q4_dlblas_weight(tensor, data, nullptr, cached)) {
            context->q4_dlblas_weights.emplace(tensor->data, cached);
        } else if (std::getenv("FLAGOS_LOG_KERNELS") != nullptr) {
            GGML_LOG_WARN("FlagOS: no buffer-local DLBLAS Q4 cache for %s [%lld,%lld]\n",
                tensor->name,
                static_cast<long long>(tensor->ne[0]),
                static_cast<long long>(tensor->ne[1]));
        }
    }
    context->revision = flagos_next_buffer_revision();
}

static void ggml_backend_flagos_buffer_get_tensor(
        ggml_backend_buffer_t buffer,
        const ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size) {
    auto * context = static_cast<ggml_backend_flagos_buffer_context *>(buffer->context);
    // Diagnostic: surface any error left pending by an earlier async op, so a sticky
    // error from a prior kernel launch is not misattributed to this copy.
    const cudaError_t pending = cudaGetLastError();
    if (pending != cudaSuccess) {
        GGML_LOG_ERROR("FlagOS: D2H entered with pending error: %s\n", cudaGetErrorString(pending));
    }
    if (!flagos_cuda_check(cudaSetDevice(context->device), "cudaSetDevice") ||
        !flagos_cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize before D2H") ||
        !flagos_cuda_check(cudaMemcpy(data, static_cast<const char *>(tensor->data) + offset, size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H")) {
        GGML_LOG_ERROR("FlagOS: D2H source %s %s[%lld,%lld,%lld,%lld] data=%p offset=%zu size=%zu base=%p buf_size=%zu dst=%p\n",
            tensor->name, ggml_type_name(tensor->type),
            static_cast<long long>(tensor->ne[0]), static_cast<long long>(tensor->ne[1]),
            static_cast<long long>(tensor->ne[2]), static_cast<long long>(tensor->ne[3]),
            tensor->data, offset, size, context->device_ptr, buffer->size, data);
        GGML_ABORT("FlagOS: device-to-host copy failed");
    }
}

static bool ggml_backend_flagos_buffer_copy_tensor(
        ggml_backend_buffer_t buffer,
        const ggml_tensor * src,
        ggml_tensor * dst) {
    if (!src->buffer || !ggml_backend_buft_is_flagos(src->buffer->buft)) {
        return false;
    }

    auto * src_context = static_cast<ggml_backend_flagos_buffer_context *>(src->buffer->context);
    auto * dst_context = static_cast<ggml_backend_flagos_buffer_context *>(buffer->context);
    if (src_context->device != dst_context->device) {
        return false;
    }

    const bool copied = flagos_cuda_check(cudaSetDevice(dst_context->device), "cudaSetDevice") &&
        flagos_cuda_check(cudaMemcpy(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice), "cudaMemcpy D2D");
    if (copied) {
        dst_context->revision = flagos_next_buffer_revision();
    }
    return copied;
}

static void ggml_backend_flagos_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * context = static_cast<ggml_backend_flagos_buffer_context *>(buffer->context);
    if (!flagos_cuda_check(cudaSetDevice(context->device), "cudaSetDevice") ||
        !flagos_cuda_check(cudaMemset(context->device_ptr, value, buffer->size), "cudaMemset")) {
        GGML_ABORT("FlagOS: device buffer clear failed");
    }
    context->revision = flagos_next_buffer_revision();
}

static const ggml_backend_buffer_i ggml_backend_flagos_buffer_interface = {
    /* .free_buffer   = */ ggml_backend_flagos_buffer_free,
    /* .get_base      = */ ggml_backend_flagos_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_flagos_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_flagos_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_flagos_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ggml_backend_flagos_buffer_copy_tensor,
    /* .clear         = */ ggml_backend_flagos_buffer_clear,
    /* .reset         = */ nullptr,
};

static ggml_backend_buffer_t ggml_backend_flagos_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft,
        size_t size) {
    auto * buft_context = static_cast<ggml_backend_flagos_buffer_type_context *>(buft->context);
    if (!flagos_cuda_check(cudaSetDevice(buft_context->device), "cudaSetDevice")) {
        return nullptr;
    }

    void * device_ptr = nullptr;
    const size_t allocation_size = std::max<size_t>(size, 1);
    if (!flagos_cuda_check(cudaMalloc(&device_ptr, allocation_size), "cudaMalloc")) {
        return nullptr;
    }

    auto * context = new ggml_backend_flagos_buffer_context {
        buft_context->device,
        device_ptr,
        flagos_next_buffer_revision(),
        {},
    };
    return ggml_backend_buffer_init(buft, ggml_backend_flagos_buffer_interface, context, size);
}

static size_t ggml_backend_flagos_buffer_type_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

static const ggml_backend_buffer_type_i ggml_backend_flagos_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_flagos_buffer_type_name,
    /* .alloc_buffer   = */ ggml_backend_flagos_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_flagos_buffer_type_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ nullptr,
};

static const char * ggml_backend_flagos_name(ggml_backend_t backend) {
    return static_cast<ggml_backend_flagos_device_context *>(backend->device->context)->name.c_str();
}

static void ggml_backend_flagos_free(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_flagos_context *>(backend->context);
    flagos_cuda_check(cudaSetDevice(context->device), "cudaSetDevice");
    flagos_cuda_check(cudaStreamSynchronize(context->stream), "cudaStreamSynchronize");
    if (context->log_kernel_stats) {
        std::fprintf(stderr,
            "FlagOS kernel stats: graphs=%llu plans=%zu plan_hits=%llu plan_misses=%llu plan_evictions=%llu "
            "graph_captures=%llu graph_replays=%llu graph_cache=%zu graph_evictions=%llu "
            "add=%llu mul=%llu scale=%llu copy=%llu "
            "swiglu=%llu set_rows=%llu get_rows=%llu flash_attn_decode=%llu "
            "rope=%llu rms_norm=%llu rms_norm_mul=%llu "
            "mul_mat_q4_k=%llu mul_mat_q6_k=%llu dequant_blas=%llu dequant_misses=%llu dequant_bytes=%zu "
            "q4_dlblas=%llu q4_dlblas_misses=%llu q4_dlblas_bytes=%zu "
            "cache_hits=%llu cache_misses=%llu cache_bytes=%zu\n",
            static_cast<unsigned long long>(context->graph_calls),
            context->graph_plans.size(),
            static_cast<unsigned long long>(context->graph_plans.hits()),
            static_cast<unsigned long long>(context->graph_plans.misses()),
            static_cast<unsigned long long>(context->graph_plans.evictions()),
            static_cast<unsigned long long>(context->graph_captures),
            static_cast<unsigned long long>(context->graph_replays),
            context->cuda_graphs.size(),
            static_cast<unsigned long long>(context->graph_cache_evictions),
            static_cast<unsigned long long>(context->add_calls),
            static_cast<unsigned long long>(context->mul_calls),
            static_cast<unsigned long long>(context->scale_calls),
            static_cast<unsigned long long>(context->copy_calls),
            static_cast<unsigned long long>(context->swiglu_calls),
            static_cast<unsigned long long>(context->set_rows_calls),
            static_cast<unsigned long long>(context->get_rows_calls),
            static_cast<unsigned long long>(context->flash_attn_decode_calls),
            static_cast<unsigned long long>(context->rope_calls),
            static_cast<unsigned long long>(context->rms_norm_calls),
            static_cast<unsigned long long>(context->rms_norm_mul_calls),
            static_cast<unsigned long long>(context->mul_mat_q4_k_calls),
            static_cast<unsigned long long>(context->mul_mat_q6_k_calls),
            static_cast<unsigned long long>(context->dequant_blas_calls),
            static_cast<unsigned long long>(context->dequant_cache_misses),
            context->dequant_cache_bytes,
            static_cast<unsigned long long>(context->q4_dlblas_calls),
            static_cast<unsigned long long>(context->q4_dlblas_cache_misses),
            context->q4_dlblas_cache_bytes,
            static_cast<unsigned long long>(context->weight_cache_hits),
            static_cast<unsigned long long>(context->weight_cache_misses),
            context->weight_cache_bytes);
    }
    for (const auto & item : context->weight_cache) {
        flagos_cuda_check(cudaFree(item.second.device_ptr), "cudaFree cached weight");
    }
    context->cuda_graphs.clear();
    for (const auto & item : context->q4_dlblas_weights) {
        flagos_cuda_check(cudaFree(item.second.packed), "cudaFree DLBLAS Q4 packed");
        flagos_cuda_check(cudaFree(item.second.scales), "cudaFree DLBLAS Q4 scales");
        flagos_cuda_check(cudaFree(item.second.zeros), "cudaFree DLBLAS Q4 zeros");
    }
    for (const auto & item : context->dequantized_weights) {
        flagos_cuda_check(cudaFree(item.second.device_ptr), "cudaFree dequantized weight");
    }
    if (context->f16_activation != nullptr) {
        flagos_cuda_check(cudaFree(context->f16_activation), "cudaFree f16 activation");
    }
    if (context->f16_output != nullptr) {
        flagos_cuda_check(cudaFree(context->f16_output), "cudaFree f16 output");
    }
    if (context->blas != nullptr) {
        flagos_blas_check(cublasDestroy(context->blas), "cublasDestroy");
    }
    context->kernels.reset();
    flagos_cuda_check(cudaStreamDestroy(context->stream), "cudaStreamDestroy");
    delete context;
    delete backend;
}

static void ggml_backend_flagos_set_tensor_async(
        ggml_backend_t backend,
        ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    auto * context = static_cast<ggml_backend_flagos_context *>(backend->context);
    if (!flagos_cuda_check(cudaSetDevice(context->device), "cudaSetDevice") ||
        !flagos_cuda_check(cudaMemcpyAsync(static_cast<char *>(tensor->data) + offset, data, size,
            cudaMemcpyHostToDevice, context->stream), "cudaMemcpyAsync H2D")) {
        GGML_ABORT("FlagOS: asynchronous host-to-device copy failed");
    }
    if (tensor->buffer != nullptr && ggml_backend_buft_is_flagos(tensor->buffer->buft)) {
        auto * buffer_context = static_cast<ggml_backend_flagos_buffer_context *>(tensor->buffer->context);
        buffer_context->revision = flagos_next_buffer_revision();
    }
}

static void ggml_backend_flagos_get_tensor_async(
        ggml_backend_t backend,
        const ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size) {
    auto * context = static_cast<ggml_backend_flagos_context *>(backend->context);
    if (!flagos_cuda_check(cudaSetDevice(context->device), "cudaSetDevice") ||
        !flagos_cuda_check(cudaMemcpyAsync(data, static_cast<const char *>(tensor->data) + offset, size,
            cudaMemcpyDeviceToHost, context->stream), "cudaMemcpyAsync D2H")) {
        GGML_ABORT("FlagOS: asynchronous device-to-host copy failed");
    }
}

static bool ggml_backend_flagos_copy_tensor_async(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        const ggml_tensor * src,
        ggml_tensor * dst) {
    auto * context = static_cast<ggml_backend_flagos_context *>(backend_dst->context);
    if (!flagos_cuda_check(cudaSetDevice(context->device), "cudaSetDevice")) {
        return false;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        if (src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            auto cached = context->weight_cache.find(src->data);
            if (cached == context->weight_cache.end()) {
                void * device_ptr = nullptr;
                const size_t size = ggml_nbytes(src);
                if (!flagos_cuda_check(cudaMalloc(&device_ptr, std::max<size_t>(size, 1)), "cudaMalloc cached weight")) {
                    return false;
                }
                if (!flagos_cuda_check(cudaMemcpyAsync(device_ptr, src->data, size,
                        cudaMemcpyHostToDevice, context->stream), "cudaMemcpyAsync cached weight H2D")) {
                    flagos_cuda_check(cudaFree(device_ptr), "cudaFree cached weight");
                    return false;
                }
                cached = context->weight_cache.emplace(src->data, flagos_cached_weight { device_ptr, size }).first;
                ++context->weight_cache_misses;
                context->weight_cache_bytes += size;
            } else {
                if (cached->second.size != ggml_nbytes(src)) {
                    GGML_LOG_ERROR("FlagOS: cached weight size changed for %s\n", src->name);
                    return false;
                }
                ++context->weight_cache_hits;
            }
            if (context->q4_dlblas_enabled && src->type == GGML_TYPE_Q4_K &&
                !flagos_cache_q4_k_for_dlblas(context, src, cached->second.device_ptr)) {
                // Some Q4_K tensors (for example an odd-row lookup table) are
                // not valid DLBLAS matrices. Keep their standard packed copy;
                // the dispatcher will use the existing AOT implementation.
                if (std::getenv("FLAGOS_LOG_KERNELS") != nullptr) {
                    GGML_LOG_WARN("FlagOS: no DLBLAS Q4 cache for %s [%lld,%lld]\n",
                        src->name,
                        static_cast<long long>(src->ne[0]),
                        static_cast<long long>(src->ne[1]));
                }
            }
            context->weight_aliases[dst] = cached->second.device_ptr;
            return true;
        }
        context->weight_aliases.erase(dst);
        const bool copied = flagos_cuda_check(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src),
            cudaMemcpyHostToDevice, context->stream), "cudaMemcpyAsync H2D");
        if (copied && dst->buffer != nullptr && ggml_backend_buft_is_flagos(dst->buffer->buft)) {
            auto * buffer_context = static_cast<ggml_backend_flagos_buffer_context *>(dst->buffer->context);
            buffer_context->revision = flagos_next_buffer_revision();
        }
        return copied;
    }
    if (flagos_denglin_is_backend(backend_src)) {
        auto * src_context = static_cast<ggml_backend_flagos_context *>(backend_src->context);
        if (src_context->device == context->device) {
            context->weight_aliases.erase(dst);
            const bool copied = flagos_cuda_check(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src),
                cudaMemcpyDeviceToDevice, context->stream), "cudaMemcpyAsync D2D");
            if (copied && dst->buffer != nullptr && ggml_backend_buft_is_flagos(dst->buffer->buft)) {
                auto * buffer_context = static_cast<ggml_backend_flagos_buffer_context *>(dst->buffer->context);
                buffer_context->revision = flagos_next_buffer_revision();
            }
            return copied;
        }
    }
    return false;
}

static void ggml_backend_flagos_synchronize(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_flagos_context *>(backend->context);
    if (!flagos_cuda_check(cudaSetDevice(context->device), "cudaSetDevice") ||
        !flagos_cuda_check(cudaStreamSynchronize(context->stream), "cudaStreamSynchronize")) {
        GGML_ABORT("FlagOS: stream synchronization failed");
    }
}

static flagos_lowering_choice flagos_denglin_query_lowering(
        void * user_data,
        const ggml_cgraph * cgraph,
        const flagos_pattern_candidate & candidate) {
    auto * context = static_cast<ggml_backend_flagos_context *>(user_data);
    flagos_lowering_choice choice;
    if (!context->fusion_enabled || candidate.id != flagos_pattern_id::rms_norm_mul ||
        candidate.node_indices.size() != 2) {
        return choice;
    }

    const ggml_tensor * norm = cgraph->nodes[candidate.node_indices[0]];
    const ggml_tensor * mul = cgraph->nodes[candidate.node_indices[1]];
    const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    if (norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || weight == nullptr ||
        norm->src[0] == nullptr || norm->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        norm->src[0]->type != GGML_TYPE_F32 || weight->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(norm, norm->src[0]) || !ggml_are_same_shape(mul, norm) ||
        ggml_nelements(weight) != norm->ne[0] || norm->ne[0] > FLAGOS_RMS_NORM_MAX_COLS ||
        norm->ne[0] % FLAGOS_ROW_WIDTH_MULTIPLE != 0 || !ggml_is_contiguous(norm) ||
        !ggml_is_contiguous(mul) || !ggml_is_contiguous(norm->src[0]) || !ggml_is_contiguous(weight)) {
        return choice;
    }

    choice.supported = true;
    choice.capture_safe = true;
    choice.implementation_id = 1;
    return choice;
}

static enum ggml_status flagos_graph_evaluate(
        ggml_backend_flagos_context * context,
        ggml_cgraph * cgraph,
        const flagos_graph_plan & plan) {
    context->last_cast_source = nullptr;
    context->last_cast_elements = 0;
    for (const auto & step : plan.steps) {
        const int i = step.candidate.node_indices[0];
        ggml_tensor * node = cgraph->nodes[i];
        bool launched = false;
        if (node->op != GGML_OP_MUL_MAT && node->data == context->last_cast_source) {
            context->last_cast_source = nullptr;
            context->last_cast_elements = 0;
        }
        // Empty nodes carry no work. Every launcher derives its grid from the
        // element count, and a zero grid makes cuLaunchKernel return
        // CUDA_ERROR_INVALID_VALUE, so skip them before dispatching.
        if (ggml_nelements(node) == 0) {
            continue;
        }
        if (step.kind == flagos_execution_kind::pattern) {
            if (step.candidate.id != flagos_pattern_id::rms_norm_mul || step.candidate.node_indices.size() != 2) {
                GGML_LOG_ERROR("FlagOS: no executor for graph pattern %s\n",
                    flagos_pattern_name(step.candidate.id));
                context->weight_aliases.clear();
                return GGML_STATUS_FAILED;
            }
            ggml_tensor * mul = cgraph->nodes[step.candidate.node_indices[1]];
            const ggml_tensor * weight = mul->src[0] == node ? mul->src[1] : mul->src[0];
            launched = context->kernels->launch_rms_norm_mul(
                node->src[0], node, mul,
                flagos_resolve_data(context, weight), context->stream);
            if (launched) {
                ++context->rms_norm_mul_calls;
            }
        } else {
            switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                continue;
            case GGML_OP_ADD:
                launched = context->kernels->launch_add(
                    node,
                    flagos_resolve_data(context, node->src[0]),
                    flagos_resolve_data(context, node->src[1]),
                    context->stream);
                context->add_calls += launched;
                break;
            case GGML_OP_MUL:
                launched = context->kernels->launch_mul(
                    node->src[1], node,
                    flagos_resolve_data(context, node->src[0]),
                    flagos_resolve_data(context, node->src[1]),
                    context->stream);
                context->mul_calls += launched;
                break;
            case GGML_OP_SUB:
            case GGML_OP_DIV:
                launched = context->kernels->launch_binary(
                    node->op == GGML_OP_SUB ? context->kernels->sub_f32
                                            : context->kernels->div_f32,
                    node->src[1], node,
                    flagos_resolve_data(context, node->src[0]),
                    flagos_resolve_data(context, node->src[1]),
                    context->stream);
                break;
            case GGML_OP_UNARY: {
                const flagos_kernel * kernel = nullptr;
                switch (ggml_get_unary_op(node)) {
                    case GGML_UNARY_OP_SIGMOID:  kernel = &context->kernels->sigmoid_f32;  break;
                    case GGML_UNARY_OP_EXP:      kernel = &context->kernels->exp_f32;      break;
                    case GGML_UNARY_OP_SOFTPLUS: kernel = &context->kernels->softplus_f32; break;
                    default: break;
                }
                GGML_ASSERT(kernel != nullptr);
                launched = context->kernels->launch_unary(
                    *kernel, node,
                    flagos_resolve_data(context, node->src[0]),
                    context->stream);
                break;
            }
            case GGML_OP_SUM_ROWS:
            case GGML_OP_L2_NORM:
            case GGML_OP_NORM:
            case GGML_OP_CUMSUM: {
                const flagos_kernel * kernel = nullptr;
                bool has_eps = false;
                switch (node->op) {
                    case GGML_OP_SUM_ROWS: kernel = &context->kernels->sum_rows_f32; break;
                    case GGML_OP_CUMSUM:   kernel = &context->kernels->cumsum_f32;   break;
                    case GGML_OP_L2_NORM:  kernel = &context->kernels->l2_norm_f32; has_eps = true; break;
                    default:               kernel = &context->kernels->norm_f32;    has_eps = true; break;
                }
                launched = context->kernels->launch_row_reduce(
                    *kernel, node,
                    flagos_resolve_data(context, node->src[0]),
                    context->stream, has_eps);
                break;
            }
            case GGML_OP_SOFT_MAX:
                launched = context->kernels->launch_soft_max(
                    node,
                    flagos_resolve_data(context, node->src[0]),
                    node->src[1] ? flagos_resolve_data(context, node->src[1]) : nullptr,
                    context->stream);
                break;
            case GGML_OP_SCALE:
                launched = context->kernels->launch_scale(
                    node,
                    flagos_resolve_data(context, node->src[0]),
                    context->stream);
                context->scale_calls += launched;
                break;
            case GGML_OP_CPY:
            case GGML_OP_CONT:
                launched = context->kernels->launch_copy(
                    node,
                    flagos_resolve_data(context, node->src[0]),
                    context->stream);
                context->copy_calls += launched;
                break;
            case GGML_OP_GLU:
                launched = context->kernels->launch_swiglu_split(
                    node,
                    flagos_resolve_data(context, node->src[0]),
                    flagos_resolve_data(context, node->src[1]),
                    context->stream);
                context->swiglu_calls += launched;
                break;
            case GGML_OP_SET_ROWS:
                launched = context->kernels->launch_set_rows(node, context->stream);
                context->set_rows_calls += launched;
                break;
            case GGML_OP_GET_ROWS:
                launched = context->kernels->launch_get_rows(node, context->stream);
                context->get_rows_calls += launched;
                break;
            case GGML_OP_SSM_CONV:
                launched = context->kernels->launch_ssm_conv(node, context->stream);
                context->ssm_conv_calls += launched;
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                launched = context->kernels->launch_flash_attn_decode(node, context->stream);
                context->flash_attn_decode_calls += launched;
                break;
            case GGML_OP_ROPE:
                launched = context->kernels->launch_rope_neox(
                    node->src[0], node,
                    flagos_resolve_data(context, node->src[0]),
                    flagos_resolve_data(context, node->src[1]),
                    context->stream);
                context->rope_calls += launched;
                break;
            case GGML_OP_MUL_MAT:
                {
                // src1->ne[1] is the token/column dimension. The DLBLAS Q4
                // path is intentionally decode-only until its batched output
                // and memory trade-offs are validated on real models.
                const int columns = static_cast<int>(node->src[1]->ne[1]);
                void * weights_data = flagos_resolve_data(context, node->src[0]);
                const flagos_dlblas_q4_weight * q4_dlblas_weight =
                    context->q4_dlblas_enabled && node->src[0]->type == GGML_TYPE_Q4_K && columns == 1
                    ? flagos_find_q4_dlblas_weight(context, node->src[0], weights_data)
                    : nullptr;
                const bool q4_dlblas_available =
                    q4_dlblas_weight != nullptr;
                if (q4_dlblas_available) {
                    launched = flagos_launch_mul_mat_q4_dlblas(
                        context, node->src[0], node, *q4_dlblas_weight,
                        flagos_resolve_data(context, node->src[1]));
                    if (!launched) {
                        // Until the common Selector owns per-signature runtime
                        // rejection state, fail closed for this backend context.
                        // Repeated vendor JIT failures are both noisy and much
                        // slower than the already validated direct AOT path.
                        context->q4_dlblas_enabled = false;
                        GGML_LOG_WARN("FlagOS: disabling optional Q4 DLBLAS variant after dispatch failure; using fallback\n");
                    }
                }
                // A missing vendor JIT toolchain or an unsupported shape must
                // not turn an optional performance variant into a graph error.
                // The same Selector order is used for both cold-start failure
                // and normal capability decline.
                if (!launched && context->dequant_blas_enabled) {
                    launched = flagos_launch_mul_mat_dequant_blas(
                        context, node->src[0], node,
                        weights_data,
                        flagos_resolve_data(context, node->src[1]));
                }
                if (!launched) {
                    if (columns == 1) {
                        // Single column (decode/GEMV)
                        launched = context->kernels->launch_mul_mat(
                            node->src[0], node,
                            weights_data,
                            flagos_resolve_data(context, node->src[1]),
                            context->stream);
                    } else {
                        // Multiple columns (prefill/GEMM)
                        launched = context->kernels->launch_mul_mat_batched(
                            node->src[0], node,
                            weights_data,
                            flagos_resolve_data(context, node->src[1]),
                            columns,
                            context->stream);
                    }
                }
                if (launched && node->src[0]->type == GGML_TYPE_Q4_K) {
                    ++context->mul_mat_q4_k_calls;
                } else if (launched) {
                    ++context->mul_mat_q6_k_calls;
                }
                break;
                }
            case GGML_OP_RMS_NORM: {
                launched = context->kernels->launch_rms_norm(node->src[0], node, context->stream);
                context->rms_norm_calls += launched;
                break;
            }
            default:
                GGML_LOG_ERROR("FlagOS: unsupported op reached graph_compute: %s\n", ggml_op_name(node->op));
                context->weight_aliases.clear();
                return GGML_STATUS_FAILED;
            }
        }
        if (!launched) {
            GGML_LOG_ERROR("FlagOS: launch returned false for op %s (%s)\n",
                ggml_op_name(node->op), node->name);
            // Which operand is in the wrong address space is otherwise invisible.
            for (int src = 0; src < GGML_MAX_SRC; ++src) {
                const ggml_tensor * operand = node->src[src];
                if (operand == nullptr) {
                    continue;
                }
                GGML_LOG_ERROR("FlagOS:   src[%d] %s %s[%lld,%lld,%lld,%lld] data=%p resolved=%p buffer=%s\n",
                    src, operand->name, ggml_type_name(operand->type),
                    static_cast<long long>(operand->ne[0]), static_cast<long long>(operand->ne[1]),
                    static_cast<long long>(operand->ne[2]), static_cast<long long>(operand->ne[3]),
                    operand->data, flagos_resolve_data(context, operand),
                    operand->buffer ? ggml_backend_buffer_name(operand->buffer) : "(none)");
            }
            GGML_LOG_ERROR("FlagOS:   dst data=%p buffer=%s\n", node->data,
                node->buffer ? ggml_backend_buffer_name(node->buffer) : "(none)");
            context->weight_aliases.clear();
            return GGML_STATUS_FAILED;
        }
        if (std::getenv("FLAGOS_TRACE_OPS") != nullptr) {
            GGML_LOG_ERROR("FLAGOSTRACE %s %s %s[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] src0nb=[%zu,%zu,%zu,%zu]\n",
                step.kind == flagos_execution_kind::pattern ? flagos_pattern_name(step.candidate.id) : ggml_op_name(node->op),
                node->name, ggml_type_name(node->type),
                static_cast<long long>(node->ne[0]), static_cast<long long>(node->ne[1]),
                static_cast<long long>(node->ne[2]), static_cast<long long>(node->ne[3]),
                node->nb[0], node->nb[1], node->nb[2], node->nb[3],
                node->src[0] ? node->src[0]->nb[0] : 0, node->src[0] ? node->src[0]->nb[1] : 0,
                node->src[0] ? node->src[0]->nb[2] : 0, node->src[0] ? node->src[0]->nb[3] : 0);
        }
        // Debug aid: launches are async, so a faulting kernel normally surfaces
        // at some unrelated later API call. Synchronizing per node attributes the
        // fault to the node that actually caused it.
        if (std::getenv("FLAGOS_SYNC_EACH_OP") != nullptr) {
            const cudaError_t sync_status = cudaStreamSynchronize(context->stream);
            if (sync_status != cudaSuccess) {
                GGML_LOG_ERROR("FlagOS: fault after op %s (%s) dst=%s[%lld,%lld,%lld,%lld]: %s\n",
                    ggml_op_name(node->op), node->name, ggml_type_name(node->type),
                    static_cast<long long>(node->ne[0]), static_cast<long long>(node->ne[1]),
                    static_cast<long long>(node->ne[2]), static_cast<long long>(node->ne[3]),
                    cudaGetErrorString(sync_status));
                context->weight_aliases.clear();
                return GGML_STATUS_FAILED;
            }
        }
    }
    context->weight_aliases.clear();
    return GGML_STATUS_SUCCESS;
}

static flagos_tensor_properties flagos_get_tensor_properties(const ggml_tensor * tensor) {
    flagos_tensor_properties properties;
    if (tensor == nullptr) {
        return properties;
    }

    properties.identity = tensor;
    properties.data = tensor->data;
    properties.type = tensor->type;
    properties.op = tensor->op;
    std::copy_n(tensor->ne, GGML_MAX_DIMS, properties.ne.begin());
    std::copy_n(tensor->nb, GGML_MAX_DIMS, properties.nb.begin());
    std::memcpy(properties.op_params.data(), tensor->op_params, properties.op_params.size());
    return properties;
}

static bool flagos_graph_properties_changed(
        ggml_backend_flagos_context * context,
        flagos_cuda_graph * graph,
        const ggml_cgraph * cgraph) {
    bool changed = graph->node_properties.size() != static_cast<size_t>(cgraph->n_nodes);
    graph->node_properties.resize(cgraph->n_nodes);

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        flagos_graph_node_properties properties {};
        properties.node = flagos_get_tensor_properties(cgraph->nodes[i]);
        for (int src = 0; src < GGML_MAX_SRC; ++src) {
            const ggml_tensor * tensor = cgraph->nodes[i]->src[src];
            properties.src[src] = flagos_get_tensor_properties(tensor);
            if (tensor != nullptr) {
                properties.resolved_src_data[src] = flagos_resolve_data(context, tensor);
            }
        }
        if (!(graph->node_properties[i] == properties)) {
            graph->node_properties[i] = properties;
            changed = true;
        }
    }
    return changed;
}

static flagos_cuda_graph * flagos_get_cuda_graph(
        ggml_backend_flagos_context * context,
        const void * key) {
    ++context->graph_cache_tick;
    const auto existing = context->cuda_graphs.find(key);
    if (existing != context->cuda_graphs.end()) {
        existing->second->last_used = context->graph_cache_tick;
        return existing->second.get();
    }

    if (context->cuda_graphs.size() >= FLAGOS_MAX_CUDA_GRAPHS) {
        const auto lru = std::min_element(
            context->cuda_graphs.begin(), context->cuda_graphs.end(),
            [](const auto & left, const auto & right) {
                return left.second->last_used < right.second->last_used;
            });
        if (!flagos_cuda_check(cudaStreamSynchronize(context->stream), "cudaStreamSynchronize graph eviction")) {
            return nullptr;
        }
        context->cuda_graphs.erase(lru);
        ++context->graph_cache_evictions;
    }

    auto graph = std::make_unique<flagos_cuda_graph>();
    graph->last_used = context->graph_cache_tick;
    auto * result = graph.get();
    context->cuda_graphs.emplace(key, std::move(graph));
    return result;
}

static enum ggml_status ggml_backend_flagos_graph_compute(
        ggml_backend_t backend,
        ggml_cgraph * cgraph) {
    auto * context = static_cast<ggml_backend_flagos_context *>(backend->context);
    if (!flagos_cuda_check(cudaSetDevice(context->device), "cudaSetDevice")) {
        return GGML_STATUS_FAILED;
    }
    ++context->graph_calls;

    bool plan_created = false;
    const flagos_graph_plan & plan = context->graph_plans.get_or_create(
        cgraph, flagos_denglin_query_lowering, context, &plan_created);
    if (plan_created && std::getenv("FLAGOS_LOG_GRAPH_PLAN") != nullptr) {
        size_t pattern_count = 0;
        for (const auto & step : plan.steps) {
            if (step.kind == flagos_execution_kind::pattern) {
                ++pattern_count;
                GGML_LOG_INFO("FlagOS: plan %016llx selected %s at node %d\n",
                    static_cast<unsigned long long>(plan.structural_fingerprint),
                    flagos_pattern_name(step.candidate.id), step.candidate.node_indices[0]);
            }
        }
        GGML_LOG_INFO("FlagOS: built plan %016llx with %zu steps and %zu patterns for %d nodes\n",
            static_cast<unsigned long long>(plan.structural_fingerprint),
            plan.steps.size(), pattern_count, cgraph->n_nodes);
    }

    if (!context->graph_capture_enabled || !plan.capture_safe() || cgraph->n_nodes < 16) {
        return flagos_graph_evaluate(context, cgraph, plan);
    }

    const void * graph_key = cgraph->nodes[0];
    flagos_cuda_graph * graph = flagos_get_cuda_graph(context, graph_key);
    if (graph == nullptr) {
        context->weight_aliases.clear();
        return GGML_STATUS_FAILED;
    }
    const bool properties_changed = flagos_graph_properties_changed(context, graph, cgraph);

    bool capture = false;
    if (!graph->warmup_complete) {
        if (!properties_changed) {
            graph->warmup_complete = true;
            capture = true;
        }
    } else if (properties_changed) {
        graph->warmup_complete = false;
    } else if (graph->instance != nullptr) {
        const bool launched = flagos_cuda_check(
            cudaGraphLaunch(graph->instance, context->stream), "cudaGraphLaunch");
        context->weight_aliases.clear();
        if (launched) {
            ++context->graph_replays;
            return GGML_STATUS_SUCCESS;
        }
        return GGML_STATUS_FAILED;
    } else {
        capture = true;
    }

    if (!capture) {
        return flagos_graph_evaluate(context, cgraph, plan);
    }

    if (graph->instance != nullptr) {
        if (!flagos_cuda_check(cudaGraphExecDestroy(graph->instance), "cudaGraphExecDestroy")) {
            context->weight_aliases.clear();
            return GGML_STATUS_FAILED;
        }
        graph->instance = nullptr;
    }
    if (graph->graph != nullptr) {
        if (!flagos_cuda_check(cudaGraphDestroy(graph->graph), "cudaGraphDestroy")) {
            context->weight_aliases.clear();
            return GGML_STATUS_FAILED;
        }
        graph->graph = nullptr;
    }

    if (!flagos_cuda_check(
            cudaStreamBeginCapture(context->stream, cudaStreamCaptureModeRelaxed),
            "cudaStreamBeginCapture")) {
        context->weight_aliases.clear();
        return GGML_STATUS_FAILED;
    }
    const enum ggml_status status = flagos_graph_evaluate(context, cgraph, plan);
    if (!flagos_cuda_check(cudaStreamEndCapture(context->stream, &graph->graph), "cudaStreamEndCapture") ||
        status != GGML_STATUS_SUCCESS || graph->graph == nullptr ||
        !flagos_cuda_check(
            flagos_cuda_graph_instantiate(&graph->instance, graph->graph),
            "cudaGraphInstantiate") ||
        !flagos_cuda_check(cudaGraphLaunch(graph->instance, context->stream), "cudaGraphLaunch captured")) {
        context->weight_aliases.clear();
        return GGML_STATUS_FAILED;
    }
    ++context->graph_captures;
    return GGML_STATUS_SUCCESS;
}

struct ggml_backend_flagos_event_context {
    int device;
    cudaEvent_t event;
};

static void ggml_backend_flagos_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * backend_context = static_cast<ggml_backend_flagos_context *>(backend->context);
    auto * event_context = static_cast<ggml_backend_flagos_event_context *>(event->context);
    if (!flagos_cuda_check(cudaSetDevice(backend_context->device), "cudaSetDevice") ||
        !flagos_cuda_check(cudaEventRecord(event_context->event, backend_context->stream), "cudaEventRecord")) {
        GGML_ABORT("FlagOS: event record failed");
    }
}

static void ggml_backend_flagos_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * backend_context = static_cast<ggml_backend_flagos_context *>(backend->context);
    auto * event_context = static_cast<ggml_backend_flagos_event_context *>(event->context);
    if (!flagos_cuda_check(cudaSetDevice(backend_context->device), "cudaSetDevice") ||
        !flagos_cuda_check(cudaStreamWaitEvent(backend_context->stream, event_context->event, 0), "cudaStreamWaitEvent")) {
        GGML_ABORT("FlagOS: event wait failed");
    }
}

static const ggml_backend_i ggml_backend_flagos_interface = {
    /* .get_name            = */ ggml_backend_flagos_name,
    /* .free                = */ ggml_backend_flagos_free,
    /* .set_tensor_async    = */ ggml_backend_flagos_set_tensor_async,
    /* .get_tensor_async    = */ ggml_backend_flagos_get_tensor_async,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ ggml_backend_flagos_copy_tensor_async,
    /* .synchronize         = */ ggml_backend_flagos_synchronize,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ ggml_backend_flagos_graph_compute,
    /* .event_record        = */ ggml_backend_flagos_event_record,
    /* .event_wait          = */ ggml_backend_flagos_event_wait,
    /* .graph_optimize      = */ nullptr,
};

static ggml_guid_t ggml_backend_flagos_guid() {
    static ggml_guid guid = { 0x46, 0x6c, 0x61, 0x67, 0x4f, 0x53, 0x2d, 0x44, 0x65, 0x6e, 0x67, 0x6c, 0x69, 0x6e, 0x01, 0x00 };
    return &guid;
}

static bool flagos_denglin_is_backend(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_flagos_guid());
}

static const char * ggml_backend_flagos_device_name(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_flagos_device_context *>(dev->context)->name.c_str();
}

static const char * ggml_backend_flagos_device_description(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_flagos_device_context *>(dev->context)->description.c_str();
}

static void ggml_backend_flagos_device_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * context = static_cast<ggml_backend_flagos_device_context *>(dev->context);
    if (!flagos_cuda_check(cudaSetDevice(context->device), "cudaSetDevice") ||
        !flagos_cuda_check(cudaMemGetInfo(free, total), "cudaMemGetInfo")) {
        *free = 0;
        *total = 0;
    }
}

static enum ggml_backend_dev_type ggml_backend_flagos_device_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_flagos_device_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    auto * context = static_cast<ggml_backend_flagos_device_context *>(dev->context);
    props->name = context->name.c_str();
    props->description = context->description.c_str();
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id = context->pci_bus_id.empty() ? nullptr : context->pci_bus_id.c_str();
    ggml_backend_flagos_device_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ true,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t ggml_backend_flagos_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    auto * device_context = static_cast<ggml_backend_flagos_device_context *>(dev->context);
    if (!flagos_cuda_check(cudaSetDevice(device_context->device), "cudaSetDevice")) {
        return nullptr;
    }

    cudaStream_t stream = nullptr;
    if (!flagos_cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) {
        return nullptr;
    }

    auto kernels = std::make_unique<flagos_kernel_registry>();
    if (!kernels->initialize()) {
        flagos_cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
        return nullptr;
    }

    auto * context = new ggml_backend_flagos_context {};
    context->device = device_context->device;
    context->stream = stream;
    context->kernels = std::move(kernels);
    context->log_kernel_stats = std::getenv("FLAGOS_LOG_KERNELS") != nullptr;
    context->fusion_enabled = std::getenv("FLAGOS_NO_GRAPH_FUSION") == nullptr;
    context->graph_capture_enabled = std::getenv("FLAGOS_GRAPH_CAPTURE") != nullptr;
    context->dequant_blas_enabled = std::getenv("FLAGOS_DEQUANT_BLAS") != nullptr;
    context->q4_dlblas_enabled = flagos_q4_dlblas_requested();
    if (context->dequant_blas_enabled || context->q4_dlblas_enabled) {
        context->f16_activation_capacity = 8 * 1024 * 1024;
        context->f16_output_capacity = context->q4_dlblas_enabled ? 1024 * 1024 : 0;
        if (!flagos_blas_check(cublasCreate(&context->blas), "cublasCreate") ||
            !flagos_blas_check(cublasSetStream(context->blas, stream), "cublasSetStream") ||
            !flagos_cuda_check(cudaMalloc(
                &context->f16_activation,
                context->f16_activation_capacity * sizeof(ggml_fp16_t)),
                "cudaMalloc f16 activation") ||
            (context->q4_dlblas_enabled && !flagos_cuda_check(cudaMalloc(
                &context->f16_output,
                context->f16_output_capacity * sizeof(ggml_fp16_t)),
                "cudaMalloc f16 output"))) {
            if (context->f16_activation != nullptr) {
                flagos_cuda_check(cudaFree(context->f16_activation), "cudaFree f16 activation");
            }
            if (context->f16_output != nullptr) {
                flagos_cuda_check(cudaFree(context->f16_output), "cudaFree f16 output");
            }
            if (context->blas != nullptr) {
                flagos_blas_check(cublasDestroy(context->blas), "cublasDestroy");
            }
            delete context;
            flagos_cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
            return nullptr;
        }
    }
    return new ggml_backend {
        /* .guid    = */ ggml_backend_flagos_guid(),
        /* .iface   = */ ggml_backend_flagos_interface,
        /* .device  = */ dev,
        /* .context = */ context,
    };
}

static ggml_backend_buffer_type_t ggml_backend_flagos_device_buffer_type(ggml_backend_dev_t dev) {
    return &static_cast<ggml_backend_flagos_device_context *>(dev->context)->buffer_type;
}

// Diagnostic: record each distinct op signature the scheduler asked us about and we
// declined, so the CPU-fallback surface is visible instead of inferred.
static void flagos_log_declined_op(const ggml_tensor * op) {
    if (!getenv("GGML_FLAGOS_LOG_DECLINED")) {
        return;
    }
    static std::set<std::string> seen;
    char key[512];
    snprintf(key, sizeof(key), "%s|%s|%lld,%lld,%lld,%lld|%s",
        ggml_op_name(op->op), ggml_type_name(op->type),
        (long long) op->ne[0], (long long) op->ne[1],
        (long long) op->ne[2], (long long) op->ne[3],
        op->src[0] ? ggml_type_name(op->src[0]->type) : "-");
    if (!seen.insert(key).second) {
        return;
    }
    GGML_LOG_INFO("FlagOS: DECLINED %s\n", key);
}

static void flagos_log_unsupported_signature(const ggml_tensor * op) {
    if (std::getenv("FLAGOS_LOG_UNSUPPORTED_OPS") == nullptr) {
        return;
    }

    std::string signature = ggml_op_name(op->op);
    auto append_tensor = [&signature](const ggml_tensor * tensor) {
        if (tensor == nullptr) {
            signature += " null";
            return;
        }
        char description[384];
        std::snprintf(description, sizeof(description),
            " %s[%lld,%lld,%lld,%lld]nb[%zu,%zu,%zu,%zu]",
            ggml_type_name(tensor->type),
            static_cast<long long>(tensor->ne[0]),
            static_cast<long long>(tensor->ne[1]),
            static_cast<long long>(tensor->ne[2]),
            static_cast<long long>(tensor->ne[3]),
            tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
        signature += description;
    };
    append_tensor(op);
    for (const ggml_tensor * src : op->src) {
        append_tensor(src);
    }

    static std::mutex log_mutex;
    static std::unordered_set<std::string> logged_signatures;
    std::lock_guard<std::mutex> lock(log_mutex);
    if (logged_signatures.insert(signature).second) {
        std::fprintf(stderr, "FlagOS unsupported signature:%s\n", signature.c_str());
    }
}

static bool flagos_device_supports_op_impl(ggml_backend_dev_t dev, const ggml_tensor * op);

static bool ggml_backend_flagos_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const bool supported = flagos_device_supports_op_impl(dev, op);
    if (!supported) {
        flagos_log_declined_op(op);
    }
    return supported;
}

static bool flagos_device_supports_op_impl(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    // Escape hatch for the ops added on top of the original kernel set, so a
    // regression can be bisected against the previous behaviour without a rebuild.
    if (std::getenv("FLAGOS_NO_NEW_OPS") != nullptr) {
        switch (op->op) {
            case GGML_OP_SUB:
            case GGML_OP_DIV:
            case GGML_OP_UNARY:
            case GGML_OP_SUM_ROWS:
            case GGML_OP_L2_NORM:
            case GGML_OP_NORM:
            case GGML_OP_CUMSUM:
            case GGML_OP_SOFT_MAX:
                return false;
            default:
                break;
        }
    }

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_ADD:
            return op->src[0] && op->src[1] &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(op, op->src[0]) &&
                ggml_are_same_shape(op->src[0], op->src[1]) &&
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_nelements(op) <= INT_MAX;
        case GGML_OP_RMS_NORM:
            return op->src[0] &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(op, op->src[0]) &&
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) &&
                // The kernel reduces a whole row in one BLOCK-wide pass, so the
                // only real constraint is the compiled block width, not one
                // specific model's hidden size.
                op->ne[0] <= FLAGOS_RMS_NORM_MAX_COLS &&
                // RMS norm is a row reduction, so it needs the same tail-lane
                // guard as SUM_ROWS/SOFT_MAX above.
                op->ne[0] % FLAGOS_ROW_WIDTH_MULTIPLE == 0 &&
                ggml_nelements(op) <= INT_MAX;
        case GGML_OP_GLU:
            return op->src[0] && op->src[1] &&
                ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(op, op->src[0]) &&
                ggml_are_same_shape(op->src[0], op->src[1]) &&
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_nelements(op) <= INT_MAX;
        case GGML_OP_GET_ROWS: {
            if (std::getenv("FLAGOS_NO_GET_ROWS") != nullptr) {
                flagos_log_unsupported_signature(op);
                return false;
            }
            // Q4_K/Q6_K table lookup dequantizes only selected rows; F32 uses a
            // dense gather. Quantized rows must contain whole 256-element blocks.
            const ggml_type table_type = op->src[0] ? op->src[0]->type : GGML_TYPE_COUNT;
            const bool quantized = table_type == GGML_TYPE_Q4_K || table_type == GGML_TYPE_Q6_K;
            const bool supported = op->src[0] && op->src[1] &&
                op->type == GGML_TYPE_F32 &&
                (quantized || table_type == GGML_TYPE_F32) &&
                op->src[1]->type == GGML_TYPE_I32 &&
                op->ne[0] == op->src[0]->ne[0] &&
                op->ne[0] > 0 && op->ne[0] <= INT_MAX &&
                (!quantized || op->ne[0] % 256 == 0) &&
                op->src[0]->ne[1] > 0 &&
                op->src[0]->ne[1] <= INT_MAX / op->src[0]->ne[0] &&
                op->src[0]->ne[2] == 1 && op->src[0]->ne[3] == 1 &&
                // One index per output row; the kernel maps program_id(0) to a
                // flat index position, so any 1-D index layout is fine.
                op->ne[2] == 1 && op->ne[3] == 1 &&
                ggml_nelements(op->src[1]) == op->ne[1] &&
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_nelements(op) <= INT_MAX;
            if (!supported) {
                flagos_log_unsupported_signature(op);
            }
            return supported;
        }
        case GGML_OP_SSM_CONV: {
            if (std::getenv("FLAGOS_NO_SSM_CONV") != nullptr) {
                flagos_log_unsupported_signature(op);
                return false;
            }
            // Depthwise causal conv1d. The kernel walks the state window with
            // element strides, so nb[0] must be unit stride on every operand.
            const bool supported = op->src[0] && op->src[1] &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->src[0]->nb[0] == sizeof(float) &&
                op->src[1]->nb[0] == sizeof(float) &&
                op->nb[0] == sizeof(float) &&
                op->ne[0] == op->src[0]->ne[1] &&
                // CPU reference asserts the state rows are densely packed.
                op->src[0]->nb[1] == op->src[0]->ne[0] * sizeof(float) &&
                op->src[0]->ne[0] == op->src[1]->ne[0] - 1 + op->ne[1] &&
                op->ne[3] == 1 &&
                ggml_nelements(op) <= INT_MAX;
            if (!supported) {
                flagos_log_unsupported_signature(op);
            }
            return supported;
        }
        case GGML_OP_SET_ROWS: {
            if (std::getenv("FLAGOS_NO_SET_ROWS") != nullptr) {
                flagos_log_unsupported_signature(op);
                return false;
            }
            const bool supported = op->src[0] && op->src[1] && op->src[2] &&
                op->type == GGML_TYPE_F16 &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_I64 &&
                op->src[2]->type == GGML_TYPE_F16 &&
                op->ne[0] > 0 &&
                op->src[0]->ne[0] == op->ne[0] &&
                op->src[0]->ne[1] > 0 &&
                op->src[0]->ne[1] <= INT_MAX / op->src[0]->ne[0] &&
                op->src[0]->ne[2] == 1 &&
                op->src[0]->ne[3] == 1 &&
                op->src[1]->ne[0] == op->src[0]->ne[1] &&
                op->src[1]->ne[1] == 1 &&
                op->src[1]->ne[2] == 1 &&
                op->src[1]->ne[3] == 1 &&
                op->ne[2] == 1 &&
                op->ne[3] == 1 &&
                op->ne[0] <= INT_MAX &&
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]);
            if (!supported) {
                flagos_log_unsupported_signature(op);
            }
            return supported;
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            float max_bias = 0.0f;
            float logit_softcap = 0.0f;
            std::memcpy(&max_bias, op->op_params + sizeof(float), sizeof(float));
            std::memcpy(&logit_softcap, op->op_params + 2 * sizeof(float), sizeof(float));
            const ggml_tensor * q = op->src[0];
            const ggml_tensor * k = op->src[1];
            const ggml_tensor * v = op->src[2];
            const ggml_tensor * mask = op->src[3];
            const bool supported = q && k && v && mask && !op->src[4] &&
                op->type == GGML_TYPE_F32 &&
                q->type == GGML_TYPE_F32 &&
                k->type == GGML_TYPE_F16 &&
                v->type == GGML_TYPE_F16 &&
                mask->type == GGML_TYPE_F16 &&
                op->ne[0] == 128 && op->ne[1] > 0 && op->ne[1] <= INT_MAX &&
                op->ne[2] == 1 && op->ne[3] == 1 &&
                q->ne[0] == 128 && q->ne[1] == 1 && q->ne[2] == op->ne[1] && q->ne[3] == 1 &&
                k->ne[0] == 128 && k->ne[1] > 0 && k->ne[1] <= INT_MAX &&
                k->ne[2] > 0 && k->ne[2] <= INT_MAX && k->ne[3] == 1 &&
                q->ne[2] % k->ne[2] == 0 &&
                ggml_are_same_shape(k, v) &&
                mask->ne[0] == k->ne[1] && mask->ne[1] == 1 && mask->ne[2] == 1 && mask->ne[3] == 1 &&
                q->nb[0] == sizeof(float) &&
                k->nb[0] == sizeof(ggml_fp16_t) && v->nb[0] == sizeof(ggml_fp16_t) &&
                k->nb[1] >= 256 * sizeof(ggml_fp16_t) && k->nb[2] == 128 * sizeof(ggml_fp16_t) &&
                v->nb[1] == k->nb[1] && v->nb[2] == k->nb[2] &&
                q->nb[1] / sizeof(float) <= INT_MAX && q->nb[2] / sizeof(float) <= INT_MAX &&
                k->nb[1] / sizeof(ggml_fp16_t) <= INT_MAX && k->nb[2] / sizeof(ggml_fp16_t) <= INT_MAX &&
                v->nb[1] / sizeof(ggml_fp16_t) <= INT_MAX && v->nb[2] / sizeof(ggml_fp16_t) <= INT_MAX &&
                op->nb[1] / sizeof(float) <= INT_MAX && op->nb[2] / sizeof(float) <= INT_MAX &&
                ggml_is_contiguous(mask) && ggml_is_contiguous(op) &&
                max_bias == 0.0f && logit_softcap == 0.0f;
            if (!supported) {
                flagos_log_unsupported_signature(op);
            }
            return supported;
        }
        case GGML_OP_ROPE: {
            if (!op->src[0] || !op->src[1] || op->src[2] ||
                op->type != GGML_TYPE_F32 ||
                op->src[0]->type != GGML_TYPE_F32 ||
                op->src[1]->type != GGML_TYPE_I32 ||
                !ggml_are_same_shape(op, op->src[0]) ||
                !ggml_is_contiguous(op) ||
                !ggml_is_contiguous(op->src[0]) ||
                !ggml_is_contiguous(op->src[1]) ||
                op->src[0]->ne[2] != op->src[1]->ne[0] ||
                ggml_nelements(op) > INT_MAX) {
                return false;
            }
            const int32_t * params = static_cast<const int32_t *>(op->op_params);
            float freq_scale = 0.0f;
            float ext_factor = 0.0f;
            float attn_factor = 0.0f;
            std::memcpy(&freq_scale, params + 6, sizeof(float));
            std::memcpy(&ext_factor, params + 7, sizeof(float));
            std::memcpy(&attn_factor, params + 8, sizeof(float));
            return params[2] == GGML_ROPE_TYPE_NEOX &&
                params[1] > 0 &&
                params[1] % 2 == 0 &&
                params[1] == op->src[0]->ne[0] &&
                freq_scale == 1.0f &&
                ext_factor == 0.0f &&
                attn_factor == 1.0f;
        }
        case GGML_OP_MUL_MAT:
            if (!op->src[0] || !op->src[1]) {
                return false;
            }
            {
                const bool dequant_blas = std::getenv("FLAGOS_DEQUANT_BLAS") != nullptr;
                const int64_t columns = op->src[1]->ne[1];
                return
                (op->src[0]->type == GGML_TYPE_Q4_K || op->src[0]->type == GGML_TYPE_Q6_K) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == op->src[1]->ne[0] &&
                op->src[0]->ne[0] % 256 == 0 &&
                op->src[0]->ne[0] <= INT_MAX &&
                op->src[0]->ne[1] > 0 &&
                op->src[0]->ne[1] <= INT_MAX &&
                op->src[0]->ne[2] == 1 &&
                op->src[0]->ne[3] == 1 &&
                columns > 0 &&
                columns <= (dequant_blas ? 512 : 512) &&
                op->src[0]->ne[0] <= INT_MAX / columns &&
                op->src[1]->ne[2] == 1 &&
                op->src[1]->ne[3] == 1 &&
                op->ne[0] == op->src[0]->ne[1] &&
                op->ne[1] == columns &&
                op->ne[2] == 1 &&
                op->ne[3] == 1 &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op);
            }
        case GGML_OP_CPY:
        case GGML_OP_CONT: {
            if (std::getenv("FLAGOS_NO_COPY") != nullptr) {
                flagos_log_unsupported_signature(op);
                return false;
            }
            // Flat-index kernel: dense in and out, same type or F32->F16.
            if (!op->src[0] ||
                ggml_nelements(op) != ggml_nelements(op->src[0]) ||
                ggml_nelements(op) > INT_MAX) {
                flagos_log_unsupported_signature(op);
                return false;
            }
            // The strided kernel indexes up to 4 dims and assumes nb[0] is a
            // whole number of elements for both sides.
            const bool dense = ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            if (!dense) {
                if (std::getenv("FLAGOS_NO_STRIDED_COPY") != nullptr ||
                    op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
                    op->nb[0] % sizeof(float) != 0 ||
                    op->src[0]->nb[0] % sizeof(float) != 0) {
                    flagos_log_unsupported_signature(op);
                    return false;
                }
                // The kernel pairs elements by flat traversal order on each side,
                // matching the CPU dup_bytes reference. That is only equivalent to
                // ggml's CPY when the innermost dimension is unit-stride on both
                // sides; a transposed nb[0] would need a different traversal.
                if (op->src[0]->nb[0] != sizeof(float) || op->nb[0] != sizeof(float)) {
                    flagos_log_unsupported_signature(op);
                    return false;
                }
                // The launcher passes element strides as int. Cache views can carry
                // multi-gigabyte nb[2]/nb[3]; truncating those to int yields a bogus
                // (often negative) stride and the kernel walks off its allocation,
                // which the device reports as a DMMU page fault.
                for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                    if (op->nb[dim] / sizeof(float) > INT_MAX ||
                        op->src[0]->nb[dim] / sizeof(float) > INT_MAX) {
                        flagos_log_unsupported_signature(op);
                        return false;
                    }
                }
                return true;
            }
            if (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F16) {
                return true;
            }
            // copy_f32 is AOT-compiled with f32 pointers, so f16->f16 would
            // reinterpret the data. Only F32->F32 is safe here.
            if (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                return true;
            }
            flagos_log_unsupported_signature(op);
            return false;
        }
        case GGML_OP_SCALE: {
            // The kernel walks a flat index range, so both sides must be dense.
            return op->src[0] &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(op, op->src[0]) &&
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_nelements(op) <= INT_MAX;
        }
        case GGML_OP_SUB:
        case GGML_OP_DIV: {
            // Same flat-index broadcast contract as GGML_OP_MUL below.
            if (!op->src[0] || !op->src[1] ||
                op->type != GGML_TYPE_F32 ||
                op->src[0]->type != GGML_TYPE_F32 ||
                op->src[1]->type != GGML_TYPE_F32 ||
                !ggml_are_same_shape(op, op->src[0]) ||
                !ggml_is_contiguous(op) ||
                !ggml_is_contiguous(op->src[0]) ||
                !ggml_is_contiguous(op->src[1]) ||
                !ggml_can_repeat(op->src[1], op->src[0]) ||
                ggml_nelements(op) > INT_MAX ||
                ggml_nelements(op->src[1]) > INT_MAX) {
                return false;
            }
            if (ggml_are_same_shape(op->src[0], op->src[1])) {
                return true;
            }
            // A flat `offset % src1_elements` index only matches ggml's repeat
            // semantics for a single leading row; higher-dim repeats would need
            // per-dimension strides.
            return op->src[1]->ne[0] == op->src[0]->ne[0] &&
                op->src[1]->ne[1] == 1 &&
                op->src[1]->ne[2] == 1 &&
                op->src[1]->ne[3] == 1;
        }
        case GGML_OP_UNARY: {
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                    return op->src[0] &&
                        op->type == GGML_TYPE_F32 &&
                        op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_are_same_shape(op, op->src[0]) &&
                        ggml_is_contiguous(op) &&
                        ggml_is_contiguous(op->src[0]) &&
                        ggml_nelements(op) <= INT_MAX;
                default:
                    return false;
            }
        }
        case GGML_OP_SUM_ROWS:
        case GGML_OP_L2_NORM:
        case GGML_OP_NORM:
        case GGML_OP_CUMSUM: {
            // One program per row with BLOCK spanning ne[0], so the row must fit
            // the AOT-compiled block width and be contiguous.
            if (!op->src[0] ||
                op->type != GGML_TYPE_F32 ||
                op->src[0]->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(op) ||
                !ggml_is_contiguous(op->src[0]) ||
                op->src[0]->ne[0] > FLAGOS_ROW_BLOCK ||
                op->src[0]->ne[0] % FLAGOS_ROW_WIDTH_MULTIPLE != 0 ||
                ggml_nelements(op->src[0]) > INT_MAX) {
                return false;
            }
            // SUM_ROWS collapses ne[0] to 1; the others keep the row width.
            if (op->op == GGML_OP_SUM_ROWS) {
                return op->ne[0] == 1 && ggml_nrows(op) == ggml_nrows(op->src[0]);
            }
            return ggml_are_same_shape(op, op->src[0]);
        }
        case GGML_OP_SOFT_MAX: {
            // ALiBi (max_bias != 0) is not implemented in the kernel.
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            if (!op->src[0] ||
                op->type != GGML_TYPE_F32 ||
                op->src[0]->type != GGML_TYPE_F32 ||
                !ggml_are_same_shape(op, op->src[0]) ||
                !ggml_is_contiguous(op) ||
                !ggml_is_contiguous(op->src[0]) ||
                op->src[0]->ne[0] > FLAGOS_ROW_BLOCK ||
                op->src[0]->ne[0] % FLAGOS_ROW_WIDTH_MULTIPLE != 0 ||
                max_bias != 0.0f ||
                ggml_nelements(op) > INT_MAX) {
                return false;
            }
            // HAS_MASK is baked into the AOT kernel as true, so the unmasked form
            // would read the mask pointer anyway. Require a real mask here; the
            // maskless case stays on the CPU until a second variant is compiled.
            // The mask is read with the same row stride as src0, so it must be
            // f32 and cover every row rather than broadcast across ne[2]/ne[3].
            // src[2] carries attention sinks, which the kernel does not apply.
            return op->src[1] != nullptr &&
                op->src[2] == nullptr &&
                op->src[1]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_are_same_shape(op->src[1], op->src[0]);
        }
        case GGML_OP_MUL: {
            if (!op->src[0] || !op->src[1] ||
                op->type != GGML_TYPE_F32 ||
                op->src[0]->type != GGML_TYPE_F32 ||
                op->src[1]->type != GGML_TYPE_F32 ||
                !ggml_are_same_shape(op, op->src[0]) ||
                !ggml_is_contiguous(op) ||
                !ggml_is_contiguous(op->src[0]) ||
                !ggml_is_contiguous(op->src[1]) ||
                !ggml_can_repeat(op->src[1], op->src[0]) ||
                ggml_nelements(op) > INT_MAX ||
                ggml_nelements(op->src[1]) > INT_MAX) {
                return false;
            }
            if (ggml_are_same_shape(op->src[0], op->src[1])) {
                return true;
            }
            return op->src[1]->ne[0] == op->src[0]->ne[0] &&
                op->src[1]->ne[1] == 1 &&
                op->src[1]->ne[2] == 1 &&
                op->src[1]->ne[3] == 1;
        }
        default: {
            flagos_log_unsupported_signature(op);
            return false;
        }
    }
}

static bool ggml_backend_flagos_device_supports_buft(
        ggml_backend_dev_t dev,
        ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_flagos(buft) && buft->device == dev;
}

static bool ggml_backend_flagos_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return ggml_backend_flagos_device_supports_op(dev, op) &&
        (op->op == GGML_OP_ADD || op->op == GGML_OP_MUL || op->op == GGML_OP_MUL_MAT);
}

static ggml_backend_event_t ggml_backend_flagos_device_event_new(ggml_backend_dev_t dev) {
    auto * device_context = static_cast<ggml_backend_flagos_device_context *>(dev->context);
    if (!flagos_cuda_check(cudaSetDevice(device_context->device), "cudaSetDevice")) {
        return nullptr;
    }

    cudaEvent_t event = nullptr;
    if (!flagos_cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreateWithFlags")) {
        return nullptr;
    }
    auto * context = new ggml_backend_flagos_event_context { device_context->device, event };
    return new ggml_backend_event { dev, context };
}

static void ggml_backend_flagos_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    auto * context = static_cast<ggml_backend_flagos_event_context *>(event->context);
    flagos_cuda_check(cudaSetDevice(context->device), "cudaSetDevice");
    flagos_cuda_check(cudaEventDestroy(context->event), "cudaEventDestroy");
    delete context;
    delete event;
}

static void ggml_backend_flagos_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    auto * context = static_cast<ggml_backend_flagos_event_context *>(event->context);
    if (!flagos_cuda_check(cudaSetDevice(context->device), "cudaSetDevice") ||
        !flagos_cuda_check(cudaEventSynchronize(context->event), "cudaEventSynchronize")) {
        GGML_ABORT("FlagOS: event synchronization failed");
    }
}

static const ggml_backend_device_i ggml_backend_flagos_device_interface = {
    /* .get_name             = */ ggml_backend_flagos_device_name,
    /* .get_description      = */ ggml_backend_flagos_device_description,
    /* .get_memory           = */ ggml_backend_flagos_device_memory,
    /* .get_type             = */ ggml_backend_flagos_device_type,
    /* .get_props            = */ ggml_backend_flagos_device_props,
    /* .init_backend         = */ ggml_backend_flagos_device_init,
    /* .get_buffer_type      = */ ggml_backend_flagos_device_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_flagos_device_supports_op,
    /* .supports_buft        = */ ggml_backend_flagos_device_supports_buft,
    /* .offload_op           = */ ggml_backend_flagos_device_offload_op,
    /* .event_new            = */ ggml_backend_flagos_device_event_new,
    /* .event_free           = */ ggml_backend_flagos_device_event_free,
    /* .event_synchronize    = */ ggml_backend_flagos_device_event_synchronize,
};

static std::vector<std::unique_ptr<ggml_backend_flagos_device_context>> flagos_device_contexts;
static std::vector<std::unique_ptr<ggml_backend_device>> flagos_devices;

static bool flagos_denglin_probe(ggml_backend_reg_t owner_reg) {
    static std::mutex mutex;
    static bool initialized = false;
    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) {
        return true;
    }
    initialized = true;

    int device_count = 0;
    if (!flagos_cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
        return false;
    }

    flagos_device_contexts.reserve(device_count);
    flagos_devices.reserve(device_count);
    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp properties {};
        std::string description = "Denglin CUDA-compatible GPU";
        if (flagos_cuda_check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties")) {
            description = properties.name;
        }

        char pci_bus_id[32] = {};
        std::string pci;
        if (flagos_cuda_check(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device), "cudaDeviceGetPCIBusId")) {
            pci = pci_bus_id;
            std::transform(pci.begin(), pci.end(), pci.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        }

        auto context = std::make_unique<ggml_backend_flagos_device_context>();
        context->device = device;
        context->name = "FlagOS" + std::to_string(device);
        context->description = description;
        context->pci_bus_id = pci;
        context->buffer_type_context = { device, context->name };

        auto backend_device = std::make_unique<ggml_backend_device>();
        backend_device->iface = ggml_backend_flagos_device_interface;
        backend_device->reg = owner_reg;
        backend_device->context = context.get();

        context->buffer_type = {
            /* .iface   = */ ggml_backend_flagos_buffer_type_interface,
            /* .device  = */ backend_device.get(),
            /* .context = */ &context->buffer_type_context,
        };

        flagos_device_contexts.push_back(std::move(context));
        flagos_devices.push_back(std::move(backend_device));
    }
    return true;
}

static bool flagos_denglin_set_device(size_t device) {
    return device <= static_cast<size_t>(INT_MAX) &&
        flagos_cuda_check(cudaSetDevice(static_cast<int>(device)), "cudaSetDevice");
}

static int flagos_denglin_get_device() {
    int device = 0;
    if (!flagos_cuda_check(cudaGetDevice(&device), "cudaGetDevice")) {
        return -1;
    }
    return device;
}

static size_t flagos_denglin_device_count() {
    return flagos_devices.size();
}

static ggml_backend_dev_t flagos_denglin_device_get(size_t index) {
    return index < flagos_devices.size() ? flagos_devices[index].get() : nullptr;
}

static uint64_t flagos_denglin_hash(const std::string & value) {
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

static constexpr uint64_t FLAGOS_DENGLIN_PROVIDER_ID = 0x64656e676c696e01ull;

static bool flagos_denglin_device_identity(size_t index, flagos_device_identity * identity) {
    if (index >= flagos_device_contexts.size() || identity == nullptr || index > UINT32_MAX) {
        return false;
    }
    const ggml_backend_flagos_device_context & device = *flagos_device_contexts[index];
    const std::string & stable_name = device.pci_bus_id.empty() ? device.name : device.pci_bus_id;
    *identity = {
        FLAGOS_DENGLIN_PROVIDER_ID,
        FLAGOS_DENGLIN_PROVIDER_ID,
        flagos_denglin_hash(stable_name),
        flagos_denglin_hash("Denglin:" + stable_name),
        static_cast<uint32_t>(index),
    };
    return true;
}

static bool flagos_denglin_device_caps(size_t index, flagos_device_caps * caps) {
    if (index >= flagos_devices.size() || caps == nullptr) {
        return false;
    }
    *caps = {
        flagos_provider_kind::gpu,
        FLAGOS_MEMORY_DEVICE_LOCAL,
        FLAGOS_EXECUTION_ASYNC_QUEUE |
            FLAGOS_EXECUTION_EVENTS |
            FLAGOS_EXECUTION_NATIVE_GRAPH |
            FLAGOS_EXECUTION_AOT_MODULE,
        "cubin",
    };
    return true;
}

static int flagos_denglin_score() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess ? device_count * 100 : 0;
}

const flagos_provider_v1 * flagos_denglin_provider() {
    static const flagos_provider_v1 provider = {
        FLAGOS_PROVIDER_API_VERSION,
        sizeof(flagos_provider_v1),
        { FLAGOS_DENGLIN_PROVIDER_ID, "Denglin", 1 },
        flagos_denglin_probe,
        flagos_denglin_device_count,
        flagos_denglin_device_get,
        flagos_denglin_device_identity,
        flagos_denglin_device_caps,
        flagos_denglin_is_backend,
        flagos_denglin_set_device,
        flagos_denglin_get_device,
        flagos_denglin_score,
    };
    return &provider;
}
