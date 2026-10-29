#pragma once

// The core-plugin interface of the SpacemiT K3 provider (plan.md "S1 design"): a pure C function table with opaque
// handles and a version, exported by the plugin through one entry symbol and loaded with dlopen.
//
// The core (compiled into libggml-flagos, built without libspert and without the IME2 toolchain) keeps the ggml
// vtables, the contract check, the op-to-steps expansion, the workspace and the capture driver, the plan cache and the
// diagnostics. The plugin keeps the device and its facts, the buffers with the repack inside set and get, the stream,
// the launch with the capture replay, the per-tile services and the kernels.
//
// C11: both sides include this file, and the test plugins of tests/plugins are C. ggml tensors cross as pointers,
// which holds while the plugin is built with the core (plan.md "S1 design" §5); the day it is built elsewhere, the
// fields the kernels read become a descriptor.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#    define FLAGOS_SPACEMIT_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#    define FLAGOS_SPACEMIT_PLUGIN_EXPORT
#endif

// the core accepts exactly this version (the loader's check, as ggml's backend loader checks its API version)
#define FLAGOS_SPACEMIT_PLUGIN_API_VERSION 1

typedef struct spacemit_plugin_device spacemit_plugin_device;  // opaque to the core
typedef struct spacemit_plugin_stream spacemit_plugin_stream;
typedef struct spacemit_plugin_buffer spacemit_plugin_buffer;

struct ggml_tensor;  // crosses as a pointer (in-tree plugin, one build)

// the mentor's spelling (flagos_status, flagos-registry.h:23-30), our meanings
enum spacemit_status {
    SPACEMIT_STATUS_SUCCESS = 0,
    SPACEMIT_STATUS_FAILED,            // a kernel returned false, or the runtime failed the launch
    SPACEMIT_STATUS_UNSUPPORTED,       // a contract, a layout or an access the plugin does not handle
    SPACEMIT_STATUS_ALLOC_FAILED,      // a buffer, the workspace or TCM
    SPACEMIT_STATUS_INVALID_ARGUMENT,  // a null handle, a range outside the tensor
};

// what the core gives the plugin at the entry: its logger and the directory the plugin was loaded from
typedef struct {
    uint32_t struct_size;
    void (*log)(int level, const char * text);  // level: an enum ggml_log_level value
    const char * plugin_dir;
} spacemit_host_v1;

// the device facts the core reports (its description, the profile) and the tiles of a launch
typedef struct {
    uint32_t     struct_size;
    uint32_t     n_tiles;
    uint32_t     vector_bits;
    size_t       tcm_bytes_per_tile;
    const char * target;
    const char * runtime;
    const char * description;
} spacemit_plugin_device_props;

struct spacemit_pairs;  // the pair waits of one launch, the plugin's (flagos-spacemit-exec.cpp)

// one AI-core tile of a launch: tile ith of nth, with that core's TCM and the workspace all tiles share.
// All tiles of a launch see the same tcm_size (0 and tcm == NULL when any tile lacks TCM), so they take the
// same kernel path.
typedef struct spacemit_tile {
    uint32_t                ith;
    uint32_t                nth;
    void *                  tcm;
    size_t                  tcm_size;
    void *                  workspace;
    size_t                  workspace_size;
    struct spacemit_pairs * pairs;  // for spacemit_pair_wait
} spacemit_tile;

// computes this tile's share of one step of a node; false means the node failed and the graph fails
typedef bool (*spacemit_step_fn)(const spacemit_tile * tile, struct ggml_tensor * node);

// one kernel the plugin offers, found by the core under its contract name
typedef struct {
    const char *     name;  // the step's contract name, "mul_mat.gemm"
    spacemit_step_fn fn;
    bool (*supports)(const struct ggml_tensor * op);  // the kernel's shape facts, or NULL
    size_t (*workspace_size)(const struct ggml_tensor * node);  // bytes of shared workspace, or NULL for none
} spacemit_plugin_kernel;

// one step of a captured split: the kernel, the node it runs on, and where its part of the workspace starts
typedef struct {
    spacemit_step_fn     fn;
    struct ggml_tensor * node;
    size_t               workspace_offset;
} spacemit_plugin_step;

// flags of replay()
enum {
    SPACEMIT_REPLAY_SYNC_LAUNCH      = 1u << 0,  // one step per launch, the host waiting for each (E3a, test-only)
    SPACEMIT_REPLAY_ORDERED_LAUNCHES = 1u << 1,  // one launch per step, ordered in the cores (E3b, test-only)
};

// test-only (structural-experiments.md E3b-diag): what the launches ordered in the cores did since the last call
typedef struct spacemit_ordered_counts {
    uint64_t launches;
    uint64_t shared_core;        // launches with two tiles on one core, at any of the points
    uint64_t moved_tiles;        // tiles found on another core than at the point before
    uint64_t tcm_elsewhere;      // launches where two tiles on different cores held the same TCM
    uint64_t pairs_on_one_core;  // launches where both tiles of a pair (2i, 2i+1) were on one core
} spacemit_ordered_counts;

struct spacemit_run_stats;  // test-only (E3c): the core's per-step clock reads, filled by the plugin's tiles

// what is not contract (plan.md "S1 design" §2, A2): the test counters and the per-tile clock reads, by their current
// names through get_proc_address, one typedef per name. NULL for every name in a release plugin
typedef uint64_t (*spacemit_steps_take_fn)(void);
typedef spacemit_ordered_counts (*spacemit_ordered_counts_take_fn)(void);
typedef void (*spacemit_mm_paths_take_fn)(uint64_t * path_a, uint64_t * path_b, uint64_t * path_c,
                                          uint64_t * path_b_staged);
typedef void (*spacemit_mm_tail_take_fn)(uint64_t * rows, uint64_t * calls, uint32_t n_tiles);
typedef void (*spacemit_run_stats_attach_fn)(spacemit_plugin_stream * stream, struct spacemit_run_stats * stats);

typedef struct {
    uint32_t api_version;  // the mentor's fields (flagos-provider.h:66-67)
    uint32_t struct_size;
    /* device */
    uint32_t (*device_count)(void);
    enum spacemit_status (*device_open)(uint32_t index, spacemit_plugin_device ** dev);  // the probe; no runtime start
    void (*device_close)(spacemit_plugin_device * dev);
    enum spacemit_status (*device_props)(spacemit_plugin_device * dev, spacemit_plugin_device_props * out);
    enum spacemit_status (*device_memory)(spacemit_plugin_device * dev, size_t * free, size_t * total);
    /* memory: buffers on huge pages from 2 MiB, 64-byte alignment, size 0 allowed */
    enum spacemit_status (*buffer_alloc)(spacemit_plugin_device * dev, size_t size, spacemit_plugin_buffer ** buf);
    void (*buffer_free)(spacemit_plugin_buffer * buf);
    void * (*buffer_base)(spacemit_plugin_buffer * buf);
    /* tensors: the layout rule and the repack inside (flagos-spacemit-weights.h:28-44) */
    uint32_t (*tensor_layout)(const struct ggml_tensor * t);  // 0 plain, else a layout id; a view answers for its storage
    size_t (*tensor_alloc_size)(const struct ggml_tensor * t);
    enum spacemit_status (*tensor_set)(struct ggml_tensor * t, const void * data, size_t offset, size_t size);
    enum spacemit_status (*tensor_get)(const struct ggml_tensor * t, void * data, size_t offset, size_t size);
    enum spacemit_status (*tensor_fill)(struct ggml_tensor * t, uint8_t value, size_t offset, size_t size);
    /* stream: one per backend context, the core grant held until destroyed */
    enum spacemit_status (*stream_create)(spacemit_plugin_device * dev, uint32_t n_tiles,
                                          spacemit_plugin_stream ** stream);
    void (*stream_destroy)(spacemit_plugin_stream * stream);
    enum spacemit_status (*stream_synchronize)(spacemit_plugin_stream * stream);
    /* kernels and the capture replay */
    const spacemit_plugin_kernel * (*kernels)(uint32_t * count);
    enum spacemit_status (*replay)(spacemit_plugin_stream * stream, const spacemit_plugin_step * steps,
                                   uint32_t n_steps, void * workspace, size_t workspace_size, uint32_t flags);
    /* what is not contract: the names above. A required member, like every one before it; it may answer NULL */
    void * (*get_proc_address)(const char * name);
    /* zero copy (plan.md "S1 design" A6): the optional tail under struct_size, NULL until S2 implements them */
    enum spacemit_status (*register_host_ptr)(spacemit_plugin_device * dev, void * ptr, size_t size,
                                              spacemit_plugin_buffer ** buf);
    enum spacemit_status (*tensor_prepare)(struct ggml_tensor * t);  // the IME copy beside a mapped weight, in t->extra
} spacemit_plugin_v1;

#ifdef __cplusplus
extern "C" {
#endif

// the entry symbol: NULL when the host table is missing or too short
FLAGOS_SPACEMIT_PLUGIN_EXPORT const spacemit_plugin_v1 * flagos_spacemit_plugin_v1(const spacemit_host_v1 * host);

// as flagos_status_name (flagos-registry.cpp:41-51); defined by the core
const char * spacemit_status_name(enum spacemit_status status);

#ifdef __cplusplus
}
#endif
