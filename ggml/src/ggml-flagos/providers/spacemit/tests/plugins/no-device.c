// Fake plugin for the loader's negative matrix (plan.md "S1 step 1 spec" §7): a complete table whose device_count is 0,
// so the loader must answer no_device. C11, built by tests.cmake.

#include "flagos-spacemit-plugin.h"

static uint32_t fake_device_count(void) {
    return 0;  // the fault: the plugin finds no device
}

static enum spacemit_status fake_device_open(uint32_t index, spacemit_plugin_device ** dev) {
    (void) index;
    (void) dev;
    return SPACEMIT_STATUS_SUCCESS;
}

static void fake_device_close(spacemit_plugin_device * dev) {
    (void) dev;
}

static enum spacemit_status fake_device_props(spacemit_plugin_device * dev, spacemit_plugin_device_props * out) {
    (void) dev;
    (void) out;
    return SPACEMIT_STATUS_SUCCESS;
}

static enum spacemit_status fake_device_memory(spacemit_plugin_device * dev, size_t * free, size_t * total) {
    (void) dev;
    *free  = 0;
    *total = 0;
    return SPACEMIT_STATUS_SUCCESS;
}

static enum spacemit_status fake_buffer_alloc(spacemit_plugin_device * dev, size_t size, spacemit_plugin_buffer ** buf) {
    (void) dev;
    (void) size;
    (void) buf;
    return SPACEMIT_STATUS_ALLOC_FAILED;
}

static void fake_buffer_free(spacemit_plugin_buffer * buf) {
    (void) buf;
}

static void * fake_buffer_base(spacemit_plugin_buffer * buf) {
    (void) buf;
    return NULL;
}

static uint32_t fake_tensor_layout(const struct ggml_tensor * t) {
    (void) t;
    return 0;
}

static size_t fake_tensor_alloc_size(const struct ggml_tensor * t) {
    (void) t;
    return 0;
}

static enum spacemit_status fake_tensor_set(struct ggml_tensor * t, const void * data, size_t offset, size_t size) {
    (void) t;
    (void) data;
    (void) offset;
    (void) size;
    return SPACEMIT_STATUS_SUCCESS;
}

static enum spacemit_status fake_tensor_get(const struct ggml_tensor * t, void * data, size_t offset, size_t size) {
    (void) t;
    (void) data;
    (void) offset;
    (void) size;
    return SPACEMIT_STATUS_SUCCESS;
}

static enum spacemit_status fake_tensor_fill(struct ggml_tensor * t, uint8_t value, size_t offset, size_t size) {
    (void) t;
    (void) value;
    (void) offset;
    (void) size;
    return SPACEMIT_STATUS_SUCCESS;
}

static enum spacemit_status fake_stream_create(spacemit_plugin_device * dev, uint32_t n_tiles,
                                               spacemit_plugin_stream ** stream) {
    (void) dev;
    (void) n_tiles;
    (void) stream;
    return SPACEMIT_STATUS_SUCCESS;
}

static void fake_stream_destroy(spacemit_plugin_stream * stream) {
    (void) stream;
}

static enum spacemit_status fake_stream_synchronize(spacemit_plugin_stream * stream) {
    (void) stream;
    return SPACEMIT_STATUS_SUCCESS;
}

static bool fake_step(const spacemit_tile * tile, struct ggml_tensor * node) {
    (void) tile;
    (void) node;
    return true;
}

static const spacemit_plugin_kernel * fake_kernels(uint32_t * count) {
    static const spacemit_plugin_kernel kernels[] = {
        { "add.f32", fake_step, NULL, NULL },
        { "mul.f32", fake_step, NULL, NULL },
        { "rms_norm.f32", fake_step, NULL, NULL },
        { "rope", fake_step, NULL, NULL },
        { "set_rows", fake_step, NULL, NULL },
        { "cpy.quant", fake_step, NULL, NULL },
        { "get_rows.f32", fake_step, NULL, NULL },
        { "swiglu.f32", fake_step, NULL, NULL },
        { "mul_mat.quantize", fake_step, NULL, NULL },
        { "mul_mat.gemm", fake_step, NULL, NULL },
        { "flash_attn", fake_step, NULL, NULL },
        { "flash_attn.split.partial", fake_step, NULL, NULL },
        { "flash_attn.split.partial_paired", fake_step, NULL, NULL },
        { "flash_attn.split.merge", fake_step, NULL, NULL },
    };
    if (count != NULL) {
        *count = sizeof(kernels) / sizeof(kernels[0]);
    }
    return kernels;
}

static enum spacemit_status fake_replay(spacemit_plugin_stream * stream, const spacemit_plugin_step * steps,
                                        uint32_t n_steps, void * workspace, size_t workspace_size, uint32_t flags) {
    (void) stream;
    (void) steps;
    (void) n_steps;
    (void) workspace;
    (void) workspace_size;
    (void) flags;
    return SPACEMIT_STATUS_SUCCESS;
}

static void * fake_get_proc_address(const char * name) {
    (void) name;
    return NULL;
}

static const spacemit_plugin_v1 fake_table = {
    /* .api_version        = */ FLAGOS_SPACEMIT_PLUGIN_API_VERSION,
    /* .struct_size        = */ sizeof(spacemit_plugin_v1),
    /* .device_count       = */ fake_device_count,
    /* .device_open        = */ fake_device_open,
    /* .device_close       = */ fake_device_close,
    /* .device_props       = */ fake_device_props,
    /* .device_memory      = */ fake_device_memory,
    /* .buffer_alloc       = */ fake_buffer_alloc,
    /* .buffer_free        = */ fake_buffer_free,
    /* .buffer_base        = */ fake_buffer_base,
    /* .tensor_layout      = */ fake_tensor_layout,
    /* .tensor_alloc_size  = */ fake_tensor_alloc_size,
    /* .tensor_set         = */ fake_tensor_set,
    /* .tensor_get         = */ fake_tensor_get,
    /* .tensor_fill        = */ fake_tensor_fill,
    /* .stream_create      = */ fake_stream_create,
    /* .stream_destroy     = */ fake_stream_destroy,
    /* .stream_synchronize = */ fake_stream_synchronize,
    /* .kernels            = */ fake_kernels,
    /* .replay             = */ fake_replay,
    /* .get_proc_address   = */ fake_get_proc_address,
    /* .register_host_ptr  = */ NULL,
    /* .tensor_prepare     = */ NULL,
};

const spacemit_plugin_v1 * flagos_spacemit_plugin_v1(const spacemit_host_v1 * host) {
    (void) host;
    return &fake_table;
}
