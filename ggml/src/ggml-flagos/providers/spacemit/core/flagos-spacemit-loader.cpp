#if defined(__linux__) && !defined(_GNU_SOURCE)
#    define _GNU_SOURCE  // dladdr
#endif

#include "flagos-spacemit-loader.h"

#include "../flagos-spacemit-api.h"  // flagos_spacemit_provider: the core symbol dladdr locates

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if !defined(_WIN32)
#    include <dlfcn.h>
#endif

// dlopen and dlsym are called here, not through ggml's dl_load_library: that lives in libggml
// (ggml/src/CMakeLists.txt:242-244), which a backend library must not depend on. Windows is not a target.

struct spacemit_plugin {
    void *                     handle = nullptr;
    const spacemit_plugin_v1 * table  = nullptr;
    std::string                path;
};

const char * spacemit_status_name(enum spacemit_status status) {
    switch (status) {
        case SPACEMIT_STATUS_SUCCESS:          return "success";
        case SPACEMIT_STATUS_FAILED:           return "failed";
        case SPACEMIT_STATUS_UNSUPPORTED:      return "unsupported";
        case SPACEMIT_STATUS_ALLOC_FAILED:     return "alloc_failed";
        case SPACEMIT_STATUS_INVALID_ARGUMENT: return "invalid_argument";
    }
    return "unknown";
}

namespace {

void set_detail(std::string * detail, const std::string & text) {
    if (detail != nullptr) {
        *detail = text;
    }
}

std::string dl_error() {
#if defined(_WIN32)
    return "no dlopen on this platform";
#else
    const char * text = dlerror();
    return text != nullptr ? text : "unknown dynamic-linker error";
#endif
}

bool file_exists(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    FILE * f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    std::fclose(f);
    return true;
}

// every member of the table before the optional zero-copy tail, in order, with its name for the detail text
struct member {
    const char * name;
    const void * value;
};

}  // namespace

std::string spacemit_plugin_default_path(bool * self) {
    if (self != nullptr) {
        *self = false;
    }
    const char * from_env = std::getenv("FLAGOS_SPACEMIT_PLUGIN");
    if (from_env != nullptr && *from_env != '\0') {
        return from_env;  // that path only, no fallback (test-only)
    }
#if defined(_WIN32)
    return "";
#else
    Dl_info info;
    if (dladdr(reinterpret_cast<const void *>(&flagos_spacemit_provider), &info) == 0 || info.dli_fname == nullptr) {
        return "";
    }
    const std::string own = info.dli_fname;
    const size_t      cut = own.find_last_of('/');
#    if defined(__APPLE__)
    const char * name = "libflagos-provider-spacemit.dylib";
#    else
    const char * name = "libflagos-provider-spacemit.so";
#    endif
    if (cut != std::string::npos) {
        const std::string beside = own.substr(0, cut + 1) + name;
        if (file_exists(beside)) {
            return beside;  // S2's plugin library, searched first
        }
    }
    if (self != nullptr) {
        *self = true;  // S1: the core's own image holds the entry
    }
    return own;
#endif
}

spacemit_load_result spacemit_plugin_load(const char * path, const spacemit_host_v1 & host,
                                          const char * const * required_kernels, size_t n_required,
                                          spacemit_plugin ** out, std::string * detail) {
    if (out == nullptr) {
        return spacemit_load_result::not_found;
    }
    *out = nullptr;
    set_detail(detail, "");

    const std::string library = path != nullptr ? std::string(path) : spacemit_plugin_default_path();
    if (library.empty()) {
        set_detail(detail, "no plugin path (dladdr found no library for the core)");
        return spacemit_load_result::not_found;
    }

#if defined(_WIN32)
    set_detail(detail, "no dlopen on this platform");
    return spacemit_load_result::not_found;
#else
    dlerror();  // clear any earlier error
    void * handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        set_detail(detail, dl_error());
        return spacemit_load_result::not_found;
    }

    // every failure below closes the library again
    const auto fail = [&](spacemit_load_result result, const std::string & text) {
        set_detail(detail, text);
        dlclose(handle);
        return result;
    };

    dlerror();
    using entry_fn = const spacemit_plugin_v1 * (*) (const spacemit_host_v1 *);
    auto entry     = reinterpret_cast<entry_fn>(dlsym(handle, "flagos_spacemit_plugin_v1"));
    if (entry == nullptr) {
        return fail(spacemit_load_result::no_entry, dl_error());
    }

    const spacemit_plugin_v1 * table = entry(&host);
    if (table == nullptr) {
        return fail(spacemit_load_result::entry_refused, "flagos_spacemit_plugin_v1 returned no table");
    }
    if (table->api_version != FLAGOS_SPACEMIT_PLUGIN_API_VERSION) {
        return fail(spacemit_load_result::bad_version,
                    "api_version " + std::to_string(table->api_version) + ", expected " +
                        std::to_string(FLAGOS_SPACEMIT_PLUGIN_API_VERSION));
    }
    const size_t required_size = offsetof(spacemit_plugin_v1, register_host_ptr);
    if (table->struct_size < required_size) {
        return fail(spacemit_load_result::short_struct, "struct_size " + std::to_string(table->struct_size) +
                                                            ", at least " + std::to_string(required_size) + " needed");
    }

    const member members[] = {
        { "device_count",       reinterpret_cast<const void *>(table->device_count)       },
        { "device_open",        reinterpret_cast<const void *>(table->device_open)        },
        { "device_close",       reinterpret_cast<const void *>(table->device_close)       },
        { "device_props",       reinterpret_cast<const void *>(table->device_props)       },
        { "device_memory",      reinterpret_cast<const void *>(table->device_memory)      },
        { "buffer_alloc",       reinterpret_cast<const void *>(table->buffer_alloc)       },
        { "buffer_free",        reinterpret_cast<const void *>(table->buffer_free)        },
        { "buffer_base",        reinterpret_cast<const void *>(table->buffer_base)        },
        { "tensor_layout",      reinterpret_cast<const void *>(table->tensor_layout)      },
        { "tensor_alloc_size",  reinterpret_cast<const void *>(table->tensor_alloc_size)  },
        { "tensor_set",         reinterpret_cast<const void *>(table->tensor_set)         },
        { "tensor_get",         reinterpret_cast<const void *>(table->tensor_get)         },
        { "tensor_fill",        reinterpret_cast<const void *>(table->tensor_fill)        },
        { "stream_create",      reinterpret_cast<const void *>(table->stream_create)      },
        { "stream_destroy",     reinterpret_cast<const void *>(table->stream_destroy)     },
        { "stream_synchronize", reinterpret_cast<const void *>(table->stream_synchronize) },
        { "kernels",            reinterpret_cast<const void *>(table->kernels)            },
        { "replay",             reinterpret_cast<const void *>(table->replay)             },
        { "get_proc_address",   reinterpret_cast<const void *>(table->get_proc_address)   },
    };
    for (const member & m : members) {
        if (m.value == nullptr) {
            return fail(spacemit_load_result::null_member, std::string(m.name) + " is NULL");
        }
    }

    uint32_t                       n_kernels = 0;
    const spacemit_plugin_kernel * kernels   = table->kernels(&n_kernels);
    for (size_t i = 0; i < n_required; i++) {
        bool found = false;
        for (uint32_t k = 0; k < n_kernels && !found; k++) {
            found = kernels != nullptr && kernels[k].name != nullptr && kernels[k].fn != nullptr &&
                    std::strcmp(kernels[k].name, required_kernels[i]) == 0;
        }
        if (!found) {
            return fail(spacemit_load_result::missing_kernel, std::string("no kernel ") + required_kernels[i]);
        }
    }

    if (table->device_count() == 0) {
        return fail(spacemit_load_result::no_device, "the plugin found no device");
    }

    auto * plugin  = new spacemit_plugin;
    plugin->handle = handle;
    plugin->table  = table;
    plugin->path   = library;
    *out           = plugin;
    return spacemit_load_result::ok;
#endif
}

void spacemit_plugin_unload(spacemit_plugin * plugin) {
    if (plugin == nullptr) {
        return;
    }
#if !defined(_WIN32)
    if (plugin->handle != nullptr) {
        dlclose(plugin->handle);
    }
#endif
    delete plugin;
}

const spacemit_plugin_v1 * spacemit_plugin_table(const spacemit_plugin * plugin) {
    return plugin != nullptr ? plugin->table : nullptr;
}

const char * spacemit_plugin_path(const spacemit_plugin * plugin) {
    return plugin != nullptr ? plugin->path.c_str() : nullptr;
}
