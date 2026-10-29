#pragma once

// The core's plugin loader (plan.md "S1 design" §6): it finds the K3 plugin next to the core's own library, or at
// FLAGOS_SPACEMIT_PLUGIN (test-only), dlopens it, resolves the entry symbol and checks the table it returns as the
// mentor's registry checks a provider (flagos-provider.cpp:5-8). Any failure registers no device.
//
// Internal to the provider; C++ because the core is C++. dlopen only (plan.md "S1 design" A5): the linked-in form is
// not built, so a statically linked executable finds no entry and gets no device.

#include "flagos-spacemit-plugin.h"

#include <string>

enum class spacemit_load_result {
    ok,
    not_found,      // no path, or the library does not open
    no_entry,       // no flagos_spacemit_plugin_v1 symbol
    entry_refused,  // the entry returned no table (it refused the host table)
    bad_version,    // api_version is not FLAGOS_SPACEMIT_PLUGIN_API_VERSION
    short_struct,   // struct_size is shorter than the required part of the table
    null_member,    // a required member is NULL
    missing_kernel, // kernels() does not offer a kernel the core's contracts name
    no_device,      // the plugin found no device (the ordinary case on a host without the AI cores)
};

struct spacemit_plugin;  // the dlopen handle, the table and the path

// Loads the plugin at path, or at spacemit_plugin_default_path() when path is nullptr, and checks its table. On ok
// *out holds the plugin, whose handle is kept until spacemit_plugin_unload; on any failure *out is nullptr, the
// library is closed again and detail names the cause (the dlerror text, the member, the kernel name).
spacemit_load_result spacemit_plugin_load(const char * path, const spacemit_host_v1 & host,
                                          const char * const * required_kernels, size_t n_required,
                                          spacemit_plugin ** out, std::string * detail);

void                       spacemit_plugin_unload(spacemit_plugin * plugin);
const spacemit_plugin_v1 * spacemit_plugin_table(const spacemit_plugin * plugin);
const char *               spacemit_plugin_path(const spacemit_plugin * plugin);

// FLAGOS_SPACEMIT_PLUGIN when set (that path only, no fallback), else the plugin library beside the core's own library
// if that file exists, else the core's own library (which holds the entry in S1); empty when the path cannot be found.
// self (nullable) is set true only for that last case, the core's own image: a statically linked executable then names
// itself and cannot be dlopened, which is A5's documented consequence (no device), not a fault to warn about
std::string spacemit_plugin_default_path(bool * self = nullptr);
