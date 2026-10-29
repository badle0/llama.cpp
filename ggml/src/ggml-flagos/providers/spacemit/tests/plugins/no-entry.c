// Fake plugin for the loader's negative matrix (plan.md "S1 step 1 spec" §7): a library that opens but holds no
// flagos_spacemit_plugin_v1, so the loader must answer no_entry. Its only symbol is the one below. C11, built by
// tests.cmake.

#include "flagos-spacemit-plugin.h"

FLAGOS_SPACEMIT_PLUGIN_EXPORT const spacemit_plugin_v1 * flagos_spacemit_not_the_entry(const spacemit_host_v1 * host);

const spacemit_plugin_v1 * flagos_spacemit_not_the_entry(const spacemit_host_v1 * host) {
    (void) host;
    return NULL;
}
