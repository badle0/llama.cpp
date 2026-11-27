#pragma once

// The common FlagOS registry only sees flagos_provider_v1 and ggml_backend_dev_t.

#include "../../../flagos-provider.h"

const flagos_provider_v1 * flagos_spacemit_provider();

// The plugin the core loaded for this device, or nullptr for any other device (flagos-check-spacemit's loader check;
// plan.md "S1 step 1 spec" §7).
struct spacemit_plugin;
const spacemit_plugin * spacemit_device_plugin(ggml_backend_dev_t dev);

// test-only: the plugin's device handle the core opened for this device, or nullptr for any other device; a stream
// made on it shares the executor of the device's backends (flagos-check-spacemit's pair-wait launches)
struct spacemit_plugin_device;
spacemit_plugin_device * spacemit_device_plugin_device(ggml_backend_dev_t dev);
