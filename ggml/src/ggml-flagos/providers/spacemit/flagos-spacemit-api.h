#pragma once

// The common FlagOS registry only sees flagos_provider_v1 and ggml_backend_dev_t.

#include "../../flagos-provider.h"

const flagos_provider_v1 * flagos_spacemit_provider();

// The plugin the core loaded for this device, or nullptr for any other device (flagos-check-spacemit's loader check;
// plan.md "S1 step 1 spec" §7).
struct spacemit_plugin;
const spacemit_plugin * spacemit_device_plugin(ggml_backend_dev_t dev);
