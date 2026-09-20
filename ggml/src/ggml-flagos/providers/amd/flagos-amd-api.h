#pragma once

// Keep HIP and ROCm types private to the AMD provider.  The common FlagOS
// registry only sees flagos_provider_v1 and ggml_backend_dev_t.

#include "../../flagos-provider.h"

const flagos_provider_v1 * flagos_amd_provider();
