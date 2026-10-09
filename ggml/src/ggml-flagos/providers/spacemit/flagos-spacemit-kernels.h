#pragma once

#include "flagos-spacemit-ops.h"

// RVV kernels; each computes the share of one node that belongs to its tile

bool spacemit_kernel_add_f32(const spacemit_tile & tile, ggml_tensor * dst);
