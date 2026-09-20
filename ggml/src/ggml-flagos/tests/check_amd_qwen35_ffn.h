#pragma once

#include "ggml-backend.h"
#include "ggml.h"

// Exercise the provider-neutral MUL_MAT + MUL_MAT + SWIGLU graph using packed
// weights.  Besides checking the terminal result, this verifies that the two
// intermediate GEMV outputs remain untouched, which proves that the graph was
// consumed by the fused provider implementation.
void flagos_check_amd_qwen35_ffn_case(
        ggml_backend_t backend, ggml_backend_dev_t device,
        ggml_backend_buffer_type_t buft, ggml_context * context,
        ggml_type quant_type);
