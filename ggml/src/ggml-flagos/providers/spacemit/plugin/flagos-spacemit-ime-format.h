#pragma once

// The IME weight layouts' formats, internal to the plugin: MUL_MAT (flagos-spacemit-ime.cpp, which defines them) and
// MUL_MAT_ID (flagos-spacemit-ime-id.cpp) take the same K blocks, row sizes, quantizers and gemm entry points
// (flagos-spacemit-ime-kernels.h) for a weight's layout.

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

// one IME weight layout: its K block, its row sizes and its kernels.
// Most layouts pack a 4-row group as 4 rows of act_row_bytes, so one row's bytes do not depend on which quantizer
// wrote it (act_4row_bytes == act_row_bytes, act_group_align 1). The q4_1 zp4 layout's 4-row kernel wants its own
// pack, 304 bytes a row against the 1-row form's 290, whose groups want a 128-byte base: a caller stores row j of a
// group at base + j * act_4row_bytes and aligns the base of a run of groups to act_group_align
struct mm_format {
    int64_t k_block;  // K values per block, in the activations and in the weights
    size_t (*act_row_bytes)(size_t k);
    size_t (*act_4row_bytes)(size_t k);  // one row inside a 4-row group; at least act_row_bytes
    size_t act_group_align;              // the alignment a run of 4-row groups wants, 1 where there is none
    size_t (*weight_row_bytes)(size_t k);
    void (*quantize_row)(const float * a, size_t count_k, uint8_t * qa, bool reference);
    void (*quantize_4rows)(const float * a, size_t count_k, uint8_t * qa, bool reference);
    size_t (*gemm)(const uint8_t * qa, const uint8_t * qb, float * c, size_t count_m, size_t count_n, size_t k_blocks,
                   size_t ldc, bool reference);
};

// the format of a weight's layout; aborts for a weight without one (spacemit_find_op admits only weights with one)
const mm_format & mm_format_of(const ggml_tensor * w);

// the scalar references: test-only GGML_FLAGOS_KERNEL_VARIANTS=0 (flagos-spacemit-kernels.h), and always without the
// IME kernels
bool mm_use_reference();
