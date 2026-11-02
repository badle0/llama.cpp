#include "flagos-spacemit-weights.h"

#include "flagos-spacemit-ime-kernels.h"

#include "../../../ggml-impl.h"

#include <cstdlib>
#include <cstring>
#include <vector>

// llama.cpp's KV cache and recurrent state, named before they are allocated (src/llama-kv-cache.cpp:231-235,
// src/llama-memory-recurrent.cpp:102-103)
static bool spacemit_is_cache_name(const char * name) {
    for (const char * prefix : { "cache_k_", "cache_v_", "cache_r_", "cache_s_" }) {
        if (std::strncmp(name, prefix, std::strlen(prefix)) == 0) {
            return true;
        }
    }
    return false;
}

spacemit_layout spacemit_weight_layout(const ggml_tensor * t) {
    if (t->view_src != nullptr || t->ne[1] % spacemit_ime::row_tile != 0 || t->ne[2] != 1 || t->ne[3] != 1 ||
        !ggml_is_contiguous(t) || spacemit_is_cache_name(t->name)) {
        return spacemit_layout::plain;
    }
    if (t->type == GGML_TYPE_Q4_0 && t->ne[0] % spacemit_ime::q4_0_k_block == 0) {
        return spacemit_layout::q4_0_32x256;
    }
    if (t->type == GGML_TYPE_Q4_1 && t->ne[0] % spacemit_ime::q8_0_k_block == 0) {
        return spacemit_layout::q4_1_q8_0_32x32;
    }
    if (t->type == GGML_TYPE_Q8_0 && t->ne[0] % spacemit_ime::q8_0_k_block == 0) {
        return spacemit_layout::q8_0_32x32;
    }
    if (t->type == GGML_TYPE_Q6_K && t->ne[0] % ggml_blck_size(GGML_TYPE_Q6_K) == 0) {  // always true for Q6_K
        return spacemit_layout::q6_k_q8_0_32x32;
    }
    return spacemit_layout::plain;
}

// the layouts requantized to Q8_0 at load: reads give the nearest weight of the tensor's type, writes must be whole
static bool spacemit_requantized(spacemit_layout layout) {
    return layout == spacemit_layout::q4_1_q8_0_32x32 || layout == spacemit_layout::q6_k_q8_0_32x32;
}

size_t spacemit_weight_alloc_size(const ggml_tensor * t) {
    if (spacemit_requantized(spacemit_weight_layout(t))) {
        return (size_t) ggml_nrows(t) * spacemit_ime::q8_0_weight_row_bytes((size_t) t->ne[0]);
    }
    return ggml_nbytes(t);
}

// the tensor that owns t's bytes, and t's byte offset inside it
static ggml_tensor * spacemit_storage(const ggml_tensor * t, size_t & offset) {
    if (t->view_src == nullptr) {
        offset = 0;
        return const_cast<ggml_tensor *>(t);
    }
    offset = static_cast<size_t>(static_cast<const char *>(t->data) - static_cast<const char *>(t->view_src->data));
    return t->view_src;
}

static void spacemit_pack(ggml_tensor * t, const void * data) {
    const size_t size = ggml_nbytes(t);
    int          rc   = -1;
    switch (spacemit_weight_layout(t)) {
        case spacemit_layout::q4_0_32x256:
            rc = spacemit_ime::repack_q4_0(t, data, size);
            break;
        case spacemit_layout::q4_1_q8_0_32x32:
            rc = spacemit_ime::repack_q4_1_q8_0(t, data, size);
            break;
        case spacemit_layout::q8_0_32x32:
            rc = spacemit_ime::repack_q8_0(t, data, size);
            break;
        case spacemit_layout::q6_k_q8_0_32x32:
            rc = spacemit_ime::repack_q6_k(t, data, size);
            break;
        case spacemit_layout::plain:
            break;
    }
    GGML_ASSERT(rc == 0 && "tensor does not fit its IME layout");
}

static void spacemit_unpack(const ggml_tensor * t, void * data) {
    switch (spacemit_weight_layout(t)) {
        case spacemit_layout::q4_0_32x256:
            spacemit_ime::unpack_q4_0(t, data);
            return;
        case spacemit_layout::q4_1_q8_0_32x32:
            spacemit_ime::unpack_q4_1_q8_0(t, data);
            return;
        case spacemit_layout::q8_0_32x32:
            spacemit_ime::unpack_q8_0(t, data);
            return;
        case spacemit_layout::q6_k_q8_0_32x32:
            spacemit_ime::unpack_q6_k(t, data);
            return;
        case spacemit_layout::plain:
            break;
    }
    GGML_ABORT("tensor has no IME layout");
}

// a partial write into a lossy layout goes through the read-back of the whole tensor. Requantized to Q8_0 (Q4_1, Q6_K):
// the read-back is approximate, so writing it back would requantize every block the write leaves alone; only a write
// of the whole tensor is allowed. llama.cpp writes weights whole on this device.
static void spacemit_check_partial(const ggml_tensor * base, size_t offset, size_t size) {
    const spacemit_layout layout = spacemit_weight_layout(base);
    GGML_ASSERT((!spacemit_requantized(layout) || (offset == 0 && size == ggml_nbytes(base))) &&
                "a partial write to a Q4_1 or Q6_K weight must cover the whole tensor");
}

spacemit_layout spacemit_storage_layout(const ggml_tensor * t) {
    size_t offset = 0;
    return spacemit_weight_layout(spacemit_storage(t, offset));
}

bool spacemit_tensor_is_repacked(const ggml_tensor * t) {
    return spacemit_storage_layout(t) != spacemit_layout::plain;
}

void spacemit_tensor_write(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    size_t        base_offset = 0;
    ggml_tensor * base        = spacemit_storage(t, base_offset);
    if (spacemit_weight_layout(base) == spacemit_layout::plain) {
        std::memcpy(static_cast<char *>(t->data) + offset, data, size);
        return;
    }
    offset += base_offset;
    const size_t nbytes = ggml_nbytes(base);
    if (offset == 0 && size == nbytes) {
        spacemit_pack(base, data);
        return;
    }
    spacemit_check_partial(base, offset, size);
    std::vector<uint8_t> whole(nbytes);
    spacemit_unpack(base, whole.data());
    std::memcpy(whole.data() + offset, data, size);
    spacemit_pack(base, whole.data());
}

void spacemit_tensor_read(const ggml_tensor * t, void * data, size_t offset, size_t size) {
    size_t              base_offset = 0;
    const ggml_tensor * base        = spacemit_storage(t, base_offset);
    if (spacemit_weight_layout(base) == spacemit_layout::plain) {
        std::memcpy(data, static_cast<const char *>(t->data) + offset, size);
        return;
    }
    offset += base_offset;
    const size_t nbytes = ggml_nbytes(base);
    if (offset == 0 && size == nbytes) {
        spacemit_unpack(base, data);
        return;
    }
    std::vector<uint8_t> whole(nbytes);
    spacemit_unpack(base, whole.data());
    std::memcpy(data, whole.data() + offset, size);
}

void spacemit_tensor_fill(ggml_tensor * t, uint8_t value, size_t offset, size_t size) {
    size_t        base_offset = 0;
    ggml_tensor * base        = spacemit_storage(t, base_offset);
    if (spacemit_weight_layout(base) == spacemit_layout::plain) {
        std::memset(static_cast<char *>(t->data) + offset, value, size);
        return;
    }
    spacemit_check_partial(base, base_offset + offset, size);
    std::vector<uint8_t> whole(ggml_nbytes(base));
    spacemit_unpack(base, whole.data());
    std::memset(whole.data() + base_offset + offset, value, size);
    spacemit_pack(base, whole.data());
}
