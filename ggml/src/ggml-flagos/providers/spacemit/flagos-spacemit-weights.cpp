#include "flagos-spacemit-weights.h"

#include "flagos-spacemit-ime-kernels.h"

#include "../../../ggml-impl.h"

#include <cstring>
#include <vector>

spacemit_layout spacemit_weight_layout(const ggml_tensor * t) {
    if (t->view_src != nullptr || t->ne[1] % spacemit_ime::row_tile != 0 || t->ne[2] != 1 || t->ne[3] != 1 ||
        !ggml_is_contiguous(t)) {
        return spacemit_layout::plain;
    }
    if (t->type == GGML_TYPE_Q4_0 && t->ne[0] % spacemit_ime::q4_0_k_block == 0) {
        return spacemit_layout::q4_0_32x256;
    }
    if (t->type == GGML_TYPE_Q4_1 && t->ne[0] % spacemit_ime::q4_1_k_block == 0) {
        return spacemit_layout::q4_1_32x32;
    }
    if (t->type == GGML_TYPE_Q8_0 && t->ne[0] % spacemit_ime::q8_0_k_block == 0) {
        return spacemit_layout::q8_0_32x32;
    }
    if (t->type == GGML_TYPE_Q6_K && t->ne[0] % ggml_blck_size(GGML_TYPE_Q6_K) == 0) {  // always true for Q6_K
        return spacemit_layout::q6_k_q8_0_32x32;
    }
    return spacemit_layout::plain;
}

size_t spacemit_weight_alloc_size(const ggml_tensor * t) {
    if (spacemit_weight_layout(t) == spacemit_layout::q6_k_q8_0_32x32) {
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
        case spacemit_layout::q4_1_32x32:
            rc = spacemit_ime::repack_q4_1(t, data, size);
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
        case spacemit_layout::q4_1_32x32:
            spacemit_ime::unpack_q4_1(t, data);
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

// a partial write into a lossy layout goes through the read-back of the whole tensor. Q4_1: it re-converts the blocks
// it touches from what the layout kept, so a write that ends inside a block would change a value the next write
// completes (a block's minimum split over two writes gets a different zero point than one whole write); it must cover
// whole blocks. Q6_K: the read-back is approximate, so writing it back would requantize every block the write leaves
// alone; only a write of the whole tensor is allowed. llama.cpp writes weights whole on this device.
static void spacemit_check_partial(const ggml_tensor * base, size_t offset, size_t size) {
    const spacemit_layout layout = spacemit_weight_layout(base);
    const size_t          blk    = ggml_type_size(base->type);
    GGML_ASSERT((layout != spacemit_layout::q4_1_32x32 || (offset % blk == 0 && size % blk == 0)) &&
                "a partial write to a Q4_1 weight must cover whole blocks");
    GGML_ASSERT((layout != spacemit_layout::q6_k_q8_0_32x32 || (offset == 0 && size == ggml_nbytes(base))) &&
                "a partial write to a Q6_K weight must cover the whole tensor");
}

bool spacemit_tensor_is_repacked(const ggml_tensor * t) {
    size_t offset = 0;
    return spacemit_weight_layout(spacemit_storage(t, offset)) != spacemit_layout::plain;
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
