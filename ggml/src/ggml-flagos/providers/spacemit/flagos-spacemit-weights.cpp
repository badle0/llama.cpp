#include "flagos-spacemit-weights.h"

#include "flagos-spacemit-ime-kernels.h"

#include "../../../ggml-impl.h"

#include <cstring>
#include <vector>

spacemit_layout spacemit_weight_layout(const ggml_tensor * t) {
    if (t->view_src == nullptr && t->type == GGML_TYPE_Q4_0 && t->ne[0] % spacemit_ime::q4_0_k_block == 0 &&
        t->ne[1] % spacemit_ime::q4_0_row_tile == 0 && t->ne[2] == 1 && t->ne[3] == 1 && ggml_is_contiguous(t)) {
        return spacemit_layout::q4_0_32x256;
    }
    return spacemit_layout::plain;
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
    const int rc = spacemit_ime::repack_q4_0(t, data, ggml_nbytes(t));
    GGML_ASSERT(rc == 0 && "tensor does not fit the q4_0 32x256 layout");
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
    std::vector<uint8_t> whole(nbytes);
    spacemit_ime::unpack_q4_0(base, whole.data());
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
        spacemit_ime::unpack_q4_0(base, data);
        return;
    }
    std::vector<uint8_t> whole(nbytes);
    spacemit_ime::unpack_q4_0(base, whole.data());
    std::memcpy(data, whole.data() + offset, size);
}

void spacemit_tensor_fill(ggml_tensor * t, uint8_t value, size_t offset, size_t size) {
    size_t        base_offset = 0;
    ggml_tensor * base        = spacemit_storage(t, base_offset);
    if (spacemit_weight_layout(base) == spacemit_layout::plain) {
        std::memset(static_cast<char *>(t->data) + offset, value, size);
        return;
    }
    std::vector<uint8_t> whole(ggml_nbytes(base));
    spacemit_ime::unpack_q4_0(base, whole.data());
    std::memset(whole.data() + base_offset + offset, value, size);
    spacemit_pack(base, whole.data());
}
