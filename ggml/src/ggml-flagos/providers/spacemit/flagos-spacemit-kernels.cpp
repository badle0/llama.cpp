#include "flagos-spacemit-kernels.h"

#include <algorithm>

#if defined(__riscv_v_intrinsic)
#    include <riscv_vector.h>
#endif

// [begin, end) of n elements for one tile, split in whole 64-byte lines so no two tiles write the same line
static void spacemit_tile_range(int64_t n, int64_t per_line, const spacemit_tile & tile, int64_t & begin, int64_t & end) {
    const int64_t lines = (n + per_line - 1) / per_line;
    begin               = std::min(n, lines * tile.ith / tile.nth * per_line);
    end                 = std::min(n, lines * (tile.ith + 1) / tile.nth * per_line);
}

bool spacemit_kernel_add_f32(const spacemit_tile & tile, ggml_tensor * dst) {
    const float * a = static_cast<const float *>(dst->src[0]->data);
    const float * b = static_cast<const float *>(dst->src[1]->data);
    float *       c = static_cast<float *>(dst->data);

    int64_t begin = 0;
    int64_t end   = 0;
    spacemit_tile_range(ggml_nelements(dst), 16, tile, begin, end);

#if defined(__riscv_v_intrinsic)
    for (int64_t i = begin; i < end;) {
        const size_t       vl = __riscv_vsetvl_e32m8(end - i);
        const vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
        const vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
        __riscv_vse32_v_f32m8(c + i, __riscv_vfadd_vv_f32m8(va, vb, vl), vl);
        i += vl;
    }
#else
    for (int64_t i = begin; i < end; i++) {
        c[i] = a[i] + b[i];
    }
#endif
    return true;
}
