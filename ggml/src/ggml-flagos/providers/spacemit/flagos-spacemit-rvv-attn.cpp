// Flash attention for VLEN 1024 (plan.md, M2e design), K3 builds only.
//
// Copied from the in-tree IME path, ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp @ 81b0d882a (called from
// ime.cpp:1151-1227; ggml-spacemit 4e782bc has the same decode code, its tiled kernel also trims masked tails):
// helpers align_up (33-35), flash_attn_ext_supported_d/shape_vlen1024_vf16 (37-43), reduce_sum_f32m4/m2_vlen1024
// (45-55), rvv_expf_approx_f32m2 (57-88), rvv_tanh_approx_f32m2 (90-100), rvv_softcap_tanh_inplace_f32 (102-116),
// rvv_softmax_exp_inplace_f32 (118-131), rvv_add_max_inplace_f32 (133-151), rvv_softcap_add_max_inplace_f32
// (153-177), rvv_zero_f32, rvv_scale_f32, rvv_add_inplace_f32, rvv_max_f32 (179-235), rvv_pack_f32_as_scaled_f16,
// rvv_pack_scaled_f16_as_f32, rvv_pack_scaled_f32_as_f32 (237-306), rvv_transposed_s16_mn_to_nm (390-470),
// rvv_qk_dot_tile_f16_x1/x4, rvv_pv_accumulate_f16_x1/x4, rvv_qk_dot_tile, rvv_pv_accumulate (472-620), memcpy2d
// (1115-1119); kernels flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow (650-911) and _m1 (913-1009),
// forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16 (1121-1303) as forward_one_chunk and
// forward_flash_attn_ext_f16_tiled_vlen1024_vf16 (1305-1628) as forward_tiled.
// Changes, each marked "flagos":
// - the interface: no ggml_compute_params; forward_tiled works in the caller's scratch (tiled_scratch_bytes) instead
//   of params->wdata or a TCM it picks itself; ggml-cpu's tile sizes as a local ggml_fa_tile_config; memcpy2d copies
//   with spacemit_ime::copy, M2b's copy of the same memcpy1d (1013-1113), kept behind a call;
// - (a) the 2- and 4-row kernel starts every row's maximum at -inf, not only the first row's;
// - (b) the dispatcher handles a missing mask (it read mask->ne without a null check);
// - (c) rows are grouped only without ALiBi (the grouped kernels apply the first row's slope to every row);
// - (d) the tiled kernel's softcap leaves the -inf padding of a short last KV tile alone (tanh made it -softcap, a
//   nonzero weight when there is no mask; ggml-cpu's tiled kernel has the same fault, ops.cpp:8894-8906);
// - (e) rvv_transposed_s16_mn_to_nm's asm declares the vector registers and memory it writes and stays behind a call;
// - (f) it advances its destination by the vl each strip gets, not by VLMAX (DK 65-127 could transpose wrongly).
// Test-only, for comparing the two (plan.md M2e.3, L2): ggml-spacemit's tiled kernel as forward_tiled_trim and its
// rvv_qk_dot_tile as rvv_qk_dot_tile_stride (spacemit-com/llama.cpp 4e782bc, ggml/src/ggml-spacemit/rvv_kernels.cpp:
// 1306-1641 and 582-598), changed only in the interface, as forward_tiled.
// flagos-spacemit-attn.cpp decides when these apply.

#include "flagos-spacemit-rvv-attn.h"

#if defined(GGML_FLAGOS_SPACEMIT_RVV)

#    include "flagos-spacemit-ime-kernels.h"

#    include "../../../ggml-impl.h"

#    include <riscv_vector.h>

#    include <algorithm>
#    include <array>
#    include <cmath>
#    include <cstdint>

#    if defined(__GNUC__)
#        pragma GCC diagnostic ignored "-Wcast-qual"
#    endif

namespace spacemit_rvv_attn {

namespace {

// flagos: ggml-cpu's flash-attention tile sizes (ggml/src/ggml-cpu/common.h:9-10 and 90-93, 64 x 64), not visible to
// the provider
struct ggml_fa_tile_config {
    static constexpr size_t Q  = 64;
    static constexpr size_t KV = 64;
};

auto align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static inline bool flash_attn_ext_supported_d_vlen1024_vf16(int64_t d) {
    return d > 0 && d <= 128;
}

static inline bool flash_attn_ext_supported_shape_vlen1024_vf16(int64_t DK, int64_t DV) {
    return flash_attn_ext_supported_d_vlen1024_vf16(DK) && flash_attn_ext_supported_d_vlen1024_vf16(DV);
}

static inline float reduce_sum_f32m4_vlen1024(vfloat32m4_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m4_f32m1(v, s_v, vl);
    return __riscv_vfmv_f_s_f32m1_f32(s_v);
}

static inline float reduce_sum_f32m2_vlen1024(vfloat32m2_t v, size_t vl) {
    vfloat32m1_t s_v = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    s_v              = __riscv_vfredusum_vs_f32m2_f32m1(v, s_v, vl);
    return __riscv_vfmv_f_s_f32m1_f32(s_v);
}

// Adapted from ggml_v_expf_m2 in vec.h. This is accurate enough for softmax.
static inline vfloat32m2_t rvv_expf_approx_f32m2(vfloat32m2_t x, size_t vl) {
    const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
    const vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
    const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    const vfloat32m2_t b =
        __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
    const vuint32m2_t  e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    const vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
    const vbool16_t    c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
    const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    const vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
        __riscv_vfmacc_vv_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl), u, vl),
        u, vl);

    if (!__riscv_vcpop_m_b16(c, vl)) {
        return __riscv_vfmacc_vv_f32m2(k, j, k, vl);
    }

    const vbool16_t    dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    const vuint32m2_t  d  = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
    const vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
    const vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
    const vfloat32m2_t r1 =
        __riscv_vmerge_vvm_f32m2(__riscv_vfmacc_vv_f32m2(k, k, j, vl),
                                 __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl), c, vl);
    return __riscv_vmerge_vvm_f32m2(r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
                                    __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl), vl);
}

static inline vfloat32m2_t rvv_tanh_approx_f32m2(vfloat32m2_t x, size_t vl) {
    const vfloat32m2_t abs_x       = __riscv_vfabs_v_f32m2(x, vl);
    const vfloat32m2_t neg_2_abs   = __riscv_vfmul_vf_f32m2(abs_x, -2.0f, vl);
    const vfloat32m2_t exp_term    = rvv_expf_approx_f32m2(neg_2_abs, vl);
    const vfloat32m2_t numerator   = __riscv_vfsub_vf_f32m2(exp_term, 1.0f, vl);
    const vfloat32m2_t denominator = __riscv_vfadd_vf_f32m2(exp_term, 1.0f, vl);
    const vfloat32m2_t tanh_abs    = __riscv_vfneg_v_f32m2(__riscv_vfdiv_vv_f32m2(numerator, denominator, vl), vl);
    const vbool16_t    neg_mask    = __riscv_vmflt_vf_f32m2_b16(x, 0.0f, vl);
    const vfloat32m2_t tanh_neg    = __riscv_vfneg_v_f32m2(tanh_abs, vl);
    return __riscv_vmerge_vvm_f32m2(tanh_abs, tanh_neg, neg_mask, vl);
}

static void rvv_softcap_tanh_inplace_f32(float * dst, int64_t dst_stride, int64_t tile_rows, int64_t n, float softcap) {
    for (int tq = 0; tq < tile_rows; ++tq, dst += dst_stride) {
        float * dst_row   = dst;
        int64_t remaining = n;
        while (remaining > 0) {
            const size_t vl = __riscv_vsetvl_e32m2(remaining);
            vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst_row, vl);
            v               = rvv_tanh_approx_f32m2(v, vl);
            v               = __riscv_vfmul_vf_f32m2(v, softcap, vl);
            __riscv_vse32_v_f32m2(dst_row, v, vl);
            dst_row += vl;
            remaining -= vl;
        }
    }
}

static inline float rvv_softmax_exp_inplace_f32(float * dst, int64_t n, float max_value) {
    float row_sum = 0.0f;
    while (n > 0) {
        const size_t vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v  = __riscv_vle32_v_f32m2(dst, vl);
        v               = __riscv_vfsub_vf_f32m2(v, max_value, vl);
        v               = rvv_expf_approx_f32m2(v, vl);
        __riscv_vse32_v_f32m2(dst, v, vl);
        row_sum += reduce_sum_f32m2_vlen1024(v, vl);
        dst += vl;
        n -= vl;
    }
    return row_sum;
}

static inline float rvv_add_max_inplace_f32(float * dst, const float * src, int64_t n) {
    float max_val = -INFINITY;
    while (n > 0) {
        const size_t vl   = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vdst = __riscv_vle32_v_f32m4(dst, vl);
        vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(src, vl);
        vdst              = __riscv_vfadd_vv_f32m4(vdst, vsrc, vl);
        __riscv_vse32_v_f32m4(dst, vdst, vl);

        vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(max_val, 1);
        seed              = __riscv_vfredmax_vs_f32m4_f32m1(vdst, seed, vl);
        max_val           = __riscv_vfmv_f_s_f32m1_f32(seed);

        dst += vl;
        src += vl;
        n -= vl;
    }
    return max_val;
}

static inline float rvv_softcap_add_max_inplace_f32(float * dst, const float * src, int64_t n, float softcap) {
    if (softcap == 0.0f) {
        return rvv_add_max_inplace_f32(dst, src, n);
    }

    float max_val = -INFINITY;
    while (n > 0) {
        const size_t vl   = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t vdst = __riscv_vle32_v_f32m2(dst, vl);
        vfloat32m2_t vsrc = __riscv_vle32_v_f32m2(src, vl);
        vdst              = rvv_tanh_approx_f32m2(vdst, vl);
        vdst              = __riscv_vfmul_vf_f32m2(vdst, softcap, vl);
        vdst              = __riscv_vfadd_vv_f32m2(vdst, vsrc, vl);
        __riscv_vse32_v_f32m2(dst, vdst, vl);

        vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(max_val, 1);
        seed              = __riscv_vfredmax_vs_f32m2_f32m1(vdst, seed, vl);
        max_val           = __riscv_vfmv_f_s_f32m1_f32(seed);

        dst += vl;
        src += vl;
        n -= vl;
    }
    return max_val;
}

static inline void rvv_zero_f32(float * dst, int64_t n) {
    while (n > 0) {
        const size_t       vl = __riscv_vsetvl_e32m4(n);
        const vfloat32m4_t z  = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        __riscv_vse32_v_f32m4(dst, z, vl);
        dst += vl;
        n -= vl;
    }
}

static inline void rvv_scale_f32(float * dst, float scale, int64_t n) {
    while (n > 0) {
        const size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t v  = __riscv_vle32_v_f32m4(dst, vl);
        v               = __riscv_vfmul_vf_f32m4(v, scale, vl);
        __riscv_vse32_v_f32m4(dst, v, vl);
        dst += vl;
        n -= vl;
    }
}

static inline void rvv_add_inplace_f32(float *       dst,
                                       int64_t       dst_stride,
                                       const float * src,
                                       int64_t       src_stride,
                                       int64_t       tile_rows,
                                       int64_t       n) {
    for (int tq = 0; tq < tile_rows; ++tq, dst += dst_stride, src += src_stride) {
        int64_t       remaining = n;
        float *       dst_row   = dst;
        const float * src_row   = src;
        while (remaining > 0) {
            const size_t vl   = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t vdst = __riscv_vle32_v_f32m4(dst_row, vl);
            vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(src_row, vl);
            vdst              = __riscv_vfadd_vv_f32m4(vdst, vsrc, vl);
            __riscv_vse32_v_f32m4(dst_row, vdst, vl);
            dst_row += vl;
            src_row += vl;
            remaining -= vl;
        }
    }
}

static inline float rvv_max_f32(const float * src, int64_t n) {
    float max_val = -INFINITY;
    while (n > 0) {
        const size_t       vl   = __riscv_vsetvl_e32m4(n);
        const vfloat32m4_t v    = __riscv_vle32_v_f32m4(src, vl);
        vfloat32m1_t       seed = __riscv_vfmv_v_f_f32m1(max_val, 1);
        seed                    = __riscv_vfredmax_vs_f32m4_f32m1(v, seed, vl);
        max_val                 = __riscv_vfmv_f_s_f32m1_f32(seed);
        src += vl;
        n -= vl;
    }
    return max_val;
}

static void rvv_pack_f32_as_scaled_f16(void *       dst,
                                       int64_t      dst_row_stride,
                                       const void * src,
                                       int64_t      src_row_stride,
                                       int64_t      tile_rows,
                                       int64_t      n,
                                       float        scale) {
    for (int tq = 0; tq < tile_rows; ++tq) {
        const float * row_ptr     = (const float *) ((const char *) src + tq * src_row_stride);
        _Float16 *    dst_row_ptr = (_Float16 *) ((char *) dst + tq * dst_row_stride);
        int64_t       remaining   = n;
        while (remaining > 0) {
            const size_t vl        = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t v32       = __riscv_vle32_v_f32m4(row_ptr, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            const vfloat16m2_t v16 = __riscv_vfncvt_f_f_w_f16m2(v32, vl);
            __riscv_vse16_v_f16m2(dst_row_ptr, v16, vl);
            dst_row_ptr += vl;
            row_ptr += vl;
            remaining -= vl;
        }
    }
}

static void rvv_pack_scaled_f16_as_f32(void *       dst,
                                       int64_t      dst_row_stride,
                                       const void * src,
                                       int64_t      src_row_stride,
                                       int64_t      tile_rows,
                                       int64_t      n,
                                       float        scale) {
    for (int tq = 0; tq < tile_rows; ++tq) {
        const _Float16 * row_ptr     = (const _Float16 *) ((const char *) src + tq * src_row_stride);
        float *          dst_row_ptr = (float *) ((char *) dst + tq * dst_row_stride);
        int64_t          remaining   = n;
        while (remaining > 0) {
            const size_t       vl  = __riscv_vsetvl_e16m2(remaining);
            const vfloat16m2_t v16 = __riscv_vle16_v_f16m2(row_ptr, vl);
            vfloat32m4_t       v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            v32                    = __riscv_vfmul_vf_f32m4(v32, scale, vl);
            __riscv_vse32_v_f32m4(dst_row_ptr, v32, vl);
            dst_row_ptr += vl;
            row_ptr += vl;
            remaining -= vl;
        }
    }
}

static void rvv_pack_scaled_f32_as_f32(void *       dst,
                                       int64_t      dst_row_stride,
                                       const void * src,
                                       int64_t      src_row_stride,
                                       int64_t      tile_rows,
                                       int64_t      n,
                                       float *      scale) {
    for (int tq = 0; tq < tile_rows; ++tq) {
        const float * row_ptr     = (const float *) ((const char *) src + tq * src_row_stride);
        float *       dst_row_ptr = (float *) ((char *) dst + tq * dst_row_stride);
        int64_t       remaining   = n;
        while (remaining > 0) {
            const size_t vl  = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t v32 = __riscv_vle32_v_f32m4(row_ptr, vl);
            v32              = __riscv_vfmul_vf_f32m4(v32, scale[tq], vl);
            __riscv_vse32_v_f32m4(dst_row_ptr, v32, vl);
            dst_row_ptr += vl;
            row_ptr += vl;
            remaining -= vl;
        }
    }
}

// flagos: (e) the asm overwrote v0-v7 and wrote memory without declaring either, so inlined into the tiled kernel it
// could destroy live vector values or be reordered against the tile's loads and stores; it also changes vl and vtype,
// which only a call boundary resets for the compiler (as M2b's copy() for memcpy1d).
// flagos: (f) the destination advanced by VLMAX rows per strip of n, computed once, whatever vl the strip got; for n
// between VLMAX and 2 VLMAX (DK 65-127) RVV 1.0 lets the first strip take any vl from ceil(n / 2) to VLMAX, and a
// shorter one left rows of the transposed tile at 0 and wrote the rest past them
static __attribute__((noinline)) void rvv_transposed_s16_mn_to_nm(int8_t * dst,
                                                                  int64_t  n_dst_stride,
                                                                  int8_t * src,
                                                                  int64_t  m_src_stride,
                                                                  int64_t  m,
                                                                  int64_t  n) {
    int8_t * in  = src;
    int8_t * out = dst;

    __asm__ volatile(
        "srli                   t2, %[isz0], 3                \n\t"
        "blez                   t2, M1%=                      \n\t"

        "LOOP_M8%=:                                           \n\t"
        "addi                   a1, %[dst], 0                 \n\t"
        "addi                   s1, %[src], 0                 \n\t"
        "add                    s2, %[src], %[is0]            \n\t"
        "add                    s3, s2, %[is0]                \n\t"
        "add                    s4, s3, %[is0]                \n\t"
        "add                    s5, s4, %[is0]                \n\t"
        "add                    s6, s5, %[is0]                \n\t"
        "add                    s7, s6, %[is0]                \n\t"
        "add                    s8, s7, %[is0]                \n\t"
        "addi                   t1, %[isz1], 0                \n\t"

        "LOOP_M8N%=:                                          \n\t"
        "vsetvli                t0, t1, e16, m1, tu, mu       \n\t"
        "mul                    t3, t0, %[os0]                \n\t"  // flagos: (f)
        "sub                    t1, t1, t0                    \n\t"
        "vle16.v                v0, (s1)                      \n\t"
        "sh1add                 s1, t0, s1                    \n\t"
        "vle16.v                v1, (s2)                      \n\t"
        "sh1add                 s2, t0, s2                    \n\t"
        "vle16.v                v2, (s3)                      \n\t"
        "sh1add                 s3, t0, s3                    \n\t"
        "vle16.v                v3, (s4)                      \n\t"
        "sh1add                 s4, t0, s4                    \n\t"
        "vle16.v                v4, (s5)                      \n\t"
        "sh1add                 s5, t0, s5                    \n\t"
        "vle16.v                v5, (s6)                      \n\t"
        "sh1add                 s6, t0, s6                    \n\t"
        "vle16.v                v6, (s7)                      \n\t"
        "sh1add                 s7, t0, s7                    \n\t"
        "vle16.v                v7, (s8)                      \n\t"
        "sh1add                 s8, t0, s8                    \n\t"
        "vssseg8e16.v           v0, (a1), %[os0]              \n\t"
        "add                    a1, a1, t3                    \n\t"
        "bnez                   t1, LOOP_M8N%=                \n\t"
        "sh3add                 %[src], %[is0], %[src]        \n\t"
        "addi                   %[dst], %[dst], 16            \n\t"
        "addi                   t2, t2, -1                    \n\t"
        "bnez                   t2, LOOP_M8%=                 \n\t"

        "M1%=:                                                \n\t"
        "andi                   t2, %[isz0], 7                \n\t"
        "blez                   t2, END%=                     \n\t"

        "LOOP_M1%=:                                           \n\t"
        "addi                   a1, %[dst], 0                 \n\t"
        "addi                   s1, %[src], 0                 \n\t"
        "addi                   t1, %[isz1], 0                \n\t"

        "LOOP_M1N%=:                                          \n\t"
        "vsetvli                t0, t1, e16, m1, tu, mu       \n\t"
        "mul                    t3, t0, %[os0]                \n\t"  // flagos: (f)
        "sub                    t1, t1, t0                    \n\t"
        "vle16.v                v0, (s1)                      \n\t"
        "sh1add                 s1, t0, s1                    \n\t"
        "vsse16.v               v0, (a1), %[os0]              \n\t"
        "add                    a1, a1, t3                    \n\t"
        "bnez                   t1, LOOP_M1N%=                \n\t"
        "add                    %[src], %[is0], %[src]        \n\t"
        "addi                   %[dst], %[dst], 2             \n\t"
        "addi                   t2, t2, -1                    \n\t"
        "bnez                   t2, LOOP_M1%=                 \n\t"
        "END%=:                                               \n\t"

        : [src] "+r"(in), [dst] "+r"(out), [isz0] "+r"(m)
        : [isz1] "r"(n), [is0] "r"(m_src_stride), [os0] "r"(n_dst_stride)
        : "cc", "t0", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "a1",
          // flagos: (e)
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory");
}

static inline void rvv_qk_dot_tile_f16_x1(float *          dst,
                                          const _Float16 * q_row,
                                          const _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    const size_t vl  = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    for (int64_t d = 0; d < dk; ++d) {
        const vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
        acc                      = __riscv_vfwmacc_vf_f32m2(acc, q_row[d], k_vec, vl);
    }

    __riscv_vse32_v_f32m2(dst, acc, vl);
}

static inline void rvv_qk_dot_tile_f16_x4(float *          dst0,
                                          float *          dst1,
                                          float *          dst2,
                                          float *          dst3,
                                          const _Float16 * q0,
                                          const _Float16 * q1,
                                          const _Float16 * q2,
                                          const _Float16 * q3,
                                          const _Float16 * k_pack,
                                          int64_t          dk,
                                          int64_t          kv_tile) {
    const size_t vl   = __riscv_vsetvl_e16m1(kv_tile);
    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    vfloat32m2_t acc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

    for (int64_t d = 0; d < dk; ++d) {
        const vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_pack + d * ggml_fa_tile_config::KV, vl);
        acc0                     = __riscv_vfwmacc_vf_f32m2(acc0, q0[d], k_vec, vl);
        acc1                     = __riscv_vfwmacc_vf_f32m2(acc1, q1[d], k_vec, vl);
        acc2                     = __riscv_vfwmacc_vf_f32m2(acc2, q2[d], k_vec, vl);
        acc3                     = __riscv_vfwmacc_vf_f32m2(acc3, q3[d], k_vec, vl);
    }

    __riscv_vse32_v_f32m2(dst0, acc0, vl);
    __riscv_vse32_v_f32m2(dst1, acc1, vl);
    __riscv_vse32_v_f32m2(dst2, acc2, vl);
    __riscv_vse32_v_f32m2(dst3, acc3, vl);
}

static inline void rvv_pv_accumulate_f16_x1(float *          dst,
                                            const float *    prob,
                                            const _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_left = dv;
    int64_t d_off  = 0;

    while (d_left > 0) {
        const size_t vl  = __riscv_vsetvl_e16m2(d_left);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        for (int64_t tk = 0; tk < kv_tile; ++tk) {
            const vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            const vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc                    = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_left -= vl;
        d_off += vl;
    }
}

static inline void rvv_pv_accumulate_f16_x4(float *          dst0,
                                            float *          dst1,
                                            float *          dst2,
                                            float *          dst3,
                                            const float *    prob0,
                                            const float *    prob1,
                                            const float *    prob2,
                                            const float *    prob3,
                                            const _Float16 * v_pack,
                                            int64_t          kv_tile,
                                            int64_t          dv) {
    int64_t d_left = dv;
    int64_t d_off  = 0;

    while (d_left > 0) {
        const size_t vl   = __riscv_vsetvl_e16m2(d_left);
        vfloat32m4_t acc0 = __riscv_vle32_v_f32m4(dst0 + d_off, vl);
        vfloat32m4_t acc1 = __riscv_vle32_v_f32m4(dst1 + d_off, vl);
        vfloat32m4_t acc2 = __riscv_vle32_v_f32m4(dst2 + d_off, vl);
        vfloat32m4_t acc3 = __riscv_vle32_v_f32m4(dst3 + d_off, vl);

        for (int64_t tk = 0; tk < kv_tile; ++tk) {
            const vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_pack + tk * dv + d_off, vl);
            const vfloat32m4_t v32 = __riscv_vfwcvt_f_f_v_f32m4(v16, vl);
            acc0                   = __riscv_vfmacc_vf_f32m4(acc0, prob0[tk], v32, vl);
            acc1                   = __riscv_vfmacc_vf_f32m4(acc1, prob1[tk], v32, vl);
            acc2                   = __riscv_vfmacc_vf_f32m4(acc2, prob2[tk], v32, vl);
            acc3                   = __riscv_vfmacc_vf_f32m4(acc3, prob3[tk], v32, vl);
        }

        __riscv_vse32_v_f32m4(dst0 + d_off, acc0, vl);
        __riscv_vse32_v_f32m4(dst1 + d_off, acc1, vl);
        __riscv_vse32_v_f32m4(dst2 + d_off, acc2, vl);
        __riscv_vse32_v_f32m4(dst3 + d_off, acc3, vl);
        d_left -= vl;
        d_off += vl;
    }
}

static inline void rvv_qk_dot_tile(float *       dst,
                                   const float * q_row,
                                   const float * k_pack,
                                   int64_t       dk,
                                   int64_t       kv_tile,
                                   float         scale) {
    const size_t vl  = __riscv_vsetvl_e32m4(kv_tile);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    for (int64_t d = 0; d < dk; ++d) {
        const vfloat32m4_t k_vec = __riscv_vle32_v_f32m4(k_pack + d * kv_tile, vl);
        acc                      = __riscv_vfmacc_vf_f32m4(acc, q_row[d] * scale, k_vec, vl);
    }

    __riscv_vse32_v_f32m4(dst, acc, vl);
}

// flagos: ggml-spacemit's rvv_qk_dot_tile (4e782bc, rvv_kernels.cpp:582-598), renamed: it takes the K tile's row
// stride apart from the number of positions; only forward_tiled_trim's F32 path calls it
static inline void rvv_qk_dot_tile_stride(float *       dst,
                                          const float * q_row,
                                          const float * k_pack,
                                          int64_t       dk,
                                          int64_t       kv_stride,
                                          int64_t       kv_tile,
                                          float         scale) {
    const size_t vl  = __riscv_vsetvl_e32m4(kv_tile);
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    for (int64_t d = 0; d < dk; ++d) {
        const vfloat32m4_t k_vec = __riscv_vle32_v_f32m4(k_pack + d * kv_stride, vl);
        acc                      = __riscv_vfmacc_vf_f32m4(acc, q_row[d] * scale, k_vec, vl);
    }

    __riscv_vse32_v_f32m4(dst, acc, vl);
}

static inline void rvv_pv_accumulate(float *       dst,
                                     const float * prob,
                                     const float * v_pack,
                                     int64_t       kv_tile,
                                     int64_t       dv) {
    int64_t d_left = dv;
    int64_t d_off  = 0;

    while (d_left > 0) {
        const size_t vl  = __riscv_vsetvl_e32m4(d_left);
        vfloat32m4_t acc = __riscv_vle32_v_f32m4(dst + d_off, vl);

        for (int64_t tk = 0; tk < kv_tile; ++tk) {
            const vfloat32m4_t v_vec = __riscv_vle32_v_f32m4(v_pack + tk * dv + d_off, vl);
            acc                      = __riscv_vfmacc_vf_f32m4(acc, prob[tk], v_vec, vl);
        }

        __riscv_vse32_v_f32m4(dst + d_off, acc, vl);
        d_left -= vl;
        d_off += vl;
    }
}

template <size_t QLEN>
static void flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow(float **            pq,
                                                                  const char *        k_data_row,
                                                                  const char *        v_data_row,
                                                                  const ggml_fp16_t * mp,
                                                                  float **            sinks,
                                                                  float **            dst,
                                                                  float               scale,
                                                                  float               logit_softcap,
                                                                  float               slope,
                                                                  int64_t             nek1,
                                                                  int64_t             nbk1,
                                                                  int64_t             nbv1,
                                                                  int64_t             DV,
                                                                  int64_t             DK,
                                                                  void *              tcm_buffer,
                                                                  size_t              tcm_buffer_size) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));
    float S[QLEN] = { 0.0f };  // sum
    float M[QLEN];             // maximum KQ value
    // flagos: (a) "= { -INFINITY }" set only M[0] and left the other rows at 0, so a row whose scores all lay far
    // below 0 was exponentiated against 0, its weights underflowed in the F16 accumulator and its output became 0
    std::fill_n(M, QLEN, -INFINITY);

    _Float16 *   kq16_buffer          = (_Float16 *) tcm_buffer;
    _Float16 *   qv_buffer            = kq16_buffer + QLEN * DV;
    const size_t qkv_temp_buffer_size = (QLEN * DV + QLEN * DK) * sizeof(_Float16);
    char *       kv_tile_buffer       = (char *) (qv_buffer + QLEN * DK);

    {
        vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);
        for (int64_t i = 0; i < QLEN; ++i) {
            __riscv_vse16_v_f16m2(kq16_buffer + i * DV, VKQ16_v, DV);
            vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq[i], DK), DK);
            __riscv_vse16_v_f16m2(qv_buffer + i * DK, Q_q_v, DK);
        }
    }

    const uintptr_t scratch_addr = reinterpret_cast<uintptr_t>(kv_tile_buffer);
    const size_t    scratch_size = tcm_buffer_size > qkv_temp_buffer_size ? tcm_buffer_size - qkv_temp_buffer_size : 0;
    const uintptr_t kq_tile_addr = align_up(scratch_addr, alignof(float));
    const size_t    scratch_prefix = kq_tile_addr - scratch_addr;
    const size_t    packed_tile_size =
        QLEN * sizeof(float) + DK * sizeof(_Float16) + DV * sizeof(_Float16) + sizeof(float);
    const int64_t max_ic_tile_step = ((int64_t) __riscv_vsetvlmax_e16m1()) & ~((int64_t) 7);
    const int64_t max_fit_by_tcm =
        scratch_size > scratch_prefix ? (int64_t) ((scratch_size - scratch_prefix) / packed_tile_size) : 0;
    const int64_t ic_tile_step = std::min(max_ic_tile_step, max_fit_by_tcm) & ~((int64_t) 7);

    const uintptr_t k_tile_addr  = kq_tile_addr + QLEN * ic_tile_step * sizeof(float);
    const uintptr_t v_tile_addr  = k_tile_addr + DK * ic_tile_step * sizeof(_Float16);
    const uintptr_t mv_tile_addr = v_tile_addr + ic_tile_step * DV * sizeof(_Float16);

    if (ic_tile_step >= 8) {
        float *    kq_tile_buffer = reinterpret_cast<float *>(kq_tile_addr);
        _Float16 * k_tile_pack    = reinterpret_cast<_Float16 *>(k_tile_addr);
        _Float16 * v_tile_pack    = reinterpret_cast<_Float16 *>(v_tile_addr);
        float *    mv_tile_pack   = reinterpret_cast<float *>(mv_tile_addr);

        const int64_t k_tile_byte_stride = ic_tile_step * (int64_t) sizeof(_Float16);

        int64_t ic_step = 0;
        for (int64_t ic = 0; ic < nek1; ++ic) {
            const float mv = mp ? slope * ((_Float16 *) mp)[ic] : 0.0f;

            if (mv != -INFINITY) {
                const _Float16 * k_data = (const _Float16 *) (k_data_row + ic * nbk1);
                const _Float16 * v_data = (const _Float16 *) (v_data_row + ic * nbv1);

                const vfloat16m2_t k_data_v = __riscv_vle16_v_f16m2(k_data, DK);
                const vfloat16m2_t v_data_v = __riscv_vle16_v_f16m2(v_data, DV);
                __riscv_vsse16_v_f16m2(k_tile_pack + ic_step, k_tile_byte_stride, k_data_v, DK);
                __riscv_vse16_v_f16m2(v_tile_pack + ic_step * DV, v_data_v, DV);
                mv_tile_pack[ic_step] = mv;
                ic_step++;
            }

            if (ic_step > 0 && (ic_step == ic_tile_step || ic == (nek1 - 1))) {
                if constexpr (QLEN == 4) {
                    const size_t qk_vl   = __riscv_vsetvl_e16m1(ic_step);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc2 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc3 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    for (int64_t d = 0; d < DK; ++d) {
                        const vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_step, qk_vl);
                        qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_buffer[0 * DK + d], k_vec, qk_vl);
                        qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_buffer[1 * DK + d], k_vec, qk_vl);
                        qk_acc2 = __riscv_vfwmacc_vf_f32m2(qk_acc2, qv_buffer[2 * DK + d], k_vec, qk_vl);
                        qk_acc3 = __riscv_vfwmacc_vf_f32m2(qk_acc3, qv_buffer[3 * DK + d], k_vec, qk_vl);
                    }

                    qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                    qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);
                    qk_acc2 = __riscv_vfmul_vf_f32m2(qk_acc2, scale, qk_vl);
                    qk_acc3 = __riscv_vfmul_vf_f32m2(qk_acc3, scale, qk_vl);

                    __riscv_vse32_v_f32m2(kq_tile_buffer + 0 * ic_tile_step, qk_acc0, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_buffer + 1 * ic_tile_step, qk_acc1, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_buffer + 2 * ic_tile_step, qk_acc2, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_buffer + 3 * ic_tile_step, qk_acc3, qk_vl);
                } else {
                    static_assert(QLEN == 2, "unsupported QLEN");

                    const size_t qk_vl   = __riscv_vsetvl_e16m1(ic_step);
                    vfloat32m2_t qk_acc0 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);
                    vfloat32m2_t qk_acc1 = __riscv_vfmv_v_f_f32m2(0.0f, qk_vl);

                    for (int64_t d = 0; d < DK; ++d) {
                        const vfloat16m1_t k_vec = __riscv_vle16_v_f16m1(k_tile_pack + d * ic_tile_step, qk_vl);
                        qk_acc0 = __riscv_vfwmacc_vf_f32m2(qk_acc0, qv_buffer[0 * DK + d], k_vec, qk_vl);
                        qk_acc1 = __riscv_vfwmacc_vf_f32m2(qk_acc1, qv_buffer[1 * DK + d], k_vec, qk_vl);
                    }

                    qk_acc0 = __riscv_vfmul_vf_f32m2(qk_acc0, scale, qk_vl);
                    qk_acc1 = __riscv_vfmul_vf_f32m2(qk_acc1, scale, qk_vl);

                    __riscv_vse32_v_f32m2(kq_tile_buffer + 0 * ic_tile_step, qk_acc0, qk_vl);
                    __riscv_vse32_v_f32m2(kq_tile_buffer + 1 * ic_tile_step, qk_acc1, qk_vl);
                }

                for (int i = 0; i < QLEN; ++i) {
                    float *     row_ptr = kq_tile_buffer + i * ic_tile_step;
                    const float tile_max =
                        rvv_softcap_add_max_inplace_f32(row_ptr, mv_tile_pack, ic_step, logit_softcap);

                    const float Mold = M[i];

                    if (tile_max > Mold) {
                        const float ms = expf(Mold - tile_max);
                        M[i]           = tile_max;
                        S[i] *= ms;

                        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_buffer + i * DV, DV);
                        VKQ16_v              = __riscv_vfmul_vf_f16m2(VKQ16_v, (_Float16) ms, DV);
                        __riscv_vse16_v_f16m2(kq16_buffer + i * DV, VKQ16_v, DV);
                    }

                    S[i] += rvv_softmax_exp_inplace_f32(row_ptr, ic_step, M[i]);
                }

                if constexpr (QLEN == 4) {
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_buffer + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_buffer + 1 * DV, DV);
                    vfloat16m2_t pv_acc2 = __riscv_vle16_v_f16m2(kq16_buffer + 2 * DV, DV);
                    vfloat16m2_t pv_acc3 = __riscv_vle16_v_f16m2(kq16_buffer + 3 * DV, DV);

                    for (int64_t tk = 0; tk < ic_step; ++tk) {
                        const vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
                        pv_acc0 =
                            __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16) kq_tile_buffer[0 * ic_tile_step + tk], v16, DV);
                        pv_acc1 =
                            __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16) kq_tile_buffer[1 * ic_tile_step + tk], v16, DV);
                        pv_acc2 =
                            __riscv_vfmacc_vf_f16m2(pv_acc2, (_Float16) kq_tile_buffer[2 * ic_tile_step + tk], v16, DV);
                        pv_acc3 =
                            __riscv_vfmacc_vf_f16m2(pv_acc3, (_Float16) kq_tile_buffer[3 * ic_tile_step + tk], v16, DV);
                    }

                    __riscv_vse16_v_f16m2(kq16_buffer + 0 * DV, pv_acc0, DV);
                    __riscv_vse16_v_f16m2(kq16_buffer + 1 * DV, pv_acc1, DV);
                    __riscv_vse16_v_f16m2(kq16_buffer + 2 * DV, pv_acc2, DV);
                    __riscv_vse16_v_f16m2(kq16_buffer + 3 * DV, pv_acc3, DV);
                } else {
                    static_assert(QLEN == 2, "unsupported QLEN");
                    vfloat16m2_t pv_acc0 = __riscv_vle16_v_f16m2(kq16_buffer + 0 * DV, DV);
                    vfloat16m2_t pv_acc1 = __riscv_vle16_v_f16m2(kq16_buffer + 1 * DV, DV);

                    for (int64_t tk = 0; tk < ic_step; ++tk) {
                        const vfloat16m2_t v16 = __riscv_vle16_v_f16m2(v_tile_pack + tk * DV, DV);
                        pv_acc0 =
                            __riscv_vfmacc_vf_f16m2(pv_acc0, (_Float16) kq_tile_buffer[0 * ic_tile_step + tk], v16, DV);
                        pv_acc1 =
                            __riscv_vfmacc_vf_f16m2(pv_acc1, (_Float16) kq_tile_buffer[1 * ic_tile_step + tk], v16, DV);
                    }

                    __riscv_vse16_v_f16m2(kq16_buffer + 0 * DV, pv_acc0, DV);
                    __riscv_vse16_v_f16m2(kq16_buffer + 1 * DV, pv_acc1, DV);
                }

                ic_step = 0;
            }
        }
    } else {
        for (int64_t ic = 0; ic < nek1; ++ic) {
            const float mv = mp ? slope * ((_Float16 *) mp)[ic] : 0.0f;

            const char * k_data = k_data_row + ic * nbk1;
            const char * v_data = v_data_row + ic * nbv1;

            vfloat16m2_t k_data_v;
            vfloat16m2_t v_data_v;

            if (mv != -INFINITY) {
                k_data_v = __riscv_vle16_v_f16m2((_Float16 *) k_data, DK);
                v_data_v = __riscv_vle16_v_f16m2((_Float16 *) v_data, DV);
            } else {
                continue;
            }

            for (int i = 0; i < QLEN; ++i) {
                vfloat16m2_t Q_q_v    = __riscv_vle16_v_f16m2(qv_buffer + i * DK, DK);
                vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_data_v, Q_q_v, DK);
                float        s        = reduce_sum_f32m4_vlen1024(qk_acc_v, DK);
                s                     = s * scale;
                if (logit_softcap != 0.0f) {
                    s = logit_softcap * tanhf(s);
                }
                s += mv;

                const float Mold = M[i];

                float ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
                float vs = 1.0f;  // post-softmax KQ value, expf(s - M)

                vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_buffer + i * DV, DV);
                if (s > M[i]) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M[i] = s;
                    ms   = expf(Mold - M[i]);

                    // V = V*expf(Mold - M)
                    VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M[i]);
                }
                VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_data_v, DV);
                __riscv_vse16_v_f16m2(kq16_buffer + i * DV, VKQ16_v, DV);
                S[i] = S[i] * ms + vs;  // scale and increment sum with partial sum
            }
        }
    }

    for (int i = 0; i < QLEN; ++i) {
        vfloat16m2_t VKQ16_v = __riscv_vle16_v_f16m2(kq16_buffer + i * DV, DV);
        vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

        // sinks
        if (sinks[i]) {
            const float s = *(sinks[i]);

            float ms = 1.0f;
            float vs = 1.0f;

            if (s > M[i]) {
                ms      = expf(M[i] - s);
                M[i]    = s;
                VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
            } else {
                vs = expf(s - M[i]);
            }

            S[i] = S[i] * ms + vs;
        }

        // V /= S
        const float S_inv = S[i] == 0.0f ? 0.0f : 1.0f / S[i];

        VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

        __riscv_vse32_v_f32m4(dst[i], VKQ32_v, DV);
    }
}

static void flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(const float *       pq,
                                                                const char *        k_data_row,
                                                                const char *        v_data_row,
                                                                const ggml_fp16_t * mp,
                                                                const float *       sinks,
                                                                float *             dst,
                                                                float               scale,
                                                                float               logit_softcap,
                                                                float               slope,
                                                                int64_t             nek1,
                                                                int64_t             nbk1,
                                                                int64_t             nbv1,
                                                                int64_t             DV,
                                                                int64_t             DK) {
    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    float S = 0.0f;       // sum
    float M = -INFINITY;  // maximum KQ value

    vfloat16m2_t VKQ16_v = __riscv_vfmv_v_f_f16m2(0.0f, DV);

    vfloat16m2_t Q_q_v = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq, DK), DK);

    for (int64_t ic = 0; ic < nek1; ++ic) {
        const float mv = mp ? slope * ((_Float16 *) mp)[ic] : 0.0f;
        if (mv == -INFINITY) {
            continue;
        }

        const char * k_data = k_data_row + ic * nbk1;

        vfloat16m2_t k_data_v = __riscv_vle16_v_f16m2((_Float16 *) k_data, DK);

        vfloat32m4_t qk_acc_v = __riscv_vfwmul_vv_f32m4(k_data_v, Q_q_v, DK);
        float        s        = reduce_sum_f32m4_vlen1024(qk_acc_v, DK);

        s = s * scale;  // scale KQ value

        if (logit_softcap != 0.0f) {
            s = logit_softcap * tanhf(s);
        }

        s += mv;  // apply mask

        const float Mold = M;

        float ms = 1.0f;  // upon new higher max val, scale VKQ and KQ sum with this value
        float vs = 1.0f;  // post-softmax KQ value, expf(s - M)

        const char * v_data = v_data_row + ic * nbv1;

        vfloat16m2_t v_data_v = __riscv_vle16_v_f16m2((_Float16 *) v_data, DV);

        if (s > M) {
            // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
            M  = s;
            ms = expf(Mold - M);

            // V = V*expf(Mold - M)
            VKQ16_v = __riscv_vfmul_vf_f16m2(VKQ16_v, ms, DV);
        } else {
            // no new maximum, ms == 1.0f, vs != 1.0f
            vs = expf(s - M);
        }

        VKQ16_v = __riscv_vfmacc_vf_f16m2(VKQ16_v, vs, v_data_v, DV);

        S = S * ms + vs;  // scale and increment sum with partial sum
    }

    vfloat32m4_t VKQ32_v = __riscv_vfwcvt_f_f_v_f32m4(VKQ16_v, DV);

    // sinks
    if (sinks) {
        const float s = *sinks;

        float ms = 1.0f;
        float vs = 1.0f;

        if (s > M) {
            ms      = expf(M - s);
            M       = s;
            VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, ms, DV);
        } else {
            vs = expf(s - M);
        }

        S = S * ms + vs;
    }

    // V /= S
    const float S_inv = S == 0.0f ? 0.0f : 1.0f / S;

    VKQ32_v = __riscv_vfmul_vf_f32m4(VKQ32_v, S_inv, DV);

    __riscv_vse32_v_f32m4(dst, VKQ32_v, DV);
}

// flagos: the rows go through spacemit_ime::copy, M2b's copy of the same memcpy1d (1013-1113), which keeps it behind a
// call because its asm does not declare the vector registers it overwrites
void memcpy2d(void * dst, int64_t dst_stride, const void * src, int64_t src_stride, int64_t tile_rows, int64_t size) {
    for (int64_t i = 0; i < tile_rows; ++i) {
        spacemit_ime::copy((char *) dst + i * dst_stride, (const char *) src + i * src_stride, (size_t) size);
    }
}

}  // namespace

bool supported_shape(int64_t DK, int64_t DV) {
    return flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV);
}

// flagos: the tiled kernel's scratch (its layout comment below), without ggml-cpu's padding between threads; a multiple
// of 256 bytes
size_t tiled_scratch_bytes(int64_t DK, int64_t DV) {
    constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
    constexpr int64_t KV_TILE_SZ = ggml_fa_tile_config::KV;
    return (size_t) (Q_TILE_SZ * DK + 2 * Q_TILE_SZ * KV_TILE_SZ + Q_TILE_SZ * DV + KV_TILE_SZ * DV + KV_TILE_SZ * DK) *
           sizeof(float);
}

// flagos: the interface of flagos-spacemit-rvv-attn.h; no ggml_compute_params (its ith was unused here)
void forward_one_chunk(const ggml_tensor * dst, int64_t ir0, int64_t ir1, void * tcm_buffer, size_t tcm_buffer_size) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    // broadcast factors
    const int64_t rk2 = neq2 / nek2;
    const int64_t rk3 = neq3 / nek3;

    const int64_t rv2 = neq2 / nev2;
    const int64_t rv3 = neq3 / nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    float scale         = *((float *) dst->op_params + 0);
    float max_bias      = *((float *) dst->op_params + 1);
    float logit_softcap = *((float *) dst->op_params + 2);

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const int KV_row_size = DK * sizeof(_Float16) + DV * sizeof(_Float16);

    // flagos: (c) the 2- and 4-row kernels apply one slope, the first row's, to all their rows, and with ALiBi each
    // query head has its own; rows are grouped only when every slope is 1
    const bool group_rows = !(max_bias > 0.0f);

    int ir_step = 1;
    for (int ir = ir0; ir < ir1; ir += ir_step) {
        // q indices
        const int iq3 = ir / (neq2 * neq1);
        const int iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        const int iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        const int iq3_1 = (ir + 1) / (neq2 * neq1);
        const int iq2_1 = (ir + 1 - iq3_1 * neq2 * neq1) / neq1;
        const int iq1_1 = (ir + 1 - iq3_1 * neq2 * neq1 - iq2_1 * neq1);

        const int iq3_2 = (ir + 2) / (neq2 * neq1);
        const int iq2_2 = (ir + 2 - iq3_2 * neq2 * neq1) / neq1;
        const int iq1_2 = (ir + 2 - iq3_2 * neq2 * neq1 - iq2_2 * neq1);

        const int iq3_3 = (ir + 3) / (neq2 * neq1);
        const int iq2_3 = (ir + 3 - iq3_3 * neq2 * neq1) / neq1;
        const int iq1_3 = (ir + 3 - iq3_3 * neq2 * neq1 - iq2_3 * neq1);

        const uint32_t h = iq2;  // head index
        const float    slope =
            (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

        const ggml_fp16_t * mp =
            mask ? (ggml_fp16_t *) ((char *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                                    (iq3 % mask->ne[3]) * mask->nb[3]) :
                   NULL;

        // flagos: (b) these read mask->ne without a null check, so a call without a mask crashed; without one the rows
        // share no mask row and group freely (the inner kernels take mp == NULL)
        const bool mp_equal_2 = !mask || (iq1_1 == iq1 && (iq2 % mask->ne[2]) == (iq2_1 % mask->ne[2]) &&
                                          (iq3 % mask->ne[3]) == (iq3_1 % mask->ne[3]));

        const bool mp_equal_4 =
            !mask || (mp_equal_2 && iq1_2 == iq1 && (iq2 % mask->ne[2]) == (iq2_2 % mask->ne[2]) &&
                      (iq3 % mask->ne[3]) == (iq3_2 % mask->ne[3]) && iq1_3 == iq1 &&
                      (iq2 % mask->ne[2]) == (iq2_3 % mask->ne[2]) && (iq3 % mask->ne[3]) == (iq3_3 % mask->ne[3]));

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        const int ik3_1 = iq3_1 / rk3;
        const int ik2_1 = iq2_1 / rk2;

        const int ik3_2 = iq3_2 / rk3;
        const int ik2_2 = iq2_2 / rk2;

        const int ik3_3 = iq3_3 / rk3;
        const int ik2_3 = iq2_3 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        const int iv3_1 = iq3_1 / rv3;
        const int iv2_1 = iq2_1 / rv2;

        const int iv3_2 = iq3_2 / rv3;
        const int iv2_2 = iq2_2 / rv2;

        const int iv3_3 = iq3_3 / rv3;
        const int iv2_3 = iq2_3 / rv2;

        const float * pq = (const float *) ((char *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));

        std::array<float *, 4> pq_buffer;
        std::array<float *, 4> sinks_buffer;
        std::array<float *, 4> dst_buffer;

        // flagos: (c) both grouped kernels only with group_rows
        if (group_rows && tcm_buffer != nullptr && 4 * KV_row_size < tcm_buffer_size && ir < (ir1 - 3) && mp_equal_4 &&
            ik3_3 == ik3 && ik2_3 == ik2 && iv3_3 == iv3 && iv2_3 == iv2 && ik3_2 == ik3 && ik2_2 == ik2 &&
            iv3_2 == iv3 && iv2_2 == iv2 && ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_step = 4;

            pq_buffer[0] = (float *) ((char *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_buffer[1] = (float *) ((char *) q->data + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));
            pq_buffer[2] = (float *) ((char *) q->data + (iq1_2 * nbq1 + iq2_2 * nbq2 + iq3_2 * nbq3));
            pq_buffer[3] = (float *) ((char *) q->data + (iq1_3 * nbq1 + iq2_3 * nbq2 + iq3_3 * nbq3));

            sinks_buffer[0] = sinks ? ((float *) ((char *) sinks->data)) + iq2 : nullptr;
            sinks_buffer[1] = sinks ? ((float *) ((char *) sinks->data)) + iq2_1 : nullptr;
            sinks_buffer[2] = sinks ? ((float *) ((char *) sinks->data)) + iq2_2 : nullptr;
            sinks_buffer[3] = sinks ? ((float *) ((char *) sinks->data)) + iq2_3 : nullptr;

            dst_buffer[0] = (float *) ((char *) dst->data + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_buffer[1] = (float *) ((char *) dst->data + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);
            dst_buffer[2] = (float *) ((char *) dst->data + (iq3_2 * ne2 * ne1 + iq2_2 + iq1_2 * ne1) * nb1);
            dst_buffer[3] = (float *) ((char *) dst->data + (iq3_3 * ne2 * ne1 + iq2_3 + iq1_3 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<4>(  //
                pq_buffer.data(),                                      //
                (const char *) k->data + (ik2 * nbk2 + ik3 * nbk3),    //
                (const char *) v->data + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_buffer.data(),                                   //
                dst_buffer.data(),                                     //
                scale, logit_softcap, slope, nek1, nbk1, nbv1, DV, DK, tcm_buffer, tcm_buffer_size);
        } else if (group_rows && tcm_buffer != nullptr && 2 * KV_row_size < tcm_buffer_size && ir < (ir1 - 1) &&
                   mp_equal_2 && ik3_1 == ik3 && ik2_1 == ik2 && iv3_1 == iv3 && iv2_1 == iv2) {
            ir_step = 2;

            pq_buffer[0] = (float *) ((char *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
            pq_buffer[1] = (float *) ((char *) q->data + (iq1_1 * nbq1 + iq2_1 * nbq2 + iq3_1 * nbq3));

            sinks_buffer[0] = sinks ? ((float *) ((char *) sinks->data)) + iq2 : nullptr;
            sinks_buffer[1] = sinks ? ((float *) ((char *) sinks->data)) + iq2_1 : nullptr;

            dst_buffer[0] = (float *) ((char *) dst->data + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1);
            dst_buffer[1] = (float *) ((char *) dst->data + (iq3_1 * ne2 * ne1 + iq2_1 + iq1_1 * ne1) * nb1);

            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_mrow<2>(  //
                pq_buffer.data(),                                      //
                (const char *) k->data + (ik2 * nbk2 + ik3 * nbk3),    //
                (const char *) v->data + (iv2 * nbv2 + iv3 * nbv3),    //
                mp,                                                    //
                sinks_buffer.data(),                                   //
                dst_buffer.data(),                                     //
                scale, logit_softcap, slope, nek1, nbk1, nbv1, DV, DK, tcm_buffer, tcm_buffer_size);
        } else {
            ir_step = 1;
            flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(                             //
                pq,                                                                          //
                (const char *) k->data + (ik2 * nbk2 + ik3 * nbk3),                          //
                (const char *) v->data + (iv2 * nbv2 + iv3 * nbv3),                          //
                mp,                                                                          //
                sinks ? ((float *) ((char *) sinks->data)) + h : nullptr,                    //
                (float *) ((char *) dst->data + (iq3 * ne2 * ne1 + iq2 + iq1 * ne1) * nb1),  //
                scale, logit_softcap, slope, nek1, nbk1, nbv1, DV, DK);
        }
    }
}

// flagos: the interface of flagos-spacemit-rvv-attn.h; no ggml_compute_params, and the caller's scratch replaces the
// thread's slice of params->wdata or its TCM
void forward_tiled(const ggml_tensor * dst, int64_t ir0, int64_t ir1, void * scratch, size_t scratch_size) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    const ggml_type kv_type = k->type;

    // broadcast factors
    const int64_t rk2 = neq2 / nek2;
    const int64_t rk3 = neq3 / nek3;

    const int64_t rv2 = neq2 / nev2;
    const int64_t rv3 = neq3 / nev3;

    float * param_list    = (float *) dst->op_params;
    float   scale         = param_list[0];
    float   max_bias      = param_list[1];
    float   logit_softcap = param_list[2];

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    static constexpr int Q_TILE_SZ  = ggml_fa_tile_config::Q;
    static constexpr int KV_TILE_SZ = ggml_fa_tile_config::KV;

    // Per-thread scratch layout:
    // Q_f32:   Q_TILE_SZ * DK
    // KQ:      Q_TILE_SZ * KV_TILE_SZ
    // mask32:  Q_TILE_SZ * KV_TILE_SZ
    // VKQ32:   Q_TILE_SZ * DV
    // V32:     KV_TILE_SZ * DV
    // K_f32:   DK * KV_TILE_SZ (transposed K tile)
    // flagos: in the caller's scratch (tiled_scratch_bytes), float-aligned for the vector loads and stores
    GGML_ASSERT(scratch != nullptr && (uintptr_t) scratch % alignof(float) == 0 &&
                scratch_size >= tiled_scratch_bytes(DK, DV));
    float * base = (float *) scratch;

    float   S_M_Buf[Q_TILE_SZ * 2];  // buffer to hold S, M, bias for one tile to reduce register pressure in main loop
    float * S = S_M_Buf;
    float * M = S_M_Buf + Q_TILE_SZ;

    int ir = ir0;
    while (ir < ir1) {
        // q indices for the start of this tile
        const int iq3 = ir / (neq2 * neq1);
        const int iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        const int iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        const int tile_rows = MIN(Q_TILE_SZ, MIN((int) (ir1 - ir), (int) (neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        const uint32_t h = iq2;  // head index
        const float    slope =
            (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

        for (int i = 0; i < Q_TILE_SZ; ++i) {
            S[i] = 0.;
            M[i] = -INFINITY;
        }

        float *    Q_f32  = base;
        float *    KQ     = (float *) ((char *) base + Q_TILE_SZ * DK * sizeof(float));
        float *    mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        float *    VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        float *    V32    = VKQ32 + Q_TILE_SZ * DV;
        float *    K_f32  = V32 + KV_TILE_SZ * DV;
        _Float16 * Q_f16  = (_Float16 *) Q_f32;
        _Float16 * V_f16  = (_Float16 *) V32;
        _Float16 * K_f16  = (_Float16 *) K_f32;

        rvv_zero_f32(VKQ32, Q_TILE_SZ * DV);

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        const float * pq = (const float *) ((char *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
        if (kv_type == GGML_TYPE_F16) {
            rvv_pack_f32_as_scaled_f16((uint8_t *) Q_f16, DK * sizeof(_Float16), (uint8_t *) pq, nbq1, tile_rows, DK,
                                       scale);
        } else {
            memcpy2d(Q_f32, DK * sizeof(float), pq, nbq1, tile_rows, DK * sizeof(float));
        }

        for (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            const int kv_tile = (int) std::min((int64_t) KV_TILE_SZ, nek1 - ic);

            rvv_zero_f32(K_f32, DK * KV_TILE_SZ);
            rvv_zero_f32(V32, KV_TILE_SZ * DV);

            // skip the tile entirely if all the masks are -inf
            if (mask) {
                bool                can_skip = true;
                const ggml_fp16_t * mp_row =
                    (const ggml_fp16_t *) ((const char *) mask->data + iq1 * mask->nb[1] +
                                           (iq2 % mask->ne[2]) * mask->nb[2] + (iq3 % mask->ne[3]) * mask->nb[3]);
                rvv_pack_scaled_f16_as_f32(mask32, KV_TILE_SZ * sizeof(float), mp_row + ic, mask->nb[1], tile_rows,
                                           kv_tile, slope);

                for (int tq = 0; tq < tile_rows; tq++) {
                    for (int tk = 0; tk < kv_tile; tk++) {
                        if (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = false;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    for (int tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                if (can_skip) {
                    continue;
                }
            }

            if (kv_type == GGML_TYPE_F16) {
                rvv_transposed_s16_mn_to_nm((int8_t *) K_f16, KV_TILE_SZ * sizeof(_Float16),
                                            (int8_t *) k->data + ic * nbk1 + ik2 * nbk2 + ik3 * nbk3, nbk1, kv_tile,
                                            DK);

                int tq = 0;
                for (; tq + 3 < tile_rows; tq += 4) {
                    rvv_qk_dot_tile_f16_x4(KQ + (tq + 0) * KV_TILE_SZ, KQ + (tq + 1) * KV_TILE_SZ,
                                           KQ + (tq + 2) * KV_TILE_SZ, KQ + (tq + 3) * KV_TILE_SZ,
                                           Q_f16 + (tq + 0) * DK, Q_f16 + (tq + 1) * DK, Q_f16 + (tq + 2) * DK,
                                           Q_f16 + (tq + 3) * DK, K_f16, DK, kv_tile);
                }
                for (; tq < tile_rows; ++tq) {
                    rvv_qk_dot_tile_f16_x1(KQ + tq * KV_TILE_SZ, Q_f16 + tq * DK, K_f16, DK, kv_tile);
                }
            } else {
                for (int tk = 0; tk < kv_tile; tk++) {
                    const char *  k_data = (const char *) k->data + (ic + tk) * nbk1 + ik2 * nbk2 + ik3 * nbk3;
                    float *       k_col  = K_f32 + tk;
                    const float * k_src  = (const float *) k_data;
                    for (int64_t dk = 0; dk < DK; ++dk) {
                        k_col[dk * KV_TILE_SZ] = k_src[dk];
                    }
                }

                for (int tq = 0; tq < tile_rows; ++tq) {
                    rvv_qk_dot_tile(KQ + tq * KV_TILE_SZ, Q_f32 + tq * DK, K_f32, DK, KV_TILE_SZ, scale);
                }
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            if (kv_tile < KV_TILE_SZ) {
                for (int tq = 0; tq < tile_rows; tq++) {
                    for (int tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            if (logit_softcap != 0.0f) {
                // flagos: (d) only the kv_tile valid columns; over all KV_TILE_SZ, tanh turned the -inf padding above
                // into -logit_softcap, which without a mask gave the padding positions weight in the row sum
                rvv_softcap_tanh_inplace_f32(KQ, KV_TILE_SZ, tile_rows, kv_tile, logit_softcap);
            }

            if (mask) {
                rvv_add_inplace_f32(KQ, KV_TILE_SZ, mask32, KV_TILE_SZ, tile_rows, KV_TILE_SZ);
            }

            bool skip[Q_TILE_SZ] = {};

            for (int tq = 0; tq < tile_rows; tq++) {
                float * kq_row = KQ + tq * KV_TILE_SZ;

                const float tile_max = rvv_max_f32(kq_row, KV_TILE_SZ);

                if (tile_max == -INFINITY) {
                    skip[tq] = true;
                    continue;
                }

                const float Mold = M[tq];
                const float Mnew = fmaxf(Mold, tile_max);

                if (Mnew > Mold) {
                    const float ms = expf(Mold - Mnew);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;

                S[tq] += rvv_softmax_exp_inplace_f32(kq_row, KV_TILE_SZ, Mnew);
            }

            // Pack V as contiguous [KV_TILE_SZ][DV].
            if (kv_type == GGML_TYPE_F16) {
                const char * v_data = (const char *) v->data + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V_f16, DV * sizeof(_Float16), v_data, nbv1, kv_tile, DV * sizeof(_Float16));

                int tq = 0;
                for (; tq + 3 < tile_rows; tq += 4) {
                    if (skip[tq + 0] || skip[tq + 1] || skip[tq + 2] || skip[tq + 3]) {
                        for (int i = 0; i < 4; ++i) {
                            if (!skip[tq + i]) {
                                rvv_pv_accumulate_f16_x1(VKQ32 + (tq + i) * DV, KQ + (tq + i) * KV_TILE_SZ, V_f16,
                                                         KV_TILE_SZ, DV);
                            }
                        }
                        continue;
                    }

                    rvv_pv_accumulate_f16_x4(VKQ32 + (tq + 0) * DV, VKQ32 + (tq + 1) * DV, VKQ32 + (tq + 2) * DV,
                                             VKQ32 + (tq + 3) * DV, KQ + (tq + 0) * KV_TILE_SZ,
                                             KQ + (tq + 1) * KV_TILE_SZ, KQ + (tq + 2) * KV_TILE_SZ,
                                             KQ + (tq + 3) * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                }
                for (; tq < tile_rows; ++tq) {
                    if (!skip[tq]) {
                        rvv_pv_accumulate_f16_x1(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V_f16, KV_TILE_SZ, DV);
                    }
                }
            } else {
                const char * v_data = (const char *) v->data + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V32, DV * sizeof(float), v_data, nbv1, kv_tile, DV * sizeof(float));

                for (int tq = 0; tq < tile_rows; ++tq) {
                    if (!skip[tq]) {
                        rvv_pv_accumulate(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V32, KV_TILE_SZ, DV);
                    }
                }
            }
        }

        // sinks (apply only to valid rows in the tile)
        if (sinks) {
            const float s = ((float *) ((char *) sinks->data))[h];

            for (int tq = 0; tq < tile_rows; tq++) {
                float ms = 1.0f;
                float vs = 1.0f;

                if (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                } else {
                    vs = expf(s - M[tq]);
                }

                float S_temp = S[tq] * ms + vs;
                S[tq]        = S_temp == 0.0f ? 0.0f : 1.0f / S_temp;
            }
        } else {
            for (int tq = 0; tq < tile_rows; tq++) {
                const float S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
                S[tq]             = S_inv;
            }
        }

        float * dst_ptr = (float *) ((char *) dst->data + (iq3 * ne2 * ne1 + iq2 + (iq1) *ne1) * nb1);
        rvv_pack_scaled_f32_as_f32(dst_ptr, nb1 * ne1, VKQ32, DV * sizeof(float), tile_rows, DV, S);

        ir += tile_rows;
    }
}

// flagos: ggml-spacemit's tiled kernel (spacemit-com/llama.cpp 4e782bc, ggml/src/ggml-spacemit/rvv_kernels.cpp:
// 1306-1641), test-only (FLAGOS_SPACEMIT_ATTN=trim, plan.md M2e.3, L2). It is forward_tiled except that each
// query row skips the masked tail of each KV tile: the K transpose, the V copy, Q.K, softcap, mask, softmax and
// P.V run over the columns up to the row's last unmasked one. Copied unchanged but for the interface (marked);
// it uses this file's helpers, so fixes (e) and (f) apply, and (d) does not arise (no column past a row's
// active ones is touched)
void forward_tiled_trim(const ggml_tensor * dst, int64_t ir0, int64_t ir1, void * scratch, size_t scratch_size) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;

    GGML_ASSERT(flash_attn_ext_supported_shape_vlen1024_vf16(DK, DV));

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    const ggml_type kv_type = k->type;

    // broadcast factors
    const int64_t rk2 = neq2 / nek2;
    const int64_t rk3 = neq3 / nek3;

    const int64_t rv2 = neq2 / nev2;
    const int64_t rv3 = neq3 / nev3;

    float * param_list    = (float *) dst->op_params;
    float   scale         = param_list[0];
    float   max_bias      = param_list[1];
    float   logit_softcap = param_list[2];

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    static constexpr int Q_TILE_SZ  = ggml_fa_tile_config::Q;
    static constexpr int KV_TILE_SZ = ggml_fa_tile_config::KV;

    // Per-thread scratch layout:
    // Q_f32:   Q_TILE_SZ * DK
    // KQ:      Q_TILE_SZ * KV_TILE_SZ
    // mask32:  Q_TILE_SZ * KV_TILE_SZ
    // VKQ32:   Q_TILE_SZ * DV
    // V32:     KV_TILE_SZ * DV
    // K_f32:   DK * KV_TILE_SZ (transposed K tile)
    // flagos: in the caller's scratch (tiled_scratch_bytes), float-aligned for the vector loads and stores, as
    // forward_tiled
    GGML_ASSERT(scratch != nullptr && (uintptr_t) scratch % alignof(float) == 0 &&
                scratch_size >= tiled_scratch_bytes(DK, DV));
    float * base = (float *) scratch;

    float   S_M_Buf[Q_TILE_SZ * 2];  // buffer to hold S, M, bias for one tile to reduce register pressure in main loop
    float * S = S_M_Buf;
    float * M = S_M_Buf + Q_TILE_SZ;

    int ir = ir0;
    while (ir < ir1) {
        // q indices for the start of this tile
        const int iq3 = ir / (neq2 * neq1);
        const int iq2 = (ir - iq3 * neq2 * neq1) / neq1;
        const int iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        const int tile_rows = MIN(Q_TILE_SZ, MIN((int) (ir1 - ir), (int) (neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        const uint32_t h = iq2;  // head index
        const float    slope =
            (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

        for (int i = 0; i < Q_TILE_SZ; ++i) {
            S[i] = 0.;
            M[i] = -INFINITY;
        }

        float *    Q_f32  = base;
        float *    KQ     = (float *) ((char *) base + Q_TILE_SZ * DK * sizeof(float));
        float *    mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        float *    VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        float *    V32    = VKQ32 + Q_TILE_SZ * DV;
        float *    K_f32  = V32 + KV_TILE_SZ * DV;
        _Float16 * Q_f16  = (_Float16 *) Q_f32;
        _Float16 * V_f16  = (_Float16 *) V32;
        _Float16 * K_f16  = (_Float16 *) K_f32;

        rvv_zero_f32(VKQ32, Q_TILE_SZ * DV);

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        const float * pq = (const float *) ((char *) q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
        if (kv_type == GGML_TYPE_F16) {
            rvv_pack_f32_as_scaled_f16((uint8_t *) Q_f16, DK * sizeof(_Float16), (uint8_t *) pq, nbq1, tile_rows, DK,
                                       scale);
        } else {
            memcpy2d(Q_f32, DK * sizeof(float), pq, nbq1, tile_rows, DK * sizeof(float));
        }

        for (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            const int kv_tile = (int) std::min((int64_t) KV_TILE_SZ, nek1 - ic);
            int       active_kv[Q_TILE_SZ];
            int       max_active_kv = kv_tile;

            rvv_zero_f32(K_f32, DK * KV_TILE_SZ);
            rvv_zero_f32(V32, KV_TILE_SZ * DV);

            // skip the tile entirely if all the masks are -inf
            if (mask) {
                const ggml_fp16_t * mp_row =
                    (const ggml_fp16_t *) ((const char *) mask->data + iq1 * mask->nb[1] +
                                           (iq2 % mask->ne[2]) * mask->nb[2] + (iq3 % mask->ne[3]) * mask->nb[3]);
                rvv_pack_scaled_f16_as_f32(mask32, KV_TILE_SZ * sizeof(float), mp_row + ic, mask->nb[1], tile_rows,
                                           kv_tile, slope);

                max_active_kv = 0;
                for (int tq = 0; tq < tile_rows; tq++) {
                    int tk = kv_tile;
                    while (tk > 0 && mask32[tq * KV_TILE_SZ + tk - 1] == -INFINITY) {
                        --tk;
                    }
                    active_kv[tq] = tk;
                    max_active_kv = std::max(max_active_kv, tk);
                }

                if (max_active_kv == 0) {
                    continue;
                }
            } else {
                std::fill_n(active_kv, tile_rows, kv_tile);
            }

            if (kv_type == GGML_TYPE_F16) {
                rvv_transposed_s16_mn_to_nm((int8_t *) K_f16, KV_TILE_SZ * sizeof(_Float16),
                                            (int8_t *) k->data + ic * nbk1 + ik2 * nbk2 + ik3 * nbk3, nbk1,
                                            max_active_kv,
                                            DK);

                int tq = 0;
                for (; tq + 3 < tile_rows; tq += 4) {
                    const int group_active = std::max(std::max(active_kv[tq + 0], active_kv[tq + 1]),
                                                      std::max(active_kv[tq + 2], active_kv[tq + 3]));
                    if (group_active > 0) {
                        rvv_qk_dot_tile_f16_x4(KQ + (tq + 0) * KV_TILE_SZ, KQ + (tq + 1) * KV_TILE_SZ,
                                               KQ + (tq + 2) * KV_TILE_SZ, KQ + (tq + 3) * KV_TILE_SZ,
                                               Q_f16 + (tq + 0) * DK, Q_f16 + (tq + 1) * DK, Q_f16 + (tq + 2) * DK,
                                               Q_f16 + (tq + 3) * DK, K_f16, DK, group_active);
                    }
                }
                for (; tq < tile_rows; ++tq) {
                    if (active_kv[tq] > 0) {
                        rvv_qk_dot_tile_f16_x1(KQ + tq * KV_TILE_SZ, Q_f16 + tq * DK, K_f16, DK, active_kv[tq]);
                    }
                }
            } else {
                for (int tk = 0; tk < max_active_kv; tk++) {
                    const char *  k_data = (const char *) k->data + (ic + tk) * nbk1 + ik2 * nbk2 + ik3 * nbk3;
                    float *       k_col  = K_f32 + tk;
                    const float * k_src  = (const float *) k_data;
                    for (int64_t dk = 0; dk < DK; ++dk) {
                        k_col[dk * KV_TILE_SZ] = k_src[dk];
                    }
                }

                for (int tq = 0; tq < tile_rows; ++tq) {
                    if (active_kv[tq] > 0) {
                        rvv_qk_dot_tile_stride(KQ + tq * KV_TILE_SZ, Q_f32 + tq * DK, K_f32, DK, KV_TILE_SZ,
                                              active_kv[tq], scale);  // flagos: renamed helper
                    }
                }
            }

            bool skip[Q_TILE_SZ] = {};

            for (int tq = 0; tq < tile_rows; tq++) {
                float * kq_row = KQ + tq * KV_TILE_SZ;
                const int row_active = active_kv[tq];

                if (row_active == 0) {
                    skip[tq] = true;
                    continue;
                }

                if (logit_softcap != 0.0f) {
                    rvv_softcap_tanh_inplace_f32(kq_row, KV_TILE_SZ, 1, row_active, logit_softcap);
                }
                if (mask) {
                    rvv_add_inplace_f32(kq_row, KV_TILE_SZ, mask32 + tq * KV_TILE_SZ, KV_TILE_SZ, 1, row_active);
                }

                const float tile_max = rvv_max_f32(kq_row, row_active);

                if (tile_max == -INFINITY) {
                    skip[tq] = true;
                    continue;
                }

                const float Mold = M[tq];
                const float Mnew = fmaxf(Mold, tile_max);

                if (Mnew > Mold) {
                    const float ms = expf(Mold - Mnew);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;

                S[tq] += rvv_softmax_exp_inplace_f32(kq_row, row_active, Mnew);
            }

            // Pack V as contiguous [KV_TILE_SZ][DV].
            if (kv_type == GGML_TYPE_F16) {
                const char * v_data = (const char *) v->data + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V_f16, DV * sizeof(_Float16), v_data, nbv1, max_active_kv, DV * sizeof(_Float16));

                int tq = 0;
                for (; tq + 3 < tile_rows; tq += 4) {
                    if (skip[tq + 0] || skip[tq + 1] || skip[tq + 2] || skip[tq + 3]) {
                        for (int i = 0; i < 4; ++i) {
                            if (!skip[tq + i]) {
                                rvv_pv_accumulate_f16_x1(VKQ32 + (tq + i) * DV, KQ + (tq + i) * KV_TILE_SZ, V_f16,
                                                         active_kv[tq + i], DV);
                            }
                        }
                        continue;
                    }

                    const int group_active = std::max(std::max(active_kv[tq + 0], active_kv[tq + 1]),
                                                      std::max(active_kv[tq + 2], active_kv[tq + 3]));
                    for (int i = 0; i < 4; ++i) {
                        std::fill(KQ + (tq + i) * KV_TILE_SZ + active_kv[tq + i],
                                  KQ + (tq + i) * KV_TILE_SZ + group_active, 0.0f);
                    }
                    rvv_pv_accumulate_f16_x4(VKQ32 + (tq + 0) * DV, VKQ32 + (tq + 1) * DV, VKQ32 + (tq + 2) * DV,
                                             VKQ32 + (tq + 3) * DV, KQ + (tq + 0) * KV_TILE_SZ,
                                             KQ + (tq + 1) * KV_TILE_SZ, KQ + (tq + 2) * KV_TILE_SZ,
                                             KQ + (tq + 3) * KV_TILE_SZ, V_f16, group_active, DV);
                }
                for (; tq < tile_rows; ++tq) {
                    if (!skip[tq]) {
                        rvv_pv_accumulate_f16_x1(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V_f16, active_kv[tq], DV);
                    }
                }
            } else {
                const char * v_data = (const char *) v->data + ic * nbv1 + iv2 * nbv2 + iv3 * nbv3;
                memcpy2d(V32, DV * sizeof(float), v_data, nbv1, max_active_kv, DV * sizeof(float));

                for (int tq = 0; tq < tile_rows; ++tq) {
                    if (!skip[tq]) {
                        rvv_pv_accumulate(VKQ32 + tq * DV, KQ + tq * KV_TILE_SZ, V32, active_kv[tq], DV);
                    }
                }
            }
        }

        // sinks (apply only to valid rows in the tile)
        if (sinks) {
            const float s = ((float *) ((char *) sinks->data))[h];

            for (int tq = 0; tq < tile_rows; tq++) {
                float ms = 1.0f;
                float vs = 1.0f;

                if (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    rvv_scale_f32(VKQ32 + tq * DV, ms, DV);
                } else {
                    vs = expf(s - M[tq]);
                }

                float S_temp = S[tq] * ms + vs;
                S[tq]        = S_temp == 0.0f ? 0.0f : 1.0f / S_temp;
            }
        } else {
            for (int tq = 0; tq < tile_rows; tq++) {
                const float S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
                S[tq]             = S_inv;
            }
        }

        float * dst_ptr = (float *) ((char *) dst->data + (iq3 * ne2 * ne1 + iq2 + (iq1) *ne1) * nb1);
        rvv_pack_scaled_f32_as_f32(dst_ptr, nb1 * ne1, VKQ32, DV * sizeof(float), tile_rows, DV, S);

        ir += tile_rows;
    }
}

}  // namespace spacemit_rvv_attn

#endif  // GGML_FLAGOS_SPACEMIT_RVV
