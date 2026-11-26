// The IME2 and RVV leaves of the FP16 path for dense Q4_0 matmuls (plan.md "The dense Q4_0 matmuls in FP16: the port",
// stage 2), for the A100 cores (VLEN 1024). The driver and the scalar references are flagos-spacemit-f16.cpp's.
//
// Copied from SpacemiT's fork (spacemit-com/llama.cpp mtmd-backend b847bf396, ggml/src/ggml-spacemit):
//   repack.h                  _spe_fp16_ (19)
//   spe_kernels_core.h        SPINE_FORCE_INLINE (9-15), SPE_ASSERT (17-21), PACK_X16_M16N8 (138-162),
//                             PACK_CAST_LOAD_F32_M8K64 to PACK_CAST_STORE_F16_M16K64 (167-266),
//                             PACK_CAST_UNPACK_U4_F16_M32K32 to PACK_CAST_STORE_F16_M16K32 (268-419),
//                             the MMT4D_FP16_* macros of the plain epilogue (427-447, 456-512, 622-632, 674-714,
//                             716-880, 903-919, 958-1059),
//                             _spe_pack_cast_q41_fixed8_f16_m32k32 (1160-1209),
//                             _spe_pack_cast_f32_f16_contiguous to _spe_pack_cast_f16_pad_blocks (1356-1811),
//                             _spe_hpgemm_impl_m16n32k8_none (1817-1818, 1901-1921),
//                             _spe_reducesum_mxn_f16_f32_16x32_1x1 and its _a100_nth8 form (2002-2099)
//   spe_kernel_dispatch.cpp   spe_pack_cast<float, _spe_fp16_, 1, 1, 16, 8> (24-79), the block loop of
//                             spe_reducesum_mxn<_spe_fp16_, float, 16, 32, 1, 1> (263-290) in reduce_tile,
//                             gemm_kernel_f16f16f16_m16_vfwmadot (341-358) in products
// Bodies are unchanged except, marked "flagos": the fork's namespaces (mlir::speir::spekernel, ::ime2, ::rvv) are
// spacemit_f16; the pack wrapper is a plain function; reduce_tile is the reduce wrapper's loop for one 16 x 32 block
// and 8 or 4 slices; products is the gemm wrapper without its accumulate flag; dequant_tile is the fork's
// _spe_pack_cast_q41_fixed8_f16_m32k32 with its two layout operands (block size, scale offset) set for our q4_0 block.
// New: the flag helpers.
// The copied asm declares no vector registers: each entry below is out of line and holds no vector values of its own
// across an asm block (as copy() in flagos-spacemit-ime-kernels.cpp). Built only with GGML_FLAGOS_SPACEMIT_IME2.

#include "flagos-spacemit-f16-kernels.h"

#if defined(GGML_FLAGOS_SPACEMIT_IME2)
#    include "../../../../ggml-impl.h"

#    include <riscv_vector.h>

#    include <algorithm>
#    include <cassert>
#    include <cstddef>

#    if defined(__GNUC__)
#        pragma GCC diagnostic ignored "-Wsign-compare"  // the copied pack's stride compares
#    endif
#endif

namespace spacemit_f16 {

#if defined(GGML_FLAGOS_SPACEMIT_IME2)

// flagos: the fork's repack.h (19)
using _spe_fp16_ = _Float16;

#ifdef _MSC_VER
#    define SPINE_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__CLANG__)
#    define SPINE_FORCE_INLINE inline __attribute__((always_inline))
#else
#    define SPINE_FORCE_INLINE inline
#endif

#ifndef NDEBUG
#    define SPE_ASSERT(expr) assert(expr)
#else
#    define SPE_ASSERT(expr) ;
#endif

// clang-format off

#define PACK_X16_M16N8                                          \
    "vpack.vv               v16, v0, v1, 1                \n\t" \
    "vpack.vv               v18, v2, v3, 1                \n\t" \
    "vpack.vv               v20, v4, v5, 1                \n\t" \
    "vpack.vv               v22, v6, v7, 1                \n\t" \
    "vpack.vv               v0, v16, v18, 2               \n\t" \
    "vpack.vv               v2, v17, v19, 2               \n\t" \
    "vpack.vv               v4, v20, v22, 2               \n\t" \
    "vpack.vv               v6, v21, v23, 2               \n\t" \
    "vpack.vv               v16, v0, v4, 3                \n\t" \
    "vpack.vv               v18, v1, v5, 3                \n\t" \
    "vpack.vv               v20, v2, v6, 3                \n\t" \
    "vpack.vv               v22, v3, v7, 3                \n\t" \
    "vpack.vv               v24, v8, v9, 1                \n\t" \
    "vpack.vv               v26, v10, v11, 1              \n\t" \
    "vpack.vv               v28, v12, v13, 1              \n\t" \
    "vpack.vv               v30, v14, v15, 1              \n\t" \
    "vpack.vv               v8, v24, v26, 2               \n\t" \
    "vpack.vv               v10, v25, v27, 2              \n\t" \
    "vpack.vv               v12, v28, v30, 2              \n\t" \
    "vpack.vv               v14, v29, v31, 2              \n\t" \
    "vpack.vv               v24, v8, v12, 3               \n\t" \
    "vpack.vv               v26, v9, v13, 3               \n\t" \
    "vpack.vv               v28, v10, v14, 3              \n\t" \
    "vpack.vv               v30, v11, v15, 3              \n\t"

// VLEN=1024; strides are in bytes. Callers set vtype.

#define PACK_CAST_LOAD_F32_M8K64                                               \
    "vle32.v        v16, (t0)                                            \n\t" \
    "vle32.v        v18, (t1)                                            \n\t" \
    "vle32.v        v20, (t5)                                            \n\t" \
    "vle32.v        v22, (t6)                                            \n\t" \
    "sh2add         t0, %[stride], t0                                    \n\t" \
    "sh2add         t1, %[stride], t1                                    \n\t" \
    "sh2add         t5, %[stride], t5                                    \n\t" \
    "sh2add         t6, %[stride], t6                                    \n\t" \
    "vle32.v        v24, (t0)                                            \n\t" \
    "vle32.v        v26, (t1)                                            \n\t" \
    "vle32.v        v28, (t5)                                            \n\t" \
    "vle32.v        v30, (t6)                                            \n\t"

#define PACK_CAST_NARROW_F32_F16_M8_LO                                         \
    "vfncvt.f.f.w   v0, v16                                              \n\t" \
    "vfncvt.f.f.w   v1, v18                                              \n\t" \
    "vfncvt.f.f.w   v2, v20                                              \n\t" \
    "vfncvt.f.f.w   v3, v22                                              \n\t" \
    "vfncvt.f.f.w   v4, v24                                              \n\t" \
    "vfncvt.f.f.w   v5, v26                                              \n\t" \
    "vfncvt.f.f.w   v6, v28                                              \n\t" \
    "vfncvt.f.f.w   v7, v30                                              \n\t"

#define PACK_CAST_NARROW_F32_F16_M8_HI                                         \
    "vfncvt.f.f.w   v8, v16                                              \n\t" \
    "vfncvt.f.f.w   v9, v18                                              \n\t" \
    "vfncvt.f.f.w   v10, v20                                             \n\t" \
    "vfncvt.f.f.w   v11, v22                                             \n\t" \
    "vfncvt.f.f.w   v12, v24                                             \n\t" \
    "vfncvt.f.f.w   v13, v26                                             \n\t" \
    "vfncvt.f.f.w   v14, v28                                             \n\t" \
    "vfncvt.f.f.w   v15, v30                                             \n\t"

#define PACK_CAST_PACK_F16_M8K64                                               \
    "vpack.vv       v8,  v0,  v1,  1                                     \n\t" \
    "vpack.vv       v10, v2,  v3,  1                                     \n\t" \
    "vpack.vv       v12, v4,  v5,  1                                     \n\t" \
    "vpack.vv       v14, v6,  v7,  1                                     \n\t" \
    "vpack.vv       v0,  v8,  v10, 2                                     \n\t" \
    "vpack.vv       v2,  v9,  v11, 2                                     \n\t" \
    "vpack.vv       v4,  v12, v14, 2                                     \n\t" \
    "vpack.vv       v6,  v13, v15, 2                                     \n\t" \
    "vpack.vv       v8,  v0,  v4,  3                                     \n\t" \
    "vpack.vv       v10, v1,  v5,  3                                     \n\t" \
    "vpack.vv       v12, v2,  v6,  3                                     \n\t" \
    "vpack.vv       v14, v3,  v7,  3                                     \n\t"

#define PACK_CAST_STORE_F16_M8K64                                              \
    "vse16.v        v8, (t2)                                             \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v9, (t2)                                             \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v10, (t2)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v11, (t2)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v12, (t2)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v13, (t2)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v14, (t2)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v15, (t2)                                            \n\t"

#define PACK_CAST_STORE_F16_M16K64                                             \
    "vse16.v        v16, (t0)                                            \n\t" \
    "vse16.v        v24, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v17, (t0)                                            \n\t" \
    "vse16.v        v25, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v18, (t0)                                            \n\t" \
    "vse16.v        v26, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v19, (t0)                                            \n\t" \
    "vse16.v        v27, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v20, (t0)                                            \n\t" \
    "vse16.v        v28, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v21, (t0)                                            \n\t" \
    "vse16.v        v29, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v22, (t0)                                            \n\t" \
    "vse16.v        v30, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v23, (t0)                                            \n\t" \
    "vse16.v        v31, (t1)                                            \n\t"

// Preserve v0: later scaling uses its paired-row mask.

#define PACK_CAST_UNPACK_U4_F16_M32K32                                         \
    "vzext.vf2      v8, v4                                               \n\t" \
    "vsrl.vi        v16, v8, 4                                           \n\t" \
    "vand.vi        v8, v8, 15                                           \n\t" \
    "vsll.vi        v16, v16, 8                                          \n\t" \
    "vor.vv         v24, v8, v16                                         \n\t" \
    "vzext.vf2      v8, v24                                              \n\t" \
    "vzext.vf2      v16, v28                                             \n\t" \
    "vsub.vx        v8, v8, %[zero_point]                                \n\t" \
    "vsub.vx        v16, v16, %[zero_point]                              \n\t" \
    "vfcvt.f.x.v    v8, v8                                               \n\t" \
    "vfcvt.f.x.v    v16, v16                                             \n\t"

#define PACK_CAST_LOAD_SCALE_F16_M16_LO                                        \
    "flh            ft0, 0(t1)                                           \n\t" \
    "flh            ft1, 2(t1)                                           \n\t" \
    "flh            ft2, 4(t1)                                           \n\t" \
    "flh            ft3, 6(t1)                                           \n\t" \
    "flh            ft4, 8(t1)                                           \n\t" \
    "flh            ft5, 10(t1)                                          \n\t" \
    "flh            ft6, 12(t1)                                          \n\t" \
    "flh            ft7, 14(t1)                                          \n\t" \
    "flh            fa0, 16(t1)                                          \n\t" \
    "flh            fa1, 18(t1)                                          \n\t" \
    "flh            fa2, 20(t1)                                          \n\t" \
    "flh            fa3, 22(t1)                                          \n\t" \
    "flh            fa4, 24(t1)                                          \n\t" \
    "flh            fa5, 26(t1)                                          \n\t" \
    "flh            fa6, 28(t1)                                          \n\t" \
    "flh            fa7, 30(t1)                                          \n\t"

#define PACK_CAST_LOAD_SCALE_F16_M16_HI                                        \
    "flh            ft0, 32(t1)                                          \n\t" \
    "flh            ft1, 34(t1)                                          \n\t" \
    "flh            ft2, 36(t1)                                          \n\t" \
    "flh            ft3, 38(t1)                                          \n\t" \
    "flh            ft4, 40(t1)                                          \n\t" \
    "flh            ft5, 42(t1)                                          \n\t" \
    "flh            ft6, 44(t1)                                          \n\t" \
    "flh            ft7, 46(t1)                                          \n\t" \
    "flh            fa0, 48(t1)                                          \n\t" \
    "flh            fa1, 50(t1)                                          \n\t" \
    "flh            fa2, 52(t1)                                          \n\t" \
    "flh            fa3, 54(t1)                                          \n\t" \
    "flh            fa4, 56(t1)                                          \n\t" \
    "flh            fa5, 58(t1)                                          \n\t" \
    "flh            fa6, 60(t1)                                          \n\t" \
    "flh            fa7, 62(t1)                                          \n\t"

// v0 selects the upper row of each pair; v1/v2 carry row scales.

#define PACK_CAST_SCALE_F16_M16K32_LO                                          \
    "vfmv.v.f       v1, ft0                                              \n\t" \
    "vfmv.v.f       v2, ft2                                              \n\t" \
    "vfmerge.vfm    v1, v1, ft1, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, ft3, v0                                      \n\t" \
    "vfmul.vv       v8, v8, v1                                           \n\t" \
    "vfmul.vv       v9, v9, v2                                           \n\t" \
    "vfmv.v.f       v1, ft4                                              \n\t" \
    "vfmv.v.f       v2, ft6                                              \n\t" \
    "vfmerge.vfm    v1, v1, ft5, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, ft7, v0                                      \n\t" \
    "vfmul.vv       v10, v10, v1                                         \n\t" \
    "vfmul.vv       v11, v11, v2                                         \n\t" \
    "vfmv.v.f       v1, fa0                                              \n\t" \
    "vfmv.v.f       v2, fa2                                              \n\t" \
    "vfmerge.vfm    v1, v1, fa1, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, fa3, v0                                      \n\t" \
    "vfmul.vv       v12, v12, v1                                         \n\t" \
    "vfmul.vv       v13, v13, v2                                         \n\t" \
    "vfmv.v.f       v1, fa4                                              \n\t" \
    "vfmv.v.f       v2, fa6                                              \n\t" \
    "vfmerge.vfm    v1, v1, fa5, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, fa7, v0                                      \n\t" \
    "vfmul.vv       v14, v14, v1                                         \n\t" \
    "vfmul.vv       v15, v15, v2                                         \n\t"

#define PACK_CAST_SCALE_F16_M16K32_HI                                          \
    "vfmv.v.f       v1, ft0                                              \n\t" \
    "vfmv.v.f       v2, ft2                                              \n\t" \
    "vfmerge.vfm    v1, v1, ft1, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, ft3, v0                                      \n\t" \
    "vfmul.vv       v16, v16, v1                                         \n\t" \
    "vfmul.vv       v17, v17, v2                                         \n\t" \
    "vfmv.v.f       v1, ft4                                              \n\t" \
    "vfmv.v.f       v2, ft6                                              \n\t" \
    "vfmerge.vfm    v1, v1, ft5, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, ft7, v0                                      \n\t" \
    "vfmul.vv       v18, v18, v1                                         \n\t" \
    "vfmul.vv       v19, v19, v2                                         \n\t" \
    "vfmv.v.f       v1, fa0                                              \n\t" \
    "vfmv.v.f       v2, fa2                                              \n\t" \
    "vfmerge.vfm    v1, v1, fa1, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, fa3, v0                                      \n\t" \
    "vfmul.vv       v20, v20, v1                                         \n\t" \
    "vfmul.vv       v21, v21, v2                                         \n\t" \
    "vfmv.v.f       v1, fa4                                              \n\t" \
    "vfmv.v.f       v2, fa6                                              \n\t" \
    "vfmerge.vfm    v1, v1, fa5, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, fa7, v0                                      \n\t" \
    "vfmul.vv       v22, v22, v1                                         \n\t" \
    "vfmul.vv       v23, v23, v2                                         \n\t"

// Preserve v0 and the next packed block in v4..v7.

#define PACK_CAST_PACK_F16_M16K32_LO                                           \
    "vpack.vv       v24, v8, v9, 1                                       \n\t" \
    "vpack.vv       v26, v10, v11, 1                                     \n\t" \
    "vpack.vv       v8, v24, v25, 1                                      \n\t" \
    "vpack.vv       v10, v26, v27, 1                                     \n\t" \
    "vpack.vv       v24, v8, v10, 3                                      \n\t" \
    "vpack.vv       v26, v9, v11, 3                                      \n\t" \
    "vpack.vv       v28, v12, v13, 1                                     \n\t" \
    "vpack.vv       v30, v14, v15, 1                                     \n\t" \
    "vpack.vv       v12, v28, v29, 1                                     \n\t" \
    "vpack.vv       v14, v30, v31, 1                                     \n\t" \
    "vpack.vv       v28, v12, v14, 3                                     \n\t" \
    "vpack.vv       v30, v13, v15, 3                                     \n\t"

#define PACK_CAST_PACK_F16_M16K32_HI                                           \
    "vpack.vv       v24, v16, v17, 1                                     \n\t" \
    "vpack.vv       v26, v18, v19, 1                                     \n\t" \
    "vpack.vv       v16, v24, v25, 1                                     \n\t" \
    "vpack.vv       v18, v26, v27, 1                                     \n\t" \
    "vpack.vv       v24, v16, v18, 3                                     \n\t" \
    "vpack.vv       v26, v17, v19, 3                                     \n\t" \
    "vpack.vv       v28, v20, v21, 1                                     \n\t" \
    "vpack.vv       v30, v22, v23, 1                                     \n\t" \
    "vpack.vv       v20, v28, v29, 1                                     \n\t" \
    "vpack.vv       v22, v30, v31, 1                                     \n\t" \
    "vpack.vv       v28, v20, v22, 3                                     \n\t" \
    "vpack.vv       v30, v21, v23, 3                                     \n\t"

// tile_stride is 512 bytes; t3 starts eight rows after t2.

#define PACK_CAST_STORE_F16_M16K32                                             \
    "vse16.v        v24, (t2)                                            \n\t" \
    "vse16.v        v28, (t3)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "addi           t3, t3, %[tile_stride]                               \n\t" \
    "vse16.v        v25, (t2)                                            \n\t" \
    "vse16.v        v29, (t3)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "addi           t3, t3, %[tile_stride]                               \n\t" \
    "vse16.v        v26, (t2)                                            \n\t" \
    "vse16.v        v30, (t3)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "addi           t3, t3, %[tile_stride]                               \n\t" \
    "vse16.v        v27, (t2)                                            \n\t" \
    "vse16.v        v31, (t3)                                            \n\t"

// VLEN=1024; A=[kc][16][8], B=[kc][32][8], C uses byte row strides.
// v0..v7 hold A, v8..v15 B, and v16..v31 FP32 accumulators.

#define _A0         "a1"
#define _A1         "a2"

#define _B0         "s1"
#define _B1         "s2"
#define _B2         "s3"
#define _B3         "s4"

#define _TMP_VL     "t0"
#define _TMP_KC     "t2"

#define _A          "%[a]"
#define _B          "%[b]"
#define _C          "%[c]"
#define _M          "%[m]"
#define _N          "%[n]"
#define _KC         "%[kc]"
#define _CS         "%[cs]"

#define MMT4D_FP16_M16N32K8_INIT_IMPL(A, A0, A1, B, B0, B1, B2, B3, KC, TMP_VL, TMP_KC)         \
    "vsetvli            " TMP_VL ", zero, e32, m8, tu, mu       \n\t"                           \
    "vxor.vv            v16, v16, v16                           \n\t"                           \
    "vxor.vv            v24, v24, v24                           \n\t"                           \
    "vsetvli            " TMP_VL ", zero, e16, m1               \n\t"                           \
    "addi               " A0 ", " A ", 0                        \n\t"                           \
    "addi               " A1 ", " A ", 8*8*2                    \n\t"                           \
    "addi               " B0 ", " B ", 0                        \n\t"                           \
    "addi               " B1 ", " B ", 8*8*2                    \n\t"                           \
    "addi               " B2 ", " B ", 8*8*4                    \n\t"                           \
    "addi               " B3 ", " B ", 8*8*6                    \n\t"                           \
    "srli               " TMP_KC ", " KC ", 2                   \n\t"

#define MMT4D_FP16_M16N32K8_INIT                                                                \
    MMT4D_FP16_M16N32K8_INIT_IMPL(_A, _A0, _A1, _B, _B0, _B1, _B2, _B3, _KC, _TMP_VL, _TMP_KC)

#define MMT4D_FP16_M16N32K8_COMPUTE_LOOP_IMPL(KC, A0, A1, B0, B1, B2, B3, TMP_VL, TMP_KC)       \
    "blez               " TMP_KC ", TAIL_KC1%=                  \n\t"                           \
    MMT4D_FP16_PRELOAD_M16N32_IMPL(_A, A0, A1, _B, B0, B1, B2, B3, TMP_VL)                      \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", TAIL_KC4%=                  \n\t"                           \
    "LOOP_KC4%=:                                                \n\t"                           \
    MMT4D_FP16_COMP_M16N32_M1_IMPL(A0, A1, B0, B1, B2, B3)                                      \
    MMT4D_FP16_COMP_M16N32_M2_IMPL(A0, A1, B0, B1, B2, B3)                                      \
    MMT4D_FP16_COMP_M16N32_M3_IMPL(A0, A1, B0, B1, B2, B3)                                      \
    MMT4D_FP16_COMP_M16N32_M4_IMPL(A0, A1, B0, B1, B2, B3)                                      \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "bnez               " TMP_KC ", LOOP_KC4%=                  \n\t"                           \
    "TAIL_KC4%=:                                                \n\t"                           \
    MMT4D_FP16_COMP_M16N32_M1_IMPL(A0, A1, B0, B1, B2, B3)                                      \
    MMT4D_FP16_COMP_M16N32_E2_IMPL(B0, B1, B2, B3)                                              \
    MMT4D_FP16_COMP_M16N32_E3_IMPL(B3)                                                          \
    MMT4D_FP16_COMP_M16N32_E4                                                                   \
    "TAIL_KC1%=:                                                \n\t"                           \
    "andi               " TMP_KC ", " KC ", 3                   \n\t"                           \
    "blez               " TMP_KC ", POST_PROCESS%=              \n\t"                           \
    "vsetvli            " TMP_VL ", zero, e16, m1               \n\t"                           \
    "LOOP_KC1%=:                                                \n\t"                           \
    "vle16.v            v6, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v7, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vle16.v            v12, (" B0 ")                           \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vle16.v            v13, (" B1 ")                           \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vle16.v            v14, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vle16.v            v15, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    MMT4D_FP16_COMP_M16N32_E4                                                                   \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "bnez               " TMP_KC ", LOOP_KC1%=                  \n\t"                           \
    "POST_PROCESS%=:                                            \n\t"

#define MMT4D_FP16_M16N32K8_COMPUTE_LOOP                                                        \
    MMT4D_FP16_M16N32K8_COMPUTE_LOOP_IMPL(_KC, _A0, _A1, _B0, _B1, _B2, _B3, _TMP_VL, _TMP_KC)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_NONE_IMPL(N, M, TMP_VL, TMP_KC)                         \
    "vsetvli            " TMP_VL ", zero, e32, m1               \n\t"                           \
    MMT4D_FP16_UNPACK_M16N32                                                                    \
    "vsetvli            " TMP_VL ", " N ", e32, m1              \n\t"                           \
    "CLIP%=:                                                    \n\t"                           \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "srli               " TMP_KC ", " M ", 4                    \n\t"                           \
    MMT4D_FP16_CONVERT_M16N32

#define MMT4D_FP16_M16N32K8_POSTPROCESS_NONE                                                    \
    MMT4D_FP16_M16N32K8_POSTPROCESS_NONE_IMPL(_N, _M, _TMP_VL, _TMP_KC)

#define MMT4D_FP16_M16N32K8_STORE_IMPL(C, CS, M, TMP_KC)                                        \
    "bnez               " TMP_KC ", STORE_M16%=                 \n\t"                           \
    MMT4D_FP16_SAVING_LT16_IMPL(C, CS, M)                                                       \
    "jal                x0, END%=                               \n\t"                           \
    "STORE_M16%=:                                               \n\t"                           \
    MMT4D_FP16_SAVING_M16_IMPL(C, CS)                                                           \
    "END%=:                                                     \n\t"

#define MMT4D_FP16_M16N32K8_STORE                                                               \
    MMT4D_FP16_M16N32K8_STORE_IMPL(_C, _CS, _M, _TMP_KC)

#define MMT4D_FP16_PRELOAD_M16N32_IMPL(A, A0, A1, B, B0, B1, B2, B3, TMP_VL)                    \
    "vle16.v            v0, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v1, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vle16.v            v2, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v3, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vle16.v            v4, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v5, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vle16.v            v8, (" B0 ")                            \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vle16.v            v9, (" B1 ")                            \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vle16.v            v10, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vle16.v            v11, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vle16.v            v12, (" B0 ")                           \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vle16.v            v13, (" B1 ")                           \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vle16.v            v14, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"

#define MMT4D_FP16_PRELOAD_M16N32                                                               \
    MMT4D_FP16_PRELOAD_M16N32_IMPL(_A, _A0, _A1, _B, _B0, _B1, _B2, _B3, _TMP_VL)

// M1-M4 preload the next tile; E2-E4 drain the final iteration.

#define MMT4D_FP16_COMP_M16N32_M1_IMPL(A0, A1, B0, B1, B2, B3)                                  \
    "vle16.v            v6, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v7, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vfwmadot           v16, v0, v8                             \n\t"                           \
    "vle16.v            v15, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v1, v8                             \n\t"                           \
    "vfwmadot           v18, v0, v9                             \n\t"                           \
    "vle16.v            v8, (" B0 ")                            \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v26, v1, v9                             \n\t"                           \
    "vfwmadot           v20, v0, v10                            \n\t"                           \
    "vle16.v            v9, (" B1 ")                            \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v28, v1, v10                            \n\t"                           \
    "vfwmadot           v22, v0, v11                            \n\t"                           \
    "vle16.v            v10, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v30, v1, v11                            \n\t"

#define MMT4D_FP16_COMP_M16N32_M1                                                               \
    MMT4D_FP16_COMP_M16N32_M1_IMPL(_A0, _A1, _B0, _B1, _B2, _B3)

#define MMT4D_FP16_COMP_M16N32_M2_IMPL(A0, A1, B0, B1, B2, B3)                                  \
    "vle16.v            v0, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v1, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vfwmadot           v16, v2, v12                            \n\t"                           \
    "vle16.v            v11, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v3, v12                            \n\t"                           \
    "vfwmadot           v18, v2, v13                            \n\t"                           \
    "vle16.v            v12, (" B0 ")                           \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v26, v3, v13                            \n\t"                           \
    "vfwmadot           v20, v2, v14                            \n\t"                           \
    "vle16.v            v13, (" B1 ")                           \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v28, v3, v14                            \n\t"                           \
    "vfwmadot           v22, v2, v15                            \n\t"                           \
    "vle16.v            v14, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v30, v3, v15                            \n\t"

#define MMT4D_FP16_COMP_M16N32_M2                                                               \
    MMT4D_FP16_COMP_M16N32_M2_IMPL(_A0, _A1, _B0, _B1, _B2, _B3)

#define MMT4D_FP16_COMP_M16N32_M3_IMPL(A0, A1, B0, B1, B2, B3)                                  \
    "vle16.v            v2, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v3, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vfwmadot           v16, v4, v8                             \n\t"                           \
    "vle16.v            v15, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v5, v8                             \n\t"                           \
    "vfwmadot           v18, v4, v9                             \n\t"                           \
    "vle16.v            v8, (" B0 ")                            \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v26, v5, v9                             \n\t"                           \
    "vfwmadot           v20, v4, v10                            \n\t"                           \
    "vle16.v            v9, (" B1 ")                            \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v28, v5, v10                            \n\t"                           \
    "vfwmadot           v22, v4, v11                            \n\t"                           \
    "vle16.v            v10, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v30, v5, v11                            \n\t"

#define MMT4D_FP16_COMP_M16N32_M3                                                               \
    MMT4D_FP16_COMP_M16N32_M3_IMPL(_A0, _A1, _B0, _B1, _B2, _B3)

#define MMT4D_FP16_COMP_M16N32_M4_IMPL(A0, A1, B0, B1, B2, B3)                                  \
    "vle16.v            v4, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v5, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vfwmadot           v16, v6, v12                            \n\t"                           \
    "vle16.v            v11, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v7, v12                            \n\t"                           \
    "vfwmadot           v18, v6, v13                            \n\t"                           \
    "vle16.v            v12, (" B0 ")                           \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v26, v7, v13                            \n\t"                           \
    "vfwmadot           v20, v6, v14                            \n\t"                           \
    "vle16.v            v13, (" B1 ")                           \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v28, v7, v14                            \n\t"                           \
    "vfwmadot           v22, v6, v15                            \n\t"                           \
    "vle16.v            v14, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v30, v7, v15                            \n\t"

#define MMT4D_FP16_COMP_M16N32_M4                                                               \
    MMT4D_FP16_COMP_M16N32_M4_IMPL(_A0, _A1, _B0, _B1, _B2, _B3)

#define MMT4D_FP16_COMP_M16N32_E2_IMPL(B0, B1, B2, B3)                                          \
    "vfwmadot           v16, v2, v12                            \n\t"                           \
    "vle16.v            v11, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v3, v12                            \n\t"                           \
    "vfwmadot           v18, v2, v13                            \n\t"                           \
    "vle16.v            v12, (" B0 ")                           \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v26, v3, v13                            \n\t"                           \
    "vfwmadot           v20, v2, v14                            \n\t"                           \
    "vle16.v            v13, (" B1 ")                           \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v28, v3, v14                            \n\t"                           \
    "vfwmadot           v22, v2, v15                            \n\t"                           \
    "vle16.v            v14, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v30, v3, v15                            \n\t"

#define MMT4D_FP16_COMP_M16N32_E2                                                               \
    MMT4D_FP16_COMP_M16N32_E2_IMPL(_B0, _B1, _B2, _B3)

#define MMT4D_FP16_COMP_M16N32_E3_IMPL(B3)                                                      \
    "vfwmadot           v16, v4, v8                             \n\t"                           \
    "vle16.v            v15, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v5, v8                             \n\t"                           \
    "vfwmadot           v18, v4, v9                             \n\t"                           \
    "vfwmadot           v26, v5, v9                             \n\t"                           \
    "vfwmadot           v20, v4, v10                            \n\t"                           \
    "vfwmadot           v28, v5, v10                            \n\t"                           \
    "vfwmadot           v22, v4, v11                            \n\t"                           \
    "vfwmadot           v30, v5, v11                            \n\t"

#define MMT4D_FP16_COMP_M16N32_E3                                                               \
    MMT4D_FP16_COMP_M16N32_E3_IMPL(_B3)

#define MMT4D_FP16_COMP_M16N32_E4                                                               \
    "vfwmadot           v16, v6, v12                            \n\t"                           \
    "vfwmadot           v24, v7, v12                            \n\t"                           \
    "vfwmadot           v18, v6, v13                            \n\t"                           \
    "vfwmadot           v26, v7, v13                            \n\t"                           \
    "vfwmadot           v20, v6, v14                            \n\t"                           \
    "vfwmadot           v28, v7, v14                            \n\t"                           \
    "vfwmadot           v22, v6, v15                            \n\t"                           \
    "vfwmadot           v30, v7, v15                            \n\t"

#define MMT4D_FP16_UNPACK_M16N32                                                                \
    "vpack.vv           v0, v16, v18, 2                         \n\t"                           \
    "vpack.vv           v2, v20, v22, 2                         \n\t"                           \
    "vpack.vv           v4, v17, v19, 2                         \n\t"                           \
    "vpack.vv           v6, v21, v23, 2                         \n\t"                           \
    "vpack.vv           v8, v24, v26, 2                         \n\t"                           \
    "vpack.vv           v10, v28, v30, 2                        \n\t"                           \
    "vpack.vv           v12, v25, v27, 2                        \n\t"                           \
    "vpack.vv           v14, v29, v31, 2                        \n\t"                           \
    "vpack.vv           v16, v0, v2, 3                          \n\t"                           \
    "vpack.vv           v18, v1, v3, 3                          \n\t"                           \
    "vpack.vv           v20, v4, v6, 3                          \n\t"                           \
    "vpack.vv           v22, v5, v7, 3                          \n\t"                           \
    "vpack.vv           v24, v8, v10, 3                         \n\t"                           \
    "vpack.vv           v26, v9, v11, 3                         \n\t"                           \
    "vpack.vv           v28, v12, v14, 3                        \n\t"                           \
    "vpack.vv           v30, v13, v15, 3                        \n\t"

#define MMT4D_FP16_CONVERT_M16N32                                                               \
    "vfncvt.f.f.w       v16, v16                                \n\t"                           \
    "vfncvt.f.f.w       v17, v17                                \n\t"                           \
    "vfncvt.f.f.w       v18, v18                                \n\t"                           \
    "vfncvt.f.f.w       v19, v19                                \n\t"                           \
    "vfncvt.f.f.w       v20, v20                                \n\t"                           \
    "vfncvt.f.f.w       v21, v21                                \n\t"                           \
    "vfncvt.f.f.w       v22, v22                                \n\t"                           \
    "vfncvt.f.f.w       v23, v23                                \n\t"                           \
    "vfncvt.f.f.w       v24, v24                                \n\t"                           \
    "vfncvt.f.f.w       v25, v25                                \n\t"                           \
    "vfncvt.f.f.w       v26, v26                                \n\t"                           \
    "vfncvt.f.f.w       v27, v27                                \n\t"                           \
    "vfncvt.f.f.w       v28, v28                                \n\t"                           \
    "vfncvt.f.f.w       v29, v29                                \n\t"                           \
    "vfncvt.f.f.w       v30, v30                                \n\t"                           \
    "vfncvt.f.f.w       v31, v31                                \n\t"

#define MMT4D_FP16_SAVING_M16_IMPL(C, CS)                                                       \
    "vse16.v            v16, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v17, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v18, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v19, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v20, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v21, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v22, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v23, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v24, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v25, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v26, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v27, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v28, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v29, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v30, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v31, (" C ")                            \n\t"

#define MMT4D_FP16_SAVING_M16                                                                   \
    MMT4D_FP16_SAVING_M16_IMPL(_C, _CS)

#define MMT4D_FP16_SAVING_LT16_IMPL(C, CS, M)                                                   \
    "vse16.v            v16, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v17, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v18, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v19, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v20, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v21, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v22, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v23, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v24, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v25, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v26, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v27, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v28, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v29, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v30, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
    "vse16.v            v31, (" C ")                            \n\t"                           \
    "jal                x0, END%=                               \n\t"

#define MMT4D_FP16_SAVING_LT16                                                                  \
    MMT4D_FP16_SAVING_LT16_IMPL(_C, _CS, _M)

// clang-format on

SPINE_FORCE_INLINE void _spe_pack_cast_q41_fixed8_f16_m32k32(const void * in,
                                                             void *       out,
                                                             int32_t      zero_point,
                                                             int64_t      blocks) {
    // flagos: our block is [32 scales][32 x 16 nibble bytes], the fork's [nibbles][zp][scales]; in points at the
    // nibbles, so the scales are 64 bytes before it and the next block 576 bytes on
    constexpr int64_t q4_0_blk_bytes    = 32 * 2 + 32 * 16;
    constexpr int64_t q4_0_scale_offset = -(32 * 2);
    static_assert(sizeof(_spe_fp16_) == 2);  // flagos: no block_type, no uint4x2_t
    if (blocks == 0) {
        return;
    }
    const auto * src = reinterpret_cast<const uint8_t *>(in);
    auto *       dst = reinterpret_cast<uint8_t *>(out);

    // clang-format off
    __asm__ volatile(
        "vsetvli        t0, zero, e16, m1, tu, mu                                \n\t"
        "vid.v          v1                                                       \n\t"
        "li             t0, 31                                                   \n\t"
        "vmsgtu.vx      v0, v1, t0                                               \n\t"
        "vl4re8.v       v4, (%[in])                                              \n\t"
        "LOOP_Q41_BLOCK%=:                                                       \n\t"
        "addi           t1, %[in], %[scale_offset]                               \n\t"
        "vsetvli        t0, zero, e16, m8, tu, mu                                \n\t"
        PACK_CAST_UNPACK_U4_F16_M32K32
        "addi           %[blocks], %[blocks], -1                                 \n\t"
        "beqz           %[blocks], Q41_SCALE_BLOCK%=                             \n\t"
        "addi           %[in], %[in], %[block_bytes]                             \n\t"
        "vl4re8.v       v4, (%[in])                                              \n\t"
        "Q41_SCALE_BLOCK%=:                                                      \n\t"
        "vsetvli        t0, zero, e16, m1, tu, mu                                \n\t"
        PACK_CAST_LOAD_SCALE_F16_M16_LO
        PACK_CAST_SCALE_F16_M16K32_LO
        PACK_CAST_PACK_F16_M16K32_LO
        "mv             t2, %[out]                                               \n\t"
        "addi           t3, t2, 128                                              \n\t"
        PACK_CAST_STORE_F16_M16K32

        PACK_CAST_LOAD_SCALE_F16_M16_HI
        PACK_CAST_SCALE_F16_M16K32_HI
        PACK_CAST_PACK_F16_M16K32_HI
        "addi           t2, %[out], 256                                          \n\t"
        "addi           t3, t2, 128                                              \n\t"
        PACK_CAST_STORE_F16_M16K32
        "addi           %[out], %[out], 1024                                     \n\t"
        "addi           %[out], %[out], 1024                                     \n\t"
        "bnez           %[blocks], LOOP_Q41_BLOCK%=                              \n\t"
        : [in] "+&r"(src), [out] "+&r"(dst), [blocks] "+&r"(blocks)
        : [zero_point] "r"(zero_point), [tile_stride] "i"(32 * 8 * sizeof(_spe_fp16_)),
          [block_bytes] "i"(q4_0_blk_bytes), [scale_offset] "i"(q4_0_scale_offset)  // flagos: our block
        : "cc", "memory", "t0", "t1", "t2", "t3",
          "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
          "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7");
    // clang-format on
}

// One K8 tile is already row-major.

SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_contiguous(const void * in, void * out, int64_t count) {
    // clang-format off
    __asm__ volatile(
        "LOOP_CONTIGUOUS%=:                                                      \n\t"
        "vsetvli        t0, %[count], e16, m4, tu, mu                            \n\t"
        "vle32.v        v0, (%[in])                                              \n\t"
        "sh2add         %[in], t0, %[in]                                         \n\t"
        "sub            %[count], %[count], t0                                   \n\t"
        "vfncvt.f.f.w   v8, v0                                                   \n\t"
        "vse16.v        v8, (%[out])                                             \n\t"
        "sh1add         %[out], t0, %[out]                                       \n\t"
        "bnez           %[count], LOOP_CONTIGUOUS%=                              \n\t"
        : [in] "+&r"(in), [out] "+&r"(out), [count] "+&r"(count)
        :
        : "cc", "memory", "t0");
    // clang-format on
}

// Partial or strided rows must retain the output row spacing.

SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_m16k8(const void * in,
                                                     int64_t      in_stride,
                                                     void *       out,
                                                     int64_t      rows,
                                                     int64_t      cols) {
    if (rows < 16 || cols < 8) {
        // clang-format off
        __asm__ volatile(
            "vsetvli        t0, zero, e16, m2, tu, mu                            \n\t"
            "vxor.vv        v4, v4, v4                                           \n\t"
            "vse16.v        v4, (%[out])                                         \n\t"
            :
            : [out] "r"(out)
            : "cc", "memory", "t0");
        // clang-format on
    }
    if (cols == 8 && in_stride == 8 * sizeof(float)) {
        const int64_t count = rows * 8;
        // clang-format off
        __asm__ volatile(
            "vsetvli        t0, %[count], e16, m2, tu, mu                        \n\t"
            "vle32.v        v0, (%[in])                                          \n\t"
            "vfncvt.f.f.w   v4, v0                                               \n\t"
            "vse16.v        v4, (%[out])                                         \n\t"
            :
            : [in] "r"(in), [out] "r"(out), [count] "r"(count)
            : "cc", "memory", "t0");
        // clang-format on
        return;
    }
    // clang-format off
    __asm__ volatile(
        "vsetvli        t0, %[cols], e16, m1, tu, mu                             \n\t"
        "LOOP_ROW%=:                                                             \n\t"
        "vle32.v        v0, (%[in])                                              \n\t"
        "add            %[in], %[in], %[stride]                                  \n\t"
        "addi           %[rows], %[rows], -1                                     \n\t"
        "vfncvt.f.f.w   v2, v0                                                   \n\t"
        "vse16.v        v2, (%[out])                                             \n\t"
        "addi           %[out], %[out], 16                                       \n\t"
        "bnez           %[rows], LOOP_ROW%=                                      \n\t"
        : [in] "+&r"(in), [out] "+&r"(out), [rows] "+&r"(rows)
        : [stride] "r"(in_stride), [cols] "r"(cols)
        : "cc", "memory", "t0");
    // clang-format on
}

// VLEN=1024; write 16 rows with dst_mb-row spacing between K8 tiles.

template <size_t dst_mb = 16>
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_m16k64(const void * in, int64_t in_stride, void * out, int64_t blocks) {
    static_assert(dst_mb == 16 || dst_mb == 32);
    constexpr int64_t kTileCols        = 8;
    constexpr int64_t kPanelCols       = 64;
    constexpr int64_t kHalfTileRows    = 8;
    constexpr int64_t kInputPanelBytes = kPanelCols * sizeof(float);
    constexpr int64_t kHalfTileBytes   = kHalfTileRows * kTileCols * sizeof(_spe_fp16_);
    constexpr int64_t kTileStride      = dst_mb * kTileCols * sizeof(_spe_fp16_);
    // clang-format off
    __asm__ volatile(
        "vsetvli        t1, zero, e16, m1, tu, mu                                \n\t"
        "LOOP_K%=:                                                               \n\t"
        "mv             t0, %[in]                                                \n\t"
        "add            t1, t0, %[stride]                                        \n\t"
        "sh1add         t5, %[stride], t0                                        \n\t"
        "add            t6, t5, %[stride]                                        \n\t"
        PACK_CAST_LOAD_F32_M8K64
        "sh2add         t0, %[stride], t0                                        \n\t"
        "sh2add         t1, %[stride], t1                                        \n\t"
        "sh2add         t5, %[stride], t5                                        \n\t"
        "sh2add         t6, %[stride], t6                                        \n\t"
        PACK_CAST_NARROW_F32_F16_M8_LO
        PACK_CAST_LOAD_F32_M8K64
        PACK_CAST_NARROW_F32_F16_M8_HI
        PACK_X16_M16N8
        "mv             t0, %[out]                                               \n\t"
        "addi           t1, t0, %[half_tile_bytes]                               \n\t"
        PACK_CAST_STORE_F16_M16K64
        "addi           %[in], %[in], %[input_panel_bytes]                       \n\t"
        "addi           %[out], t0, %[tile_stride]                               \n\t"
        "addi           %[blocks], %[blocks], -1                                 \n\t"
        "bnez           %[blocks], LOOP_K%=                                      \n\t"
        : [in] "+&r"(in), [out] "+&r"(out), [blocks] "+&r"(blocks)
        : [stride] "r"(in_stride), [tile_stride] "i"(kTileStride), [input_panel_bytes] "i"(kInputPanelBytes),
          [half_tile_bytes] "i"(kHalfTileBytes)
        : "cc", "memory", "t0", "t1", "t5", "t6");
    // clang-format on
}

// VLEN=1024; pack complete Rows x K64 panels into tightly packed [M/dst_mb][K/8][dst_mb][8].

template <size_t dst_mb, size_t Rows>
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_m8k64_rows(const void * in,
                                                          int64_t      in_stride,
                                                          void *       out,
                                                          int64_t      k_blocks) {
    static_assert(dst_mb == 16 || dst_mb == 32);
    static_assert(Rows == 64 || Rows == 128);
    if (k_blocks == 0) {
        return;
    }
    constexpr int64_t kBatchRows                 = 8;
    constexpr int64_t kTileCols                  = 8;
    constexpr int64_t kPanelCols                 = 64;
    constexpr int64_t kOutputElementBytes        = sizeof(_spe_fp16_);
    constexpr int64_t kInputPanelBytes           = kPanelCols * sizeof(float);
    constexpr int64_t kOutputBatchBytes          = kBatchRows * kTileCols * kOutputElementBytes;
    constexpr int64_t kOutputPanelRemainderBytes = (kPanelCols - kTileCols) * kOutputElementBytes;
    constexpr int64_t kTileStride                = dst_mb * kTileCols * kOutputElementBytes;
    int64_t           remaining                  = k_blocks * (Rows / kBatchRows);
    const int64_t     k                          = k_blocks * kPanelCols;

    const int64_t     in_col_fix                 = kInputPanelBytes - int64_t(Rows) * in_stride;
    const int64_t     out_mb_advance             = int64_t(dst_mb) * (k - kTileCols) * kOutputElementBytes;
    const int64_t     out_col_fix =
        int64_t(dst_mb) * kOutputPanelRemainderBytes - int64_t(Rows - dst_mb) * k * kOutputElementBytes;
    // clang-format off
    __asm__ volatile(
        "vsetvli        t1, zero, e16, m1, tu, mu                                \n\t"

        "li             t3, %[group_segments]                                    \n\t"
        "li             t4, %[mb_segments]                                       \n\t"
        "mv             t0, %[in]                                                \n\t"
        "add            t1, t0, %[stride]                                        \n\t"
        "sh1add         t5, %[stride], t0                                        \n\t"
        "add            t6, t5, %[stride]                                        \n\t"

        PACK_CAST_LOAD_F32_M8K64
        PACK_CAST_NARROW_F32_F16_M8_LO
        "LOOP_GROUPED%=:                                                         \n\t"
        "mv             t2, %[out]                                               \n\t"
        "addi           %[remaining], %[remaining], -1                           \n\t"
        "beqz           %[remaining], PACK_GROUPED%=                             \n\t"
        "sh2add         t0, %[stride], t0                                        \n\t"
        "sh2add         t1, %[stride], t1                                        \n\t"
        "sh2add         t5, %[stride], t5                                        \n\t"
        "sh2add         t6, %[stride], t6                                        \n\t"
        "addi           %[out], %[out], %[batch_out_bytes]                       \n\t"
        "addi           t3, t3, -1                                               \n\t"
        "addi           t4, t4, -1                                               \n\t"
        "beqz           t3, COLUMN_GROUPED%=                                     \n\t"
        "bnez           t4, LOAD_GROUPED%=                                       \n\t"
        "add            %[out], %[out], %[out_mb_advance]                        \n\t"
        "li             t4, %[mb_segments]                                       \n\t"
        "j              LOAD_GROUPED%=                                           \n\t"
        "COLUMN_GROUPED%=:                                                       \n\t"
        "add            t0, t0, %[in_col_fix]                                    \n\t"
        "add            t1, t1, %[in_col_fix]                                    \n\t"
        "add            t5, t5, %[in_col_fix]                                    \n\t"
        "add            t6, t6, %[in_col_fix]                                    \n\t"
        "add            %[out], %[out], %[out_col_fix]                           \n\t"
        "li             t3, %[group_segments]                                    \n\t"
        "li             t4, %[mb_segments]                                       \n\t"

        "LOAD_GROUPED%=:                                                         \n\t"
        PACK_CAST_LOAD_F32_M8K64
        "PACK_GROUPED%=:                                                         \n\t"
        PACK_CAST_PACK_F16_M8K64
        PACK_CAST_STORE_F16_M8K64

        "beqz           %[remaining], END_GROUPED%=                              \n\t"
        PACK_CAST_NARROW_F32_F16_M8_LO
        "j              LOOP_GROUPED%=                                           \n\t"
        "END_GROUPED%=:                                                          \n\t"
        : [out] "+&r"(out), [remaining] "+&r"(remaining)
        : [in] "r"(in), [stride] "r"(in_stride), [group_segments] "i"(Rows / kBatchRows),
          [mb_segments] "i"(dst_mb / kBatchRows), [in_col_fix] "r"(in_col_fix), [out_mb_advance] "r"(out_mb_advance),
          [out_col_fix] "r"(out_col_fix), [tile_stride] "i"(kTileStride), [batch_out_bytes] "i"(kOutputBatchBytes)
        : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6");
    // clang-format on
}

// Tail: rows=1..16, cols=1..64; zero-fill inactive lanes before packing.

template <size_t dst_mb = 16>
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_m16k64_tail(const void * in,
                                                           int64_t      in_stride,
                                                           void *       out,
                                                           int64_t      rows,
                                                           int64_t      cols,
                                                           int64_t      tiles) {
    static_assert(dst_mb == 16 || dst_mb == 32);
    constexpr int64_t kTileCols      = 8;
    constexpr int64_t kHalfTileRows  = 8;
    constexpr int64_t kHalfTileBytes = kHalfTileRows * kTileCols * sizeof(_spe_fp16_);
    constexpr int64_t kTileStride    = dst_mb * kTileCols * sizeof(_spe_fp16_);
    // clang-format off
    __asm__ volatile(
        "vsetvli        t0, zero, e16, m8, tu, mu                                \n\t"
        "vxor.vv        v0, v0, v0                                               \n\t"
        "vxor.vv        v8, v8, v8                                               \n\t"
        "vsetvli        t0, %[cols], e16, m1, tu, mu                             \n\t"
        "mv             t0, %[in]                                                \n\t"
        "mv             t2, %[rows]                                              \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v0, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v1, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v2, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v3, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v4, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v5, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v6, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v7, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v8, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v9, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v10, v16                                                 \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v11, v16                                                 \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v12, v16                                                 \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v13, v16                                                 \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v14, v16                                                 \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v15, v16                                                 \n\t"
        "PACK%=:                                                                 \n\t"
        "vsetvli        t0, zero, e16, m1, tu, mu                                \n\t"
        PACK_X16_M16N8
        "mv             t0, %[out]                                               \n\t"
        "addi           t1, t0, %[half_tile_bytes]                               \n\t"
        "mv             t2, %[tiles]                                             \n\t"
        "vse16.v        v16, (t0)                                                \n\t"
        "vse16.v        v24, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v17, (t0)                                                \n\t"
        "vse16.v        v25, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v18, (t0)                                                \n\t"
        "vse16.v        v26, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v19, (t0)                                                \n\t"
        "vse16.v        v27, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v20, (t0)                                                \n\t"
        "vse16.v        v28, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v21, (t0)                                                \n\t"
        "vse16.v        v29, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v22, (t0)                                                \n\t"
        "vse16.v        v30, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v23, (t0)                                                \n\t"
        "vse16.v        v31, (t1)                                                \n\t"
        "END%=:                                                                  \n\t"
        :
        : [in] "r"(in), [stride] "r"(in_stride), [out] "r"(out), [rows] "r"(rows), [cols] "r"(cols), [tiles] "r"(tiles),
          [tile_stride] "i"(kTileStride), [half_tile_bytes] "i"(kHalfTileBytes)
        : "cc", "memory", "t0", "t1", "t2");
    // clang-format on
}

// Pack complete M blocks; crop/pad K to dst_kc K8 tiles.

template <size_t mb>
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_full_blocks(const void * in,
                                                           int64_t      in_stride,
                                                           void *       out,
                                                           int64_t      out_stride,
                                                           int64_t      mc,
                                                           int64_t      k,
                                                           int64_t      dst_kc) {
    static_assert(mb == 16 || mb == 32);
    constexpr int64_t kb         = 8;
    constexpr int64_t panel_k    = 64;
    constexpr int64_t inner_rows = 16;
    const int64_t     rows       = mc * mb;
    if (k == kb && dst_kc == 1 && in_stride == kb * sizeof(float) && out_stride == mb * kb * sizeof(_spe_fp16_)) {
        _spe_pack_cast_f32_f16_contiguous(in, out, rows * kb);
        return;
    }

    auto *  src      = reinterpret_cast<const int8_t *>(in);
    auto *  dst      = reinterpret_cast<int8_t *>(out);
    int64_t first_mo = 0;

    if (k % panel_k == 0 && dst_kc == k / kb && in_stride == k * sizeof(float) &&
        out_stride == k * mb * sizeof(_spe_fp16_)) {
        int64_t row = 0;
        for (; row + 128 <= rows; row += 128) {
            _spe_pack_cast_f32_f16_m8k64_rows<mb, 128>(src + row * in_stride, in_stride, dst + (row / mb) * out_stride,
                                                       k / panel_k);
        }
        if (row + 64 <= rows) {
            _spe_pack_cast_f32_f16_m8k64_rows<mb, 64>(src + row * in_stride, in_stride, dst + (row / mb) * out_stride,
                                                      k / panel_k);
            row += 64;
        }
        first_mo = row / mb;
    }

    const int64_t panels      = k / panel_k;
    const int64_t col         = panels * panel_k;
    const int64_t cols        = k - col;
    const int64_t valid_tiles = (k + kb - 1) / kb;
    for (int64_t mo = first_mo; mo < mc; ++mo) {
        auto * src_block = src + mo * mb * in_stride;
        auto * dst_block = reinterpret_cast<_spe_fp16_ *>(dst + mo * out_stride);
        // MB32 consists of two M16 halves sharing the MB32 tile stride.
        for (int64_t mi = 0; mi < mb; mi += inner_rows) {
            auto * src_half = src_block + mi * in_stride;
            auto * dst_half = dst_block + mi * kb;
            if (panels != 0) {
                _spe_pack_cast_f32_f16_m16k64<mb>(src_half, in_stride, dst_half, panels);
            }
            if (cols != 0) {
                if (cols <= kb) {
                    _spe_pack_cast_f32_f16_m16k8(src_half + col * sizeof(float), in_stride, dst_half + col * mb,
                                                 inner_rows, cols);
                } else {
                    _spe_pack_cast_f32_f16_m16k64_tail<mb>(src_half + col * sizeof(float), in_stride,
                                                           dst_half + col * mb, inner_rows, cols, (cols + kb - 1) / kb);
                }
            }
        }
        std::fill_n(dst_block + valid_tiles * mb * kb, (dst_kc - valid_tiles) * mb * kb, static_cast<_spe_fp16_>(0.0f));
    }
}

// Successive K8 tiles are mb*8 elements apart; rows are not contiguous.

template <size_t mb>
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_row(const void * in, void * out, int64_t k, int64_t dst_kc) {
    static_assert(mb == 16 || mb == 32);
    constexpr int64_t kb  = 8;
    auto *            src = reinterpret_cast<const float *>(in);
    auto *            dst = reinterpret_cast<_spe_fp16_ *>(out);
    for (int64_t ko = 0; ko < dst_kc; ++ko) {
        const int64_t cols     = std::max<int64_t>(0, std::min<int64_t>(kb, k - ko * kb));
        auto *        dst_tile = dst + ko * mb * kb;
        if (cols != 0) {
            _spe_pack_cast_f32_f16_contiguous(src + ko * kb, dst_tile, cols);
        }
        std::fill_n(dst_tile + cols, kb - cols, static_cast<_spe_fp16_>(0.0f));
    }
}

template <size_t mb> SPINE_FORCE_INLINE void _spe_pack_cast_f16_pad_row(void * out, int64_t dst_kc) {
    static_assert(mb == 16 || mb == 32);
    constexpr int64_t kb  = 8;
    auto *            dst = reinterpret_cast<_spe_fp16_ *>(out);
    for (int64_t ko = 0; ko < dst_kc; ++ko) {
        std::fill_n(dst + ko * mb * kb, kb, static_cast<_spe_fp16_>(0.0f));
    }
}

// Zero logical output elements; preserve gaps between M blocks.
template <size_t mb>
SPINE_FORCE_INLINE void _spe_pack_cast_f16_pad_blocks(void * out, int64_t out_stride, int64_t mc, int64_t dst_kc) {
    static_assert(mb == 16 || mb == 32);
    constexpr int64_t kb  = 8;
    auto *            dst = reinterpret_cast<int8_t *>(out);
    for (int64_t mo = 0; mo < mc; ++mo) {
        std::fill_n(reinterpret_cast<_spe_fp16_ *>(dst + mo * out_stride), dst_kc * mb * kb,
                    static_cast<_spe_fp16_>(0.0f));
    }
}

// GEMM: m=1..16, n=1..32, kc counts K8 groups; c_stride is in bytes.
// Bias may be null; non-HP epilogues take FP32 bias.
SPINE_FORCE_INLINE void _spe_hpgemm_impl_m16n32k8_none(size_t       m,
                                                       size_t       n,
                                                       size_t       kc,
                                                       const void * a,
                                                       const void * b,
                                                       int8_t *     c,
                                                       size_t       c_stride) {
    __asm__ volatile(
        // clang-format off
        MMT4D_FP16_M16N32K8_INIT

        MMT4D_FP16_M16N32K8_COMPUTE_LOOP

        MMT4D_FP16_M16N32K8_POSTPROCESS_NONE

        MMT4D_FP16_M16N32K8_STORE
        // clang-format on
        : [c] "+r"(c), [m] "+r"(m)
        : [a] "r"(a), [b] "r"(b), [n] "r"(n), [kc] "r"(kc), [cs] "r"(c_stride)
        : "memory", "cc", "t0", "t2", "a1", "a2", "s1", "s2", "s3", "s4");
}

// Split stride stays 32 FP16 elements even when vl < 32.
SPINE_FORCE_INLINE void _spe_reducesum_mxn_f16_f32_16x32_1x1(const void * in, void * out, int64_t n_size, int64_t nth) {

    constexpr int64_t src_nb = 32;
    __asm__ volatile(
        "L_N%=:                                               \n\t"
        "vsetvli                t0, %[nsize], e32, m1, tu, mu \n\t"
        "vxor.vv                v8, v8, v8                    \n\t"
        "vsetvli                zero, t0, e16, mf2, tu, mu    \n\t"
        "mv                     t1, %[nth]                    \n\t"
        "mv                     a1, %[in]                     \n\t"

        "L_THREADS%=:                                         \n\t"
        "vle16.v                v0, (a1)                      \n\t"
        "vfwadd.wv              v8, v8, v0                    \n\t"
        "sh1add                 a1, %[snb], a1                \n\t"
        "addi                   t1, t1, -1                    \n\t"
        "bnez                   t1, L_THREADS%=               \n\t"

        "vsetvli                zero, t0, e32, m1, tu, mu     \n\t"
        "vse32.v                v8, (%[out])                  \n\t"
        "sh1add                 %[in], t0, %[in]              \n\t"
        "sh2add                 %[out], t0, %[out]            \n\t"
        "sub                    %[nsize], %[nsize], t0        \n\t"
        "bnez                   %[nsize], L_N%=               \n\t"
        : [in] "+&r"(in), [out] "+&r"(out), [nsize] "+&r"(n_size)
        : [nth] "r"(nth), [snb] "r"(src_nb)
        : "cc", "memory", "t0", "t1", "a1");
}

// VLEN=1024; input=[16,8,32], out_stride is in bytes; preserve t=0..7 summation order.
// Widen all inputs before reuse and restore VL=32 before clearing registers.
SPINE_FORCE_INLINE void _spe_reducesum_mxn_f16_f32_16x32_1x1_a100_nth8(const void * in,
                                                                       void *       out,
                                                                       int64_t      out_stride) {
    constexpr int64_t src_nb       = 32;
    constexpr int64_t load_nb      = 2 * src_nb;
    constexpr int64_t row_elements = 8 * src_nb;
    __asm__ volatile(
        "li                     t1, 8                         \n\t"
        "L_PAIR%=:                                            \n\t"
        "mv                     a1, %[in]                     \n\t"
        "sh1add                 a2, %[row_elements], %[in]    \n\t"
        "vsetvli                zero, %[lnb], e16, m1, tu, mu \n\t"
        "vle16.v                v0, (a1)                      \n\t"
        "sh2add                 a1, %[snb], a1                \n\t"
        "vle16.v                v4, (a2)                      \n\t"
        "sh2add                 a2, %[snb], a2                \n\t"
        "vle16.v                v1, (a1)                      \n\t"
        "sh2add                 a1, %[snb], a1                \n\t"
        "vle16.v                v5, (a2)                      \n\t"
        "sh2add                 a2, %[snb], a2                \n\t"
        "vle16.v                v2, (a1)                      \n\t"
        "sh2add                 a1, %[snb], a1                \n\t"
        "vle16.v                v6, (a2)                      \n\t"
        "sh2add                 a2, %[snb], a2                \n\t"
        "vle16.v                v3, (a1)                      \n\t"
        "sh2add                 a1, %[snb], a1                \n\t"
        "vle16.v                v7, (a2)                      \n\t"
        "sh2add                 a2, %[snb], a2                \n\t"
        "vfwcvt.f.f.v           v16, v0                       \n\t"
        "vfwcvt.f.f.v           v24, v4                       \n\t"
        "vfwcvt.f.f.v           v18, v1                       \n\t"
        "vfwcvt.f.f.v           v26, v5                       \n\t"
        "vfwcvt.f.f.v           v20, v2                       \n\t"
        "vfwcvt.f.f.v           v28, v6                       \n\t"
        "vfwcvt.f.f.v           v22, v3                       \n\t"
        "vfwcvt.f.f.v           v30, v7                       \n\t"
        "vsetvli                zero, %[snb], e32, m1, tu, mu \n\t"
        "vxor.vv                v0, v0, v0                    \n\t"
        "vxor.vv                v1, v1, v1                    \n\t"
        "vfadd.vv               v0, v0, v16                   \n\t"
        "vfadd.vv               v1, v1, v24                   \n\t"
        "vfadd.vv               v0, v0, v17                   \n\t"
        "vfadd.vv               v1, v1, v25                   \n\t"
        "vfadd.vv               v0, v0, v18                   \n\t"
        "vfadd.vv               v1, v1, v26                   \n\t"
        "vfadd.vv               v0, v0, v19                   \n\t"
        "vfadd.vv               v1, v1, v27                   \n\t"
        "vfadd.vv               v0, v0, v20                   \n\t"
        "vfadd.vv               v1, v1, v28                   \n\t"
        "vfadd.vv               v0, v0, v21                   \n\t"
        "vfadd.vv               v1, v1, v29                   \n\t"
        "vfadd.vv               v0, v0, v22                   \n\t"
        "vfadd.vv               v1, v1, v30                   \n\t"
        "vfadd.vv               v0, v0, v23                   \n\t"
        "vfadd.vv               v1, v1, v31                   \n\t"
        "vse32.v                v0, (%[out])                  \n\t"
        "add                    a1, %[out], %[os]             \n\t"
        "vse32.v                v1, (a1)                      \n\t"
        "sh1add                 %[out], %[os], %[out]         \n\t"
        "mv                     %[in], a2                     \n\t"
        "addi                   t1, t1, -1                    \n\t"
        "bnez                   t1, L_PAIR%=                  \n\t"
        : [in] "+&r"(in), [out] "+&r"(out)
        : [snb] "r"(src_nb), [lnb] "r"(load_nb), [row_elements] "r"(row_elements), [os] "r"(out_stride)
        : "cc", "memory", "t1", "a1", "a2");
}

// FP32 [M,K] -> FP16 [dst_mc,dst_kc,16,8]; offsets and strides are bytes.
// flagos: spe_pack_cast<float, _spe_fp16_, 1, 1, 16, 8> as a plain function, the helpers without ime2::
static void spe_pack_cast_f32_f16_16_8(void *  in,
                                       int64_t in_offset,
                                       int64_t in_stride,
                                       void *  out,
                                       int64_t out_offset,
                                       int64_t out_stride,
                                       int64_t m,
                                       int64_t k,
                                       int64_t dst_mc,
                                       int64_t dst_kc) {
    constexpr int64_t mb = 16;
    constexpr int64_t kb = 8;
    SPE_ASSERT(m >= 0 && k >= 0 && dst_mc >= 0 && dst_kc >= 0);
    if (dst_mc == 0 || dst_kc == 0) {
        return;
    }
    SPE_ASSERT(__riscv_vlenb() == 128);
    SPE_ASSERT(in_stride >= k * static_cast<int64_t>(sizeof(float)));
    SPE_ASSERT(out_stride >= dst_kc * mb * kb * static_cast<int64_t>(sizeof(_spe_fp16_)));
    SPE_ASSERT(out != nullptr);
    SPE_ASSERT(m == 0 || k == 0 || in != nullptr);

    const int64_t valid_m = std::min<int64_t>(m, dst_mc * mb);
    const int64_t valid_k = std::min<int64_t>(k, dst_kc * kb);
    auto *        dst     = reinterpret_cast<int8_t *>(out) + out_offset;
    if (valid_m == 0 || valid_k == 0) {
        _spe_pack_cast_f16_pad_blocks<mb>(dst, out_stride, dst_mc, dst_kc);
        return;
    }
    auto *        src      = reinterpret_cast<const int8_t *>(in) + in_offset;
    const int64_t mc       = valid_m / mb;
    const int64_t m_remain = valid_m % mb;

    if (mc != 0) {
        _spe_pack_cast_f32_f16_full_blocks<mb>(src, in_stride, dst, out_stride, mc, valid_k, dst_kc);
    }

    if (m_remain != 0) {
        auto *  src_tail = src + mc * mb * in_stride;
        auto *  dst_tail = reinterpret_cast<_spe_fp16_ *>(dst + mc * out_stride);
        int64_t mi       = 0;
        for (; mi < m_remain; ++mi) {
            _spe_pack_cast_f32_f16_row<mb>(src_tail + mi * in_stride, dst_tail + mi * kb, valid_k, dst_kc);
        }
        for (; mi < mb; ++mi) {
            _spe_pack_cast_f16_pad_row<mb>(dst_tail + mi * kb, dst_kc);
        }
    }

    const int64_t written_mc = mc + (m_remain != 0);
    if (written_mc < dst_mc) {
        _spe_pack_cast_f16_pad_blocks<mb>(dst + written_mc * out_stride, out_stride, dst_mc - written_mc, dst_kc);
    }
}

// flagos: the call of the fork's driver for one 16-row tile of a slice (ime.cpp 2352-2359)
__attribute__((noinline)) void cast_tile(const float * x, int64_t x_stride_floats, int64_t rows, int64_t ks,
                                         uint16_t * a) {
    spe_pack_cast_f32_f16_16_8(const_cast<float *>(x), 0, x_stride_floats * (int64_t) sizeof(float), a, 0,
                               ks * 16 * (int64_t) sizeof(_spe_fp16_), rows, ks, 1, ks / 8);
}

// flagos: the fork's _spe_pack_cast_q41_fixed8_f16_m32k32 on our block, from its nibbles (w + 64). It converts q - 8,
// an exact fp16 integer in [-8, 7], and multiplies it by the row's d in one fp16 vfmul (frm: to nearest even): the
// exact product rounded once. GGML_FP32_TO_FP16(d * (float) (q - 8)) rounds the same exact product once (11 by 4 bits
// fit in float), so the bits are equal, infinity on overflow included. A zero takes the xor of the two signs in both:
// q = 8 gives +0, so -0 for a negative d; d = -0 gives -0 for q >= 8 and +0 below. A |d| above 8188 overflows here and
// raises OF, which the products step reports
__attribute__((noinline)) void dequant_tile(const uint8_t * w, int64_t ks, uint16_t * b) {
    GGML_ASSERT(ks % 32 == 0);
    SPE_ASSERT(__riscv_vlenb() == 128);
    _spe_pack_cast_q41_fixed8_f16_m32k32(w + 64, b, 8, ks / 32);
}

// flagos: gemm_kernel_f16f16f16_m16_vfwmadot without its accumulate flag (always false there)
__attribute__((noinline)) void products(int64_t ks, const uint16_t * b, const uint16_t * a, uint16_t * c, int64_t rows,
                                        int64_t cols, int64_t c_stride_elems) {
    GGML_ASSERT(rows > 0 && rows <= 16);
    GGML_ASSERT(cols > 0 && cols <= 32);
    GGML_ASSERT(ks % 8 == 0);

    // Align c_ptr to 64 bytes so each FP16 row store stays within one DDR bank.
    _spe_hpgemm_impl_m16n32k8_none((size_t) rows, (size_t) cols, (size_t) (ks / 8), a, b, reinterpret_cast<int8_t *>(c),
                                   (size_t) c_stride_elems * sizeof(_spe_fp16_));
}

// flagos: the block loop of spe_reducesum_mxn<_spe_fp16_, float, 16, 32, 1, 1> for one block (lhs strides 0) and
// nth = slices: the _a100_nth8 form for 8 slices on a full 16 x 32 tile, else the generic kernel per row (the fork's
// pairs, g = 4)
__attribute__((noinline)) void reduce_tile(const uint16_t * partials, float * out, int64_t out_stride_bytes,
                                           int64_t rows, int64_t cols, int64_t slices) {
    const int64_t     nth           = slices;
    constexpr int64_t block_m       = 16;
    constexpr int64_t block_n       = 32;
    constexpr int64_t thread_stride = block_n * sizeof(_spe_fp16_);
    const auto *      in_block      = reinterpret_cast<const int8_t *>(partials);
    auto *            out_block     = reinterpret_cast<int8_t *>(out);
    const int64_t     row_stride    = nth * thread_stride;

    const bool use_a100_preload = nth == 8 && __riscv_vlenb() == 128;
    if (use_a100_preload && rows == block_m && cols == block_n) {
        _spe_reducesum_mxn_f16_f32_16x32_1x1_a100_nth8(in_block, out_block, out_stride_bytes);
        return;
    }
    for (int64_t i = 0; i < rows; ++i) {
        auto * out_row = out_block + i * out_stride_bytes;
        _spe_reducesum_mxn_f16_f32_16x32_1x1(in_block + i * row_stride, out_row, cols, nth);
    }
}

// flagos: new. fflags: NX bit 0, UF 1, OF 2, DZ 3, NV 4
void clear_flags() {
    __asm__ volatile("csrw fflags, zero" : : : "memory");
}

bool overflow_raised() {
    unsigned long flags = 0;
    __asm__ volatile("csrr %0, fflags" : "=r"(flags) : : "memory");
    return (flags & 0x4) != 0;
}

#endif  // GGML_FLAGOS_SPACEMIT_IME2

bool have() {
#if defined(GGML_FLAGOS_SPACEMIT_IME2)
    return true;
#else
    return false;
#endif
}

}  // namespace spacemit_f16
