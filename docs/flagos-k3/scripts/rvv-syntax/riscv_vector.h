// Stand-in for clang's riscv_vector.h, which Apple clang does not ship: the RVV types and vsetvl macros, with the
// intrinsics themselves provided by clang's `#pragma clang riscv intrinsic vector`. For rvv-syntax-check.sh only.
#ifndef __RISCV_VECTOR_H
#define __RISCV_VECTOR_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#pragma clang riscv intrinsic vector
#define __riscv_vlenb() __builtin_rvv_vlenb()
#define __riscv_vsetvl_e8m1(avl) __builtin_rvv_vsetvli((size_t)(avl), 0, 0)
#define __riscv_vsetvlmax_e8m1() __builtin_rvv_vsetvlimax(0, 0)
#define __riscv_vsetvl_e8m2(avl) __builtin_rvv_vsetvli((size_t)(avl), 0, 1)
#define __riscv_vsetvlmax_e8m2() __builtin_rvv_vsetvlimax(0, 1)
#define __riscv_vsetvl_e8m4(avl) __builtin_rvv_vsetvli((size_t)(avl), 0, 2)
#define __riscv_vsetvlmax_e8m4() __builtin_rvv_vsetvlimax(0, 2)
#define __riscv_vsetvl_e8m8(avl) __builtin_rvv_vsetvli((size_t)(avl), 0, 3)
#define __riscv_vsetvlmax_e8m8() __builtin_rvv_vsetvlimax(0, 3)
#define __riscv_vsetvl_e8mf8(avl) __builtin_rvv_vsetvli((size_t)(avl), 0, 5)
#define __riscv_vsetvlmax_e8mf8() __builtin_rvv_vsetvlimax(0, 5)
#define __riscv_vsetvl_e8mf4(avl) __builtin_rvv_vsetvli((size_t)(avl), 0, 6)
#define __riscv_vsetvlmax_e8mf4() __builtin_rvv_vsetvlimax(0, 6)
#define __riscv_vsetvl_e8mf2(avl) __builtin_rvv_vsetvli((size_t)(avl), 0, 7)
#define __riscv_vsetvlmax_e8mf2() __builtin_rvv_vsetvlimax(0, 7)
#define __riscv_vsetvl_e16m1(avl) __builtin_rvv_vsetvli((size_t)(avl), 1, 0)
#define __riscv_vsetvlmax_e16m1() __builtin_rvv_vsetvlimax(1, 0)
#define __riscv_vsetvl_e16m2(avl) __builtin_rvv_vsetvli((size_t)(avl), 1, 1)
#define __riscv_vsetvlmax_e16m2() __builtin_rvv_vsetvlimax(1, 1)
#define __riscv_vsetvl_e16m4(avl) __builtin_rvv_vsetvli((size_t)(avl), 1, 2)
#define __riscv_vsetvlmax_e16m4() __builtin_rvv_vsetvlimax(1, 2)
#define __riscv_vsetvl_e16m8(avl) __builtin_rvv_vsetvli((size_t)(avl), 1, 3)
#define __riscv_vsetvlmax_e16m8() __builtin_rvv_vsetvlimax(1, 3)
#define __riscv_vsetvl_e16mf8(avl) __builtin_rvv_vsetvli((size_t)(avl), 1, 5)
#define __riscv_vsetvlmax_e16mf8() __builtin_rvv_vsetvlimax(1, 5)
#define __riscv_vsetvl_e16mf4(avl) __builtin_rvv_vsetvli((size_t)(avl), 1, 6)
#define __riscv_vsetvlmax_e16mf4() __builtin_rvv_vsetvlimax(1, 6)
#define __riscv_vsetvl_e16mf2(avl) __builtin_rvv_vsetvli((size_t)(avl), 1, 7)
#define __riscv_vsetvlmax_e16mf2() __builtin_rvv_vsetvlimax(1, 7)
#define __riscv_vsetvl_e32m1(avl) __builtin_rvv_vsetvli((size_t)(avl), 2, 0)
#define __riscv_vsetvlmax_e32m1() __builtin_rvv_vsetvlimax(2, 0)
#define __riscv_vsetvl_e32m2(avl) __builtin_rvv_vsetvli((size_t)(avl), 2, 1)
#define __riscv_vsetvlmax_e32m2() __builtin_rvv_vsetvlimax(2, 1)
#define __riscv_vsetvl_e32m4(avl) __builtin_rvv_vsetvli((size_t)(avl), 2, 2)
#define __riscv_vsetvlmax_e32m4() __builtin_rvv_vsetvlimax(2, 2)
#define __riscv_vsetvl_e32m8(avl) __builtin_rvv_vsetvli((size_t)(avl), 2, 3)
#define __riscv_vsetvlmax_e32m8() __builtin_rvv_vsetvlimax(2, 3)
#define __riscv_vsetvl_e32mf8(avl) __builtin_rvv_vsetvli((size_t)(avl), 2, 5)
#define __riscv_vsetvlmax_e32mf8() __builtin_rvv_vsetvlimax(2, 5)
#define __riscv_vsetvl_e32mf4(avl) __builtin_rvv_vsetvli((size_t)(avl), 2, 6)
#define __riscv_vsetvlmax_e32mf4() __builtin_rvv_vsetvlimax(2, 6)
#define __riscv_vsetvl_e32mf2(avl) __builtin_rvv_vsetvli((size_t)(avl), 2, 7)
#define __riscv_vsetvlmax_e32mf2() __builtin_rvv_vsetvlimax(2, 7)
#define __riscv_vsetvl_e64m1(avl) __builtin_rvv_vsetvli((size_t)(avl), 3, 0)
#define __riscv_vsetvlmax_e64m1() __builtin_rvv_vsetvlimax(3, 0)
#define __riscv_vsetvl_e64m2(avl) __builtin_rvv_vsetvli((size_t)(avl), 3, 1)
#define __riscv_vsetvlmax_e64m2() __builtin_rvv_vsetvlimax(3, 1)
#define __riscv_vsetvl_e64m4(avl) __builtin_rvv_vsetvli((size_t)(avl), 3, 2)
#define __riscv_vsetvlmax_e64m4() __builtin_rvv_vsetvlimax(3, 2)
#define __riscv_vsetvl_e64m8(avl) __builtin_rvv_vsetvli((size_t)(avl), 3, 3)
#define __riscv_vsetvlmax_e64m8() __builtin_rvv_vsetvlimax(3, 3)
#define __riscv_vsetvl_e64mf8(avl) __builtin_rvv_vsetvli((size_t)(avl), 3, 5)
#define __riscv_vsetvlmax_e64mf8() __builtin_rvv_vsetvlimax(3, 5)
#define __riscv_vsetvl_e64mf4(avl) __builtin_rvv_vsetvli((size_t)(avl), 3, 6)
#define __riscv_vsetvlmax_e64mf4() __builtin_rvv_vsetvlimax(3, 6)
#define __riscv_vsetvl_e64mf2(avl) __builtin_rvv_vsetvli((size_t)(avl), 3, 7)
#define __riscv_vsetvlmax_e64mf2() __builtin_rvv_vsetvlimax(3, 7)
typedef __rvv_bool1_t vbool1_t;
typedef __rvv_bool2_t vbool2_t;
typedef __rvv_bool4_t vbool4_t;
typedef __rvv_bool8_t vbool8_t;
typedef __rvv_bool16_t vbool16_t;
typedef __rvv_bool32_t vbool32_t;
typedef __rvv_bool64_t vbool64_t;
typedef __rvv_int8mf8_t vint8mf8_t;
typedef __rvv_int8mf4_t vint8mf4_t;
typedef __rvv_int8mf2_t vint8mf2_t;
typedef __rvv_int8m1_t vint8m1_t;
typedef __rvv_int8m2_t vint8m2_t;
typedef __rvv_int8m4_t vint8m4_t;
typedef __rvv_int8m8_t vint8m8_t;
typedef __rvv_int16mf4_t vint16mf4_t;
typedef __rvv_int16mf2_t vint16mf2_t;
typedef __rvv_int16m1_t vint16m1_t;
typedef __rvv_int16m2_t vint16m2_t;
typedef __rvv_int16m4_t vint16m4_t;
typedef __rvv_int16m8_t vint16m8_t;
typedef __rvv_int32mf2_t vint32mf2_t;
typedef __rvv_int32m1_t vint32m1_t;
typedef __rvv_int32m2_t vint32m2_t;
typedef __rvv_int32m4_t vint32m4_t;
typedef __rvv_int32m8_t vint32m8_t;
typedef __rvv_int64m1_t vint64m1_t;
typedef __rvv_int64m2_t vint64m2_t;
typedef __rvv_int64m4_t vint64m4_t;
typedef __rvv_int64m8_t vint64m8_t;
typedef __rvv_uint8mf8_t vuint8mf8_t;
typedef __rvv_uint8mf4_t vuint8mf4_t;
typedef __rvv_uint8mf2_t vuint8mf2_t;
typedef __rvv_uint8m1_t vuint8m1_t;
typedef __rvv_uint8m2_t vuint8m2_t;
typedef __rvv_uint8m4_t vuint8m4_t;
typedef __rvv_uint8m8_t vuint8m8_t;
typedef __rvv_uint16mf4_t vuint16mf4_t;
typedef __rvv_uint16mf2_t vuint16mf2_t;
typedef __rvv_uint16m1_t vuint16m1_t;
typedef __rvv_uint16m2_t vuint16m2_t;
typedef __rvv_uint16m4_t vuint16m4_t;
typedef __rvv_uint16m8_t vuint16m8_t;
typedef __rvv_uint32mf2_t vuint32mf2_t;
typedef __rvv_uint32m1_t vuint32m1_t;
typedef __rvv_uint32m2_t vuint32m2_t;
typedef __rvv_uint32m4_t vuint32m4_t;
typedef __rvv_uint32m8_t vuint32m8_t;
typedef __rvv_uint64m1_t vuint64m1_t;
typedef __rvv_uint64m2_t vuint64m2_t;
typedef __rvv_uint64m4_t vuint64m4_t;
typedef __rvv_uint64m8_t vuint64m8_t;
typedef __rvv_float16mf4_t vfloat16mf4_t;
typedef __rvv_float16mf2_t vfloat16mf2_t;
typedef __rvv_float16m1_t vfloat16m1_t;
typedef __rvv_float16m2_t vfloat16m2_t;
typedef __rvv_float16m4_t vfloat16m4_t;
typedef __rvv_float16m8_t vfloat16m8_t;
typedef __rvv_float32mf2_t vfloat32mf2_t;
typedef __rvv_float32m1_t vfloat32m1_t;
typedef __rvv_float32m2_t vfloat32m2_t;
typedef __rvv_float32m4_t vfloat32m4_t;
typedef __rvv_float32m8_t vfloat32m8_t;
typedef __rvv_float64m1_t vfloat64m1_t;
typedef __rvv_float64m2_t vfloat64m2_t;
typedef __rvv_float64m4_t vfloat64m4_t;
typedef __rvv_float64m8_t vfloat64m8_t;
#ifdef __cplusplus
}
#endif
#endif
