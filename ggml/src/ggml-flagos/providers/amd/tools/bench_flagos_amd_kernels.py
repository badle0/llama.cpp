#!/usr/bin/env python3
"""Small gfx1150 microbenchmarks for candidate FlagOS kernels.

This is intentionally independent of the AOT packager: it lets us compare
Triton launch shapes before changing the package or llama.cpp selector.  The
layout matches the AMD provider ABI: weights are row-major [rows, k], inputs
are contiguous GGML columns [columns, k], and outputs are [columns, rows].
"""

import argparse
import statistics
import time
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

QK_K = 256
KERNEL_DIR = Path(__file__).resolve().parents[3] / "kernels"
sys.path.insert(0, str(KERNEL_DIR))
import generate_flagos_kernels as common  # noqa: E402
import generate_flagos_amd_kernels as amd_kernels  # noqa: E402


def make_q40_weights(rows: int, blocks: int, seed: int,
                     with_reference: bool = False):
    """Build valid Q4_0 bytes without a production-shape Python scalar loop."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    packed = torch.zeros(
        (rows, blocks, common.Q4_0_BLOCK_BYTES), dtype=torch.uint8)
    scales = torch.rand(
        (rows, blocks), generator=generator, dtype=torch.float32).mul_(0.1).half()
    packed[:, :, :2] = scales.view(torch.uint8).reshape(rows, blocks, 2)
    quant = torch.randint(
        0, 256, (rows, blocks, common.Q4_0_BLOCK_BYTES - 2),
        generator=generator, dtype=torch.uint8)
    packed[:, :, 2:] = quant
    packed_device = packed.reshape(-1).cuda()
    if not with_reference:
        return packed_device, None
    low = (quant & 15).to(torch.float32) - 8.0
    high = (quant >> 4).to(torch.float32) - 8.0
    dequantized = torch.cat((low, high), dim=2) * scales.float().unsqueeze(2)
    return packed_device, dequantized.reshape(rows, blocks * common.QK4_0).cuda()


@triton.jit
def f16_gemm(
    weights, x, output, k, rows, columns,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, k, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            weights + row[:, None] * k + kk[None, :],
            mask=(row[:, None] < rows) & (kk[None, :] < k), other=0.0,
        ).to(tl.float16)
        b = tl.load(
            x + col[None, :] * k + kk[:, None],
            mask=(col[None, :] < columns) & (kk[:, None] < k), other=0.0,
        ).to(tl.float16)
        acc += tl.dot(a, b)
    tl.store(
        output + col[None, :] * rows + row[:, None], acc,
        mask=(row[:, None] < rows) & (col[None, :] < columns),
    )


@triton.jit
def f16_gemm_grouped(
    weights, x, output, k, rows, columns,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """AMD tutorial-style grouped dense GEMM for prefill retuning."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(rows, BLOCK_M)
    num_pid_n = tl.cdiv(columns, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, k, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            weights + row[:, None] * k + kk[None, :],
            mask=(row[:, None] < rows) & (kk[None, :] < k), other=0.0,
        ).to(tl.float16)
        b = tl.load(
            x + col[None, :] * k + kk[:, None],
            mask=(col[None, :] < columns) & (kk[:, None] < k), other=0.0,
        ).to(tl.float16)
        acc += tl.dot(a, b)
    tl.store(
        output + col[None, :] * rows + row[:, None], acc,
        mask=(row[:, None] < rows) & (col[None, :] < columns),
    )


@triton.jit
def ffn_swiglu(
    gate_weights, up_weights, x, output, k, rows, columns,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, k, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        row_mask = row[:, None] < rows
        k_mask = kk[None, :] < k
        gate = tl.load(
            gate_weights + row[:, None] * k + kk[None, :],
            mask=row_mask & k_mask, other=0.0,
        ).to(tl.float16)
        up = tl.load(
            up_weights + row[:, None] * k + kk[None, :],
            mask=row_mask & k_mask, other=0.0,
        ).to(tl.float16)
        activation = tl.load(
            x + col[None, :] * k + kk[:, None],
            mask=(col[None, :] < columns) & (kk[:, None] < k), other=0.0,
        ).to(tl.float16)
        gate_acc += tl.dot(gate, activation)
        up_acc += tl.dot(up, activation)
    result = gate_acc * tl.sigmoid(gate_acc) * up_acc
    tl.store(
        output + col[None, :] * rows + row[:, None], result,
        mask=(row[:, None] < rows) & (col[None, :] < columns),
    )


@triton.jit
def ffn_swiglu_grouped(
    gate_weights, up_weights, x, output, k, rows, columns,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Grouped-schedule variant of the fused dual-projection SwiGLU."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(rows, BLOCK_M)
    num_pid_n = tl.cdiv(columns, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, k, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        row_mask = row[:, None] < rows
        k_mask = kk[None, :] < k
        gate = tl.load(
            gate_weights + row[:, None] * k + kk[None, :],
            mask=row_mask & k_mask, other=0.0,
        ).to(tl.float16)
        up = tl.load(
            up_weights + row[:, None] * k + kk[None, :],
            mask=row_mask & k_mask, other=0.0,
        ).to(tl.float16)
        activation = tl.load(
            x + col[None, :] * k + kk[:, None],
            mask=(col[None, :] < columns) & (kk[:, None] < k), other=0.0,
        ).to(tl.float16)
        gate_acc += tl.dot(gate, activation)
        up_acc += tl.dot(up, activation)
    result = gate_acc * tl.sigmoid(gate_acc) * up_acc
    tl.store(
        output + col[None, :] * rows + row[:, None], result,
        mask=(row[:, None] < rows) & (col[None, :] < columns),
    )


@triton.jit
def f16_gemv(
    weights, x, output, k, rows,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Dense F16-weight/F32-activation GEMV candidate.

    This is intentionally separate from the production F16 batched GEMM:
    decode has one activation column, so a [M, 1] dot tile can lower poorly
    on AMD.  Each program owns a small row tile and reuses the one activation
    vector across those rows.
    """
    pid = tl.program_id(0)
    rows_i = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in tl.range(0, k, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        mask = (rows_i[:, None] < rows) & (kk[None, :] < k)
        w = tl.load(weights + rows_i[:, None] * k + kk[None, :],
                    mask=mask, other=0.0).to(tl.float16)
        a = tl.load(x + kk, mask=kk < k, other=0.0).to(tl.float16)
        acc += tl.sum(w * a[None, :], axis=1)
    tl.store(output + rows_i, acc, mask=rows_i < rows)


@triton.jit
def q4_gemv_fixed_blocks(
    weights_u8, weights_f16, x, output, rows,
    BLOCKS: tl.constexpr,
):
    """Q4_K GEMV with a compile-time number of 256-value blocks."""
    row = tl.program_id(0)
    packed_lanes = tl.arange(0, 128)
    chunk = packed_lanes // 32
    lane = packed_lanes % 32
    low_index = chunk * 64 + lane
    high_index = low_index + 32
    low_group = chunk * 2
    high_group = low_group + 1
    accumulator = tl.zeros((128,), dtype=tl.float32)
    for block in tl.range(0, BLOCKS):
        block_byte = (row * BLOCKS + block) * 144
        block_half = block_byte // 2
        d = tl.load(weights_f16 + block_half).to(tl.float32)
        dmin = tl.load(weights_f16 + block_half + 1).to(tl.float32)
        low_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                  tl.where(low_group < 4, low_group, low_group + 4))
        low_scale_hi = tl.load(weights_u8 + block_byte + 4 + tl.maximum(low_group - 4, 0))
        low_scale = tl.where(low_group < 4, low_scale_byte & 63,
                             (low_scale_byte & 15) | ((low_scale_hi >> 6) << 4)).to(tl.float32)
        low_min_byte = tl.load(weights_u8 + block_byte + 4 + low_group + 4)
        low_min_hi = tl.load(weights_u8 + block_byte + 4 + low_group)
        low_minimum = tl.where(low_group < 4, low_min_byte & 63,
                               (low_min_byte >> 4) | ((low_min_hi >> 6) << 4)).to(tl.float32)
        high_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                   tl.where(high_group < 4, high_group, high_group + 4))
        high_scale_hi = tl.load(weights_u8 + block_byte + 4 + tl.maximum(high_group - 4, 0))
        high_scale = tl.where(high_group < 4, high_scale_byte & 63,
                              (high_scale_byte & 15) | ((high_scale_hi >> 6) << 4)).to(tl.float32)
        high_min_byte = tl.load(weights_u8 + block_byte + 4 + high_group + 4)
        high_min_hi = tl.load(weights_u8 + block_byte + 4 + high_group)
        high_minimum = tl.where(high_group < 4, high_min_byte & 63,
                                (high_min_byte >> 4) | ((high_min_hi >> 6) << 4)).to(tl.float32)
        quant_byte = tl.load(weights_u8 + block_byte + 16 + packed_lanes)
        low_value = d * low_scale * (quant_byte & 15).to(tl.float32) - dmin * low_minimum
        high_value = d * high_scale * (quant_byte >> 4).to(tl.float32) - dmin * high_minimum
        low_activation = tl.load(x + block * 256 + low_index).to(tl.float32)
        high_activation = tl.load(x + block * 256 + high_index).to(tl.float32)
        accumulator += low_value * low_activation + high_value * high_activation
    tl.store(output + row, tl.sum(accumulator, axis=0), mask=row < rows)


@triton.jit
def q6_gemv_fixed_blocks(
    weights_u8, weights_f16, x, output, rows,
    BLOCKS: tl.constexpr,
):
    """Q6_K GEMV with a compile-time number of 256-value blocks."""
    row = tl.program_id(0)
    packed_lanes = tl.arange(0, 64)
    half = packed_lanes // 32
    lane = packed_lanes % 32
    lane_half = lane // 16
    accumulator = tl.zeros((64,), dtype=tl.float32)
    for block in tl.range(0, BLOCKS):
        block_byte = (row * BLOCKS + block) * 210
        d = tl.load(weights_f16 + block_byte // 2 + 104).to(tl.float32)
        ql_a = tl.load(weights_u8 + block_byte + half * 64 + lane)
        ql_b = tl.load(weights_u8 + block_byte + half * 64 + lane + 32)
        qh = tl.load(weights_u8 + block_byte + 128 + half * 32 + lane)
        q0 = ((ql_a & 15) | (((qh >> 0) & 3) << 4)).to(tl.int32) - 32
        q1 = ((ql_b & 15) | (((qh >> 2) & 3) << 4)).to(tl.int32) - 32
        q2 = ((ql_a >> 4) | (((qh >> 4) & 3) << 4)).to(tl.int32) - 32
        q3 = ((ql_b >> 4) | (((qh >> 6) & 3) << 4)).to(tl.int32) - 32
        scale_base = block_byte + 192 + half * 8 + lane_half
        s0 = tl.load(weights_u8 + scale_base + 0).to(tl.int8).to(tl.float32)
        s1 = tl.load(weights_u8 + scale_base + 2).to(tl.int8).to(tl.float32)
        s2 = tl.load(weights_u8 + scale_base + 4).to(tl.int8).to(tl.float32)
        s3 = tl.load(weights_u8 + scale_base + 6).to(tl.int8).to(tl.float32)
        base = block * 256 + half * 128 + lane
        a0 = tl.load(x + base + 0).to(tl.float32)
        a1 = tl.load(x + base + 32).to(tl.float32)
        a2 = tl.load(x + base + 64).to(tl.float32)
        a3 = tl.load(x + base + 96).to(tl.float32)
        accumulator += d * (s0 * q0.to(tl.float32) * a0
                            + s1 * q1.to(tl.float32) * a1
                            + s2 * q2.to(tl.float32) * a2
                            + s3 * q3.to(tl.float32) * a3)
    tl.store(output + row, tl.sum(accumulator, axis=0), mask=row < rows)


@triton.jit
def q4_gemv_rows(
    weights_u8, weights_f16, x, output, k, rows,
    BLOCK_M: tl.constexpr,
):
    """Q4_K GEMV tile that shares the activation loads across rows."""
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_lanes = tl.arange(0, 128)
    chunk = packed_lanes // 32
    lane = packed_lanes % 32
    low_index = chunk * 64 + lane
    high_index = low_index + 32
    low_group = chunk * 2
    high_group = low_group + 1
    acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    blocks = k // 256
    for block in tl.range(0, blocks):
        block_byte = (row_ids[:, None] * blocks + block) * 144
        block_half = block_byte // 2
        d = tl.load(weights_f16 + block_half).to(tl.float32)
        dmin = tl.load(weights_f16 + block_half + 1).to(tl.float32)
        low_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                  tl.where(low_group < 4, low_group, low_group + 4)[None, :])
        low_scale_hi = tl.load(weights_u8 + block_byte + 4 +
                               tl.maximum(low_group - 4, 0)[None, :])
        low_scale = tl.where(low_group[None, :] < 4, low_scale_byte & 63,
                             (low_scale_byte & 15) | ((low_scale_hi >> 6) << 4)).to(tl.float32)
        low_min_byte = tl.load(weights_u8 + block_byte + 4 + low_group[None, :] + 4)
        low_min_hi = tl.load(weights_u8 + block_byte + 4 + low_group[None, :])
        low_minimum = tl.where(low_group[None, :] < 4, low_min_byte & 63,
                               (low_min_byte >> 4) | ((low_min_hi >> 6) << 4)).to(tl.float32)
        high_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                   tl.where(high_group < 4, high_group, high_group + 4)[None, :])
        high_scale_hi = tl.load(weights_u8 + block_byte + 4 +
                                tl.maximum(high_group - 4, 0)[None, :])
        high_scale = tl.where(high_group[None, :] < 4, high_scale_byte & 63,
                              (high_scale_byte & 15) | ((high_scale_hi >> 6) << 4)).to(tl.float32)
        high_min_byte = tl.load(weights_u8 + block_byte + 4 + high_group[None, :] + 4)
        high_min_hi = tl.load(weights_u8 + block_byte + 4 + high_group[None, :])
        high_minimum = tl.where(high_group[None, :] < 4, high_min_byte & 63,
                                (high_min_byte >> 4) | ((high_min_hi >> 6) << 4)).to(tl.float32)
        qbyte = tl.load(weights_u8 + block_byte + 16 + packed_lanes[None, :])
        lo = d * low_scale * (qbyte & 15).to(tl.float32) - dmin * low_minimum
        hi = d * high_scale * (qbyte >> 4).to(tl.float32) - dmin * high_minimum
        a0 = tl.load(x + block * 256 + low_index).to(tl.float32)
        a1 = tl.load(x + block * 256 + high_index).to(tl.float32)
        acc += lo * a0[None, :] + hi * a1[None, :]
    tl.store(output + row_ids, tl.sum(acc, axis=1), mask=row_ids < rows)


@triton.jit
def q6_gemv_rows(
    weights_u8, weights_f16, x, output, k, rows,
    BLOCK_M: tl.constexpr,
):
    """Q6_K GEMV tile that shares the activation loads across rows."""
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    lanes = tl.arange(0, 64)
    half = lanes // 32
    lane = lanes % 32
    lane_half = lane // 16
    acc = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    blocks = k // 256
    for block in tl.range(0, blocks):
        block_byte = (row_ids[:, None] * blocks + block) * 210
        d = tl.load(weights_f16 + block_byte // 2 + 104).to(tl.float32)
        ql_a = tl.load(weights_u8 + block_byte + half[None, :] * 64 + lane[None, :])
        ql_b = tl.load(weights_u8 + block_byte + half[None, :] * 64 + lane[None, :] + 32)
        qh = tl.load(weights_u8 + block_byte + 128 + half[None, :] * 32 + lane[None, :])
        q0 = ((ql_a & 15) | (((qh >> 0) & 3) << 4)).to(tl.int32) - 32
        q1 = ((ql_b & 15) | (((qh >> 2) & 3) << 4)).to(tl.int32) - 32
        q2 = ((ql_a >> 4) | (((qh >> 4) & 3) << 4)).to(tl.int32) - 32
        q3 = ((ql_b >> 4) | (((qh >> 6) & 3) << 4)).to(tl.int32) - 32
        scale_base = block_byte + 192 + half[None, :] * 8 + lane_half[None, :]
        s0 = tl.load(weights_u8 + scale_base + 0).to(tl.int8).to(tl.float32)
        s1 = tl.load(weights_u8 + scale_base + 2).to(tl.int8).to(tl.float32)
        s2 = tl.load(weights_u8 + scale_base + 4).to(tl.int8).to(tl.float32)
        s3 = tl.load(weights_u8 + scale_base + 6).to(tl.int8).to(tl.float32)
        base = block * 256 + half[None, :] * 128 + lane[None, :]
        a0 = tl.load(x + base + 0).to(tl.float32)
        a1 = tl.load(x + base + 32).to(tl.float32)
        a2 = tl.load(x + base + 64).to(tl.float32)
        a3 = tl.load(x + base + 96).to(tl.float32)
        acc += d * (s0 * q0.to(tl.float32) * a0 + s1 * q1.to(tl.float32) * a1
                    + s2 * q2.to(tl.float32) * a2 + s3 * q3.to(tl.float32) * a3)
    tl.store(output + row_ids, tl.sum(acc, axis=1), mask=row_ids < rows)


@triton.jit
def q4_gemv_narrow(
    weights_u8, weights_f16, x, output, k, rows,
    BLOCK_M: tl.constexpr,
):
    """Q4_K GEMV using a physical-wave-sized vector and explicit subchunks."""
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    lanes = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    blocks = k // 256
    for block in tl.range(0, blocks):
        block_byte = (row_ids[:, None] * blocks + block) * 144
        block_half = block_byte // 2
        d = tl.load(weights_f16 + block_half).to(tl.float32)
        dmin = tl.load(weights_f16 + block_half + 1).to(tl.float32)
        for chunk in tl.range(0, 4):
            low_group = chunk * 2
            high_group = low_group + 1
            low_index = chunk * 64 + lanes
            high_index = low_index + 32
            low_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                      (low_group if low_group < 4 else low_group + 4))
            low_scale_hi = tl.load(weights_u8 + block_byte + 4 +
                                   max(low_group - 4, 0))
            low_scale = (low_scale_byte & 63 if low_group < 4 else
                         (low_scale_byte & 15) | ((low_scale_hi >> 6) << 4)).to(tl.float32)
            low_min_byte = tl.load(weights_u8 + block_byte + 4 + low_group + 4)
            low_min_hi = tl.load(weights_u8 + block_byte + 4 + low_group)
            low_minimum = (low_min_byte & 63 if low_group < 4 else
                           (low_min_byte >> 4) | ((low_min_hi >> 6) << 4)).to(tl.float32)
            high_scale_byte = tl.load(weights_u8 + block_byte + 4 +
                                       (high_group if high_group < 4 else high_group + 4))
            high_scale_hi = tl.load(weights_u8 + block_byte + 4 +
                                    max(high_group - 4, 0))
            high_scale = (high_scale_byte & 63 if high_group < 4 else
                          (high_scale_byte & 15) | ((high_scale_hi >> 6) << 4)).to(tl.float32)
            high_min_byte = tl.load(weights_u8 + block_byte + 4 + high_group + 4)
            high_min_hi = tl.load(weights_u8 + block_byte + 4 + high_group)
            high_minimum = (high_min_byte & 63 if high_group < 4 else
                            (high_min_byte >> 4) | ((high_min_hi >> 6) << 4)).to(tl.float32)
            qbyte = tl.load(weights_u8 + block_byte + 16 + chunk * 32 + lanes[None, :])
            lo = d * low_scale * (qbyte & 15).to(tl.float32) - dmin * low_minimum
            hi = d * high_scale * (qbyte >> 4).to(tl.float32) - dmin * high_minimum
            a0 = tl.load(x + block * 256 + low_index).to(tl.float32)
            a1 = tl.load(x + block * 256 + high_index).to(tl.float32)
            acc += lo * a0[None, :] + hi * a1[None, :]
    tl.store(output + row_ids, tl.sum(acc, axis=1), mask=row_ids < rows)


@triton.jit
def q6_gemv_narrow(
    weights_u8, weights_f16, x, output, k, rows,
    BLOCK_M: tl.constexpr,
):
    """Q6_K GEMV using a physical-wave-sized vector and explicit halves."""
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    lanes = tl.arange(0, 32)
    lane_half = lanes // 16
    acc = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    blocks = k // 256
    for block in tl.range(0, blocks):
        block_byte = (row_ids[:, None] * blocks + block) * 210
        d = tl.load(weights_f16 + block_byte // 2 + 104).to(tl.float32)
        for half in tl.range(0, 2):
            ql_a = tl.load(weights_u8 + block_byte + half * 64 + lanes[None, :])
            ql_b = tl.load(weights_u8 + block_byte + half * 64 + lanes[None, :] + 32)
            qh = tl.load(weights_u8 + block_byte + 128 + half * 32 + lanes[None, :])
            q0 = ((ql_a & 15) | (((qh >> 0) & 3) << 4)).to(tl.int32) - 32
            q1 = ((ql_b & 15) | (((qh >> 2) & 3) << 4)).to(tl.int32) - 32
            q2 = ((ql_a >> 4) | (((qh >> 4) & 3) << 4)).to(tl.int32) - 32
            q3 = ((ql_b >> 4) | (((qh >> 6) & 3) << 4)).to(tl.int32) - 32
            scale_base = block_byte + 192 + half * 8 + lane_half[None, :]
            s0 = tl.load(weights_u8 + scale_base + 0).to(tl.int8).to(tl.float32)
            s1 = tl.load(weights_u8 + scale_base + 2).to(tl.int8).to(tl.float32)
            s2 = tl.load(weights_u8 + scale_base + 4).to(tl.int8).to(tl.float32)
            s3 = tl.load(weights_u8 + scale_base + 6).to(tl.int8).to(tl.float32)
            base = block * 256 + half * 128 + lanes
            a0 = tl.load(x + base + 0).to(tl.float32)
            a1 = tl.load(x + base + 32).to(tl.float32)
            a2 = tl.load(x + base + 64).to(tl.float32)
            a3 = tl.load(x + base + 96).to(tl.float32)
            acc += d * (s0 * q0.to(tl.float32) * a0[None, :]
                        + s1 * q1.to(tl.float32) * a1[None, :]
                        + s2 * q2.to(tl.float32) * a2[None, :]
                        + s3 * q3.to(tl.float32) * a3[None, :])
    tl.store(output + row_ids, tl.sum(acc, axis=1), mask=row_ids < rows)


@triton.jit
def q6_gemv_split_k_f32(
    weights_u8, weights_f16, x, output, k, rows,
    SPLIT_K: tl.constexpr,
):
    """Q6_K GEMV with one physical wave per explicit K partition."""
    row = tl.program_id(0)
    split_ids = tl.arange(0, SPLIT_K)[:, None]
    lanes = tl.arange(0, 32)[None, :]
    lane_half = lanes // 16
    accumulator = tl.zeros((SPLIT_K, 32), dtype=tl.float32)
    blocks = k // 256
    for block0 in tl.range(0, blocks, SPLIT_K):
        block = block0 + split_ids
        valid_block = block < blocks
        block_byte = (row * blocks + block) * 210
        d = tl.load(weights_f16 + block_byte // 2 + 104,
                    mask=valid_block, other=0.0).to(tl.float32)
        for half in tl.range(0, 2):
            ql_a = tl.load(weights_u8 + block_byte + half * 64 + lanes,
                           mask=valid_block, other=0)
            ql_b = tl.load(weights_u8 + block_byte + half * 64 + lanes + 32,
                           mask=valid_block, other=0)
            qh = tl.load(weights_u8 + block_byte + 128 + half * 32 + lanes,
                         mask=valid_block, other=0)
            q0 = ((ql_a & 15) | (((qh >> 0) & 3) << 4)).to(tl.int32) - 32
            q1 = ((ql_b & 15) | (((qh >> 2) & 3) << 4)).to(tl.int32) - 32
            q2 = ((ql_a >> 4) | (((qh >> 4) & 3) << 4)).to(tl.int32) - 32
            q3 = ((ql_b >> 4) | (((qh >> 6) & 3) << 4)).to(tl.int32) - 32
            scale_base = block_byte + 192 + half * 8 + lane_half
            s0 = tl.load(weights_u8 + scale_base + 0,
                         mask=valid_block, other=0).to(tl.int8).to(tl.float32)
            s1 = tl.load(weights_u8 + scale_base + 2,
                         mask=valid_block, other=0).to(tl.int8).to(tl.float32)
            s2 = tl.load(weights_u8 + scale_base + 4,
                         mask=valid_block, other=0).to(tl.int8).to(tl.float32)
            s3 = tl.load(weights_u8 + scale_base + 6,
                         mask=valid_block, other=0).to(tl.int8).to(tl.float32)
            base = block * 256 + half * 128 + lanes
            a0 = tl.load(x + base + 0, mask=valid_block, other=0.0).to(tl.float32)
            a1 = tl.load(x + base + 32, mask=valid_block, other=0.0).to(tl.float32)
            a2 = tl.load(x + base + 64, mask=valid_block, other=0.0).to(tl.float32)
            a3 = tl.load(x + base + 96, mask=valid_block, other=0.0).to(tl.float32)
            accumulator += d * (s0 * q0.to(tl.float32) * a0 +
                                s1 * q1.to(tl.float32) * a1 +
                                s2 * q2.to(tl.float32) * a2 +
                                s3 * q3.to(tl.float32) * a3)
    partials = tl.sum(accumulator, axis=1)
    tl.store(output + row, tl.sum(partials, axis=0), mask=row < rows)


@triton.jit
def quantize_f32_q8_32(x, q8, scales, sums, k):
    """Quantize one contiguous 32-value activation block per program."""
    block = tl.program_id(0)
    lanes = tl.arange(0, 32)
    offsets = block * 32 + lanes
    values = tl.load(x + offsets, mask=offsets < k, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=0)
    scale = amax / 127.0
    inverse = tl.where(amax > 0.0, 1.0 / scale, 0.0)
    quantized = libdevice.nearbyint(values * inverse)
    quantized = tl.maximum(-127.0, tl.minimum(127.0, quantized)).to(tl.int8)
    tl.store(q8 + offsets, quantized, mask=offsets < k)
    tl.store(scales + block, scale)
    quad_sums = tl.sum(quantized.to(tl.int32).reshape((8, 4)), axis=1)
    tl.store(sums + block * 8 + tl.arange(0, 8), quad_sums)


@triton.jit
def quantize_f32_q8_1_32(x, q8, ds, k):
    """Quantize one Q8_1 block and store its FP16 scale and source sum."""
    block = tl.program_id(0)
    lanes = tl.arange(0, 32)
    offsets = block * 32 + lanes
    values = tl.load(x + offsets, mask=offsets < k, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=0)
    scale = amax / 127.0
    inverse = tl.where(amax > 0.0, 1.0 / scale, 0.0)
    quantized = libdevice.nearbyint(values * inverse)
    quantized = tl.maximum(-127.0, tl.minimum(127.0, quantized)).to(tl.int8)
    tl.store(q8 + offsets, quantized, mask=offsets < k)
    tl.store(ds + block * 2, scale)
    tl.store(ds + block * 2 + 1, tl.sum(values, axis=0))


@triton.jit
def pack_i8x4(v0, v1, v2, v3):
    """Pack four signed byte values into one 32-bit dot-product operand."""
    return ((v0 & 255) | ((v1 & 255) << 8) |
            ((v2 & 255) << 16) | ((v3 & 255) << 24)).to(tl.int32)


@triton.jit
def amd_sdot4_i8(a, b, accumulator):
    """RDNA4 signed int8 dot4, matching HIP's __builtin_amdgcn_sudot4."""
    return tl.inline_asm_elementwise(
        "v_dot4_i32_iu8 $0, $1, $2, $3 neg_lo:[1,1,0]",
        "=v,v,v,v", [a, b, accumulator], dtype=tl.int32,
        is_pure=True, pack=1,
    )


@triton.jit
def q6_gemv_q8_dot4(
    weights_u8, weights_f16, q8_i32, q8_scales, output, k, rows,
    BLOCK_M: tl.constexpr,
):
    """Q6_K GEMV over Q8 activations using RDNA4 packed int8 dot4."""
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    quads = tl.arange(0, 8)
    byte_lane = quads * 4
    scale_lane = quads // 4
    accumulator = tl.zeros((BLOCK_M, 8), dtype=tl.float32)
    blocks = k // 256
    zero = tl.zeros((BLOCK_M, 8), dtype=tl.int32)

    for block in tl.range(0, blocks):
        block_byte = (row_ids[:, None] * blocks + block) * 210
        d6 = tl.load(weights_f16 + block_byte // 2 + 104).to(tl.float32)
        for half in tl.range(0, 2):
            ql_a0 = tl.load(weights_u8 + block_byte + half * 64 + byte_lane[None, :] + 0)
            ql_a1 = tl.load(weights_u8 + block_byte + half * 64 + byte_lane[None, :] + 1)
            ql_a2 = tl.load(weights_u8 + block_byte + half * 64 + byte_lane[None, :] + 2)
            ql_a3 = tl.load(weights_u8 + block_byte + half * 64 + byte_lane[None, :] + 3)
            ql_b0 = tl.load(weights_u8 + block_byte + half * 64 + byte_lane[None, :] + 32)
            ql_b1 = tl.load(weights_u8 + block_byte + half * 64 + byte_lane[None, :] + 33)
            ql_b2 = tl.load(weights_u8 + block_byte + half * 64 + byte_lane[None, :] + 34)
            ql_b3 = tl.load(weights_u8 + block_byte + half * 64 + byte_lane[None, :] + 35)
            qh0 = tl.load(weights_u8 + block_byte + 128 + half * 32 + byte_lane[None, :] + 0)
            qh1 = tl.load(weights_u8 + block_byte + 128 + half * 32 + byte_lane[None, :] + 1)
            qh2 = tl.load(weights_u8 + block_byte + 128 + half * 32 + byte_lane[None, :] + 2)
            qh3 = tl.load(weights_u8 + block_byte + 128 + half * 32 + byte_lane[None, :] + 3)

            q0 = pack_i8x4(
                ((ql_a0 & 15) | (((qh0 >> 0) & 3) << 4)).to(tl.int32) - 32,
                ((ql_a1 & 15) | (((qh1 >> 0) & 3) << 4)).to(tl.int32) - 32,
                ((ql_a2 & 15) | (((qh2 >> 0) & 3) << 4)).to(tl.int32) - 32,
                ((ql_a3 & 15) | (((qh3 >> 0) & 3) << 4)).to(tl.int32) - 32,
            )
            q1 = pack_i8x4(
                ((ql_b0 & 15) | (((qh0 >> 2) & 3) << 4)).to(tl.int32) - 32,
                ((ql_b1 & 15) | (((qh1 >> 2) & 3) << 4)).to(tl.int32) - 32,
                ((ql_b2 & 15) | (((qh2 >> 2) & 3) << 4)).to(tl.int32) - 32,
                ((ql_b3 & 15) | (((qh3 >> 2) & 3) << 4)).to(tl.int32) - 32,
            )
            q2 = pack_i8x4(
                ((ql_a0 >> 4) | (((qh0 >> 4) & 3) << 4)).to(tl.int32) - 32,
                ((ql_a1 >> 4) | (((qh1 >> 4) & 3) << 4)).to(tl.int32) - 32,
                ((ql_a2 >> 4) | (((qh2 >> 4) & 3) << 4)).to(tl.int32) - 32,
                ((ql_a3 >> 4) | (((qh3 >> 4) & 3) << 4)).to(tl.int32) - 32,
            )
            q3 = pack_i8x4(
                ((ql_b0 >> 4) | (((qh0 >> 6) & 3) << 4)).to(tl.int32) - 32,
                ((ql_b1 >> 4) | (((qh1 >> 6) & 3) << 4)).to(tl.int32) - 32,
                ((ql_b2 >> 4) | (((qh2 >> 6) & 3) << 4)).to(tl.int32) - 32,
                ((ql_b3 >> 4) | (((qh3 >> 6) & 3) << 4)).to(tl.int32) - 32,
            )

            q8_base = block * 64 + half * 32 + quads
            a0 = tl.load(q8_i32 + q8_base + 0)
            a1 = tl.load(q8_i32 + q8_base + 8)
            a2 = tl.load(q8_i32 + q8_base + 16)
            a3 = tl.load(q8_i32 + q8_base + 24)
            d80 = tl.load(q8_scales + block * 8 + half * 4 + 0)
            d81 = tl.load(q8_scales + block * 8 + half * 4 + 1)
            d82 = tl.load(q8_scales + block * 8 + half * 4 + 2)
            d83 = tl.load(q8_scales + block * 8 + half * 4 + 3)
            scale_base = block_byte + 192 + half * 8 + scale_lane[None, :]
            s0 = tl.load(weights_u8 + scale_base + 0).to(tl.int8).to(tl.float32)
            s1 = tl.load(weights_u8 + scale_base + 2).to(tl.int8).to(tl.float32)
            s2 = tl.load(weights_u8 + scale_base + 4).to(tl.int8).to(tl.float32)
            s3 = tl.load(weights_u8 + scale_base + 6).to(tl.int8).to(tl.float32)
            dot0 = amd_sdot4_i8(q0, a0[None, :], zero).to(tl.float32)
            dot1 = amd_sdot4_i8(q1, a1[None, :], zero).to(tl.float32)
            dot2 = amd_sdot4_i8(q2, a2[None, :], zero).to(tl.float32)
            dot3 = amd_sdot4_i8(q3, a3[None, :], zero).to(tl.float32)
            accumulator += d6 * (d80 * s0 * dot0 + d81 * s1 * dot1 +
                                 d82 * s2 * dot2 + d83 * s3 * dot3)
    tl.store(output + row_ids, tl.sum(accumulator, axis=1), mask=row_ids < rows)


@triton.jit
def q6_gemv_q8_dot4_packed(
    weights_u8, weights_i16, weights_f16, q8_i32, q8_scales, q8_sums,
    output, k, rows, BLOCK_M: tl.constexpr,
):
    """Q6_K/Q8 dot4 with halfword weight loads and unsigned Q6 packing."""
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    quads = tl.arange(0, 8)
    scale_lane = quads // 4
    accumulator = tl.zeros((BLOCK_M, 8), dtype=tl.float32)
    blocks = k // 256
    zero = tl.zeros((BLOCK_M, 8), dtype=tl.int32)

    for block in tl.range(0, blocks):
        block_byte = (row_ids[:, None] * blocks + block) * 210
        block_half = block_byte // 2
        d6 = tl.load(weights_f16 + block_half + 104).to(tl.float32)
        for half in tl.range(0, 2):
            ql_a_lo = tl.load(weights_i16 + block_half + half * 32 + quads[None, :] * 2 + 0)
            ql_a_hi = tl.load(weights_i16 + block_half + half * 32 + quads[None, :] * 2 + 1)
            ql_b_lo = tl.load(weights_i16 + block_half + half * 32 + quads[None, :] * 2 + 16)
            ql_b_hi = tl.load(weights_i16 + block_half + half * 32 + quads[None, :] * 2 + 17)
            qh_lo = tl.load(weights_i16 + block_half + 64 + half * 16 + quads[None, :] * 2 + 0)
            qh_hi = tl.load(weights_i16 + block_half + 64 + half * 16 + quads[None, :] * 2 + 1)
            ql_a = ((ql_a_lo.to(tl.int32) & 65535) |
                    ((ql_a_hi.to(tl.int32) & 65535) << 16))
            ql_b = ((ql_b_lo.to(tl.int32) & 65535) |
                    ((ql_b_hi.to(tl.int32) & 65535) << 16))
            qh = ((qh_lo.to(tl.int32) & 65535) |
                  ((qh_hi.to(tl.int32) & 65535) << 16))
            q0 = (ql_a & 0x0F0F0F0F) | ((qh << 4) & 0x30303030)
            q1 = (ql_b & 0x0F0F0F0F) | ((qh << 2) & 0x30303030)
            q2 = ((ql_a >> 4) & 0x0F0F0F0F) | (qh & 0x30303030)
            q3 = ((ql_b >> 4) & 0x0F0F0F0F) | ((qh >> 2) & 0x30303030)

            q8_base = block * 64 + half * 32 + quads
            a0 = tl.load(q8_i32 + q8_base + 0)
            a1 = tl.load(q8_i32 + q8_base + 8)
            a2 = tl.load(q8_i32 + q8_base + 16)
            a3 = tl.load(q8_i32 + q8_base + 24)
            d80 = tl.load(q8_scales + block * 8 + half * 4 + 0)
            d81 = tl.load(q8_scales + block * 8 + half * 4 + 1)
            d82 = tl.load(q8_scales + block * 8 + half * 4 + 2)
            d83 = tl.load(q8_scales + block * 8 + half * 4 + 3)
            sum0 = tl.load(q8_sums + q8_base + 0)
            sum1 = tl.load(q8_sums + q8_base + 8)
            sum2 = tl.load(q8_sums + q8_base + 16)
            sum3 = tl.load(q8_sums + q8_base + 24)
            scale_base = block_byte + 192 + half * 8 + scale_lane[None, :]
            s0 = tl.load(weights_u8 + scale_base + 0).to(tl.int8).to(tl.float32)
            s1 = tl.load(weights_u8 + scale_base + 2).to(tl.int8).to(tl.float32)
            s2 = tl.load(weights_u8 + scale_base + 4).to(tl.int8).to(tl.float32)
            s3 = tl.load(weights_u8 + scale_base + 6).to(tl.int8).to(tl.float32)
            # Raw Q6 bytes are in [0, 63], so signed dot4 is equivalent to
            # unsigned-Q6/signed-Q8 here; subtract the shared 32 zero-point.
            dot0 = amd_sdot4_i8(q0, a0[None, :], zero) - 32 * sum0
            dot1 = amd_sdot4_i8(q1, a1[None, :], zero) - 32 * sum1
            dot2 = amd_sdot4_i8(q2, a2[None, :], zero) - 32 * sum2
            dot3 = amd_sdot4_i8(q3, a3[None, :], zero) - 32 * sum3
            accumulator += d6 * (d80 * s0 * dot0.to(tl.float32) +
                                 d81 * s1 * dot1.to(tl.float32) +
                                 d82 * s2 * dot2.to(tl.float32) +
                                 d83 * s3 * dot3.to(tl.float32))
    tl.store(output + row_ids, tl.sum(accumulator, axis=1), mask=row_ids < rows)


@triton.jit
def q6_gemv_q8_dot4_split_k(
    weights_u8, weights_i16, weights_f16, q8_i32, q8_scales, q8_sums,
    output, k, rows, SPLIT_K: tl.constexpr,
):
    """RDNA4-style Q6_K/Q8: one wave per explicit K partition."""
    row = tl.program_id(0)
    split_ids = tl.arange(0, SPLIT_K)[:, None]
    lanes = tl.arange(0, 32)[None, :]
    quadrant = lanes // 8
    quad = lanes % 8
    accumulator = tl.zeros((SPLIT_K, 32), dtype=tl.float32)
    zero = tl.zeros((SPLIT_K, 32), dtype=tl.int32)
    blocks = k // 256

    for block0 in tl.range(0, blocks, SPLIT_K):
        block = block0 + split_ids
        valid_block = block < blocks
        block_byte = (row * blocks + block) * 210
        block_half = block_byte // 2
        d6 = tl.load(weights_f16 + block_half + 104,
                     mask=valid_block, other=0.0).to(tl.float32)
        for half in tl.range(0, 2):
            ql_offset = (block_half + half * 32 + (quadrant % 2) * 16 +
                         quad * 2)
            ql_lo = tl.load(weights_i16 + ql_offset,
                            mask=valid_block, other=0)
            ql_hi = tl.load(weights_i16 + ql_offset + 1,
                            mask=valid_block, other=0)
            qh_offset = block_half + 64 + half * 16 + quad * 2
            qh_lo = tl.load(weights_i16 + qh_offset,
                            mask=valid_block, other=0)
            qh_hi = tl.load(weights_i16 + qh_offset + 1,
                            mask=valid_block, other=0)
            ql = ((ql_lo.to(tl.int32) & 65535) |
                  ((ql_hi.to(tl.int32) & 65535) << 16))
            qh = ((qh_lo.to(tl.int32) & 65535) |
                  ((qh_hi.to(tl.int32) & 65535) << 16))
            low = (ql >> (4 * (quadrant // 2))) & 0x0F0F0F0F
            high = ((qh >> (2 * quadrant)) << 4) & 0x30303030
            raw_q6 = low | high
            group = half * 4 + quadrant
            q8_index = block * 64 + group * 8 + quad
            activation = tl.load(q8_i32 + q8_index,
                                 mask=valid_block, other=0)
            activation_sum = tl.load(q8_sums + q8_index,
                                     mask=valid_block, other=0)
            d8 = tl.load(q8_scales + block * 8 + group,
                         mask=valid_block, other=0.0)
            q6_scale = tl.load(
                weights_u8 + block_byte + 192 + half * 8 + quadrant * 2 + quad // 4,
                mask=valid_block, other=0,
            ).to(tl.int8).to(tl.float32)
            dot = amd_sdot4_i8(raw_q6, activation, zero) - 32 * activation_sum
            accumulator += d6 * d8 * q6_scale * dot.to(tl.float32)
    partials = tl.sum(accumulator, axis=1)
    tl.store(output + row, tl.sum(partials, axis=0), mask=row < rows)


@triton.jit
def q6_gemv_q8_dot4_partials(
    weights_u8, weights_i16, weights_f16, q8_i32, q8_scales, q8_sums,
    partials, k, rows, SPLIT_K: tl.constexpr,
):
    """One-wave Q6_K/Q8 split-K partial, reduced by a second kernel."""
    pid = tl.program_id(0)
    row = pid // SPLIT_K
    split_id = pid % SPLIT_K
    lanes = tl.arange(0, 32)
    quadrant = lanes // 8
    quad = lanes % 8
    accumulator = tl.zeros((32,), dtype=tl.float32)
    zero = tl.zeros((32,), dtype=tl.int32)
    blocks = k // 256

    for block0 in tl.range(0, blocks, SPLIT_K):
        block = block0 + split_id
        block_byte = (row * blocks + block) * 210
        block_half = block_byte // 2
        d6 = tl.load(weights_f16 + block_half + 104).to(tl.float32)
        for half in tl.range(0, 2):
            ql_offset = (block_half + half * 32 + (quadrant % 2) * 16 +
                         quad * 2)
            ql_lo = tl.load(weights_i16 + ql_offset)
            ql_hi = tl.load(weights_i16 + ql_offset + 1)
            qh_offset = block_half + 64 + half * 16 + quad * 2
            qh_lo = tl.load(weights_i16 + qh_offset)
            qh_hi = tl.load(weights_i16 + qh_offset + 1)
            ql = ((ql_lo.to(tl.int32) & 65535) |
                  ((ql_hi.to(tl.int32) & 65535) << 16))
            qh = ((qh_lo.to(tl.int32) & 65535) |
                  ((qh_hi.to(tl.int32) & 65535) << 16))
            low = (ql >> (4 * (quadrant // 2))) & 0x0F0F0F0F
            high = ((qh >> (2 * quadrant)) << 4) & 0x30303030
            raw_q6 = low | high
            group = half * 4 + quadrant
            q8_index = block * 64 + group * 8 + quad
            activation = tl.load(q8_i32 + q8_index)
            activation_sum = tl.load(q8_sums + q8_index)
            d8 = tl.load(q8_scales + block * 8 + group)
            q6_scale = tl.load(
                weights_u8 + block_byte + 192 + half * 8 + quadrant * 2 + quad // 4,
            ).to(tl.int8).to(tl.float32)
            dot = amd_sdot4_i8(raw_q6, activation, zero) - 32 * activation_sum
            accumulator += d6 * d8 * q6_scale * dot.to(tl.float32)
    tl.store(partials + row * SPLIT_K + split_id,
             tl.sum(accumulator, axis=0), mask=row < rows)


@triton.jit
def reduce_split_k_f32(partials, output, rows, SPLIT_K: tl.constexpr):
    row = tl.program_id(0)
    split_ids = tl.arange(0, SPLIT_K)
    values = tl.load(partials + row * SPLIT_K + split_ids,
                     mask=row < rows, other=0.0)
    tl.store(output + row, tl.sum(values, axis=0), mask=row < rows)


@triton.jit
def q4_ffn_swiglu_q8_dot4(
    gate_u8, gate_i32, gate_f16, up_u8, up_i32, up_f16,
    q8_i32, q8_scales, q8_sums, output, k, rows,
    BLOCK_M: tl.constexpr,
):
    """Fused Q4_K gate/up/SwiGLU over a shared Q8 activation."""
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    quads = tl.arange(0, 8)
    gate_acc = tl.zeros((BLOCK_M, 8), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, 8), dtype=tl.float32)
    zero = tl.zeros((BLOCK_M, 8), dtype=tl.int32)
    blocks = k // 256

    for block in tl.range(0, blocks):
        block_byte = (row_ids[:, None] * blocks + block) * 144
        block_half = block_byte // 2
        block_word = block_byte // 4
        gate_d = tl.load(gate_f16 + block_half).to(tl.float32)
        gate_dmin = tl.load(gate_f16 + block_half + 1).to(tl.float32)
        up_d = tl.load(up_f16 + block_half).to(tl.float32)
        up_dmin = tl.load(up_f16 + block_half + 1).to(tl.float32)
        for chunk in tl.range(0, 4):
            low_group = chunk * 2
            high_group = low_group + 1
            low_scale_index = low_group if low_group < 4 else low_group + 4
            low_scale_hi_index = low_group - 4 if low_group >= 4 else 0
            high_scale_index = high_group if high_group < 4 else high_group + 4
            high_scale_hi_index = high_group - 4 if high_group >= 4 else 0

            gate_low_scale_byte = tl.load(gate_u8 + block_byte + 4 + low_scale_index)
            gate_low_scale_hi = tl.load(gate_u8 + block_byte + 4 + low_scale_hi_index)
            gate_low_scale = (gate_low_scale_byte & 63 if low_group < 4 else
                              (gate_low_scale_byte & 15) |
                              ((gate_low_scale_hi >> 6) << 4)).to(tl.float32)
            gate_low_min_byte = tl.load(gate_u8 + block_byte + 4 + low_group + 4)
            gate_low_min_hi = tl.load(gate_u8 + block_byte + 4 + low_group)
            gate_low_min = (gate_low_min_byte & 63 if low_group < 4 else
                            (gate_low_min_byte >> 4) |
                            ((gate_low_min_hi >> 6) << 4)).to(tl.float32)
            gate_high_scale_byte = tl.load(gate_u8 + block_byte + 4 + high_scale_index)
            gate_high_scale_hi = tl.load(gate_u8 + block_byte + 4 + high_scale_hi_index)
            gate_high_scale = (gate_high_scale_byte & 63 if high_group < 4 else
                               (gate_high_scale_byte & 15) |
                               ((gate_high_scale_hi >> 6) << 4)).to(tl.float32)
            gate_high_min_byte = tl.load(gate_u8 + block_byte + 4 + high_group + 4)
            gate_high_min_hi = tl.load(gate_u8 + block_byte + 4 + high_group)
            gate_high_min = (gate_high_min_byte & 63 if high_group < 4 else
                             (gate_high_min_byte >> 4) |
                             ((gate_high_min_hi >> 6) << 4)).to(tl.float32)

            up_low_scale_byte = tl.load(up_u8 + block_byte + 4 + low_scale_index)
            up_low_scale_hi = tl.load(up_u8 + block_byte + 4 + low_scale_hi_index)
            up_low_scale = (up_low_scale_byte & 63 if low_group < 4 else
                            (up_low_scale_byte & 15) |
                            ((up_low_scale_hi >> 6) << 4)).to(tl.float32)
            up_low_min_byte = tl.load(up_u8 + block_byte + 4 + low_group + 4)
            up_low_min_hi = tl.load(up_u8 + block_byte + 4 + low_group)
            up_low_min = (up_low_min_byte & 63 if low_group < 4 else
                          (up_low_min_byte >> 4) |
                          ((up_low_min_hi >> 6) << 4)).to(tl.float32)
            up_high_scale_byte = tl.load(up_u8 + block_byte + 4 + high_scale_index)
            up_high_scale_hi = tl.load(up_u8 + block_byte + 4 + high_scale_hi_index)
            up_high_scale = (up_high_scale_byte & 63 if high_group < 4 else
                             (up_high_scale_byte & 15) |
                             ((up_high_scale_hi >> 6) << 4)).to(tl.float32)
            up_high_min_byte = tl.load(up_u8 + block_byte + 4 + high_group + 4)
            up_high_min_hi = tl.load(up_u8 + block_byte + 4 + high_group)
            up_high_min = (up_high_min_byte & 63 if high_group < 4 else
                           (up_high_min_byte >> 4) |
                           ((up_high_min_hi >> 6) << 4)).to(tl.float32)

            gate_q = tl.load(gate_i32 + block_word + 4 + chunk * 8 + quads[None, :])
            up_q = tl.load(up_i32 + block_word + 4 + chunk * 8 + quads[None, :])
            gate_low_q = gate_q & 0x0F0F0F0F
            gate_high_q = (gate_q >> 4) & 0x0F0F0F0F
            up_low_q = up_q & 0x0F0F0F0F
            up_high_q = (up_q >> 4) & 0x0F0F0F0F
            q8_base = block * 64 + low_group * 8 + quads
            low_a = tl.load(q8_i32 + q8_base)
            high_a = tl.load(q8_i32 + q8_base + 8)
            low_sum = tl.load(q8_sums + q8_base)
            high_sum = tl.load(q8_sums + q8_base + 8)
            low_d8 = tl.load(q8_scales + block * 8 + low_group)
            high_d8 = tl.load(q8_scales + block * 8 + high_group)
            gate_low_dot = amd_sdot4_i8(gate_low_q, low_a[None, :], zero)
            gate_high_dot = amd_sdot4_i8(gate_high_q, high_a[None, :], zero)
            up_low_dot = amd_sdot4_i8(up_low_q, low_a[None, :], zero)
            up_high_dot = amd_sdot4_i8(up_high_q, high_a[None, :], zero)
            gate_acc += low_d8 * (gate_d * gate_low_scale * gate_low_dot.to(tl.float32) -
                                  gate_dmin * gate_low_min * low_sum) + \
                        high_d8 * (gate_d * gate_high_scale * gate_high_dot.to(tl.float32) -
                                   gate_dmin * gate_high_min * high_sum)
            up_acc += low_d8 * (up_d * up_low_scale * up_low_dot.to(tl.float32) -
                                up_dmin * up_low_min * low_sum) + \
                      high_d8 * (up_d * up_high_scale * up_high_dot.to(tl.float32) -
                                 up_dmin * up_high_min * high_sum)
    gate = tl.sum(gate_acc, axis=1)
    up = tl.sum(up_acc, axis=1)
    tl.store(output + row_ids, gate * tl.sigmoid(gate) * up, mask=row_ids < rows)


@triton.jit
def q40_ffn_swiglu_q8_dot4(
    gate_u8, gate_f16, up_u8, up_f16,
    q8_i32, q8_ds, output, k, rows,
    BLOCK_M: tl.constexpr,
):
    """Fused Q4_0 gate/up/SwiGLU over one shared Q8 activation."""
    row_ids = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    quads = tl.arange(0, 4)
    byte_lane = quads * 4
    gate_acc = tl.zeros((BLOCK_M, 4), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, 4), dtype=tl.float32)
    zero = tl.zeros((BLOCK_M, 4), dtype=tl.int32)
    blocks = k // 32

    for block in tl.range(0, blocks):
        block_byte = (row_ids[:, None] * blocks + block) * 18
        gate_d = tl.load(gate_f16 + block_byte // 2).to(tl.float32)
        up_d = tl.load(up_f16 + block_byte // 2).to(tl.float32)

        gate_q = pack_i8x4(
            tl.load(gate_u8 + block_byte + 2 + byte_lane[None, :] + 0),
            tl.load(gate_u8 + block_byte + 2 + byte_lane[None, :] + 1),
            tl.load(gate_u8 + block_byte + 2 + byte_lane[None, :] + 2),
            tl.load(gate_u8 + block_byte + 2 + byte_lane[None, :] + 3),
        )
        up_q = pack_i8x4(
            tl.load(up_u8 + block_byte + 2 + byte_lane[None, :] + 0),
            tl.load(up_u8 + block_byte + 2 + byte_lane[None, :] + 1),
            tl.load(up_u8 + block_byte + 2 + byte_lane[None, :] + 2),
            tl.load(up_u8 + block_byte + 2 + byte_lane[None, :] + 3),
        )
        low_a = tl.load(q8_i32 + block * 8 + quads)
        high_a = tl.load(q8_i32 + block * 8 + 4 + quads)
        d8 = tl.load(q8_ds + block * 2).to(tl.float32)
        input_sum = tl.load(q8_ds + block * 2 + 1).to(tl.float32)

        gate_dot = (amd_sdot4_i8(gate_q & 0x0F0F0F0F, low_a[None, :], zero) +
                    amd_sdot4_i8((gate_q >> 4) & 0x0F0F0F0F,
                                 high_a[None, :], zero))
        up_dot = (amd_sdot4_i8(up_q & 0x0F0F0F0F, low_a[None, :], zero) +
                  amd_sdot4_i8((up_q >> 4) & 0x0F0F0F0F,
                               high_a[None, :], zero))
        zero_point = tl.where(quads == 0, 8.0 * input_sum, 0.0)
        gate_acc += gate_d * (d8 * gate_dot.to(tl.float32) - zero_point[None, :])
        up_acc += up_d * (d8 * up_dot.to(tl.float32) - zero_point[None, :])

    gate = tl.sum(gate_acc, axis=1)
    up = tl.sum(up_acc, axis=1)
    tl.store(output + row_ids, gate * tl.sigmoid(gate) * up, mask=row_ids < rows)


@triton.jit
def q4_ffn_swiglu_decode(
    gate_u8, gate_f16, up_u8, up_f16, x, output, k, rows,
    BLOCK_M: tl.constexpr,
):
    """Fused Q4_K gate/up GEMV plus SwiGLU for one decode column."""
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    lanes = tl.arange(0, 32)
    gate_acc = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    blocks = k // 256
    for block in tl.range(0, blocks):
        block_offset = (row_ids[:, None] * blocks + block) * 144
        block_half = block_offset // 2
        gate_d = tl.load(gate_f16 + block_half).to(tl.float32)
        gate_dmin = tl.load(gate_f16 + block_half + 1).to(tl.float32)
        up_d = tl.load(up_f16 + block_half).to(tl.float32)
        up_dmin = tl.load(up_f16 + block_half + 1).to(tl.float32)
        for chunk in tl.range(0, 4):
            low_group = chunk * 2
            high_group = low_group + 1
            low_index = chunk * 64 + lanes
            high_index = low_index + 32
            low_scale_index = low_group if low_group < 4 else low_group + 4
            low_scale_hi_index = low_group - 4 if low_group >= 4 else 0
            high_scale_index = high_group if high_group < 4 else high_group + 4
            high_scale_hi_index = high_group - 4 if high_group >= 4 else 0

            gate_low_scale_byte = tl.load(gate_u8 + block_offset + 4 + low_scale_index)
            gate_low_scale_hi = tl.load(gate_u8 + block_offset + 4 + low_scale_hi_index)
            gate_low_scale = (gate_low_scale_byte & 63 if low_group < 4 else
                              (gate_low_scale_byte & 15) | ((gate_low_scale_hi >> 6) << 4)).to(tl.float32)
            gate_low_min_byte = tl.load(gate_u8 + block_offset + 4 + low_group + 4)
            gate_low_min_hi = tl.load(gate_u8 + block_offset + 4 + low_group)
            gate_low_min = (gate_low_min_byte & 63 if low_group < 4 else
                            (gate_low_min_byte >> 4) | ((gate_low_min_hi >> 6) << 4)).to(tl.float32)
            gate_high_scale_byte = tl.load(gate_u8 + block_offset + 4 + high_scale_index)
            gate_high_scale_hi = tl.load(gate_u8 + block_offset + 4 + high_scale_hi_index)
            gate_high_scale = (gate_high_scale_byte & 63 if high_group < 4 else
                               (gate_high_scale_byte & 15) | ((gate_high_scale_hi >> 6) << 4)).to(tl.float32)
            gate_high_min_byte = tl.load(gate_u8 + block_offset + 4 + high_group + 4)
            gate_high_min_hi = tl.load(gate_u8 + block_offset + 4 + high_group)
            gate_high_min = (gate_high_min_byte & 63 if high_group < 4 else
                             (gate_high_min_byte >> 4) | ((gate_high_min_hi >> 6) << 4)).to(tl.float32)

            up_low_scale_byte = tl.load(up_u8 + block_offset + 4 + low_scale_index)
            up_low_scale_hi = tl.load(up_u8 + block_offset + 4 + low_scale_hi_index)
            up_low_scale = (up_low_scale_byte & 63 if low_group < 4 else
                            (up_low_scale_byte & 15) | ((up_low_scale_hi >> 6) << 4)).to(tl.float32)
            up_low_min_byte = tl.load(up_u8 + block_offset + 4 + low_group + 4)
            up_low_min_hi = tl.load(up_u8 + block_offset + 4 + low_group)
            up_low_min = (up_low_min_byte & 63 if low_group < 4 else
                          (up_low_min_byte >> 4) | ((up_low_min_hi >> 6) << 4)).to(tl.float32)
            up_high_scale_byte = tl.load(up_u8 + block_offset + 4 + high_scale_index)
            up_high_scale_hi = tl.load(up_u8 + block_offset + 4 + high_scale_hi_index)
            up_high_scale = (up_high_scale_byte & 63 if high_group < 4 else
                             (up_high_scale_byte & 15) | ((up_high_scale_hi >> 6) << 4)).to(tl.float32)
            up_high_min_byte = tl.load(up_u8 + block_offset + 4 + high_group + 4)
            up_high_min_hi = tl.load(up_u8 + block_offset + 4 + high_group)
            up_high_min = (up_high_min_byte & 63 if high_group < 4 else
                           (up_high_min_byte >> 4) | ((up_high_min_hi >> 6) << 4)).to(tl.float32)

            gate_q = tl.load(gate_u8 + block_offset + 16 + chunk * 32 + lanes[None, :])
            up_q = tl.load(up_u8 + block_offset + 16 + chunk * 32 + lanes[None, :])
            a0 = tl.load(x + block * 256 + low_index).to(tl.float32)
            a1 = tl.load(x + block * 256 + high_index).to(tl.float32)
            gate_low = gate_d * gate_low_scale * (gate_q & 15).to(tl.float32) - gate_dmin * gate_low_min
            gate_high = gate_d * gate_high_scale * (gate_q >> 4).to(tl.float32) - gate_dmin * gate_high_min
            up_low = up_d * up_low_scale * (up_q & 15).to(tl.float32) - up_dmin * up_low_min
            up_high = up_d * up_high_scale * (up_q >> 4).to(tl.float32) - up_dmin * up_high_min
            gate_acc += gate_low * a0[None, :] + gate_high * a1[None, :]
            up_acc += up_low * a0[None, :] + up_high * a1[None, :]
    gate = tl.sum(gate_acc, axis=1)
    up = tl.sum(up_acc, axis=1)
    tl.store(output + row_ids, gate * tl.sigmoid(gate) * up, mask=row_ids < rows)


@triton.jit
def q4_ffn_swiglu_split_k_f32(
    gate_u8, gate_f16, up_u8, up_f16, x, output, k, rows,
    SPLIT_K: tl.constexpr,
):
    """Fused Q4_K FFN with one physical wave per K partition."""
    row = tl.program_id(0)
    split_ids = tl.arange(0, SPLIT_K)[:, None]
    lanes = tl.arange(0, 32)[None, :]
    gate_acc = tl.zeros((SPLIT_K, 32), dtype=tl.float32)
    up_acc = tl.zeros((SPLIT_K, 32), dtype=tl.float32)
    blocks = k // 256
    for block0 in tl.range(0, blocks, SPLIT_K):
        block = block0 + split_ids
        valid_block = block < blocks
        block_offset = (row * blocks + block) * 144
        block_half = block_offset // 2
        gate_d = tl.load(gate_f16 + block_half,
                         mask=valid_block, other=0.0).to(tl.float32)
        gate_dmin = tl.load(gate_f16 + block_half + 1,
                            mask=valid_block, other=0.0).to(tl.float32)
        up_d = tl.load(up_f16 + block_half,
                       mask=valid_block, other=0.0).to(tl.float32)
        up_dmin = tl.load(up_f16 + block_half + 1,
                          mask=valid_block, other=0.0).to(tl.float32)
        for chunk in tl.range(0, 4):
            low_group = chunk * 2
            high_group = low_group + 1
            low_index = chunk * 64 + lanes
            high_index = low_index + 32
            low_scale_index = low_group if low_group < 4 else low_group + 4
            low_scale_hi_index = low_group - 4 if low_group >= 4 else 0
            high_scale_index = high_group if high_group < 4 else high_group + 4
            high_scale_hi_index = high_group - 4 if high_group >= 4 else 0

            gate_low_scale_byte = tl.load(
                gate_u8 + block_offset + 4 + low_scale_index,
                mask=valid_block, other=0)
            gate_low_scale_hi = tl.load(
                gate_u8 + block_offset + 4 + low_scale_hi_index,
                mask=valid_block, other=0)
            gate_low_scale = (gate_low_scale_byte & 63 if low_group < 4 else
                              (gate_low_scale_byte & 15) |
                              ((gate_low_scale_hi >> 6) << 4)).to(tl.float32)
            gate_low_min_byte = tl.load(
                gate_u8 + block_offset + 4 + low_group + 4,
                mask=valid_block, other=0)
            gate_low_min_hi = tl.load(
                gate_u8 + block_offset + 4 + low_group,
                mask=valid_block, other=0)
            gate_low_min = (gate_low_min_byte & 63 if low_group < 4 else
                            (gate_low_min_byte >> 4) |
                            ((gate_low_min_hi >> 6) << 4)).to(tl.float32)
            gate_high_scale_byte = tl.load(
                gate_u8 + block_offset + 4 + high_scale_index,
                mask=valid_block, other=0)
            gate_high_scale_hi = tl.load(
                gate_u8 + block_offset + 4 + high_scale_hi_index,
                mask=valid_block, other=0)
            gate_high_scale = (gate_high_scale_byte & 63 if high_group < 4 else
                               (gate_high_scale_byte & 15) |
                               ((gate_high_scale_hi >> 6) << 4)).to(tl.float32)
            gate_high_min_byte = tl.load(
                gate_u8 + block_offset + 4 + high_group + 4,
                mask=valid_block, other=0)
            gate_high_min_hi = tl.load(
                gate_u8 + block_offset + 4 + high_group,
                mask=valid_block, other=0)
            gate_high_min = (gate_high_min_byte & 63 if high_group < 4 else
                             (gate_high_min_byte >> 4) |
                             ((gate_high_min_hi >> 6) << 4)).to(tl.float32)

            up_low_scale_byte = tl.load(
                up_u8 + block_offset + 4 + low_scale_index,
                mask=valid_block, other=0)
            up_low_scale_hi = tl.load(
                up_u8 + block_offset + 4 + low_scale_hi_index,
                mask=valid_block, other=0)
            up_low_scale = (up_low_scale_byte & 63 if low_group < 4 else
                            (up_low_scale_byte & 15) |
                            ((up_low_scale_hi >> 6) << 4)).to(tl.float32)
            up_low_min_byte = tl.load(
                up_u8 + block_offset + 4 + low_group + 4,
                mask=valid_block, other=0)
            up_low_min_hi = tl.load(
                up_u8 + block_offset + 4 + low_group,
                mask=valid_block, other=0)
            up_low_min = (up_low_min_byte & 63 if low_group < 4 else
                          (up_low_min_byte >> 4) |
                          ((up_low_min_hi >> 6) << 4)).to(tl.float32)
            up_high_scale_byte = tl.load(
                up_u8 + block_offset + 4 + high_scale_index,
                mask=valid_block, other=0)
            up_high_scale_hi = tl.load(
                up_u8 + block_offset + 4 + high_scale_hi_index,
                mask=valid_block, other=0)
            up_high_scale = (up_high_scale_byte & 63 if high_group < 4 else
                             (up_high_scale_byte & 15) |
                             ((up_high_scale_hi >> 6) << 4)).to(tl.float32)
            up_high_min_byte = tl.load(
                up_u8 + block_offset + 4 + high_group + 4,
                mask=valid_block, other=0)
            up_high_min_hi = tl.load(
                up_u8 + block_offset + 4 + high_group,
                mask=valid_block, other=0)
            up_high_min = (up_high_min_byte & 63 if high_group < 4 else
                           (up_high_min_byte >> 4) |
                           ((up_high_min_hi >> 6) << 4)).to(tl.float32)

            gate_q = tl.load(gate_u8 + block_offset + 16 + chunk * 32 + lanes,
                             mask=valid_block, other=0)
            up_q = tl.load(up_u8 + block_offset + 16 + chunk * 32 + lanes,
                           mask=valid_block, other=0)
            a0 = tl.load(x + block * 256 + low_index,
                         mask=valid_block, other=0.0).to(tl.float32)
            a1 = tl.load(x + block * 256 + high_index,
                         mask=valid_block, other=0.0).to(tl.float32)
            gate_low = (gate_d * gate_low_scale * (gate_q & 15).to(tl.float32) -
                        gate_dmin * gate_low_min)
            gate_high = (gate_d * gate_high_scale * (gate_q >> 4).to(tl.float32) -
                         gate_dmin * gate_high_min)
            up_low = (up_d * up_low_scale * (up_q & 15).to(tl.float32) -
                      up_dmin * up_low_min)
            up_high = (up_d * up_high_scale * (up_q >> 4).to(tl.float32) -
                       up_dmin * up_high_min)
            gate_acc += gate_low * a0 + gate_high * a1
            up_acc += up_low * a0 + up_high * a1
    gate = tl.sum(tl.sum(gate_acc, axis=1), axis=0)
    up = tl.sum(tl.sum(up_acc, axis=1), axis=0)
    tl.store(output + row, gate * tl.sigmoid(gate) * up, mask=row < rows)


def bench(rows: int, columns: int, k: int, m: int, n: int, bk: int,
          warps: int, warmup: int, rep: int, ffn: bool) -> float:
    device = torch.device("cuda")
    weights = torch.randn((rows, k), device=device, dtype=torch.float16)
    up_weights = torch.randn((rows, k), device=device, dtype=torch.float16)
    x = torch.randn((columns, k), device=device, dtype=torch.float32)
    output = torch.empty((columns, rows), device=device, dtype=torch.float32)
    grid = (triton.cdiv(rows, m), triton.cdiv(columns, n))
    for _ in range(warmup):
        kernel = ffn_swiglu if ffn else f16_gemm
        args = (weights, up_weights, x, output) if ffn else (weights, x, output)
        kernel[grid](*args, k, rows, columns,
                     BLOCK_M=m, BLOCK_N=n, BLOCK_K=bk, num_warps=warps)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        kernel = ffn_swiglu if ffn else f16_gemm
        args = (weights, up_weights, x, output) if ffn else (weights, x, output)
        kernel[grid](*args, k, rows, columns,
                     BLOCK_M=m, BLOCK_N=n, BLOCK_K=bk, num_warps=warps)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / rep


def bench_quant(rows: int, k: int, kind: str, warmup: int, rep: int) -> None:
    device = torch.device("cuda")
    blocks = k // QK_K
    block_bytes = 144 if kind == "q4" else 210
    packed = torch.randint(0, 256, (rows * blocks * block_bytes,),
                           device=device, dtype=torch.uint8)
    weights_f16 = packed.view(torch.float16)
    x = torch.randn((k,), device=device, dtype=torch.float32)
    out = torch.empty((rows,), device=device, dtype=torch.float32)
    kernel = common.flagos_mul_mat_q4_k_f32 if kind == "q4" else common.flagos_mul_mat_q6_k_f32
    # Validate the production kernel on the same packed layout used by the
    # common generator before collecting timings.
    test_rows, test_blocks = 16, 3
    if kind == "q4":
        test_packed, test_reference = common.make_q4_k_weights(test_rows, test_blocks)
    else:
        test_packed, test_reference = common.make_q6_k_weights(test_rows, test_blocks)
    test_k = test_blocks * QK_K
    test_x = torch.randn((test_k,), device=device, dtype=torch.float32)
    test_out = torch.empty((test_rows,), device=device, dtype=torch.float32)
    kernel[(test_rows,)](test_packed, test_packed.view(torch.float16), test_x,
                         test_out, test_k, test_rows, num_warps=1)
    torch.testing.assert_close(test_out, test_reference @ test_x, rtol=2e-5, atol=2e-3)
    fixed_kernel = q4_gemv_fixed_blocks if kind == "q4" else q6_gemv_fixed_blocks
    fixed_out = torch.empty((test_rows,), device=device, dtype=torch.float32)
    fixed_kernel[(test_rows,)](
        test_packed, test_packed.view(torch.float16), test_x, fixed_out,
        test_rows, BLOCKS=test_blocks, num_warps=1)
    torch.testing.assert_close(fixed_out, test_reference @ test_x,
                               rtol=2e-5, atol=2e-3)
    rows_kernel = q4_gemv_rows if kind == "q4" else q6_gemv_rows
    rows_out = torch.empty((test_rows,), device=device, dtype=torch.float32)
    rows_kernel[(triton.cdiv(test_rows, 4),)](
        test_packed, test_packed.view(torch.float16), test_x, rows_out,
        test_k, test_rows, BLOCK_M=4, num_warps=1)
    torch.testing.assert_close(rows_out, test_reference @ test_x,
                               rtol=2e-5, atol=2e-3)
    narrow_kernel = q4_gemv_narrow if kind == "q4" else q6_gemv_narrow
    narrow_out = torch.empty((test_rows,), device=device, dtype=torch.float32)
    narrow_kernel[(test_rows,)](
        test_packed, test_packed.view(torch.float16), test_x, narrow_out,
        test_k, test_rows, BLOCK_M=1, num_warps=1)
    torch.testing.assert_close(narrow_out, test_reference @ test_x,
                               rtol=2e-5, atol=2e-3)
    if kind == "q6":
        for split_k in (2, 4, 8):
            split_out = torch.empty((test_rows,), device=device, dtype=torch.float32)
            q6_gemv_split_k_f32[(test_rows,)](
                test_packed, test_packed.view(torch.float16), test_x, split_out,
                test_k, test_rows, SPLIT_K=split_k, num_warps=split_k)
            torch.testing.assert_close(split_out, test_reference @ test_x,
                                       rtol=2e-5, atol=2e-3)
    fixed_blocks = k // QK_K
    cases = [("legacy_w1", kernel, False, 1),
             ("legacy_w2", kernel, False, 2),
             ("legacy_w4", kernel, False, 4),
             ("fixed_w1", fixed_kernel, True, 1),
             ("rows2_w1", q4_gemv_rows if kind == "q4" else q6_gemv_rows, "rows", 1),
             ("rows4_w1", q4_gemv_rows if kind == "q4" else q6_gemv_rows, "rows", 1),
             ("rows8_w1", q4_gemv_rows if kind == "q4" else q6_gemv_rows, "rows", 1),
             ("narrow1_w1", narrow_kernel, "narrow", 1),
             ("narrow4_w1", narrow_kernel, "narrow", 1),
             ("narrow4_w2", narrow_kernel, "narrow", 2),
             ("narrow4_w4", narrow_kernel, "narrow", 4),
             ("narrow8_w1", narrow_kernel, "narrow", 1),
             ("narrow16_w1", narrow_kernel, "narrow", 1),
             ("narrow32_w1", narrow_kernel, "narrow", 1)]
    if kind == "q4":
        # Native RDNA4 MMVQ assigns one output row to each of eight waves.
        # The validated FlagOS narrow8 symbol instead maps all eight rows onto
        # one wave. Sweep the matching multi-wave layouts explicitly rather
        # than assuming the one-wave Triton schedule remains optimal.
        cases.extend([
            ("narrow8_w2", narrow_kernel, "narrow", 2),
            ("narrow8_w4", narrow_kernel, "narrow", 4),
            ("narrow8_w8", narrow_kernel, "narrow", 8),
        ])
    if kind == "q6":
        # Native RDNA4 MMVQ uses eight waves per block.  Sweep matching
        # multi-wave row tiles explicitly; the earlier one-wave-only sweep
        # cannot reveal whether Triton distributes the (rows, 32) logical
        # tensor across physical waves without the high register pressure of
        # carrying every output row in one wave.
        cases.extend([
            ("narrow8_w2", narrow_kernel, "narrow", 2),
            ("narrow8_w4", narrow_kernel, "narrow", 4),
            ("narrow8_w8", narrow_kernel, "narrow", 8),
            ("narrow16_w4", narrow_kernel, "narrow", 4),
            ("narrow16_w8", narrow_kernel, "narrow", 8),
            ("narrow32_w8", narrow_kernel, "narrow", 8),
        ])
        cases.extend((f"split{split_k}", q6_gemv_split_k_f32, "split", split_k)
                     for split_k in (2, 4, 8))
    print(f"shape kind={kind} rows={rows} k={k}")
    print("variant ms GFLOP/s")
    for name, fn, fixed, warps in cases:
        tile = (int(name.split("_", 1)[0].replace("rows", ""))
                if fixed == "rows" else
                int(name.split("_", 1)[0].replace("narrow", ""))
                if fixed == "narrow" else 1)
        grid = (triton.cdiv(rows, tile),)
        for _ in range(warmup):
            if fixed is True:
                fn[grid](packed, weights_f16, x, out, rows,
                         BLOCKS=fixed_blocks, num_warps=warps)
            elif fixed == "rows":
                fn[grid](packed, weights_f16, x, out, k, rows,
                         BLOCK_M=tile, num_warps=warps)
            elif fixed == "narrow":
                fn[grid](packed, weights_f16, x, out, k, rows,
                         BLOCK_M=tile, num_warps=warps)
            elif fixed == "split":
                fn[grid](packed, weights_f16, x, out, k, rows,
                         SPLIT_K=warps, num_warps=warps)
            else:
                fn[grid](packed, weights_f16, x, out, k, rows,
                         num_warps=warps)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            if fixed is True:
                fn[grid](packed, weights_f16, x, out, rows,
                         BLOCKS=fixed_blocks, num_warps=warps)
            elif fixed == "rows":
                fn[grid](packed, weights_f16, x, out, k, rows,
                         BLOCK_M=tile, num_warps=warps)
            elif fixed == "narrow":
                fn[grid](packed, weights_f16, x, out, k, rows,
                         BLOCK_M=tile, num_warps=warps)
            elif fixed == "split":
                fn[grid](packed, weights_f16, x, out, k, rows,
                         SPLIT_K=warps, num_warps=warps)
            else:
                fn[grid](packed, weights_f16, x, out, k, rows,
                         num_warps=warps)
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end) / rep
        gflops = 2.0 * rows * k / (ms * 1e6)
        print(f"{name:7s} {ms:8.4f} {gflops:8.3f}", flush=True)


def bench_q6_q8(rows: int, k: int, warmup: int, rep: int) -> None:
    """Compare F32 Q6_K GEMV with quantize-plus-dot4 Q8 candidates."""
    if k % QK_K != 0:
        raise ValueError("Q6_K benchmark k must be divisible by 256")
    device = torch.device("cuda")
    blocks = k // QK_K
    packed = torch.randint(0, 256, (rows * blocks * 210,),
                           device=device, dtype=torch.uint8)
    weights_f16 = packed.view(torch.float16)
    x = torch.randn((k,), device=device, dtype=torch.float32)
    q8 = torch.empty((k,), device=device, dtype=torch.int8)
    q8_scales = torch.empty((k // 32,), device=device, dtype=torch.float32)
    q8_sums = torch.empty((k // 4,), device=device, dtype=torch.int32)
    out = torch.empty((rows,), device=device, dtype=torch.float32)

    # Validate both quantization and dot4 mapping on valid generated Q6_K data.
    test_rows, test_blocks = 16, 8
    test_k = test_blocks * QK_K
    test_packed, test_reference = common.make_q6_k_weights(test_rows, test_blocks)
    test_x = torch.randn((test_k,), device=device, dtype=torch.float32)
    test_q8 = torch.empty((test_k,), device=device, dtype=torch.int8)
    test_scales = torch.empty((test_k // 32,), device=device, dtype=torch.float32)
    test_sums = torch.empty((test_k // 4,), device=device, dtype=torch.int32)
    test_partials = torch.empty((test_rows * 8,), device=device, dtype=torch.float32)
    test_out = torch.empty((test_rows,), device=device, dtype=torch.float32)
    quantize_f32_q8_32[(test_k // 32,)](
        test_x, test_q8, test_scales, test_sums, test_k, num_warps=1)
    q6_gemv_q8_dot4[(test_rows,)](
        test_packed, test_packed.view(torch.float16), test_q8.view(torch.int32),
        test_scales, test_out, test_k, test_rows, BLOCK_M=1, num_warps=1)
    torch.cuda.synchronize()
    dequantized_x = test_q8.float() * test_scales.repeat_interleave(32)
    expected_q8 = test_reference @ dequantized_x
    torch.testing.assert_close(test_out, expected_q8, rtol=3e-5, atol=3e-2)
    q6_gemv_q8_dot4_packed[(test_rows,)](
        test_packed, test_packed.view(torch.int16), test_packed.view(torch.float16),
        test_q8.view(torch.int32), test_scales, test_sums, test_out,
        test_k, test_rows, BLOCK_M=1, num_warps=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(test_out, expected_q8, rtol=3e-5, atol=3e-2)
    for split_k in (2, 4, 8):
        q6_gemv_q8_dot4_split_k[(test_rows,)](
            test_packed, test_packed.view(torch.int16),
            test_packed.view(torch.float16), test_q8.view(torch.int32),
            test_scales, test_sums, test_out, test_k, test_rows,
            SPLIT_K=split_k, num_warps=split_k)
        torch.cuda.synchronize()
        torch.testing.assert_close(test_out, expected_q8, rtol=3e-5, atol=3e-2)
        q6_gemv_q8_dot4_partials[(test_rows * split_k,)](
            test_packed, test_packed.view(torch.int16),
            test_packed.view(torch.float16), test_q8.view(torch.int32),
            test_scales, test_sums, test_partials, test_k, test_rows,
            SPLIT_K=split_k, num_warps=1)
        reduce_split_k_f32[(test_rows,)](
            test_partials, test_out, test_rows, SPLIT_K=split_k, num_warps=1)
        torch.cuda.synchronize()
        torch.testing.assert_close(test_out, expected_q8, rtol=3e-5, atol=3e-2)
    expected_f32 = test_reference @ test_x
    q8_error = (test_out - expected_f32).abs()
    print(f"validation q8 mean_abs={q8_error.mean().item():.6f} "
          f"max_abs={q8_error.max().item():.6f} "
          f"relative_l2={(torch.linalg.vector_norm(test_out - expected_f32) / torch.linalg.vector_norm(expected_f32)).item():.6f}")

    quant_grid = (k // 32,)

    def timed(launch) -> float:
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            launch()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / rep

    quantize = lambda: quantize_f32_q8_32[quant_grid](
        x, q8, q8_scales, q8_sums, k, num_warps=1)
    quant_ms = timed(quantize)
    legacy_ms = timed(lambda: common.flagos_mul_mat_q6_k_f32[(rows,)](
        packed, weights_f16, x, out, k, rows, num_warps=1))

    print(f"shape q6_q8 rows={rows} k={k}")
    print(f"quantize_ms {quant_ms:.4f}")
    print(f"legacy_f32  {legacy_ms:.4f}")
    print("variant m warps gemv_ms total_ms speedup total_GFLOP/s")
    partial_storage = torch.empty((rows * 8,), device=device, dtype=torch.float32)
    for variant, kernel in (("byte", q6_gemv_q8_dot4),
                            ("packed", q6_gemv_q8_dot4_packed)):
        for m in (1, 2, 4, 8):
            for warps in (1, 2, 4, 8):
                grid = (triton.cdiv(rows, m),)

                def gemv() -> None:
                    if variant == "byte":
                        kernel[grid](
                            packed, weights_f16, q8.view(torch.int32), q8_scales,
                            out, k, rows, BLOCK_M=m, num_warps=warps)
                    else:
                        kernel[grid](
                            packed, packed.view(torch.int16), weights_f16,
                            q8.view(torch.int32), q8_scales, q8_sums, out,
                            k, rows, BLOCK_M=m, num_warps=warps)

                # Quantize before the GEMV-only timing so its input is initialized.
                quantize()
                gemv_ms = timed(gemv)

                def quantize_and_gemv() -> None:
                    quantize()
                    gemv()

                total_ms = timed(quantize_and_gemv)
                gflops = 2.0 * rows * k / (total_ms * 1e6)
                print(f"{variant:7s} {m:2d} {warps:5d} {gemv_ms:8.4f} {total_ms:8.4f} "
                      f"{legacy_ms / total_ms:7.3f}x {gflops:13.3f}", flush=True)
    for split_k in (2, 4, 8):
        grid = (rows,)

        def split_gemv() -> None:
            q6_gemv_q8_dot4_split_k[grid](
                packed, packed.view(torch.int16), weights_f16,
                q8.view(torch.int32), q8_scales, q8_sums, out,
                k, rows, SPLIT_K=split_k, num_warps=split_k)

        quantize()
        gemv_ms = timed(split_gemv)

        def quantize_and_split_gemv() -> None:
            quantize()
            split_gemv()

        total_ms = timed(quantize_and_split_gemv)
        gflops = 2.0 * rows * k / (total_ms * 1e6)
        print(f"split   {split_k:2d} {split_k:5d} {gemv_ms:8.4f} {total_ms:8.4f} "
              f"{legacy_ms / total_ms:7.3f}x {gflops:13.3f}", flush=True)
    for split_k in (2, 4, 8):
        def partial_gemv() -> None:
            q6_gemv_q8_dot4_partials[(rows * split_k,)](
                packed, packed.view(torch.int16), weights_f16,
                q8.view(torch.int32), q8_scales, q8_sums, partial_storage,
                k, rows, SPLIT_K=split_k, num_warps=1)
            reduce_split_k_f32[(rows,)](
                partial_storage, out, rows, SPLIT_K=split_k, num_warps=1)

        quantize()
        gemv_ms = timed(partial_gemv)

        def quantize_and_partial_gemv() -> None:
            quantize()
            partial_gemv()

        total_ms = timed(quantize_and_partial_gemv)
        gflops = 2.0 * rows * k / (total_ms * 1e6)
        print(f"partial {split_k:2d} {1:5d} {gemv_ms:8.4f} {total_ms:8.4f} "
              f"{legacy_ms / total_ms:7.3f}x {gflops:13.3f}", flush=True)


def bench_f16_gemv(rows: int, k: int, warmup: int, rep: int) -> None:
    device = torch.device("cuda")
    weights = torch.randn((rows, k), device=device, dtype=torch.float16)
    x = torch.randn((k,), device=device, dtype=torch.float32)
    out = torch.empty((rows,), device=device, dtype=torch.float32)
    cases = []
    for m in (1, 2, 4, 8, 16, 32):
        for bk in (64, 128, 256):
            for warps in (1, 2, 4):
                cases.append((m, bk, warps))
    print(f"shape f16_gemv rows={rows} k={k}")
    print("m bk warps ms GFLOP/s")
    for m, bk, warps in cases:
        grid = (triton.cdiv(rows, m),)
        for _ in range(warmup):
            f16_gemv[grid](weights, x, out, k, rows,
                           BLOCK_M=m, BLOCK_K=bk, num_warps=warps)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            f16_gemv[grid](weights, x, out, k, rows,
                           BLOCK_M=m, BLOCK_K=bk, num_warps=warps)
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end) / rep
        gflops = 2.0 * rows * k / (ms * 1e6)
        print(f"{m:2d} {bk:3d} {warps:5d} {ms:8.4f} {gflops:8.3f}", flush=True)


def bench_q4_ffn_decode(rows: int, k: int, warmup: int, rep: int) -> None:
    device = torch.device("cuda")
    blocks = k // QK_K
    gate, _ = common.make_q4_k_weights(rows, blocks)
    up, _ = common.make_q4_k_weights(rows, blocks)
    x = torch.randn((k,), device=device, dtype=torch.float32)
    out = torch.empty((rows,), device=device, dtype=torch.float32)

    test_rows, test_blocks = 16, 3
    test_gate, gate_ref = common.make_q4_k_weights(test_rows, test_blocks)
    test_up, up_ref = common.make_q4_k_weights(test_rows, test_blocks)
    test_k = test_blocks * QK_K
    test_x = torch.randn((test_k,), device=device, dtype=torch.float32)
    expected_gate = gate_ref @ test_x
    expected_up = up_ref @ test_x
    expected = torch.nn.functional.silu(expected_gate) * expected_up

    print(f"shape q4_ffn_decode rows={rows} k={k}")
    print("m warps fused_ms separate_ms speedup")
    for m in (1, 2, 4, 8, 16, 32):
        for warps in (1, 2):
            test_out = torch.empty((test_rows,), device=device, dtype=torch.float32)
            q4_ffn_swiglu_decode[(triton.cdiv(test_rows, m),)](
                test_gate, test_gate.view(torch.float16),
                test_up, test_up.view(torch.float16), test_x, test_out,
                test_k, test_rows, BLOCK_M=m, num_warps=warps)
            torch.testing.assert_close(test_out, expected, rtol=5e-4, atol=5e-2)

            grid = (triton.cdiv(rows, m),)
            for _ in range(warmup):
                q4_ffn_swiglu_decode[grid](
                    gate, gate.view(torch.float16), up, up.view(torch.float16),
                    x, out, k, rows, BLOCK_M=m, num_warps=warps)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(rep):
                q4_ffn_swiglu_decode[grid](
                    gate, gate.view(torch.float16), up, up.view(torch.float16),
                    x, out, k, rows, BLOCK_M=m, num_warps=warps)
            end.record()
            end.synchronize()
            fused_ms = start.elapsed_time(end) / rep

            gate_out = torch.empty_like(out)
            up_out = torch.empty_like(out)
            separate_grid = (triton.cdiv(rows, 8),)
            for _ in range(warmup):
                q4_gemv_narrow[separate_grid](gate, gate.view(torch.float16), x, gate_out,
                                              k, rows, BLOCK_M=8, num_warps=1)
                q4_gemv_narrow[separate_grid](up, up.view(torch.float16), x, up_out,
                                              k, rows, BLOCK_M=8, num_warps=1)
            torch.cuda.synchronize()
            start.record()
            for _ in range(rep):
                q4_gemv_narrow[separate_grid](gate, gate.view(torch.float16), x, gate_out,
                                              k, rows, BLOCK_M=8, num_warps=1)
                q4_gemv_narrow[separate_grid](up, up.view(torch.float16), x, up_out,
                                              k, rows, BLOCK_M=8, num_warps=1)
            end.record()
            end.synchronize()
            separate_ms = start.elapsed_time(end) / rep
            print(f"{m:2d} {warps:5d} {fused_ms:8.4f} {separate_ms:11.4f} "
                  f"{separate_ms / fused_ms:7.3f}x", flush=True)
    for split_k in (2, 4, 8):
        test_out = torch.empty((test_rows,), device=device, dtype=torch.float32)
        q4_ffn_swiglu_split_k_f32[(test_rows,)](
            test_gate, test_gate.view(torch.float16),
            test_up, test_up.view(torch.float16), test_x, test_out,
            test_k, test_rows, SPLIT_K=split_k, num_warps=split_k)
        torch.testing.assert_close(test_out, expected, rtol=5e-4, atol=5e-2)
        grid = (rows,)
        for _ in range(warmup):
            q4_ffn_swiglu_split_k_f32[grid](
                gate, gate.view(torch.float16), up, up.view(torch.float16),
                x, out, k, rows, SPLIT_K=split_k, num_warps=split_k)
        torch.cuda.synchronize()
        start.record()
        for _ in range(rep):
            q4_ffn_swiglu_split_k_f32[grid](
                gate, gate.view(torch.float16), up, up.view(torch.float16),
                x, out, k, rows, SPLIT_K=split_k, num_warps=split_k)
        end.record()
        end.synchronize()
        fused_ms = start.elapsed_time(end) / rep
        print(f"s{split_k:<1d} {split_k:5d} {fused_ms:8.4f} {separate_ms:11.4f} "
              f"{separate_ms / fused_ms:7.3f}x", flush=True)


def bench_q40_ffn_decode(rows: int, k: int, warmup: int, rep: int) -> None:
    """Benchmark the provider-local packed-Q4_0 FFN decode fusion."""
    if rows <= 0 or k <= 0 or k % common.QK4_0 != 0:
        raise ValueError("Q4_0 FFN benchmark requires positive rows and k divisible by 32")
    device = torch.device("cuda")
    blocks = k // common.QK4_0
    gate, _ = make_q40_weights(rows, blocks, 20260902)
    up, _ = make_q40_weights(rows, blocks, 20260903)
    x = torch.randn((k,), device=device, dtype=torch.float32)
    out = torch.empty((rows,), device=device, dtype=torch.float32)
    test_rows, test_blocks = 16, 5
    test_gate, gate_ref = make_q40_weights(
        test_rows, test_blocks, 20260904, with_reference=True)
    test_up, up_ref = make_q40_weights(
        test_rows, test_blocks, 20260905, with_reference=True)
    test_k = test_blocks * common.QK4_0
    test_x = torch.randn((test_k,), device=device, dtype=torch.float32)
    test_out = torch.empty((test_rows,), device=device, dtype=torch.float32)
    amd_kernels.flagos_ffn_swiglu_q4_0_f32_decode[(2,)](
        test_gate, test_gate.view(torch.float16), test_up, test_up.view(torch.float16),
        test_x, test_out, test_k, test_rows, BLOCK_M=8, num_warps=1, waves_per_eu=4)
    expected = torch.nn.functional.silu(gate_ref @ test_x) * (up_ref @ test_x)
    torch.testing.assert_close(test_out, expected, rtol=5e-4, atol=5e-2)
    print(f"shape q40_ffn_decode rows={rows} k={k}")
    print("m warps fused_ms separate_ms speedup")
    for m in (1, 2, 4, 8, 16):
        for warps in (1, 2):
            fused_grid = (triton.cdiv(rows, m),)
            gate_out, up_out = torch.empty_like(out), torch.empty_like(out)
            separate_grid = (rows,)
            split_grid = (triton.cdiv(rows, common.BLOCK_SIZE),)

            def fused() -> None:
                amd_kernels.flagos_ffn_swiglu_q4_0_f32_decode[fused_grid](
                    gate, gate.view(torch.float16), up, up.view(torch.float16),
                    x, out, k, rows, BLOCK_M=m, num_warps=warps, waves_per_eu=4)

            def separate() -> None:
                common.flagos_mul_mat_q4_0_f32[separate_grid](
                    gate, gate.view(torch.float16), x, gate_out, k, rows,
                    num_warps=1)
                common.flagos_mul_mat_q4_0_f32[separate_grid](
                    up, up.view(torch.float16), x, up_out, k, rows,
                    num_warps=1)
                common.flagos_swiglu_split_f32[split_grid](
                    gate_out, up_out, out, rows, BLOCK=common.BLOCK_SIZE,
                    num_warps=common.NUM_WARPS)

            def timed(operation) -> float:
                for _ in range(warmup):
                    operation()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(rep):
                    operation()
                end.record()
                end.synchronize()
                return start.elapsed_time(end) / rep

            samples = {"fused": [], "separate": []}
            orders = (
                (("fused", fused), ("separate", separate)),
                (("separate", separate), ("fused", fused)),
            )
            for operations in orders:
                for name, operation in operations:
                    samples[name].append(timed(operation))
            fused_ms = sum(samples["fused"]) / len(samples["fused"])
            separate_ms = sum(samples["separate"]) / len(samples["separate"])
            print(f"{m:2d} {warps:5d} {fused_ms:8.4f} {separate_ms:11.4f} "
                  f"{separate_ms / fused_ms:7.3f}x", flush=True)


def bench_q4_ffn_staged(rows: int, k: int, warmup: int, rep: int) -> None:
    """Focused reversed-order screen of low-VGPR staged Q4 FFN shapes."""
    if rows <= 0 or rows % 8 != 0 or k <= 0 or k % QK_K != 0:
        raise ValueError("staged Q4 FFN requires rows divisible by 8 and k divisible by 256")
    device = torch.device("cuda")
    blocks = k // QK_K
    gate, _ = common.make_q4_k_weights(rows, blocks)
    up, _ = common.make_q4_k_weights(rows, blocks)
    x = torch.randn((k,), device=device, dtype=torch.float32)
    baseline_out = torch.empty((rows,), device=device, dtype=torch.float32)
    staged_m8_w2_out = torch.empty_like(baseline_out)
    staged_m8_w4_out = torch.empty_like(baseline_out)
    staged_m4_w2_out = torch.empty_like(baseline_out)
    grid8 = (rows // 8,)
    grid4 = (rows // 4,)

    baseline = lambda: amd_kernels.flagos_ffn_swiglu_q4_k_f32_decode[grid8](
        gate, gate.view(torch.float16), up, up.view(torch.float16),
        x, baseline_out, k, rows, BLOCK_M=8, num_warps=2, waves_per_eu=4)
    staged_m8_w2 = lambda: amd_kernels.flagos_ffn_swiglu_q4_k_f32_decode_staged[grid8](
        gate, gate.view(torch.float16), up, up.view(torch.float16),
        x, staged_m8_w2_out, k, rows, BLOCK_M=8, num_warps=2, waves_per_eu=4)
    staged_m8_w4 = lambda: amd_kernels.flagos_ffn_swiglu_q4_k_f32_decode_staged[grid8](
        gate, gate.view(torch.float16), up, up.view(torch.float16),
        x, staged_m8_w4_out, k, rows, BLOCK_M=8, num_warps=4, waves_per_eu=4)
    staged_m4_w2 = lambda: amd_kernels.flagos_ffn_swiglu_q4_k_f32_decode_staged[grid4](
        gate, gate.view(torch.float16), up, up.view(torch.float16),
        x, staged_m4_w2_out, k, rows, BLOCK_M=4, num_warps=2, waves_per_eu=4)

    baseline()
    staged_m8_w2()
    staged_m8_w4()
    staged_m4_w2()
    torch.cuda.synchronize()
    for candidate in (staged_m8_w2_out, staged_m8_w4_out, staged_m4_w2_out):
        torch.testing.assert_close(candidate, baseline_out, rtol=2e-5, atol=2e-3)

    def timed(operation) -> float:
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            operation()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / rep

    operations = [
        ("baseline_m8_w2", baseline),
        ("staged_m8_w2", staged_m8_w2),
        ("staged_m8_w4", staged_m8_w4),
        ("staged_m4_w2", staged_m4_w2),
    ]
    samples = {name: [] for name, _ in operations}
    for order in (operations, list(reversed(operations))):
        for name, operation in order:
            samples[name].append(timed(operation))

    averages = {name: sum(values) / len(values) for name, values in samples.items()}
    baseline_ms = averages["baseline_m8_w2"]
    print(f"shape q4_ffn_staged rows={rows} k={k} waves_per_eu=4")
    print("candidate         avg_ms    speedup")
    for name, _ in operations:
        candidate_ms = averages[name]
        print(f"{name:16s} {candidate_ms:8.6f} {baseline_ms / candidate_ms:8.4f}x")


def bench_q40_ffn_staged(rows: int, k: int, warmup: int, rep: int) -> None:
    """Screen sequential-accumulator Q4_0 FFN candidates."""
    if rows <= 0 or rows % 8 != 0 or k <= 0 or k % common.QK4_0 != 0:
        raise ValueError("staged Q4_0 FFN requires rows divisible by 8 and k divisible by 32")
    device = torch.device("cuda")
    blocks = k // common.QK4_0
    gate, _ = make_q40_weights(rows, blocks, 20260906)
    up, _ = make_q40_weights(rows, blocks, 20260907)
    x = torch.randn((k,), device=device, dtype=torch.float32)
    baseline_out = torch.empty((rows,), device=device, dtype=torch.float32)
    staged_m8_w1_out = torch.empty_like(baseline_out)
    staged_m8_w2_out = torch.empty_like(baseline_out)
    staged_m4_w1_out = torch.empty_like(baseline_out)
    grid8 = (rows // 8,)
    grid4 = (rows // 4,)

    test_rows, test_blocks = 16, 5
    test_k = test_blocks * common.QK4_0
    test_gate, gate_ref = make_q40_weights(
        test_rows, test_blocks, 20260908, with_reference=True)
    test_up, up_ref = make_q40_weights(
        test_rows, test_blocks, 20260909, with_reference=True)
    test_x = torch.randn((test_k,), device=device, dtype=torch.float32)
    expected = torch.nn.functional.silu(gate_ref @ test_x) * (up_ref @ test_x)
    for block_m, warps in ((8, 1), (8, 2), (4, 1)):
        test_output = torch.empty((test_rows,), device=device, dtype=torch.float32)
        amd_kernels.flagos_ffn_swiglu_q4_0_f32_decode_staged[
            (triton.cdiv(test_rows, block_m),)
        ](
            test_gate, test_gate.view(torch.float16),
            test_up, test_up.view(torch.float16), test_x, test_output,
            test_k, test_rows, BLOCK_M=block_m, num_warps=warps,
            waves_per_eu=4)
        torch.testing.assert_close(test_output, expected, rtol=5e-4, atol=5e-2)

    baseline = lambda: amd_kernels.flagos_ffn_swiglu_q4_0_f32_decode[grid8](
        gate, gate.view(torch.float16), up, up.view(torch.float16),
        x, baseline_out, k, rows, BLOCK_M=8, num_warps=1, waves_per_eu=4)
    staged_m8_w1 = lambda: amd_kernels.flagos_ffn_swiglu_q4_0_f32_decode_staged[grid8](
        gate, gate.view(torch.float16), up, up.view(torch.float16),
        x, staged_m8_w1_out, k, rows, BLOCK_M=8, num_warps=1, waves_per_eu=4)
    staged_m8_w2 = lambda: amd_kernels.flagos_ffn_swiglu_q4_0_f32_decode_staged[grid8](
        gate, gate.view(torch.float16), up, up.view(torch.float16),
        x, staged_m8_w2_out, k, rows, BLOCK_M=8, num_warps=2, waves_per_eu=4)
    staged_m4_w1 = lambda: amd_kernels.flagos_ffn_swiglu_q4_0_f32_decode_staged[grid4](
        gate, gate.view(torch.float16), up, up.view(torch.float16),
        x, staged_m4_w1_out, k, rows, BLOCK_M=4, num_warps=1, waves_per_eu=4)

    baseline()
    staged_m8_w1()
    staged_m8_w2()
    staged_m4_w1()
    torch.cuda.synchronize()
    for candidate in (staged_m8_w1_out, staged_m8_w2_out, staged_m4_w1_out):
        torch.testing.assert_close(candidate, baseline_out, rtol=2e-5, atol=2e-3)

    def timed(operation) -> float:
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            operation()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / rep

    operations = [
        ("baseline_m8_w1", baseline),
        ("staged_m8_w1", staged_m8_w1),
        ("staged_m8_w2", staged_m8_w2),
        ("staged_m4_w1", staged_m4_w1),
    ]
    samples = {name: [] for name, _ in operations}
    for order in (operations, list(reversed(operations))):
        for name, operation in order:
            samples[name].append(timed(operation))

    averages = {name: sum(values) / len(values) for name, values in samples.items()}
    baseline_ms = averages["baseline_m8_w1"]
    print(f"shape q40_ffn_staged rows={rows} k={k} waves_per_eu=4")
    print("candidate         avg_ms    speedup")
    for name, _ in operations:
        candidate_ms = averages[name]
        print(f"{name:16s} {candidate_ms:8.6f} {baseline_ms / candidate_ms:8.4f}x")


def bench_q40_q8_ffn(rows: int, k: int, warmup: int, rep: int) -> None:
    """Screen a native-style Q8 activation path for the Q4_0 decode FFN."""
    if rows <= 0 or rows % 8 != 0 or k <= 0 or k % common.QK4_0 != 0:
        raise ValueError("Q4_0/Q8 FFN requires rows divisible by 8 and k divisible by 32")
    device = torch.device("cuda")
    blocks = k // common.QK4_0
    gate, _ = make_q40_weights(rows, blocks, 20260912)
    up, _ = make_q40_weights(rows, blocks, 20260913)
    x = torch.randn((k,), device=device, dtype=torch.float32)
    q8 = torch.empty((k,), device=device, dtype=torch.int8)
    q8_ds = torch.empty((blocks, 2), device=device, dtype=torch.float16)
    output = torch.empty((rows,), device=device, dtype=torch.float32)
    baseline_output = torch.empty_like(output)

    test_rows = 16
    test_blocks = 5
    test_k = test_blocks * common.QK4_0
    test_gate, gate_reference = make_q40_weights(
        test_rows, test_blocks, 20260914, with_reference=True)
    test_up, up_reference = make_q40_weights(
        test_rows, test_blocks, 20260915, with_reference=True)
    test_x = torch.randn((test_k,), device=device, dtype=torch.float32)
    test_q8 = torch.empty((test_k,), device=device, dtype=torch.int8)
    test_ds = torch.empty((test_blocks, 2), device=device, dtype=torch.float16)
    quantize_f32_q8_1_32[(test_blocks,)](
        test_x, test_q8, test_ds, test_k, num_warps=1)
    test_scales = test_ds[:, 0].float()
    input_sums = test_ds[:, 1].float()
    dequantized_x = test_q8.float() * test_scales.repeat_interleave(32)
    dequantized_sums = dequantized_x.reshape(test_blocks, 32).sum(dim=1)
    sum_delta = input_sums - dequantized_sums
    gate_scales = test_gate.view(torch.float16).reshape(
        test_rows, test_blocks, common.Q4_0_BLOCK_BYTES // 2)[:, :, 0].float()
    up_scales = test_up.view(torch.float16).reshape(
        test_rows, test_blocks, common.Q4_0_BLOCK_BYTES // 2)[:, :, 0].float()
    gate_q8 = gate_reference @ dequantized_x - 8.0 * (gate_scales * sum_delta).sum(dim=1)
    up_q8 = up_reference @ dequantized_x - 8.0 * (up_scales * sum_delta).sum(dim=1)
    expected_q8 = torch.nn.functional.silu(gate_q8) * up_q8
    expected_f32 = (torch.nn.functional.silu(gate_reference @ test_x) *
                    (up_reference @ test_x))
    for block_m, warps in ((1, 1), (2, 1), (4, 1), (8, 1), (8, 2), (8, 4)):
        test_output = torch.empty((test_rows,), device=device, dtype=torch.float32)
        q40_ffn_swiglu_q8_dot4[(triton.cdiv(test_rows, block_m),)](
            test_gate, test_gate.view(torch.float16),
            test_up, test_up.view(torch.float16),
            test_q8.view(torch.int32), test_ds, test_output,
            test_k, test_rows, BLOCK_M=block_m, num_warps=warps,
            waves_per_eu=4)
        torch.testing.assert_close(test_output, expected_q8, rtol=5e-4, atol=5e-2)
    relative_l2 = (torch.linalg.vector_norm(test_output - expected_f32) /
                   torch.linalg.vector_norm(expected_f32)).item()
    print(f"validation q40_ffn_q8 relative_l2={relative_l2:.6f}")

    def quantize() -> None:
        quantize_f32_q8_1_32[(blocks,)](
            x, q8, q8_ds, k, num_warps=1)

    def baseline() -> None:
        amd_kernels.flagos_ffn_swiglu_q4_0_f32_decode[(rows // 8,)](
            gate, gate.view(torch.float16), up, up.view(torch.float16),
            x, baseline_output, k, rows, BLOCK_M=8, num_warps=1,
            waves_per_eu=4)

    def timed(operation) -> float:
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            operation()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / rep

    quantize_ms = timed(quantize)
    print(f"shape q40_q8_ffn rows={rows} k={k}")
    print(f"quantize_ms {quantize_ms:.6f}")
    print("candidate  f32_ms gemv_ms total_ms speedup")
    for block_m, warps in ((1, 1), (2, 1), (4, 1), (8, 1), (8, 2), (8, 4)):
        grid = (triton.cdiv(rows, block_m),)

        def candidate() -> None:
            q40_ffn_swiglu_q8_dot4[grid](
                gate, gate.view(torch.float16), up, up.view(torch.float16),
                q8.view(torch.int32), q8_ds, output,
                k, rows, BLOCK_M=block_m, num_warps=warps, waves_per_eu=4)

        def combined() -> None:
            quantize()
            candidate()

        quantize()
        samples = {"baseline": [], "candidate": [], "combined": []}
        for operations in (
                (("baseline", baseline), ("candidate", candidate), ("combined", combined)),
                (("combined", combined), ("candidate", candidate), ("baseline", baseline))):
            for name, operation in operations:
                samples[name].append(timed(operation))
        averages = {name: sum(values) / len(values) for name, values in samples.items()}
        print(f"m{block_m}_w{warps:<1d} {averages['baseline']:7.6f} "
              f"{averages['candidate']:7.6f} {averages['combined']:8.6f} "
              f"{averages['baseline'] / averages['combined']:7.4f}x", flush=True)


def bench_q4_narrow8_warps(row_shapes: tuple[int, ...], k: int,
                           warmup: int, rep: int) -> None:
    """Compare the exact production Q4 narrow8 source at 1, 4, and 8 waves."""
    if not row_shapes or any(rows <= 0 or rows % 8 != 0 for rows in row_shapes):
        raise ValueError("Q4 narrow8 row shapes must be positive multiples of 8")
    if k <= 0 or k % QK_K != 0:
        raise ValueError("Q4 narrow8 requires k divisible by 256")
    device = torch.device("cuda")

    # First validate the alternate workgroup mappings against a small unpacked
    # reference. The performance shapes then reuse valid Q4_K data and compare
    # bitwise-equivalent math against the production one-wave schedule.
    test_rows, test_blocks = 16, 3
    test_k = test_blocks * QK_K
    test_packed, test_reference = common.make_q4_k_weights(test_rows, test_blocks)
    test_x = torch.randn((test_k,), device=device, dtype=torch.float32)
    expected = test_reference @ test_x
    for warps in (1, 4, 8):
        test_out = torch.empty((test_rows,), device=device, dtype=torch.float32)
        common.flagos_mul_mat_q4_k_f32_narrow8[(test_rows // 8,)](
            test_packed, test_packed.view(torch.float16), test_x, test_out,
            test_k, test_rows, num_warps=warps)
        torch.testing.assert_close(test_out, expected, rtol=2e-5, atol=2e-3)
    torch.cuda.synchronize()

    def timed(operation) -> float:
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            operation()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / rep

    for rows in row_shapes:
        packed, dequantized = common.make_q4_k_weights(rows, k // QK_K)
        del dequantized
        x = torch.randn((k,), device=device, dtype=torch.float32)
        outputs = {
            warps: torch.empty((rows,), device=device, dtype=torch.float32)
            for warps in (1, 4, 8)
        }
        grid = (rows // 8,)

        def launch(warps: int) -> None:
            common.flagos_mul_mat_q4_k_f32_narrow8[grid](
                packed, packed.view(torch.float16), x, outputs[warps],
                k, rows, num_warps=warps)

        for warps in (1, 4, 8):
            launch(warps)
        torch.cuda.synchronize()
        for warps in (4, 8):
            torch.testing.assert_close(
                outputs[warps], outputs[1], rtol=2e-5, atol=2e-3)

        operations = [(f"narrow8_w{warps}", lambda warps=warps: launch(warps))
                      for warps in (1, 4, 8)]
        samples = {name: [] for name, _ in operations}
        for order in (operations, list(reversed(operations))):
            for name, operation in order:
                samples[name].append(timed(operation))
        averages = {name: sum(values) / len(values) for name, values in samples.items()}
        baseline_ms = averages["narrow8_w1"]
        print(f"shape q4_narrow8_warps rows={rows} k={k}")
        print("candidate        avg_ms    speedup")
        for name, _ in operations:
            candidate_ms = averages[name]
            print(f"{name:15s} {candidate_ms:8.6f} {baseline_ms / candidate_ms:8.4f}x")


def bench_q40_narrow_tiles(rows: int, k: int, warmup: int, rep: int) -> None:
    """Compare row-tiled Q4_0 GEMV candidates with the one-row kernel."""
    tiles = (4, 8, 16, 32)
    if rows <= 0 or any(rows % tile != 0 for tile in tiles):
        raise ValueError("Q4_0 narrow tile benchmark requires rows divisible by 32")
    if k <= 0 or k % common.QK4_0 != 0:
        raise ValueError("Q4_0 narrow tile benchmark requires k divisible by 32")
    device = torch.device("cuda")

    test_rows = 32
    test_k = 3 * common.QK4_0
    test_packed, test_reference = make_q40_weights(
        test_rows, 3, 20260910, with_reference=True)
    test_x = torch.randn((test_k,), device=device, dtype=torch.float32)
    expected = test_reference @ test_x
    for tile in tiles:
        test_output = torch.empty((test_rows,), device=device, dtype=torch.float32)
        common.flagos_mul_mat_q4_0_f32_narrow[(test_rows // tile,)](
            test_packed, test_packed.view(torch.float16), test_x, test_output,
            test_k, test_rows, BLOCK_M=tile, num_warps=1, waves_per_eu=4)
        torch.testing.assert_close(test_output, expected, rtol=2e-5, atol=2e-4)
    torch.cuda.synchronize()

    packed, _ = make_q40_weights(rows, k // common.QK4_0, 20260911)
    x = torch.randn((k,), device=device, dtype=torch.float32)
    outputs = {
        tile: torch.empty((rows,), device=device, dtype=torch.float32)
        for tile in tiles
    }
    baseline_output = torch.empty((rows,), device=device, dtype=torch.float32)

    def baseline() -> None:
        common.flagos_mul_mat_q4_0_f32[(rows,)](
            packed, packed.view(torch.float16), x, baseline_output,
            k, rows, num_warps=1)

    def candidate(tile: int) -> None:
        common.flagos_mul_mat_q4_0_f32_narrow[(rows // tile,)](
            packed, packed.view(torch.float16), x, outputs[tile],
            k, rows, BLOCK_M=tile, num_warps=1, waves_per_eu=4)

    baseline()
    for tile in tiles:
        candidate(tile)
    torch.cuda.synchronize()
    for tile in tiles:
        torch.testing.assert_close(outputs[tile], baseline_output, rtol=2e-5, atol=2e-4)

    def timed(operation) -> float:
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            operation()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / rep

    operations = [("baseline_m1", baseline)] + [
        (f"narrow_m{tile}", lambda tile=tile: candidate(tile))
        for tile in tiles
    ]
    samples = {name: [] for name, _ in operations}
    for order in (operations, list(reversed(operations))):
        for name, operation in order:
            samples[name].append(timed(operation))
    averages = {name: sum(values) / len(values) for name, values in samples.items()}
    baseline_ms = averages["baseline_m1"]
    print(f"shape q40_narrow_tiles rows={rows} k={k} waves_per_eu=4")
    print("candidate        avg_ms    speedup")
    for name, _ in operations:
        candidate_ms = averages[name]
        print(f"{name:15s} {candidate_ms:8.6f} {baseline_ms / candidate_ms:8.4f}x")


def bench_q40_narrow_qwen35(warmup: int, rep: int) -> None:
    """Screen row tiles on the direct Q4_0 GEMV shapes in Qwen3.5-4B."""
    # (rows, k): K/V, recurrent gate, Q/QKV, attention output, and FFN down.
    # The 9216x2560 gate/up pair is covered by bench_q40_ffn_staged instead.
    shapes = (
        (1024, 2560),
        (4096, 2560),
        (8192, 2560),
        (2560, 4096),
        (2560, 9216),
    )
    for rows, k in shapes:
        bench_q40_narrow_tiles(rows, k, warmup, rep)


def bench_q4_q8_ffn(rows: int, k: int, warmup: int, rep: int) -> None:
    """Benchmark shared Q8 activation quantization plus fused Q4_K FFN."""
    if k % QK_K != 0:
        raise ValueError("Q4_K benchmark k must be divisible by 256")
    device = torch.device("cuda")
    blocks = k // QK_K
    packed_bytes = rows * blocks * 144
    gate = torch.randint(0, 256, (packed_bytes,), device=device, dtype=torch.uint8)
    up = torch.randint(0, 256, (packed_bytes,), device=device, dtype=torch.uint8)
    x = torch.randn((k,), device=device, dtype=torch.float32)
    q8 = torch.empty((k,), device=device, dtype=torch.int8)
    q8_scales = torch.empty((k // 32,), device=device, dtype=torch.float32)
    q8_sums = torch.empty((k // 4,), device=device, dtype=torch.int32)
    out = torch.empty((rows,), device=device, dtype=torch.float32)

    test_rows, test_blocks = 16, 3
    test_k = test_blocks * QK_K
    test_gate, gate_reference = common.make_q4_k_weights(test_rows, test_blocks)
    test_up, up_reference = common.make_q4_k_weights(test_rows, test_blocks)
    test_x = torch.randn((test_k,), device=device, dtype=torch.float32)
    test_q8 = torch.empty((test_k,), device=device, dtype=torch.int8)
    test_scales = torch.empty((test_k // 32,), device=device, dtype=torch.float32)
    test_sums = torch.empty((test_k // 4,), device=device, dtype=torch.int32)
    test_out = torch.empty((test_rows,), device=device, dtype=torch.float32)
    quantize_f32_q8_32[(test_k // 32,)](
        test_x, test_q8, test_scales, test_sums, test_k, num_warps=1)
    q4_ffn_swiglu_q8_dot4[(test_rows,)](
        test_gate, test_gate.view(torch.int32), test_gate.view(torch.float16),
        test_up, test_up.view(torch.int32), test_up.view(torch.float16),
        test_q8.view(torch.int32), test_scales, test_sums, test_out,
        test_k, test_rows, BLOCK_M=1, num_warps=1)
    torch.cuda.synchronize()
    dequantized_x = test_q8.float() * test_scales.repeat_interleave(32)
    expected_gate_q8 = gate_reference @ dequantized_x
    expected_up_q8 = up_reference @ dequantized_x
    expected_q8 = torch.nn.functional.silu(expected_gate_q8) * expected_up_q8
    torch.testing.assert_close(test_out, expected_q8, rtol=5e-4, atol=8e-2)
    expected_f32 = (torch.nn.functional.silu(gate_reference @ test_x) *
                    (up_reference @ test_x))
    relative_l2 = (torch.linalg.vector_norm(test_out - expected_f32) /
                   torch.linalg.vector_norm(expected_f32)).item()
    print(f"validation q4_ffn_q8 relative_l2={relative_l2:.6f}")

    quant_grid = (k // 32,)

    def timed(launch) -> float:
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            launch()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / rep

    def quantize() -> None:
        quantize_f32_q8_32[quant_grid](
            x, q8, q8_scales, q8_sums, k, num_warps=1)

    quant_ms = timed(quantize)
    legacy_grid = (triton.cdiv(rows, 8),)
    legacy_ms = timed(lambda: q4_ffn_swiglu_decode[legacy_grid](
        gate, gate.view(torch.float16), up, up.view(torch.float16),
        x, out, k, rows, BLOCK_M=8, num_warps=2))
    print(f"shape q4_q8_ffn rows={rows} k={k}")
    print(f"quantize_ms {quant_ms:.4f}")
    print(f"legacy_f32  {legacy_ms:.4f}")
    print("m warps gemv_ms total_ms speedup")
    for m in (1, 2, 4, 8, 16):
        for warps in (1, 2, 4):
            grid = (triton.cdiv(rows, m),)

            def fused() -> None:
                q4_ffn_swiglu_q8_dot4[grid](
                    gate, gate.view(torch.int32), gate.view(torch.float16),
                    up, up.view(torch.int32), up.view(torch.float16),
                    q8.view(torch.int32), q8_scales, q8_sums, out,
                    k, rows, BLOCK_M=m, num_warps=warps)

            quantize()
            fused_ms = timed(fused)

            def quantize_and_fused() -> None:
                quantize()
                fused()

            total_ms = timed(quantize_and_fused)
            print(f"{m:2d} {warps:5d} {fused_ms:8.4f} {total_ms:8.4f} "
                  f"{legacy_ms / total_ms:7.3f}x", flush=True)


def bench_grouped_f16(rows: int, columns: int, k: int, warmup: int, rep: int) -> None:
    """Evaluate FlagTree's AMD tutorial GEMM configurations on GGML layout."""
    device = torch.device("cuda")
    weights = torch.randn((rows, k), device=device, dtype=torch.float16)
    x = torch.randn((columns, k), device=device, dtype=torch.float32)
    output = torch.empty((columns, rows), device=device, dtype=torch.float32)
    configs = [
        (32, 32, 64, 6, 8, 2),
        (64, 32, 64, 4, 8, 2),
        (32, 64, 64, 6, 8, 2),
        (64, 64, 64, 6, 8, 2),
        (128, 64, 64, 4, 8, 2),
        (128, 128, 64, 4, 8, 2),
        (256, 128, 64, 4, 8, 2),
        (256, 256, 64, 6, 8, 2),
        (64, 128, 32, 8, 4, 2),
        (64, 128, 64, 8, 8, 2),
    ]
    print(f"shape grouped_f16 rows={rows} columns={columns} k={k}")
    print("m n bk group warps stages ms TFLOP/s")
    for m, n, bk, group, warps, stages in configs:
        grid = (triton.cdiv(rows, m) * triton.cdiv(columns, n),)
        try:
            for _ in range(warmup):
                f16_gemm_grouped[grid](
                    weights, x, output, k, rows, columns,
                    BLOCK_M=m, BLOCK_N=n, BLOCK_K=bk, GROUP_M=group,
                    num_warps=warps, num_stages=stages,
                    matrix_instr_nonkdim=16,
                )
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(rep):
                f16_gemm_grouped[grid](
                    weights, x, output, k, rows, columns,
                    BLOCK_M=m, BLOCK_N=n, BLOCK_K=bk, GROUP_M=group,
                    num_warps=warps, num_stages=stages,
                    matrix_instr_nonkdim=16,
                )
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end) / rep
            tflops = 2.0 * rows * columns * k / (ms * 1e9)
            print(f"{m:3d} {n:3d} {bk:2d} {group:5d} {warps:5d} {stages:6d} "
                  f"{ms:8.4f} {tflops:8.3f}", flush=True)
        except Exception as exc:
            print(f"{m:3d} {n:3d} {bk:2d} {group:5d} {warps:5d} {stages:6d} "
                  f"ERROR {exc}", flush=True)
    print("waves_per_eu sweep for 64x128x32, group=8, warps=4")
    for stages in (1, 2, 3):
        for waves_per_eu in (1, 2, 3, 4):
            grid = (triton.cdiv(rows, 64) * triton.cdiv(columns, 128),)
            try:
                for _ in range(warmup):
                    f16_gemm_grouped[grid](
                        weights, x, output, k, rows, columns,
                        BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, GROUP_M=8,
                        num_warps=4, num_stages=stages,
                        matrix_instr_nonkdim=16, waves_per_eu=waves_per_eu,
                    )
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(rep):
                    f16_gemm_grouped[grid](
                        weights, x, output, k, rows, columns,
                        BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, GROUP_M=8,
                        num_warps=4, num_stages=stages,
                        matrix_instr_nonkdim=16, waves_per_eu=waves_per_eu,
                    )
                end.record()
                end.synchronize()
                ms = start.elapsed_time(end) / rep
                tflops = 2.0 * rows * columns * k / (ms * 1e9)
                print(f"stages={stages} waves={waves_per_eu} ms={ms:.4f} "
                      f"TFLOP/s={tflops:.3f}", flush=True)
            except Exception as exc:
                print(f"stages={stages} waves={waves_per_eu} ERROR {exc}", flush=True)


def bench_grouped_ffn(rows: int, columns: int, k: int, warmup: int, rep: int) -> None:
    """Sweep grouped scheduling for the Qwen dual-projection SwiGLU tile."""
    device = torch.device("cuda")
    gate = torch.randn((rows, k), device=device, dtype=torch.float16)
    up = torch.randn((rows, k), device=device, dtype=torch.float16)
    x = torch.randn((columns, k), device=device, dtype=torch.float32)
    output = torch.empty((columns, rows), device=device, dtype=torch.float32)
    print(f"shape grouped_ffn rows={rows} columns={columns} k={k}")
    print("m n bk group warps stages waves ms TFLOP/s")
    configs = []
    for m, n, bk, warps in ((32, 64, 32, 2), (64, 128, 32, 8)):
        for group in (4, 8, 16):
            for stages in (1, 2):
                for waves in (1, 2, 3, 4):
                    configs.append((m, n, bk, group, warps, stages, waves))
    for m, n, bk, group, warps, stages, waves in configs:
        grid = (triton.cdiv(rows, m) * triton.cdiv(columns, n),)
        try:
            for _ in range(warmup):
                ffn_swiglu_grouped[grid](
                    gate, up, x, output, k, rows, columns,
                    BLOCK_M=m, BLOCK_N=n, BLOCK_K=bk, GROUP_M=group,
                    num_warps=warps, num_stages=stages,
                    matrix_instr_nonkdim=16, waves_per_eu=waves,
                )
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(rep):
                ffn_swiglu_grouped[grid](
                    gate, up, x, output, k, rows, columns,
                    BLOCK_M=m, BLOCK_N=n, BLOCK_K=bk, GROUP_M=group,
                    num_warps=warps, num_stages=stages,
                    matrix_instr_nonkdim=16, waves_per_eu=waves,
                )
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end) / rep
            tflops = 4.0 * rows * columns * k / (ms * 1e9)
            print(f"{m:3d} {n:3d} {bk:2d} {group:5d} {warps:5d} {stages:6d} "
                  f"{waves:5d} {ms:8.4f} {tflops:8.3f}", flush=True)
        except Exception as exc:
            print(f"{m:3d} {n:3d} {bk:2d} {group:5d} {warps:5d} {stages:6d} "
                  f"{waves:5d} ERROR {exc}", flush=True)


def _elapsed_ms(launch, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / rep


def bench_gdn_cache(
        token_counts: list[int], warmup: int, rep: int, rounds: int) -> None:
    """Compare model-shape GDN+copy with cache-write fusion in one HIP process."""
    torch.manual_seed(0)
    state_size = 128
    q_heads = 16
    heads = 32
    sequences = 1
    snapshot_count = 1
    scale = state_size ** -0.5
    decode_search_configs = (
        (1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1),
        (2, 2), (4, 2), (8, 4), (16, 4), (32, 4),
    )
    full_wave_configs = (
        (4, 4), (8, 8), (16, 8), (32, 8), (64, 8),
    )

    for tokens in token_counts:
        q = torch.randn(
            sequences, tokens, q_heads, state_size,
            device="cuda", dtype=torch.float32).mul_(0.02)
        k = torch.randn_like(q).mul_(0.02)
        v = torch.randn(
            sequences, tokens, heads, state_size,
            device="cuda", dtype=torch.float32).mul_(0.02)
        gate = torch.randn(
            sequences, tokens, heads, 1,
            device="cuda", dtype=torch.float32).mul_(0.02)
        beta = torch.sigmoid(torch.randn_like(gate))
        state = torch.randn(
            sequences, heads, state_size, state_size,
            device="cuda", dtype=torch.float32).mul_(0.02)
        attention_elements = state_size * heads * tokens * sequences
        snapshot_stride = state_size * state_size * heads * sequences
        direct_output = torch.empty(
            attention_elements + snapshot_stride,
            device="cuda", dtype=torch.float32)
        direct_cache = torch.empty(snapshot_stride, device="cuda", dtype=torch.float32)
        direct_grid = (heads, sequences, triton.cdiv(state_size, 4))

        def launch_direct() -> None:
            common.flagos_gated_delta_net_scalar_f32[direct_grid](
                q, k, v, gate, beta, state, direct_output,
                state_size, heads, tokens, sequences,
                q.stride(2), q.stride(1), q.stride(0),
                v.stride(2), v.stride(1), v.stride(0),
                beta.stride(2), beta.stride(1), beta.stride(0),
                q_heads, 1, snapshot_count, scale,
                BLOCK=state_size, COLS=4, num_warps=4)
            direct_cache.copy_(direct_output[attention_elements:])

        print(
            f"shape gdn state={state_size} q_heads={q_heads} heads={heads} "
            f"tokens={tokens} sequences={sequences}", flush=True)

        launch_direct()
        torch.cuda.synchronize()

        # Materialize and validate every candidate before timing any of them.
        # This keeps Triton compilation and first-launch setup outside the
        # samples. Store the closures and their tensors so Python does not
        # release a candidate allocation while later rounds still use it.
        candidates = []

        configs = full_wave_configs + (decode_search_configs if tokens == 1 else ())
        for cols, warps in configs:
            grid = (heads, sequences, triton.cdiv(state_size, cols))
            fused_output = None
            fused_cache = None
            cache_only_output = None
            cache_only_cache = None
            launch_fused = None
            launch_cache_only = None
            if (cols, warps) in full_wave_configs:
                fused_output = torch.empty_like(direct_output)
                fused_cache = torch.empty_like(direct_cache)

                def launch_fused(
                        cols=cols, warps=warps, grid=grid,
                        fused_output=fused_output, fused_cache=fused_cache) -> None:
                    common.flagos_gated_delta_net_scalar_f32_cache[grid](
                        q, k, v, gate, beta, state, fused_output, fused_cache,
                        state_size, heads, tokens, sequences,
                        q.stride(2), q.stride(1), q.stride(0),
                        v.stride(2), v.stride(1), v.stride(0),
                        beta.stride(2), beta.stride(1), beta.stride(0),
                        q_heads, 1, snapshot_count, scale, 0,
                        BLOCK=state_size, COLS=cols, num_warps=warps)

                launch_fused()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    fused_output, direct_output, rtol=3e-4, atol=3e-4)
                torch.testing.assert_close(
                    fused_cache, direct_cache, rtol=3e-4, atol=3e-4)
                cache_only_output = torch.full_like(direct_output, float("nan"))
                cache_only_cache = torch.empty_like(direct_cache)

                def launch_cache_only(
                        cols=cols, warps=warps, grid=grid,
                        cache_only_output=cache_only_output,
                        cache_only_cache=cache_only_cache) -> None:
                    common.flagos_gated_delta_net_scalar_f32_cache_only[grid](
                        q, k, v, gate, beta, state, cache_only_output, cache_only_cache,
                        state_size, heads, tokens, sequences,
                        q.stride(2), q.stride(1), q.stride(0),
                        v.stride(2), v.stride(1), v.stride(0),
                        beta.stride(2), beta.stride(1), beta.stride(0),
                        q_heads, 1, snapshot_count, scale, 0,
                        BLOCK=state_size, COLS=cols, EXACT_STATE_SIZE=state_size,
                        EXACT_SCALE=scale,
                        num_warps=warps)

                launch_cache_only()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    cache_only_output[:attention_elements],
                    direct_output[:attention_elements], rtol=3e-4, atol=3e-4)
                torch.testing.assert_close(
                    cache_only_cache, direct_cache, rtol=3e-4, atol=3e-4)
                if not torch.isnan(cache_only_output[attention_elements:]).all().item():
                    raise AssertionError("cache-only GDN wrote the temporary snapshot output")
            decode_output = None
            decode_cache = None
            launch_decode = None
            if tokens == 1:
                decode_output = torch.full_like(direct_output, float("nan"))
                decode_cache = torch.empty_like(direct_cache)

                def launch_decode(
                        cols=cols, warps=warps, grid=grid,
                        decode_output=decode_output,
                        decode_cache=decode_cache) -> None:
                    common.flagos_gated_delta_net_scalar_f32_cache_only_decode[grid](
                        q, k, v, gate, beta, state, decode_output, decode_cache,
                        state_size, heads, tokens, sequences,
                        q.stride(2), q.stride(1), q.stride(0),
                        v.stride(2), v.stride(1), v.stride(0),
                        beta.stride(2), beta.stride(1), beta.stride(0),
                        q_heads, 1, snapshot_count, scale, 0,
                        BLOCK=state_size, COLS=cols, EXACT_STATE_SIZE=state_size,
                        EXACT_SCALE=scale,
                        num_warps=warps)

                launch_decode()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    decode_output[:attention_elements],
                    direct_output[:attention_elements], rtol=3e-4, atol=3e-4)
                torch.testing.assert_close(
                    decode_cache, direct_cache, rtol=3e-4, atol=3e-4)
                if not torch.isnan(decode_output[attention_elements:]).all().item():
                    raise AssertionError(
                        "decode cache-only GDN wrote the temporary snapshot output")
            candidates.append({
                "cols": cols,
                "warps": warps,
                "full": launch_fused,
                "cache_only": launch_cache_only,
                "decode": launch_decode,
                "tensors": (
                    fused_output, fused_cache, cache_only_output, cache_only_cache,
                    decode_output, decode_cache),
            })

        benchmarks = [("direct", launch_direct)]
        for candidate in candidates:
            tag = (candidate["cols"], candidate["warps"])
            if candidate["full"] is not None:
                benchmarks.append((("full", *tag), candidate["full"]))
                benchmarks.append((("cache_only", *tag), candidate["cache_only"]))
            if candidate["decode"] is not None:
                benchmarks.append((("decode", *tag), candidate["decode"]))
        samples = {key: [] for key, _ in benchmarks}
        for round_index in range(rounds):
            offset = ((round_index // 2) * max(1, len(benchmarks) // 2)) % len(benchmarks)
            rotated = benchmarks[offset:] + benchmarks[:offset]
            ordered = rotated if round_index % 2 == 0 else list(reversed(rotated))
            for key, launch in ordered:
                samples[key].append(_elapsed_ms(
                    launch, warmup if round_index == 0 else 0, rep))

        direct_samples = samples["direct"]
        direct_ms = statistics.median(direct_samples)
        print(
            f"direct+c4-copy median_ms={direct_ms:.4f} "
            f"range=[{min(direct_samples):.4f},{max(direct_samples):.4f}] rounds={rounds}")
        header = (
            "cols warps full_median_ms full_range cache_only_median_ms "
            "cache_only_range full_speedup cache_only_speedup")
        print(header)
        for candidate in candidates:
            if candidate["full"] is None:
                continue
            cols = candidate["cols"]
            warps = candidate["warps"]
            full_samples = samples[("full", cols, warps)]
            cache_only_samples = samples[("cache_only", cols, warps)]
            fused_ms = statistics.median(full_samples)
            cache_only_ms = statistics.median(cache_only_samples)
            row = (
                f"{cols:4d} {warps:5d} {fused_ms:14.4f} "
                f"[{min(full_samples):.4f},{max(full_samples):.4f}] "
                f"{cache_only_ms:20.4f} "
                f"[{min(cache_only_samples):.4f},{max(cache_only_samples):.4f}] "
                f"{direct_ms / fused_ms:12.3f}x "
                f"{direct_ms / cache_only_ms:18.3f}x")
            print(row, flush=True)
        if tokens == 1:
            print("cols warps decode_median_ms decode_range decode_speedup")
            for candidate in candidates:
                cols = candidate["cols"]
                warps = candidate["warps"]
                decode_samples = samples[("decode", cols, warps)]
                decode_ms = statistics.median(decode_samples)
                print(
                    f"{cols:4d} {warps:5d} {decode_ms:16.4f} "
                    f"[{min(decode_samples):.4f},{max(decode_samples):.4f}] "
                    f"{direct_ms / decode_ms:14.3f}x", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=6144)
    parser.add_argument("--columns", type=int, default=512)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--ffn", action="store_true",
                        help="benchmark the fused dual-projection SwiGLU kernel")
    parser.add_argument("--quant", choices=("q4", "q6"),
                        help="benchmark production quantized GEMV at 1, 2, and 4 warps")
    parser.add_argument("--f16-gemv", action="store_true",
                        help="benchmark dense F16-weight GEMV row tiles")
    parser.add_argument("--q4-ffn-decode", action="store_true",
                        help="benchmark fused Q4_K gate/up GEMV plus SwiGLU")
    parser.add_argument("--q40-ffn-decode", action="store_true",
                        help="benchmark fused Q4_0 gate/up GEMV plus SwiGLU")
    parser.add_argument("--q4-ffn-staged", action="store_true",
                        help="compare validated and low-VGPR staged Q4 FFN kernels")
    parser.add_argument("--q40-ffn-staged", action="store_true",
                        help="compare validated and sequential-accumulator Q4_0 FFN kernels")
    parser.add_argument("--q40-q8-ffn", action="store_true",
                        help="compare Q4_0 FFN with a shared Q8 activation path")
    parser.add_argument("--q4-narrow8-warps", action="store_true",
                        help="compare the production Q4 narrow8 source at 1, 4, and 8 waves")
    parser.add_argument("--q40-narrow-tiles", action="store_true",
                        help="compare Q4_0 row tiles with the one-row GEMV")
    parser.add_argument("--decode-candidates", action="store_true",
                        help="screen staged FFN and Qwen Q4 narrow8 wave candidates in one process")
    parser.add_argument("--qwen35-candidates", action="store_true",
                        help="screen Q4_0 FFN/GEMV and GDN candidates in one process")
    parser.add_argument("--q6-q8", action="store_true",
                        help="benchmark F32-to-Q8 plus RDNA4 Q6_K int8 dot4 GEMV")
    parser.add_argument("--q4-q8-ffn", action="store_true",
                        help="benchmark shared Q8 plus fused Q4_K gate/up/SwiGLU")
    parser.add_argument("--grouped-f16", action="store_true",
                        help="benchmark AMD tutorial-style grouped F16 GEMM")
    parser.add_argument("--grouped-ffn", action="store_true",
                        help="benchmark grouped fused F16 FFN/SwiGLU")
    parser.add_argument("--gdn-cache", action="store_true",
                        help="compare GDN+copy with cache-write fusion tiles")
    parser.add_argument("--gdn-tokens", action="append", type=int,
                        help="token count for --gdn-cache (repeatable; default: 1 and 512)")
    parser.add_argument("--gdn-rounds", type=int, default=3,
                        help="alternating timing rounds for --gdn-cache (default: 3)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("HIP device is unavailable")
    if args.gdn_tokens is not None and any(tokens <= 0 for tokens in args.gdn_tokens):
        parser.error("--gdn-tokens values must be positive")
    if args.gdn_rounds <= 0:
        parser.error("--gdn-rounds must be positive")
    if args.gdn_cache:
        bench_gdn_cache(
            args.gdn_tokens or [1, 512], args.warmup, args.rep, args.gdn_rounds)
        return
    if args.qwen35_candidates:
        bench_q40_ffn_staged(args.rows, args.k, args.warmup, args.rep)
        bench_q40_q8_ffn(args.rows, args.k, args.warmup, args.rep)
        bench_q40_narrow_qwen35(args.warmup, args.rep)
        bench_gdn_cache(
            args.gdn_tokens or [1, 512], args.warmup, args.rep, args.gdn_rounds)
        return
    if args.decode_candidates:
        bench_q4_ffn_staged(args.rows, args.k, args.warmup, args.rep)
        bench_q4_narrow8_warps((1024, 2048), args.k, args.warmup, args.rep)
        return
    if args.quant:
        bench_quant(args.rows, args.k, args.quant, args.warmup, args.rep)
        return
    if args.f16_gemv:
        bench_f16_gemv(args.rows, args.k, args.warmup, args.rep)
        return
    if args.q4_ffn_decode:
        bench_q4_ffn_decode(args.rows, args.k, args.warmup, args.rep)
        return
    if args.q40_ffn_decode:
        bench_q40_ffn_decode(args.rows, args.k, args.warmup, args.rep)
        return
    if args.q4_ffn_staged:
        bench_q4_ffn_staged(args.rows, args.k, args.warmup, args.rep)
        return
    if args.q40_ffn_staged:
        bench_q40_ffn_staged(args.rows, args.k, args.warmup, args.rep)
        return
    if args.q40_q8_ffn:
        bench_q40_q8_ffn(args.rows, args.k, args.warmup, args.rep)
        return
    if args.q4_narrow8_warps:
        bench_q4_narrow8_warps((args.rows,), args.k, args.warmup, args.rep)
        return
    if args.q40_narrow_tiles:
        bench_q40_narrow_tiles(args.rows, args.k, args.warmup, args.rep)
        return
    if args.q6_q8:
        bench_q6_q8(args.rows, args.k, args.warmup, args.rep)
        return
    if args.q4_q8_ffn:
        bench_q4_q8_ffn(args.rows, args.k, args.warmup, args.rep)
        return
    if args.grouped_f16:
        bench_grouped_f16(args.rows, args.columns, args.k, args.warmup, args.rep)
        return
    if args.grouped_ffn:
        bench_grouped_ffn(args.rows, args.columns, args.k, args.warmup, args.rep)
        return
    candidates = [
        (32, 64, 32, 2),
        (32, 128, 32, 2),
        (64, 64, 32, 2),
        (64, 128, 32, 4),
        (64, 128, 32, 8),
        (128, 128, 32, 8),
        (64, 256, 32, 8),
        (128, 64, 32, 8),
        (64, 128, 16, 4),
        (32, 64, 64, 4),
        (32, 64, 64, 2),
        (64, 64, 64, 4),
        (128, 64, 32, 4),
    ]
    print(f"shape rows={args.rows} columns={args.columns} k={args.k}")
    print("m n bk warps ms TFLOP/s")
    for m, n, bk, warps in candidates:
        try:
            ms = bench(args.rows, args.columns, args.k, m, n, bk, warps,
                       args.warmup, args.rep, args.ffn)
            flop_factor = 4.0 if args.ffn else 2.0
            tflops = flop_factor * args.rows * args.columns * args.k / (ms * 1e9)
            print(f"{m:3d} {n:3d} {bk:2d} {warps:5d} {ms:8.4f} {tflops:8.3f}", flush=True)
        except Exception as exc:
            print(f"{m:3d} {n:3d} {bk:2d} {warps:5d} ERROR {exc}", flush=True)


if __name__ == "__main__":
    main()
